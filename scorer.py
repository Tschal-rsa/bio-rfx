import config
import torch
from torch.nn import MSELoss
import numpy as np
import random
from entity_models import NMS
from preprocess.const import task_tup_limits, task_ent_labels
from utils import compute_f1_and_auc

seed = config.train_config["seed"]
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def evaluate_entity_aware(model, data, ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, rel_tuples, sample_indices, rel_types, batch_size, device, tokenizer, dataset) -> tuple:
	model_config = config.model["entity_detector"]
	complex_mode: bool
	if model_config["ent_cls_mode"] == "complex":
		complex_mode = True
	elif model_config["ent_cls_mode"] == "simple":
		complex_mode = False
	else:
		raise NotImplementedError

	def _search_backward(vector, ent_start, coord_start, targets) -> list:
		tuples = []
		for i in range(ent_start - 1, -1, -1):
			ent_id, coord = vector[i]
			if ent_id in targets:
				tuples.append((coord_start, coord))
			# elif ent_id in [1, 2, 3]:
			# 	break
		return tuples

	def _search_forward(vector, ent_start, coord_start, targets) -> list:
		tuples = []
		for i in range(ent_start + 1, len(vector)):
			ent_id, coord = vector[i]
			if ent_id in targets:
				tuples.append((coord_start, coord))
			# elif ent_id in [1, 2, 3]:
			# 	break
		return tuples

	def _search(vector, rel_type) -> list:
		'''
		- vectors: list of tuples after NMS. [(ent_id, coord) * k].
		- ent_id: entity type, either ENT_GENE or ENT_CHEM.
		- coord: index of span. coord_1 < coord_2 <=> (start_1 < start_2) || (start_1 == start_2 && end_1 < end_2)
		- k: predicted number for subjects and objects.
		'''
		tuples = []
		if complex_mode:
			for i in range(len(vector)):
				ent_id, coord = vector[i]
				if ent_id in [1, 3]:
					tuples.extend(_search_backward(vector, i, coord, [5, 6]))
				if ent_id in [2, 3]:
					tuples.extend(_search_forward(vector, i, coord, [4, 6]))
		else:
			sbj_targets = task_tup_limits[dataset][rel_type]['sbj_targets']
			obj_targets = task_tup_limits[dataset][rel_type]['obj_targets']
			for i in range(len(vector)):
				ent_id, coord = vector[i]
				if ent_id in sbj_targets:
					tuples.extend(_search_backward(vector, i, coord, obj_targets))
					tuples.extend(_search_forward(vector, i, coord, obj_targets))
		return tuples
	
	tag_chemical, tag_gene = 'CHEMICAL', 'GENE'
	def _search_entity(vector, mask, score=None, max_cnt=None, coord2span=None) -> list:
		entities, exact_entities = [], []
		arg_vector = np.argwhere(vector * mask)[0].tolist()
		if score is not None:
			pred_entities = [(score[i], i) for i in arg_vector]
			arg_vector = NMS(pred_entities, max_cnt + 1, coord2span)
			# arg_vector = NMS(pred_entities, 13, coord2span)
		if complex_mode:
			for i in arg_vector:
				ent_id = vector[i]
				exact_entities.append((ent_id, i))
				if ent_id in [1, 2, 3]:
					entities.append((tag_chemical, i))
				elif ent_id in [4, 5, 6]:
					entities.append((tag_gene, i))
		else:
			for i in arg_vector:
				ent_id = vector[i]
				exact_entities.append((ent_id, i))
				entities.append((task_ent_labels[dataset][ent_id - 1], i))
		return entities, exact_entities

	model.eval()
	all_input_ids = [i["input_ids"] for i in data]
	all_attn_mask = [i["attention_mask"] for i in data]
	all_token_type_ids = [i["token_type_ids"] for i in data]
	st, ed = 0, 0
	all_loss, all_f1 = [], []
	true_pos, pred_len, target_len = [], [], []
	ent_true_len, ent_pred_len, ent_target_len = [], [], []
	id2ent = {}
	while ed < len(data):
		st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
		with torch.no_grad():
			input_ids = torch.tensor(all_input_ids[st:ed]).to(device)
			attn_mask = torch.tensor(all_attn_mask[st:ed]).to(device)
			token_type_ids = torch.tensor(all_token_type_ids[st:ed]).to(device)
			spans = torch.tensor(ent_spans[st:ed]).to(device)
			widths = torch.tensor(ent_widths[st:ed]).to(device)
			scores = torch.tensor(ent_scores[st:ed]).to(device)
			labels = torch.tensor(ent_labels[st:ed]).to(device)
			masks = torch.tensor(loss_masks[st:ed]).to(device)
			counts = ent_counts[st:ed]
			tuples = rel_tuples[st:ed]
			indices = sample_indices[st:ed]
			rtypes = rel_types[st:ed]
			outputs = model(input_ids=input_ids, attn_mask=attn_mask, token_type_ids=token_type_ids, ent_spans=spans, ent_widths=widths, ent_scores=scores, ent_labels=labels, loss_masks=masks)
				
			# logits: [B, N, 7], scores: [B, N]
			loss, logits, scores = outputs["loss"], outputs["logits"], outputs["scores"]
			all_loss.append(loss.cpu().numpy())

			# scores, logits = logits.max(dim=-1)
			# scores = 1 - logits[:, :, 0]
			scores = scores[:, :, 1]
			logits = logits.argmax(dim=-1) # [B, N]
			pred_tuples = []
			for input_id, span, score, logit, label, mask, count, true_tup, sample_idx, rtype in zip(input_ids, spans, scores.cpu(), logits.cpu(), labels.cpu(), masks.cpu(), counts, tuples, indices, rtypes):
				if sample_idx not in id2ent.keys():
					id2ent[sample_idx] = {
						"true_ent": set(),
						"pred_ent": set()
					}
				true_ent = id2ent[sample_idx]["true_ent"]
				pred_ent = id2ent[sample_idx]["pred_ent"]
				true_ent_list, true_ex_ent_list = _search_entity(label, mask)
				pred_ent_list, pred_ex_ent_list = _search_entity(logit, mask, score, count, span)
				true_ent.update(true_ent_list)
				pred_ent.update(pred_ent_list)
				pred_tup = _search(pred_ex_ent_list, rtype)
				true_pos.append(len(set(true_tup) & set(pred_tup)))
				pred_len.append(len(pred_tup))
				target_len.append(len(true_tup))
				pred_tuples.append(pred_tup)
				if not config.common["do_train"]:
					# print(input_ids[0].size(), masks[0].size())
					sentence = input_id
					print(tokenizer.decode(sentence, skip_special_tokens=True))
					# print(logit.tolist()[138:172])
					# print(label.tolist()[138:172])
					print('=' * 22, 'TRUE', '=' * 22)
					for tup in true_tup:
						print(tup)
						sbj_tok_span_adj = span[tup[0]]
						obj_tok_span_adj = span[tup[1]]
						print(
							tokenizer.decode(sentence[sbj_tok_span_adj[0]: sbj_tok_span_adj[1] + 1]),
							'<=>',
							tokenizer.decode(sentence[obj_tok_span_adj[0]: obj_tok_span_adj[1] + 1])
						)
					for true_ent in true_ent_list:
						tok_span_adj = span[true_ent[1]]
						print(true_ent, tokenizer.decode(sentence[tok_span_adj[0]: tok_span_adj[1] + 1]))
					print('=' * 22, 'PRED', '=' * 22)
					for tup in pred_tup:
						print(tup)
						sbj_tok_span_adj = span[tup[0]]
						obj_tok_span_adj = span[tup[1]]
						print(
							tokenizer.decode(sentence[sbj_tok_span_adj[0]: sbj_tok_span_adj[1] + 1]),
							'<=>',
							tokenizer.decode(sentence[obj_tok_span_adj[0]: obj_tok_span_adj[1] + 1])
						)
					for pred_ent in pred_ent_list:
						tok_span_adj = span[pred_ent[1]]
						print(pred_ent, tokenizer.decode(sentence[tok_span_adj[0]: tok_span_adj[1] + 1]))
					print()


	loss = np.mean(all_loss)
	print('=' * 22, 'Triple', '=' * 22)
	print(true_pos[:20])
	print(pred_len[:20])
	print(target_len[:20])
	precision = np.sum(true_pos) / np.sum(pred_len)
	recall = np.sum(true_pos) / np.sum(target_len)
	f1 = 2 * np.sum(true_pos) / (np.sum(pred_len) + np.sum(target_len))
	print(f"P:  {precision}\nR:  {recall}\nF1: {f1}")

	for sample in id2ent.values():
		true_ent, pred_ent = sample["true_ent"], sample["pred_ent"]
		ent_target_len.append(len(true_ent))
		ent_pred_len.append(len(pred_ent))
		ent_true_len.append(len(true_ent & pred_ent))
	print('=' * 22, 'Entity', '=' * 22)
	print(ent_true_len[:20])
	print(ent_pred_len[:20])
	print(ent_target_len[:20])
	ent_precision = np.sum(ent_true_len) / np.sum(ent_pred_len)
	ent_recall = np.sum(ent_true_len) / np.sum(ent_target_len)
	ent_f1 = 2 * np.sum(ent_true_len) / (np.sum(ent_pred_len) + np.sum(ent_target_len))
	print(f"P:  {ent_precision}\nR:  {ent_recall}\nF1: {ent_f1}")

	model.train()
	return loss, f1, ent_f1


def evaluate_entity_blind(model, data, ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, sample_indices, batch_size, device, tokenizer, dataset) -> tuple:
	model_config = config.model["entity_detector"]
	complex_mode: bool
	if model_config["ent_cls_mode"] == "complex":
		complex_mode = True
	elif model_config["ent_cls_mode"] == "simple":
		complex_mode = False
	else:
		raise NotImplementedError

	tag_chemical, tag_gene = 'CHEMICAL', 'GENE'
	def _search_entity(vector, mask, score=None, max_cnt=None, coord2span=None) -> list:
		entities, exact_entities = [], []
		arg_vector = np.argwhere(vector * mask)[0].tolist()
		if score is not None:
			pred_entities = [(score[i], i) for i in arg_vector]
			arg_vector = NMS(pred_entities, max_cnt + 1, coord2span)
			# arg_vector = NMS(pred_entities, 6, coord2span)
		if complex_mode:
			for i in arg_vector:
				ent_id = vector[i]
				exact_entities.append((ent_id, i))
				if ent_id in [1, 2, 3]:
					entities.append((tag_chemical, i))
				elif ent_id in [4, 5, 6]:
					entities.append((tag_gene, i))
		else:
			for i in arg_vector:
				ent_id = vector[i]
				exact_entities.append((ent_id, i))
				entities.append((task_ent_labels[dataset][ent_id - 1], i))
		return entities, exact_entities

	model.eval()
	all_input_ids = [i["input_ids"] for i in data]
	all_attn_mask = [i["attention_mask"] for i in data]
	all_token_type_ids = [i["token_type_ids"] for i in data]
	st, ed = 0, 0
	all_loss = []
	true_pos, pred_len, target_len = [], [], []
	ent_true_len, ent_pred_len, ent_target_len = [], [], []
	id2ent = {}
	while ed < len(data):
		st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
		with torch.no_grad():
			input_ids = torch.tensor(all_input_ids[st:ed]).to(device)
			attn_mask = torch.tensor(all_attn_mask[st:ed]).to(device)
			token_type_ids = torch.tensor(all_token_type_ids[st:ed]).to(device)
			spans = torch.tensor(ent_spans[st:ed]).to(device)
			widths = torch.tensor(ent_widths[st:ed]).to(device)
			scores = torch.tensor(ent_scores[st:ed]).to(device)
			labels = torch.tensor(ent_labels[st:ed]).to(device)
			masks = torch.tensor(loss_masks[st:ed]).to(device)
			counts = ent_counts[st:ed]
			indices = sample_indices[st:ed]
			outputs = model(input_ids=input_ids, attn_mask=attn_mask, token_type_ids=token_type_ids, ent_spans=spans, ent_widths=widths, ent_scores=scores, ent_labels=labels, loss_masks=masks)
				
			# logits: [B, N, 7], scores: [B, N]
			loss, logits, scores = outputs["loss"], outputs["logits"], outputs["scores"]
			all_loss.append(loss.cpu().numpy())

			# scores, logits = logits.max(dim=-1)
			scores = 1 - logits[:, :, 0]
			# scores = scores[:, :, 1]
			logits = logits.argmax(dim=-1) # [B, N]
			for input_id, span, score, logit, label, mask, count, sample_idx in zip(input_ids, spans, scores.cpu(), logits.cpu(), labels.cpu(), masks.cpu(), counts, indices):
				if sample_idx not in id2ent.keys():
					id2ent[sample_idx] = {
						"true_ent": set(),
						"pred_ent": set()
					}
				true_ent = id2ent[sample_idx]["true_ent"]
				pred_ent = id2ent[sample_idx]["pred_ent"]
				true_ent_list, true_ex_ent_list = _search_entity(label, mask)
				pred_ent_list, pred_ex_ent_list = _search_entity(logit, mask, score, count, span)
				true_ent.update(true_ent_list)
				pred_ent.update(pred_ent_list)

	loss = np.mean(all_loss)

	print('=' * 22, 'Entity', '=' * 22)
	for sample in id2ent.values():
		true_ent, pred_ent = sample["true_ent"], sample["pred_ent"]
		ent_target_len.append(len(true_ent))
		ent_pred_len.append(len(pred_ent))
		ent_true_len.append(len(true_ent & pred_ent))

	print(ent_true_len[:20])
	print(ent_pred_len[:20])
	print(ent_target_len[:20])
	ent_precision = np.sum(ent_true_len) / np.sum(ent_pred_len)
	ent_recall = np.sum(ent_true_len) / np.sum(ent_target_len)
	ent_f1 = 2 * np.sum(ent_true_len) / (np.sum(ent_pred_len) + np.sum(ent_target_len))
	print(f"P:  {ent_precision}\nR:  {ent_recall}\nF1: {ent_f1}")

	model.train()
	return loss, None, ent_f1


def evaluate_relation(model, data, rel_labels, batch_size, device, tokenizer) -> tuple:
	model.eval()
	all_input_ids = [i["input_ids"] for i in data]
	all_attn_mask = [i["attention_mask"] for i in data]
	st, ed = 0, 0
	all_predictions = None
	all_loss, all_auc = [], []
	rel_num = len(rel_labels[0])
	while ed < len(data):
		st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
		with torch.no_grad():
			input_ids = torch.tensor(all_input_ids[st:ed]).to(device)
			attn_mask = torch.tensor(all_attn_mask[st:ed]).to(device)
			labels = torch.tensor(rel_labels[st:ed]).to(device)
			outputs = model(input_ids=input_ids, attn_mask=attn_mask, rel_labels=labels)
				
			loss, logits = outputs["loss"], outputs["logits"]
			all_loss.append(loss.cpu().numpy())

			thresholds = []
			for i in range(rel_num):
				label = np.array(labels[:,i].cpu()).astype(int)
				prob  = np.array(logits[:,i].cpu()).astype(float)
				f1, auc, t = compute_f1_and_auc(y_prob=prob, y_true=label)

				if auc is not None:
					all_auc.append(auc)
				thresholds.append(t)

			probs = np.array(logits.cpu()).astype(float)
			preds =  np.where(probs >= thresholds, 1, 0)

			if all_predictions is None:
				all_predictions = preds
			else:
				all_predictions = np.concatenate([all_predictions, preds], axis=0)

			if st == 0:
				end_idx = np.where(input_ids[0].cpu().numpy()==config.bert_config["sep_token"])[0][0]
				sentence = input_ids[0][:end_idx+1]
				print("Sentence: ")
				print(tokenizer.decode(sentence))
				print('=' * 22, 'PRED', '=' * 22)
				print(logits[0].view(-1))
				print('=' * 22, 'TRUE', '=' * 22)
				print(labels[0].view(-1))

	loss = np.mean(all_loss)
	auc = np.mean(all_auc)
	model.train()
	print("auc: ", auc)
	rel_labels = np.array(rel_labels)
	tp = np.count_nonzero(rel_labels * all_predictions)
	fp = np.count_nonzero((1-rel_labels) * all_predictions)
	fn = np.count_nonzero((rel_labels * (1-all_predictions)))
	tn = np.count_nonzero((1-rel_labels) * (1-all_predictions))
	print("Relation Classification: ")
	print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
	precision = tp / (tp  + fp)
	recall  = tp / (tp + fn)
	f1 = 2*tp / (2*tp + fp + fn)
	print(f"P:  {precision}\nR:  {recall}\nF1: {f1}")
	return loss, f1, all_predictions


def evaluate_count(model, data, batch_size, device, tokenizer) -> tuple:
	FACTOR = 1

	model.eval()
	all_input_ids = [i["tokens"]["input_ids"] for i in data]
	all_attn_mask = [i["tokens"]["attention_mask"] for i in data]
	all_token_type_ids = [i["tokens"]["token_type_ids"] for i in data]
	all_cnt_label = [i["count"] for i in data]
	st, ed = 0, 0
	all_loss, all_mse_score = [], []
	predictions = None
	mse_loss = MSELoss()
	while ed < len(data):
		st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
		with torch.no_grad():
			input_ids = torch.tensor(all_input_ids[st:ed]).to(device)
			attn_mask = torch.tensor(all_attn_mask[st:ed]).to(device)
			token_type_ids = torch.tensor(all_token_type_ids[st:ed]).to(device)
			labels = torch.tensor(all_cnt_label[st:ed]).to(device)
			outputs = model(input_ids=input_ids, attn_mask=attn_mask, token_type_ids=token_type_ids, cnt_label=labels)
				
			loss, logits = outputs["loss"], outputs["logits"]
			all_loss.append(loss.cpu().numpy())

			preds = torch.round(logits*FACTOR).to(torch.float)
			mse_score = mse_loss(preds, FACTOR*labels.reshape(-1,1).to(torch.float)).cpu().numpy()
			all_mse_score.append(mse_score)
			if predictions is None:
				predictions = preds.view(-1).to(torch.int)
			else:
				predictions = torch.cat([predictions, preds.view(-1).to(torch.int)], dim=-1)

			if st == 0:
				attn_mask = torch.where(attn_mask == 1, True, False)
				sentence = input_ids[0][attn_mask[0]]
				print("Sentence: ")
				print(tokenizer.decode(sentence))
				print('=' * 22, 'PRED', '=' * 22)
				print(logits.view(-1))
				print(preds.view(-1).to(torch.int))
				print('=' * 22, 'TRUE', '=' * 22)
				print(((FACTOR*labels).to(torch.int)).view(-1))
				

	loss = np.mean(all_loss)
	mse_score = np.mean(all_mse_score)
	model.train()
	print("mse: ", mse_score)
	return loss, mse_score, predictions
