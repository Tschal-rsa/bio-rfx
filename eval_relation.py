import config
from tqdm import tqdm
import torch
import torch.nn as nn
import json
import numpy as np
from transformers import AutoTokenizer

import os
from entity_models import NMS
from preprocess.umls import append_marker_tokens
from preprocess.const import task_umls_rels, task_umls_ent_labels, task_tup_limits, task_rel_labels, task_ent_labels, get_labelmap, get_shifted_labelmap
from utils import compute_f1_and_auc, generate_quesiton_and_context
import argparse

def load_relation_data(data_base_path, tokenizer, dataset, field_list=["dev"], maxlen=512):
	encoded_dict, relation_labels = {}, {}
	rel2id, _ = get_labelmap(task_rel_labels[dataset])
	for name in field_list:
		encoded_dict[name], relation_labels[name] = [], []
		input_file_path = os.path.join(data_base_path, "%s"%(dataset), "%s.json"%(name))
		input_data = json.load(open(input_file_path, "r", encoding = "utf-8"))
		for sample in tqdm(input_data, desc="Loading data from %s" % (input_file_path)):
			tokens = tokenizer.encode_plus(sample["text"], 
										max_length=maxlen,
										padding="max_length",
										truncation=True)
			
			rel_label = [0] * len(rel2id)
			for rel in sample["relation_list"]:
				rel_id = rel2id[rel["rel_type"]]
				rel_label[rel_id] = 1
			
			encoded_dict[name].append(tokens)
			relation_labels[name].append(rel_label)

	return encoded_dict, relation_labels

def predict_relation(model, data, rel_labels, batch_size, device):
	model.eval()
	all_input_ids = [i["input_ids"] for i in data]
	all_attn_mask = [i["attention_mask"] for i in data]
	all_predictions = None
	rel_num = len(rel_labels[0])
	all_f1 = []
	st, ed = 0, 0
	while ed < len(data):
		st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
		with torch.no_grad():
			input_ids = torch.tensor(all_input_ids[st:ed]).to(device)
			attn_mask = torch.tensor(all_attn_mask[st:ed]).to(device)
			labels = torch.tensor(rel_labels[st:ed]).to(device)
			outputs = model(input_ids=input_ids, attn_mask=attn_mask, rel_labels=labels)
				
			logits = outputs["logits"]

			labels =  np.array(labels.cpu()).astype(int)
			probs = np.array(logits.cpu()).astype(float)

			thresholds = []
			for i in range(rel_num):
				f1, auc, threshold = compute_f1_and_auc(y_prob=probs[:,i], y_true=labels[:,i])
				thresholds.append(threshold)
				if f1 is not None: all_f1.append(f1)

			preds =  np.where(probs >= thresholds, 1, 0)

			if all_predictions is None:
				all_predictions = preds
			else:
				all_predictions = np.concatenate([all_predictions, preds], axis=0)

	
	rel_labels = np.array(rel_labels)
	tp = np.count_nonzero(rel_labels * all_predictions)
	fp = np.count_nonzero((1-rel_labels) * all_predictions)
	fn = np.count_nonzero((rel_labels * (1-all_predictions)))
	tn = np.count_nonzero((1-rel_labels) * (1-all_predictions))
	f1 = np.mean(all_f1)
	print("Relation Classification: ")
	print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
	precision = tp / (tp  + fp)
	recall  = tp / (tp + fn)
	f1 = 2*tp / (2*tp + fp + fn)
	print(f"P:  {precision}\nR:  {recall}\nF1: {f1}")
	return all_predictions


def make_relation_list(data_base_path, dataset, pred_rel, field_list=["dev"]):
	relation_list, false_negs = {}, {}
	_, id2rel = get_labelmap(task_rel_labels[dataset])

	for name in field_list:
		relation_list[name], false_negs[name] = [], []
		input_file_path = os.path.join(data_base_path, dataset, f"{name}.json")
		input_data = json.load(open(input_file_path, "r", encoding="utf-8"))
		relation_hit_rate = []

		for sample_id, sample in enumerate(tqdm(input_data, desc=f"Loading data from {input_file_path}")):
			pred_relation_list = [id2rel[i] for i in np.where(pred_rel[sample_id] == 1)[0]]
			relation_set = set(rel["rel_type"] for rel in sample["relation_list"])
			relation_set.update(pred_relation_list)
			rlist, fnegs = zip(*((rel, rel not in pred_relation_list) for rel in relation_set))
			relation_list[name].append(rlist)
			false_negs[name].append(fnegs)

			hit_rel_cnt = 0
			for rel in sample["relation_list"]:
				rel_type = rel["rel_type"]
				if rel_type in pred_relation_list:
					hit_rel_cnt += 1
			
			relation_hit_rate.append(hit_rel_cnt/len(sample["relation_list"]))
		
		print("Average relation hit rate: ", np.mean(relation_hit_rate))
	
	return relation_list, false_negs

def load_count_data(data_base_path, tokenizer, dataset, rel_list, false_negs, split_sentence, field_list=["dev"], maxlen=512):
	'''
	Return values:
	 - encode_dict: Dictionary. Data and labels.
	   {
			"train":
			[
				{
					"tokens": Result for tokenzier().
					"count": (number of subjects and objects) / FACTOR. Label for regression model.
					"mask": To mask question.
				},
				...
			],
			"dev": [...]
	   }
	 - split_map: List (if split_sentence = True), else None.
	 			  If the `i`th sample is splitted into `c` sentences, then split_map[i] = c.
	'''
	FACTOR = 1
	
	def _clip(left, right):
		"""
		Clip the numbers in a pair to maxlen - 1
		"""
		if left >= maxlen:
			left = maxlen - 1
		if right >= maxlen:
			right = maxlen - 1
		return (left, right)

	encoded_dict, subj_and_obj_cnt = {}, {}
	for name in field_list:
		encoded_dict[name], subj_and_obj_cnt[name] = [], []
		input_file_path = os.path.join(data_base_path, "%s"%(dataset), "%s.json"%(name))
		input_data = json.load(open(input_file_path, "r", encoding = "utf-8"))
		sep_token = config.bert_config["sep_token"]

		for sample_id, sample in enumerate(tqdm(input_data, desc="Loading data from %s" % (input_file_path))):
			
			# Count subjects and objects			
			rel2id, _ = get_labelmap(rel_list[name][sample_id])

			subj_and_obj = [set() for _ in rel_list[name][sample_id]]
			for rel in sample["relation_list"]:
				rel_id = rel2id[rel["rel_type"]]
				entity_set = subj_and_obj[rel_id]
				entity_set.add(_clip(rel["obj_tok_span"][0], rel["obj_tok_span"][1]))
				entity_set.add(_clip(rel["sbj_tok_span"][0], rel["sbj_tok_span"][1]))

			# Construct relation-specific tokens
			for rel_type in rel_list[name][sample_id]:
				tokens = tokenizer(
					# text=generate_quesiton(rel_type, False),
					text="How many triplets exist for relation type " + task_umls_rels[dataset][rel_type] + " ?" if config.common["natural_questions"] else rel_type,
					text_pair=sample["text"],
					max_length=maxlen,
					padding="max_length",
					truncation=True,
					return_special_tokens_mask = True
				)
				offset = tokens["input_ids"].index(sep_token) + 1
				length = tokens["input_ids"].index(sep_token, offset) - offset
				remain = maxlen - offset - length
				rel_id = rel2id[rel_type]
				encoded_dict[name].append({
					"tokens": tokens,
					"count": len(subj_and_obj[rel_id]) / FACTOR,
					"mask": [False] * offset + [True] * length + [False] * remain,
				})

			
	sentence = torch.tensor(encoded_dict["dev"][0]["tokens"]["input_ids"])
	mask = torch.tensor(encoded_dict["dev"][0]["mask"])
	print(tokenizer.decode(sentence[mask]))
	print(encoded_dict["dev"][0]["count"])
	return encoded_dict

def evaluate_count(model, data, batch_size, device, tokenizer):
	FACTOR = 1

	model.eval()
	all_input_ids = [i["tokens"]["input_ids"] for i in data]
	all_attn_mask = [i["tokens"]["attention_mask"] for i in data]
	all_token_type_ids = [i["tokens"]["token_type_ids"] for i in data]
	all_cnt_label = [i["count"] for i in data]
	st, ed = 0, 0
	all_loss, all_mse_score = [], []
	predictions = None
	mse_loss = nn.MSELoss()
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
				predictions = preds.view(-1).to(torch.int).cpu()
			else:
				predictions = torch.cat([predictions, preds.view(-1).to(torch.int).cpu()], dim=-1)

			# valid_rel_mask = labels.reshape(-1,1) > 0
			# valid_rel_label = labels.reshape(-1,1)[valid_rel_mask]
			# valid_rel_pred = preds[valid_rel_mask]
			# if len(valid_rel_label) > 0:
			# 	mse_score = mse_loss(valid_rel_pred, valid_rel_label.to(torch.float)).cpu().numpy()
			# 	all_valid_rel_mse.append(mse_score)

			if st == 0:
				attn_mask = torch.where(attn_mask == 1, True, False)
				sentence = input_ids[0][attn_mask[0]]
				print("Sentence: ")
				print(tokenizer.decode(sentence))
				print('=' * 22, 'PRED', '=' * 22)
				print(logits.view(-1))
				print(preds.view(-1).to(torch.int))
				print('=' * 22, 'TRUE', '=' * 22)
				print((FACTOR*labels).view(-1))
			
	print("mse: ", mse_score)
	# print("non-zero relation mse: ", np.mean(all_valid_rel_mse))
	return predictions

def load_entity_data(data_base_path, tokenizer, dataset, rel_list, false_negs, field_list=["dev"], maxlen=512):
	model_config = config.model["entity_detector"]
	complex_mode: bool
	if model_config["ent_cls_mode"] == "complex":
		complex_mode = True
	elif model_config["ent_cls_mode"] == "simple":
		complex_mode = False
	else:
		raise NotImplementedError

	def _clip(left, right):
		if left >= maxlen:
			left = maxlen - 1
		if right >= maxlen:
			right = maxlen - 1
		return (left, right)

	CP_LEFT, CP_RIGHT = 0, 1
	def _get_class(ori_cls, ent_id, cp_loc):
		new_cls = 0
		if ori_cls == 0:
			new_cls = ent_id * 3 - 2 if cp_loc == CP_LEFT else ent_id * 3 - 1
		elif ori_cls == ent_id * 3 - 2:
			new_cls = ent_id * 3 - 2 if cp_loc == CP_LEFT else ent_id * 3
		elif ori_cls == ent_id * 3 - 1:
			new_cls = ent_id * 3 if cp_loc == CP_LEFT else ent_id * 3 - 1
		elif ori_cls == ent_id * 3:
			new_cls = ent_id * 3
		return new_cls

	if config.common["add_umls_marker"]:
		append_marker_tokens(tokenizer, task_umls_ent_labels[dataset])	
	
	encoded_dict, entity_spans, entity_widths, entity_scores, entity_labels, loss_masks, entity_counts, relation_tuples, sample_indices, rel_types, sample_false_negs = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
	ent2id, _ = get_shifted_labelmap(task_ent_labels[dataset])
	sep_token = config.bert_config["sep_token"]
	model_config["ent_cls_num"] = 1 + len(task_ent_labels[dataset]) * (3 if complex_mode else 1)
	
	for name in field_list:
		encoded_dict[name], entity_spans[name], entity_widths[name], entity_scores[name], entity_labels[name], loss_masks[name], entity_counts[name], relation_tuples[name], sample_indices[name], rel_types[name], sample_false_negs[name] = [], [], [], [], [], [], [], [], [], [], []
		input_file_path = os.path.join(data_base_path, dataset, f"{name}.json")
		input_data = json.load(open(input_file_path, "r", encoding="utf-8"))
		
		max_num_span = 0
		samples = []
		success, failure = 0, 0
		for sample_id, sample in enumerate(tqdm(input_data, desc=f"Loading data from {input_file_path}")):
			span2id = {}

			if config.common["add_umls_marker"]:
				text = sample["new_text"]
			else:
				text = sample["text"]		
			
			for ent in sample["entity_list"]:
				
				if config.common["add_umls_marker"]:
					tok_span, ent_type = _clip(*ent["new_tok_span"]), ent["ent_type"]
				else:
					tok_span, ent_type = _clip(*ent["tok_span"]), ent["ent_type"]
				span2id[tok_span] = ent2id[ent_type]
			ent_label = []

			rel2id, _ = get_labelmap(rel_list[name][sample_id])

			for rel_type, sample_fn in zip(rel_list[name][sample_id], false_negs[name][sample_id]):
				question, context = generate_quesiton_and_context(rel_type, config.common["add_umls_details"], config.common["natural_questions"], sample["umls_entity_list"])
				if context is not None:
					text = context + " " + sample["text"]
					context_len = len(tokenizer.encode(context)) - 2 # -2 for CLS and SEP
				else:
					text = sample["text"]
				indices = tokenizer(
					text = question,
					text_pair = text,
					max_length = maxlen,
					padding = "max_length",
					truncation = True,
					return_special_tokens_mask = True
				)
				tokens = tokenizer.convert_ids_to_tokens(indices["input_ids"])
				# offset = indices["input_ids"].index(sep_token) + 1 + context_len
				offset = indices["input_ids"].index(sep_token) + 1
				length = indices["input_ids"].index(sep_token, offset) - offset
				prefix = offset + length
				# Select spans for every sample
				# Every sample has a coord2span, coord2width, span2coord
				span2coord, coord2span, coord2width = {}, [], []
				lbound = offset
				num_span = 0
				while lbound < prefix:
					while tokens[lbound][:2] == "##":
						lbound += 1
					if lbound >= prefix:
						break
					rbound = lbound
					for width in range(8):
						while rbound < prefix and tokens[rbound + 1][:2] == "##":
							rbound += 1
						if rbound >= prefix:
							break
						span = (lbound, rbound)
						span2coord[span] = num_span
						coord2span.append(span)
						coord2width.append(width)
						num_span += 1
						rbound += 1
					lbound += 1
				# update the max num_span, pad all samples at the end
				if max_num_span < num_span:
					max_num_span = num_span
				ent_label.append({
					"tokens": indices,
					"offset": offset,
					"num_span": num_span,
					"spans": coord2span,
					"widths": coord2width,
					"span2coord": span2coord,
					"scores": [0] * num_span,
					"labels": [0] * num_span,
					"mask": [True] * num_span,
					# "count": 0,
					"coords": set(),
					"tuples": [],
					"sample_id": sample_id,
					"rel_type": rel_type,
					"sample_fn": sample_fn
				})
			for rel in sample["relation_list"]:
				rel_id = rel2id[rel["rel_type"]]
				offset = ent_label[rel_id]["offset"]
				span2coord = ent_label[rel_id]["span2coord"]
				labels = ent_label[rel_id]["labels"]
				scores = ent_label[rel_id]["scores"]
				coords = ent_label[rel_id]["coords"]
				tuples = ent_label[rel_id]["tuples"]
				if config.common["add_umls_marker"]:
					sbj_tok_span, obj_tok_span = _clip(*rel["new_sbj_tok_span"]), _clip(*rel["new_obj_tok_span"])
				else:
					sbj_tok_span, obj_tok_span = _clip(*rel["sbj_tok_span"]), _clip(*rel["obj_tok_span"])
				# We need to minus 1 at span's end
				sbj_tok_span_adj = _clip(sbj_tok_span[0] + offset, sbj_tok_span[1] + offset - 1)
				obj_tok_span_adj = _clip(obj_tok_span[0] + offset, obj_tok_span[1] + offset - 1)
				# If not exist, maybe the span is too long, throw it away
				try:
					sbj_coord = span2coord[sbj_tok_span_adj]
					obj_coord = span2coord[obj_tok_span_adj]
					success += 1
				except KeyError:
					failure += 1
					continue
				scores[sbj_coord] = 1
				scores[obj_coord] = 1
				coords.update([sbj_coord, obj_coord])
				sbj_ent_id, obj_ent_id = span2id[sbj_tok_span], span2id[obj_tok_span]
				
				if complex_mode:
					cp_of_sbj = CP_LEFT if obj_tok_span[1] < sbj_tok_span[1] else CP_RIGHT
					cp_of_obj = CP_LEFT if sbj_tok_span[1] < obj_tok_span[1] else CP_RIGHT
					sbj_ori_cls = labels[sbj_coord]
					obj_ori_cls = labels[obj_coord]
					sbj_new_cls = _get_class(sbj_ori_cls, sbj_ent_id, cp_of_sbj)
					obj_new_cls = _get_class(obj_ori_cls, obj_ent_id, cp_of_obj)
					labels[sbj_coord] = sbj_new_cls
					labels[obj_coord] = obj_new_cls
				else:
					labels[sbj_coord] = sbj_ent_id
					labels[obj_coord] = obj_ent_id
				rel_tuple = (sbj_coord, obj_coord)
				tuples.append(rel_tuple)
			samples.append(ent_label)

		print(success, failure)
		for ent_label in samples:
			for query in ent_label:
				pad_len = max_num_span - query["num_span"]
				sample_spans = query["spans"] + [(0, 0)] * pad_len
				sample_widths = query["widths"] + [0] * pad_len
				sample_scores = query["scores"] + [0] * pad_len
				sample_labels = query["labels"] + [0] * pad_len
				sample_mask = query["mask"] + [False] * pad_len

				encoded_dict[name].append(query["tokens"])
				entity_spans[name].append(sample_spans)
				entity_widths[name].append(sample_widths)
				entity_scores[name].append(sample_scores)
				entity_labels[name].append(sample_labels)
				loss_masks[name].append(sample_mask)
				entity_counts[name].append(len(query["coords"]))
				# entity_counts[name].append(12)
				relation_tuples[name].append(query["tuples"])
				sample_indices[name].append(query["sample_id"])
				rel_types[name].append(query["rel_type"])
				sample_false_negs[name].append(query["sample_fn"])
	
	return encoded_dict, entity_spans, entity_widths, entity_scores, entity_labels, loss_masks, entity_counts, relation_tuples, sample_indices, rel_types, sample_false_negs

def evaluate_entity(model, data, ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, rel_tuples, sample_indices, rel_types, false_negs, pred_counts, batch_size, device, tokenizer, dataset):
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
	_, id2label = get_shifted_labelmap(task_ent_labels[dataset])
	def _search_entity(vector, mask, score=None, max_cnt=None, coord2span=None) -> list:
		entities, exact_entities = [], []
		arg_vector = np.argwhere(vector * mask)[0].tolist()
		if score is not None:
			pred_entities = [(score[i], i) for i in arg_vector]
			arg_vector = NMS(pred_entities, max_cnt + 1, coord2span)
			# arg_vector = NMS(pred_entities, 12, coord2span)
		if complex_mode:
			for i in arg_vector:
				ent_id = vector[i].item()
				exact_entities.append((ent_id, i))
				if ent_id in [1, 2, 3]:
					entities.append((tag_chemical, i))
				elif ent_id in [4, 5, 6]:
					entities.append((tag_gene, i))
		else:
			for i in arg_vector:
				ent_id = vector[i].item()
				exact_entities.append((ent_id, i))
				entities.append((id2label[ent_id], i))
		return entities, exact_entities

	model.eval()
	all_input_ids = [i["input_ids"] for i in data]
	all_attn_mask = [i["attention_mask"] for i in data]
	all_token_type_ids = [i["token_type_ids"] for i in data]
	st, ed = 0, 0
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
			true_cnts = ent_counts[st:ed]
			pred_cnts = pred_counts[st:ed]
			tuples = rel_tuples[st:ed]
			indices = sample_indices[st:ed]
			rtypes = rel_types[st:ed]
			fnegs = false_negs[st:ed]
			outputs = model(input_ids=input_ids, attn_mask=attn_mask, token_type_ids=token_type_ids, ent_spans=spans, ent_widths=widths, ent_scores=scores, ent_labels=labels, loss_masks=masks)
				
			# logits: [B, N, 7], scores: [B, N]
			logits, scores = outputs["logits"], outputs["scores"]

			# scores, logits = logits.max(dim=-1)
			# scores = 1 - logits[:, :, 0]
			scores = scores[:, :, 1]
			logits = logits.argmax(dim=-1) # [B, N]
			pred_tuples = []
			for input_id, span, score, logit, label, mask, true_cnt, pred_cnt, true_tup, sample_idx, rtype, fneg in zip(input_ids, spans, scores.cpu(), logits.cpu(), labels.cpu(), masks.cpu(), true_cnts, pred_cnts, tuples, indices, rtypes, fnegs):
				if sample_idx not in id2ent.keys():
					id2ent[sample_idx] = {
						"true_ent": set(),
						"pred_ent": set()
					}
				true_ent = id2ent[sample_idx]["true_ent"]
				pred_ent = id2ent[sample_idx]["pred_ent"]
				true_ent_list, true_ex_ent_list = _search_entity(label, mask)
				true_ent.update(true_ent_list)
				if fneg:
					pred_ent_list, pred_ex_ent_list, pred_tup = [], [], []
				else:
					pred_ent_list, pred_ex_ent_list = _search_entity(logit, mask, score, pred_cnt, span)
					pred_ent.update(pred_ent_list)
					pred_tup = _search(pred_ex_ent_list, rtype)
				true_pos.append(len(set(true_tup) & set(pred_tup)))
				pred_len.append(len(pred_tup))
				target_len.append(len(true_tup))
				pred_tuples.append(pred_tup)

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

	# return f1, ent_f1

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluate RE')
	parser.add_argument('--dataset', type=str, help='DrugVar or DrugProt')
	parser.add_argument('--rel_name', type=str, default='', help='The name of the relation classifier model')
	parser.add_argument('--rel_id', type=str, help='The run ID of the relation classifier model')
	parser.add_argument('--ent_name', type=str, default='', help='The name of the entity detector model')
	parser.add_argument('--ent_id', type=str, help='The run ID of the entity detector model')
	parser.add_argument('--cnt_name', type=str, default='', help='The name of the number predictor model')
	parser.add_argument('--cnt_id', type=str, help='The run ID of the number predictor model')

	args = parser.parse_args()
	config.common['exp_name'] = args.dataset

	eval_config = config.eval_config
	eval_config['rel_model_name'] = args.rel_name
	eval_config['rel_run_id'] = args.rel_id
	eval_config['ent_model_name'] = args.ent_name
	eval_config['ent_run_id'] = args.ent_id
	eval_config['cnt_model_name'] = args.cnt_name
	eval_config['cnt_run_id'] = args.cnt_id

	device = config.common["device"]

	rel_model_path = os.path.join(
		eval_config["saved_model_dir"], 
		'checkpoint_%s_%s.pth.tar' % (eval_config["rel_model_name"], eval_config["rel_run_id"])
	)
	cnt_model_path = os.path.join(
		eval_config["saved_model_dir"], 
		'checkpoint_%s_%s.pth.tar' % (eval_config["cnt_model_name"], eval_config["cnt_run_id"])
	)
	ent_model_path = os.path.join(
		eval_config["saved_model_dir"], 
		'checkpoint_%s_%s.pth.tar' % (eval_config["ent_model_name"], eval_config["ent_run_id"])
	)
	if not os.path.exists(rel_model_path):
		raise RuntimeError("No such relation checkpoint: %s. Please check `config.py`." % rel_model_path)
	if not os.path.exists(cnt_model_path):
		raise RuntimeError("No such count checkpoint: %s. Please check `config.py`." % cnt_model_path)
	if not os.path.exists(ent_model_path):
		raise RuntimeError("No such entity checkpoint: %s. Please check `config.py`." % ent_model_path)
	
	# Load data
	tokenizer = AutoTokenizer.from_pretrained(config.bert_config["bert_path"], do_lower_case=False)
	data, rel_labels = load_relation_data(
		data_base_path=config.data["data_base_dir"], 
		tokenizer=tokenizer,
		dataset=config.common["exp_name"],
		field_list=["dev"]
	)

	print("Loading relation model from %s" % rel_model_path)
	rel_model = torch.load(rel_model_path)

	rel_model.to(device)
	rel_predictions = predict_relation(
		model=rel_model,
		data=data["dev"], 
		rel_labels=rel_labels["dev"], 
		batch_size=eval_config["batch_size"], 
		device=device
	)
	# rel_predictions = rel_labels["dev"]
	relation_list, false_negs = make_relation_list(
		data_base_path=config.data["data_base_dir"],
		dataset=config.common["exp_name"],
		pred_rel=rel_predictions
	)

	data = load_count_data(
		data_base_path=config.data["data_base_dir"], 
		tokenizer=tokenizer,
		dataset=config.common["exp_name"],
		rel_list=relation_list,
		false_negs=false_negs,
		split_sentence=False
	)

	print("Loading count model from %s" % cnt_model_path)
	cnt_model = torch.load(cnt_model_path)

	cnt_model.to(device)
	pred_counts = evaluate_count(
		model=cnt_model,
		data=data["dev"], 
		batch_size=eval_config["batch_size"],
		device=device,
		tokenizer=tokenizer
	)
	
	data, ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, rel_tuples, sample_indices, rel_types, sample_fns = load_entity_data(
		data_base_path=config.data["data_base_dir"], 
		tokenizer=tokenizer, 
		dataset=config.common["exp_name"],
		rel_list=relation_list,
		false_negs=false_negs
	)

	print("Loading entity model from %s" % ent_model_path)
	ent_model = torch.load(ent_model_path)

	ent_model.to(device)
	evaluate_entity(
		model=ent_model,
		data=data["dev"],
		ent_spans=ent_spans["dev"],
		ent_widths=ent_widths["dev"],
		ent_scores=ent_scores["dev"],
		ent_labels=ent_labels["dev"],
		loss_masks=loss_masks["dev"],
		ent_counts=ent_counts["dev"],
		rel_tuples=rel_tuples["dev"],
		false_negs=sample_fns["dev"],
		pred_counts=pred_counts,
		# pred_counts=ent_counts["dev"],
		sample_indices=sample_indices["dev"],
		rel_types=rel_types["dev"],
		batch_size=eval_config["batch_size"],
		device=device,
		tokenizer=tokenizer,
		dataset=config.common["exp_name"]
	)
