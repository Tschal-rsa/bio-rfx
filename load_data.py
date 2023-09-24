import config
from tqdm import tqdm
import torch
import json
import numpy as np
import random
import os
from preprocess.umls import append_marker_tokens
from preprocess.const import task_umls_ent_labels, get_labelmap, task_ent_labels, task_ent_to_id, task_cnt_question, task_rel_labels, task_umls_rels
from utils import generate_quesiton_and_context

seed = config.train_config["seed"]
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def _clip(left, right, maxlen):
	"""
	Clip the numbers in a pair to maxlen - 1
	"""
	if left >= maxlen:
		left = maxlen - 1
	if right >= maxlen:
		right = maxlen - 1
	return (left, right)


def load_entity_data_aware(data_base_path, tokenizer, dataset, field_list, maxlen):
	"""
	Load entity data

	Returns: dictionaries, with keys = ["train", "dev"]
		encoded_dict: tokenized QA pairs, padded or truncated
		entity_spans: [(0, 0), (0, 1), ..., (0, 7), (1, 1), (1, 2), ...]
		entity_widths: [0, 1, ..., 7, 0, 1, ...], length = num_span, same as entity_spans
		entity_scores: 0/1, length = num_span, same as entity_spans
		entity_labels: 0~6, length = num_span, same as entity_spans
		loss_masks: T/F, length = num_span, same as entity_spans
		entity_counts: the number of entities in every samples
		relation_tuples: [(coord of subject, coord of object)]
		sample_indices: [(PMID, id)]
	"""
	model_config = config.model["entity_detector"]
	complex_mode: bool
	if model_config["ent_cls_mode"] == "complex":
		complex_mode = True
	elif model_config["ent_cls_mode"] == "simple":
		complex_mode = False
	else:
		raise NotImplementedError

	CP_LEFT, CP_RIGHT = 0, 1
	def _get_class(ori_cls, ent_id, cp_loc):
		"""
		Get the current class (0-7)

		Parameters:
			ori_cls: original class (0-7)
			ent_id: 0 for chemical, 1 for gene
			cp_loc: location of the counterpart, either CP_LEFT or CP_RIGHT
		"""
		new_cls = 0
		if ori_cls == 0:
			new_cls = ent_id * 3 + 1 if cp_loc == CP_LEFT else ent_id * 3 + 2
		elif ori_cls == ent_id * 3 + 1:
			new_cls = ent_id * 3 + 1 if cp_loc == CP_LEFT else ent_id * 3 + 3
		elif ori_cls == ent_id * 3 + 2:
			new_cls = ent_id * 3 + 3 if cp_loc == CP_LEFT else ent_id * 3 + 2
		elif ori_cls == ent_id * 3 + 3:
			new_cls = ent_id * 3 + 3
		return new_cls

	if config.common["add_umls_marker"]:
		append_marker_tokens(tokenizer, task_umls_ent_labels[dataset])
	
	# Maintain a hashtable: tok_span_adj(closed interval) -> coordinate_of_label
	# Future entity_widths: coord2width
	# Future entity_spans:  coord2span
	# span2coord, coord2span, coord2width = {}, [], []
	# num_span = 0
	# for i in range(maxlen):
	# 	for j in range(i, min(i + 8, maxlen)):
	# 		span2coord[(i, j)] = num_span
	# 		coord2span.append((i, j))
	# 		coord2width.append(j - i)
	# 		num_span += 1
	
	encoded_dict, entity_spans, entity_widths, entity_scores, entity_labels, loss_masks, entity_counts, relation_tuples, sample_indices, rel_types = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
	sep_token = config.bert_config["sep_token"]
	# sample_span_num = model_config["sample_span_num"]
	model_config["ent_cls_num"] = 1 + len(task_ent_labels[dataset]) * (3 if complex_mode else 1)
	for name in field_list:
		encoded_dict[name], entity_spans[name], entity_widths[name], entity_scores[name], entity_labels[name], loss_masks[name], entity_counts[name], relation_tuples[name], sample_indices[name], rel_types[name] = [], [], [], [], [], [], [], [], [], []
		input_file_path = os.path.join(data_base_path, dataset, f"{name}.json")
		input_data = json.load(open(input_file_path, "r", encoding="utf-8"))
		# random.shuffle(input_data)
		max_num_span = 0
		samples = []
		success, failure = 0, 0
		for sample_id, sample in enumerate(tqdm(input_data, desc=f"Loading data from {input_file_path}")):
			# Maintain a hashtable: tok_span(left closed & right open) -> entity_id(0/1)
			span2id = {}

			if config.common["add_umls_marker"]:
				text = sample["new_text"]
			else:
				text = sample["text"]
			
			for ent in sample["entity_list"]:
				
				if config.common["add_umls_marker"]:
					tok_span, ent_type = _clip(*ent["new_tok_span"], maxlen), ent["ent_type"]
				else:
					tok_span, ent_type = _clip(*ent["tok_span"], maxlen), ent["ent_type"]
				span2id[tok_span] = task_ent_to_id[dataset][ent_type]
			ent_label = {}
			
			for rel in sample["relation_list"]:
				rel_type = rel["rel_type"]
				# rel_type = "_lexicallyChainedTo"
				if rel_type not in ent_label.keys():
					# text = generate_quesiton(rel_type, config.common["add_umls_details"], sample["umls_entity_list"])
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
					# mask = []
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
					# for coord in range(num_span):
					# 	lbound, rbound = coord2span[coord]
					# 	if lbound >= offset and rbound < prefix and tokens[lbound][:2] != "##" and tokens[rbound + 1][:2] != "##":
					# 		mask.append(True)
					# 	else:
					# 		mask.append(False)
					ent_label[rel_type] = {
						# "words": tokens,
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
						"rel_type": rel_type
					}
				offset = ent_label[rel_type]["offset"]
				span2coord = ent_label[rel_type]["span2coord"]
				labels = ent_label[rel_type]["labels"]
				scores = ent_label[rel_type]["scores"]
				# mask = ent_label[rel_type]["mask"]
				coords = ent_label[rel_type]["coords"]
				tuples = ent_label[rel_type]["tuples"]
				if config.common["add_umls_marker"]:
					sbj_tok_span, obj_tok_span = _clip(*rel["new_sbj_tok_span"], maxlen), _clip(*rel["new_obj_tok_span"], maxlen)
				else:
					sbj_tok_span, obj_tok_span = _clip(*rel["sbj_tok_span"], maxlen), _clip(*rel["obj_tok_span"], maxlen)
				# We need to minus 1 at span's end
				sbj_tok_span_adj = _clip(sbj_tok_span[0] + offset, sbj_tok_span[1] + offset - 1, maxlen)
				obj_tok_span_adj = _clip(obj_tok_span[0] + offset, obj_tok_span[1] + offset - 1, maxlen)
				# If not exist, maybe the span is too long, throw it away
				try:
					sbj_coord = span2coord[sbj_tok_span_adj]
					obj_coord = span2coord[obj_tok_span_adj]
					success += 1
				except KeyError:
					# words = ent_label[rel_type]["words"]
					# print(words[sbj_tok_span_adj[0]: sbj_tok_span_adj[1] + 1])
					# print(words[obj_tok_span_adj[0]: obj_tok_span_adj[1] + 1])
					failure += 1
					continue
				# if sbj_tok_span[1] > sbj_tok_span[0] + 8:
				# 	mask[sbj_coord] = False
				# if obj_tok_span[1] > obj_tok_span[0] + 8:
				# 	mask[obj_coord] = False
				scores[sbj_coord] = 1
				scores[obj_coord] = 1
				# ent_label[rel_type]["count"] += 2
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
					labels[sbj_coord] = sbj_ent_id + 1
					labels[obj_coord] = obj_ent_id + 1
				rel_tuple = (sbj_coord, obj_coord)
				tuples.append(rel_tuple)
			samples.append(ent_label)
		print(success, failure)
		for ent_label in samples:
			for query in ent_label.values():
				pad_len = max_num_span - query["num_span"]
				sample_spans = query["spans"] + [(0, 0)] * pad_len
				sample_widths = query["widths"] + [0] * pad_len
				sample_scores = query["scores"] + [0] * pad_len
				sample_labels = query["labels"] + [0] * pad_len
				sample_mask = query["mask"] + [False] * pad_len
				# if name == "train":
				# 	sample_mask = query["mask"].copy()
				# 	entity_coords = np.array(list(query["coords"]))
				# 	for coord in entity_coords:
				# 		sample_mask[coord] = False
				# 	negative_coords = np.nonzero(sample_mask)[0]
				# 	if len(entity_coords) + len(negative_coords) < sample_span_num:
				# 		continue
				# 	selected_negative_coords = np.random.choice(
				# 		negative_coords,
				# 		size=sample_span_num - len(entity_coords),
				# 		replace=False
				# 	)
				# 	valid_coords = np.concatenate((entity_coords, selected_negative_coords))
				# 	sample_spans = np.array(sample_spans)[valid_coords].tolist()
				# 	sample_widths = np.array(sample_widths)[valid_coords].tolist()
				# 	sample_scores = np.array(sample_scores)[valid_coords].tolist()
				# 	sample_labels = np.array(sample_labels)[valid_coords].tolist()
				# 	sample_mask = [True] * sample_span_num

				encoded_dict[name].append(query["tokens"])
				entity_spans[name].append(sample_spans)
				entity_widths[name].append(sample_widths)
				entity_scores[name].append(sample_scores)
				entity_labels[name].append(sample_labels)
				loss_masks[name].append(sample_mask)
				entity_counts[name].append(len(query["coords"]))
				relation_tuples[name].append(query["tuples"])
				sample_indices[name].append(query["sample_id"])
				rel_types[name].append(query["rel_type"])
	
	
	return encoded_dict, entity_spans, entity_widths, entity_scores, entity_labels, loss_masks, entity_counts, relation_tuples, sample_indices, rel_types


def load_entity_data_blind(data_base_path, tokenizer, dataset, field_list, maxlen):
	"""
	Load entity data

	Returns: dictionaries, with keys = ["train", "dev"]
		encoded_dict: tokenized QA pairs, padded or truncated
		entity_spans: [(0, 0), (0, 1), ..., (0, 7), (1, 1), (1, 2), ...]
		entity_widths: [0, 1, ..., 7, 0, 1, ...], length = num_span, same as entity_spans
		entity_scores: 0/1, length = num_span, same as entity_spans
		entity_labels: 0~6, length = num_span, same as entity_spans
		loss_masks: T/F, length = num_span, same as entity_spans
		entity_counts: the number of entities in every samples
		sample_indices: [(PMID, id)]
	"""
	model_config = config.model["entity_detector"]
	complex_mode: bool
	if model_config["ent_cls_mode"] == "complex":
		complex_mode = True
	elif model_config["ent_cls_mode"] == "simple":
		complex_mode = False
	else:
		raise NotImplementedError

	if config.common["add_umls_marker"]:
		append_marker_tokens(tokenizer, task_umls_ent_labels[dataset])
	
	
	encoded_dict, entity_spans, entity_widths, entity_scores, entity_labels, loss_masks, entity_counts, relation_tuples, sample_indices, rel_types = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
	sep_token = config.bert_config["sep_token"]
	# sample_span_num = model_config["sample_span_num"]
	model_config["ent_cls_num"] = 1 + len(task_ent_labels[dataset]) * (3 if complex_mode else 1)
	for name in field_list:
		encoded_dict[name], entity_spans[name], entity_widths[name], entity_scores[name], entity_labels[name], loss_masks[name], entity_counts[name], relation_tuples[name], sample_indices[name], rel_types[name] = [], [], [], [], [], [], [], [], [], []
		input_file_path = os.path.join(data_base_path, dataset, f"{name}.json")
		input_data = json.load(open(input_file_path, "r", encoding="utf-8"))
		# random.shuffle(input_data)
		max_num_span = 0
		samples = []
		success, failure = 0, 0
		for sample_id, sample in enumerate(tqdm(input_data, desc=f"Loading data from {input_file_path}")):

			text = sample["new_text"] if config.common["add_umls_marker"] else sample["text"]					
			indices = tokenizer(
				text = "What are the %s in the sentence?" % task_cnt_question[dataset],
				text_pair = text,
				max_length = maxlen,
				padding = "max_length",
				truncation = True,
				return_special_tokens_mask = True
			)
			if name == "dev" and sample_id == 0:
				print(text)

			tokens = tokenizer.convert_ids_to_tokens(indices["input_ids"])
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

			ent_label = {
				"tokens": indices,
				"offset": offset,
				"num_span": num_span,
				"spans": coord2span,
				"widths": coord2width,
				"span2coord": span2coord,
				"scores": [0] * num_span,
				"labels": [0] * num_span,
				"mask": [True] * num_span,
				"coords": set(),
				"sample_id": sample_id,
			}
			offset = ent_label["offset"]
			span2coord = ent_label["span2coord"]
			labels =  ent_label["labels"]
			scores =  ent_label["scores"]
			coords =  ent_label["coords"]

			for ent in sample["entity_list"]:
				if config.common["add_umls_marker"]:
					tok_span = _clip(*ent["new_tok_span"], maxlen)
				else:
					tok_span = _clip(*ent["tok_span"], maxlen)
				# We need to minus 1 at span's end
				tok_span_adj = _clip(tok_span[0] + offset, tok_span[1] + offset - 1, maxlen)
				
				# If not exist, maybe the span is too long, throw it away
				try:
					ent_coord = span2coord[tok_span_adj]
					success += 1
				except KeyError:
					failure += 1
					continue
				scores[ent_coord] = 1
				coords.add(str(ent_coord))
				ent_id = task_ent_to_id[dataset][ent["ent_type"]]
				labels[ent_coord] = ent_id + 1

			
			samples.append(ent_label)
			assert(len(coords) <= len(sample["entity_list"]))

		print(success, failure)
		for query in samples:
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
			# entity_counts[name].append(len(query["coords"]))
			entity_counts[name].append(20)
			sample_indices[name].append(query["sample_id"])
	
	
	return encoded_dict, entity_spans, entity_widths, entity_scores, entity_labels, \
		   loss_masks, entity_counts, relation_tuples, sample_indices, rel_types


def load_relation_data(data_base_path, tokenizer, dataset, field_list=["train", "dev"], maxlen=512):
	encoded_dict, relation_labels = {}, {}
	rel2id, id2rel = get_labelmap(task_rel_labels[dataset])
	for name in field_list:
		encoded_dict[name], relation_labels[name] = [], []
		input_file_path = os.path.join(data_base_path, "%s"%(dataset), "%s.json"%(name))
		input_data = json.load(open(input_file_path, "r", encoding = "utf-8"))
		# random.shuffle(input_data)
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
		# json.dump(input_data, open(os.path.join(data_base_path, "%s"%(dataset), "%s.json"%(name+"-shuffled")), "w", encoding = "utf-8"), ensure_ascii = False)

	return encoded_dict, relation_labels


def load_count_data(data_base_path, tokenizer, dataset, relation_aware, field_list=["train", "dev"], maxlen=512):
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
	'''
	if config.common["exp_name"] == "BB":
		FACTOR = 10
	elif "DrugProt" in config.common["exp_name"]:
		FACTOR = 1
	elif "DrugVar" in config.common["exp_name"]:
		FACTOR = 1
	else:
		FACTOR = None
		raise NotImplementedError
	
	encoded_dict, subj_and_obj_cnt = {}, {}
	for name in field_list:
		encoded_dict[name], subj_and_obj_cnt[name] = [], []
		input_file_path = os.path.join(data_base_path, "%s"%(dataset), "%s.json"%(name))
		input_data = json.load(open(input_file_path, "r", encoding = "utf-8"))
		# random.shuffle(input_data)
		sep_token = config.bert_config["sep_token"]

		for sample in tqdm(input_data, desc="Loading data from %s" % (input_file_path)):
			
			if relation_aware:
				# Count subjects and objects
				subj_and_obj = {}
				for rel_type in task_rel_labels[dataset]:
					subj_and_obj[rel_type] = []
				for rel in sample["relation_list"]:
					rel_type = rel["rel_type"]
					subj_and_obj[rel_type].append(_clip(*rel["obj_tok_span"], maxlen))
					subj_and_obj[rel_type].append(_clip(*rel["sbj_tok_span"], maxlen))
				for rel_type in task_rel_labels[dataset]:
					subj_and_obj[rel_type] = len(set(subj_and_obj[rel_type]))
					if subj_and_obj[rel_type] == 0:
						subj_and_obj.pop(rel_type, None)

				# Construct relation-specific tokens
				for rel_type in subj_and_obj.keys():
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
					encoded_dict[name].append({
						"tokens": tokens,
						"count": subj_and_obj[rel_type] / FACTOR,
						"mask": [False] * offset + [True] * length + [False] * remain,
					})
				
			else:
				tokens = tokenizer(
						# text=generate_quesiton(rel_type, False),
						text="How many " + task_cnt_question[dataset] + " are there in the sentence?",
						text_pair=sample["text"],
						max_length=maxlen,
						padding="max_length",
						truncation=True,
						return_special_tokens_mask = True
					)
				offset = tokens["input_ids"].index(sep_token) + 1
				length = tokens["input_ids"].index(sep_token, offset) - offset
				remain = maxlen - offset - length
				encoded_dict[name].append({
					"tokens": tokens,
					"count": len(sample["entity_list"]) / FACTOR,
					"mask": [False] * offset + [True] * length + [False] * remain,
				})

		
	sentence = torch.tensor(encoded_dict["dev"][0]["tokens"]["input_ids"])
	mask = torch.tensor(encoded_dict["dev"][0]["mask"])
	print("[Example of count data]:")	
	print("[Sentence]:")
	print(tokenizer.decode(sentence[mask]))
	print("[Entity Count]:", encoded_dict["dev"][0]["count"])
	return encoded_dict
