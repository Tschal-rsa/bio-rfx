import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout
from allennlp.nn.util import batched_index_select, batched_span_select
import config

from transformers import BertModel

def debug(count=1):
	def _debug(*args):
		nonlocal count
		for tensor in args:
			print(tensor.size())
		count -= 1
		if count <= 0:
			exit(0)
	return _debug
dbg = debug(1)

class EntityModel(nn.Module):
	def __init__(self, model_config):
		super().__init__()
		self.num_ent_cls = model_config["ent_cls_num"]
		self.bert = BertModel.from_pretrained(config.bert_config["bert_path"])
		self.dropout = Dropout(model_config["detector_dropout"])
		self.start_detector = nn.Sequential(
			nn.Linear(model_config["hidden_size"], model_config["head_hidden_size"]),
			nn.LayerNorm(model_config["head_hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["head_dropout"]),
			nn.Linear(model_config["head_hidden_size"], self.num_ent_cls)
		)
		self.end_detector = nn.Sequential(
			nn.Linear(model_config["hidden_size"], model_config["head_hidden_size"]),
			nn.LayerNorm(model_config["head_hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["head_dropout"]),
			nn.Linear(model_config["head_hidden_size"], self.num_ent_cls)
		)
		# self.start_detector = TransposeLinear(self.num_ent_cls, model_config["hidden_size"])
		# self.end_detector = TransposeLinear(self.num_ent_cls, model_config["hidden_size"])
		ce_weight = torch.tensor([0.1, 1, 1, 1, 1, 1, 1])
		self.cross_entropy = CrossEntropyLoss(weight=ce_weight, reduction="none")
	
	def resize_token_embeddings(self, vocab_length):
		self.bert.resize_token_embeddings(vocab_length)
	
	def forward(self, input_ids, attn_mask, ent_labels=None, loss_masks=None):
		outputs = self.bert(
            input_ids,
            attention_mask=attn_mask,
    	)
		bert_output = outputs[0]

		bert_output = self.dropout(bert_output)

		start_logits = self.start_detector(bert_output)
		end_logits = self.end_detector(bert_output)
		logits = torch.stack((start_logits, end_logits), dim=1)

		batch_size = logits.size(0)
		loss = self.cross_entropy(logits.reshape(-1, self.num_ent_cls), ent_labels.reshape(-1))
		loss = loss[loss_masks.view(-1)].sum() / batch_size
		
		return { 
			"logits": logits, # [batch_size, 2, seq_len, 7]
			"loss": loss 
		}

class EntitySpanModel(nn.Module):
	def __init__(self, model_config):
		super().__init__()
		self.num_ent_cls = model_config["ent_cls_num"]
		self.bert = BertModel.from_pretrained(config.bert_config["bert_path"])
		self.cls_embed = nn.Embedding(model_config["width_embed_num"], model_config["width_embed_dim"])
		# self.scr_embed = nn.Embedding(model_config["width_embed_num"], model_config["width_embed_dim"])
		self.bert_head = nn.Sequential(
			nn.Dropout(model_config["detector_dropout"])
		)
		# self.bilstm = nn.LSTM(
		# 	input_size=model_config["hidden_size"],
		# 	hidden_size=model_config["lstm_hidden_size"],
		# 	num_layers=3,
		# 	batch_first=True,
		# 	dropout=model_config["detector_dropout"],
		# 	bidirectional=True
		# )
		self.hidden_scorer = nn.Sequential(
			nn.Linear(model_config["hidden_size"], model_config["head_hidden_size"]),
			nn.LayerNorm(model_config["head_hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["head_dropout"]),
			nn.Linear(model_config["head_hidden_size"], 1)
		)
		self.span_detector = nn.Sequential(
			nn.Linear(3 * model_config["hidden_size"] + model_config["width_embed_dim"], model_config["head_hidden_size"]),
			nn.LayerNorm(model_config["head_hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["head_dropout"]),
			nn.Linear(model_config["head_hidden_size"], model_config["head_hidden_size"]),
			nn.LayerNorm(model_config["head_hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["head_dropout"]),
			nn.Linear(model_config["head_hidden_size"], self.num_ent_cls)
		)
		# self.score_generator = nn.Sequential(
		# 	nn.Linear(2 * model_config["hidden_size"] + model_config["width_embed_dim"], model_config["head_hidden_size"]),
		# 	nn.LayerNorm(model_config["head_hidden_size"]),
		# 	nn.GELU(),
		# 	nn.Dropout(model_config["head_dropout"]),
		# 	nn.Linear(model_config["head_hidden_size"], 2)
		# )
		cls_ce_weight = torch.tensor([0.1] + [1] * (self.num_ent_cls - 1), dtype=torch.float)
		# cls_ce_weight = torch.tensor([0.01, 1, 1, 5, 10, 10], dtype=torch.float)
		# scr_ce_weight = torch.tensor([0.1, 1])
		self.cls_ce_loss = CrossEntropyLoss(weight=cls_ce_weight, reduction="mean")
		# self.scr_ce_loss = CrossEntropyLoss(weight=scr_ce_weight, reduction="mean")
	
	def resize_token_embeddings(self, vocab_length):
		self.bert.resize_token_embeddings(vocab_length)
	
	def forward(self, input_ids, attn_mask, token_type_ids, ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, **kwargs):
		outputs = self.bert(
            input_ids,
            attention_mask=attn_mask,
			token_type_ids=token_type_ids
    	)
		bert_output = outputs[0]

		bert_output = self.bert_head(bert_output)

		# lstm_output, _ = self.bilstm(bert_output)
		# # [B, L, h_bert] | [B, L, 2h_lstm] -> [B, L, H] where H = h_bert + 2 * h_lstm
		# total_output = torch.cat((bert_output, lstm_output), dim=-1)

		# [B, L, H] @ [B, N, 2] -> [B, N, 2, H]
		output_spans = batched_index_select(bert_output, ent_spans)
		# [B, L, H] @ [B, N, 2] -> [B, N, W, H] & [B, N, W]
		span_embeddings, span_mask = batched_span_select(bert_output, ent_spans)
		# [B, N] -> [B, N, W]
		cls_width_embed = self.cls_embed(ent_widths)
		# scr_width_embed = self.scr_embed(ent_widths)
		# Solution 1: span boundaries
		# [B, N, H] | [B, N, H] | [B, N, W] -> [B, N, 2 * H + W]
		# cls_span_reps = torch.cat((
		# 	output_spans[:, :, 0, :],
		# 	output_spans[:, :, 1, :],
		# 	cls_width_embed
		# ), dim=-1)
		# Solution 2: sum of the whole span
		# span_embeddings.masked_fill_(~span_mask.unsqueeze(-1), 0.0)
		# cls_span_reps = torch.cat((
		# 	output_spans[:, :, 0, :],
		# 	torch.sum(span_embeddings, dim=-2),
		# 	output_spans[:, :, 1, :],
		# 	cls_width_embed
		# ), dim=-1)
		# Solution 3: mean of the whole span
		# cls_span_reps = torch.cat((
		# 	output_spans[:, :, 0, :],
		# 	masked_mean(span_embeddings, span_mask.unsqueeze(-1), dim=-2),
		# 	output_spans[:, :, 1, :],
		# 	cls_width_embed
		# ), dim=-1)
		# Solution 4: max pool the whole span
		# span_embeddings.masked_fill_(~span_mask.unsqueeze(-1), -100.0)
		# cls_span_reps = torch.cat((
		# 	output_spans[:, :, 0, :],
		# 	torch.max(span_embeddings, dim=-2).values,
		# 	output_spans[:, :, 1, :],
		# 	cls_width_embed
		# ), dim=-1)
		# Solution 5: weighted sum of the whole span
		hidden_score = self.hidden_scorer(span_embeddings)
		hidden_score.masked_fill_(~span_mask.unsqueeze(-1), -1e5)
		span_score = torch.softmax(hidden_score, dim=-2)
		aggr_span_embed = torch.sum(span_score * span_embeddings, dim=-2)
		cls_span_reps = torch.cat((
			output_spans[:, :, 0, :],
			aggr_span_embed,
			output_spans[:, :, 1, :],
			cls_width_embed
		), dim=-1)
		# scr_span_reps = torch.cat((
		# 	output_spans[:, :, 0, :],
		# 	output_spans[:, :, 1, :],
		# 	scr_width_embed
		# ), dim=-1)
		# [B, N, 2 * H + W] -> [B, N, C]
		cls_logits = self.span_detector(cls_span_reps)
		# [B, N, 2 * H + W] -> [B, N, 2]
		# scr_logits = self.score_generator(scr_span_reps)
		scr_logits = torch.stack((cls_logits[:, :, 0], 1 - cls_logits[:, :, 0]), dim=-1)

		labels = torch.where(loss_masks, ent_labels, torch.tensor(self.cls_ce_loss.ignore_index).type_as(ent_labels))
		# scores = torch.where(loss_masks, ent_scores, torch.tensor(self.scr_ce_loss.ignore_index).type_as(ent_scores))
		cls_loss = self.cls_ce_loss(cls_logits.reshape(-1, self.num_ent_cls), labels.reshape(-1))
		# scr_loss = self.scr_ce_loss(scr_logits.reshape(-1, 2), scores.reshape(-1))
		scr_loss = 0
		
		return { 
			"logits": cls_logits, # [batch_size, num_span, 7]
			"scores": scr_logits,
			"loss": cls_loss + scr_loss
		}

def NMS(pred_entities, max_cnt, coord2span):
	'''
	Parameters:
	 - pred_entities: predicted entity list. composed of tuples: (score, coord).
	 - max_cnt: max span number.
	 - coord2span: coord2span[coord] denotes the token span for a predicted entity.
	Return:
	 - filtered_entities: length = max_cnt.
	'''
	def is_overlap(span1, span2):
		return not (span1[1] < span2[0] or span2[1] < span1[0])
	
	filtered_entities = set()
	pred_entities = set(pred_entities)
	while len(filtered_entities) < max_cnt and len(pred_entities) > 0:
		candidate_list = list(pred_entities)
		candidate_list.sort(key=lambda item:item[0], reverse=True)
		candidate = candidate_list[0]
		filtered_entities.add(candidate[1])
		pred_entities.remove(candidate)
		for span in pred_entities.copy():
			if is_overlap(coord2span[candidate[1]], coord2span[span[1]]):
				pred_entities.remove(span)
	return sorted(filtered_entities)
