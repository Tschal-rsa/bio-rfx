import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss
import config
import math

from transformers import BertModel

class RelationModel(nn.Module):
	def __init__(self, model_config):
		super().__init__()
		self.num_rels = model_config["rel_num"]
		self.bert = BertModel.from_pretrained(config.bert_config["bert_path"])
		self.dropout = nn.Dropout(model_config["classifier_dropout"])
		self.classifiers =  nn.ModuleList(
								[nn.Linear(model_config["hidden_size"], 1)
								.to(config.common["device"])
							for _ in range(self.num_rels)])

	def forward(self, input_ids, attn_mask, rel_labels=None):
		outputs = self.bert(
			input_ids,
			attention_mask=attn_mask,
		)
		bert_output = outputs[1]

		bert_output = self.dropout(bert_output)

		logits = None
		loss = 0
		bce_loss_fct = BCELoss()
		sigmoid = nn.Sigmoid()

		for i, classifier in enumerate(self.classifiers):
			logit = classifier(bert_output)  # [batch_size, 1]
			logit = sigmoid(logit)

			if logits is None:
				logits = logit
			else:
				logits = torch.cat((logits, logit), dim=1)	

			if rel_labels is not None:
				rel_label = rel_labels[:,i].reshape(-1, 1).to(torch.float)
				loss += bce_loss_fct(logit, rel_label)
		
		
		return { 
			"logits": logits, # [batch_size, rel_num]
			"loss": loss 
		}

class CountModel(nn.Module):
	def __init__(self, model_config):
		super().__init__()
		self.bert = BertModel.from_pretrained(config.bert_config["bert_path"])		
		self.dropout = nn.Dropout(model_config["predictor_dropout"])
		self.predictor =  nn.Sequential(
            nn.Linear(model_config["hidden_size"], 2 * model_config["hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["predictor_dropout"]),
			nn.Linear(2 * model_config["hidden_size"], 2 * model_config["hidden_size"]),
			nn.GELU(),
			nn.Dropout(model_config["predictor_dropout"]),
            nn.Linear(2 * model_config["hidden_size"], 1),
        )
		self.mse_loss = MSELoss()

	def forward(self, input_ids, attn_mask, token_type_ids, cnt_label=None):
		outputs = self.bert(
			input_ids,
			attention_mask=attn_mask,
			token_type_ids=token_type_ids
		)
		bert_output = outputs[1]
		pred_input = self.dropout(bert_output)
		logits = self.predictor(pred_input)

		loss = None
		if cnt_label is not None:
			cnt_label = cnt_label.reshape(-1, 1).to(torch.float)
			loss = self.mse_loss(logits, cnt_label)
		
		return { 
			"logits": logits, # [batch_size, 1]
			"loss": loss.to(torch.float)
		}

class CountModelCNN(nn.Module):		
	def __init__(self, model_config):
		super().__init__()
		self.bert = BertModel.from_pretrained(config.bert_config["bert_path"])		
		self.dropout = nn.Dropout(model_config["predicctor_dropout"])
		self.conv2d = nn.Conv2d(in_channels=1, out_channels=1,
								kernel_size=[5, 5],
								stride=[1, 2],
								padding=[2, 256], padding_mode="circular")
		predictor_in_dim = math.floor((model_config["hidden_size"] + 512 - 5)/2) + 1 + model_config["hidden_size"]
		self.predictor = nn.Linear(predictor_in_dim, 1)
		self.mse_loss = MSELoss()

	def forward(self, input_ids, attn_mask, token_type_ids, cnt_label=None):
		outputs = self.bert(
			input_ids,
			attention_mask=attn_mask,
			token_type_ids=token_type_ids
		)
		bert_output = outputs[0]
		sequence_output = self.dropout(bert_output)
		conv_input = sequence_output.unsqueeze(1)
		conv_output = self.conv2d(conv_input)
		conv_output = conv_output.squeeze(1)
		pred_input = torch.cat([sequence_output,conv_output],-1)
			
		pred_input = self.dropout(pred_input)
		pred_input = torch.max(pred_input, dim=1).values
		logits = self.predictor(pred_input)

		loss = None
		if cnt_label is not None:
			cnt_label = cnt_label.reshape(-1, 1).to(torch.float)
			loss = self.mse_loss(logits, cnt_label)
		
		return { 
			"logits": logits, # [batch_size, 1]
			"loss": loss.to(torch.float)
		}