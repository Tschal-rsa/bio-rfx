import config
import torch
from torch import optim
import numpy as np
import time
import random
from transformers import AutoTokenizer
import os
from entity_models import EntitySpanModel
from utils import init_logger, get_lr_schedular
from load_data import load_entity_data_aware, load_entity_data_blind
from scorer import evaluate_entity_aware, evaluate_entity_blind
import argparse

seed = config.train_config["seed"]
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def evaluate_entity(model, data, 
					ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, rel_tuples, sample_indices, rel_types, 
					batch_size, device, tokenizer, dataset, relation_aware) -> tuple:
	if relation_aware:
		return evaluate_entity_aware(model, data, 
									ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, rel_tuples, sample_indices, rel_types, 
									batch_size, device, tokenizer, dataset)
	else:
		return evaluate_entity_blind(model, data, 
									ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, sample_indices, 
									batch_size, device, tokenizer, dataset)

def load_entity_data(data_base_path, tokenizer, dataset, relation_aware, field_list=["train", "dev"], maxlen=512):
	if relation_aware:
		return load_entity_data_aware(data_base_path, tokenizer, dataset, field_list, maxlen)
	else:
		return load_entity_data_blind(data_base_path, tokenizer, dataset, field_list, maxlen)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train an entity detector')
	parser.add_argument('--dataset', type=str, help='DrugVar, DrugProt, BC5CDR or CRAFT')
	parser.add_argument('--name', type=str, default='', help='Create a name for the model file')
	parser.add_argument('--NER', action='store_true', help='Whether to only conduct NER')

	args = parser.parse_args()
	config.common['exp_name'] = args.dataset
	config.common['do_train'] = True
	config.common['run_name'] = args.name
	config.common['relation_aware'] = not args.NER

	model_config = config.model["entity_detector"]
	device = config.common["device"]

	# Initialize logger
	run_name = "Ent Dtct Span" if config.common["run_name"] is None else config.common["run_name"]
	logger = init_logger(run_name, model_key="entity_detector")
	

	tokenizer = AutoTokenizer.from_pretrained(config.bert_config["bert_path"], do_lower_case=False)
	data, ent_spans, ent_widths, ent_scores, ent_labels, loss_masks, ent_counts, rel_tuples, sample_indices, rel_types = load_entity_data(data_base_path=config.data["data_base_dir"], tokenizer=tokenizer, dataset=config.common["exp_name"], relation_aware=config.common["relation_aware"])
	if config.common["do_train"] is True:	# training
		train_config = config.train_config
		
		if not train_config["is_pretrain"]:
			print("Created model with fresh parameters.")
			# model_config = config.model["entity_detector"]
			# model_config["ent_cls_num"] = 2 + 1
			model = EntitySpanModel(model_config)
			model.resize_token_embeddings(len(tokenizer)) 
		else:
			model_path = os.path.join(train_config["pretrain_dir"], 'checkpoint_%s.pth.tar' % train_config["pretrain_model"])
			if os.path.exists(model_path):
				print("Loading model from %s" % model_path)
				model = torch.load(model_path)
			else:
				raise RuntimeError("No such checkpoint: %s" % model_path)

		model.to(device)

		input_ids = [i["input_ids"] for i in data["train"]]
		attn_mask = [i["attention_mask"] for i in data["train"]]
		token_type_ids = [i["token_type_ids"] for i in data["train"]]

		train_config = config.train_config
		batch_size = train_config["batch_size"]
		num_epochs = train_config["num_epochs"]
		num_training_steps = (len(data["train"]) + batch_size - 1) // batch_size * num_epochs
		optimizer = optim.AdamW(
			[
				{
					"params": [param for name, param in model.named_parameters() if "bert" in name],
					"lr": train_config["learning_rate"]
				},
				{
					"params": [param for name, param in model.named_parameters() if "bert" not in name],
					"lr": train_config["head_learning_rate"]
				}
			], 
			weight_decay=train_config["weight_decay"]
		)
		lr_schedular = get_lr_schedular(train_config, optimizer, num_training_steps)
		
		best_val_f1 = -1 * float("inf")
		best_epoch = -1

		if not os.path.exists(train_config["train_dir"]):
			os.mkdir(train_config["train_dir"])

		for epoch in range(1, num_epochs + 1):
			start_time = time.time()
			st, ed, batch_num = 0, 0, 0
			losses = []
			while ed < len(data["train"]):
				batch_num += 1
				st_time = time.time()
				st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data["train"])) else len(data["train"])
				batched_input_ids = torch.tensor(input_ids[st:ed]).to(device)
				batched_attn_mask = torch.tensor(attn_mask[st:ed]).to(device)
				batched_token_type_ids = torch.tensor(token_type_ids[st:ed]).to(device)
				batched_ent_span = torch.tensor(ent_spans["train"][st:ed]).to(device)
				batched_ent_width = torch.tensor(ent_widths["train"][st:ed]).to(device)
				batched_ent_score = torch.tensor(ent_scores["train"][st:ed]).to(device)
				batched_ent_label = torch.tensor(ent_labels["train"][st:ed]).to(device)
				batched_loss_mask = torch.tensor(loss_masks["train"][st:ed]).to(device)

				optimizer.zero_grad()
				outputs = model(input_ids=batched_input_ids, 
								attn_mask=batched_attn_mask, 
								token_type_ids=batched_token_type_ids, 
								ent_spans=batched_ent_span, 
								ent_widths=batched_ent_width, 
								ent_scores=batched_ent_score, 
								ent_labels=batched_ent_label, 
								loss_masks=batched_loss_mask)
				loss = outputs["loss"]
				loss.backward()
				optimizer.step()
				lr_schedular.step()
				losses.append(loss.tolist())

				if (batch_num) % 10 == 0:
					print("Epoch %d Batch %d, train loss %f" % (epoch, batch_num, np.mean(losses[-100:])))

			train_loss = np.mean(losses)

			val_loss, val_rel_f1, val_ent_f1 = evaluate_entity(model=model, data=data["dev"], 
											ent_spans=ent_spans["dev"], ent_widths=ent_widths["dev"], ent_scores=ent_scores["dev"], 
											ent_labels=ent_labels["dev"], loss_masks=loss_masks["dev"], ent_counts=ent_counts["dev"], 
											rel_tuples=rel_tuples["dev"], sample_indices=sample_indices["dev"], rel_types=rel_types["dev"], 
											batch_size=batch_size, device=device, tokenizer=tokenizer, dataset=config.common["exp_name"],
											relation_aware=config.common["relation_aware"])
			
			if val_rel_f1 is not None and val_rel_f1 == np.nan:	val_rel_f1 = 0
			if val_ent_f1 == np.nan: val_ent_f1 = 0
			if (config.common["relation_aware"] and val_rel_f1 >= best_val_f1) or (not config.common["relation_aware"] and val_ent_f1 >= best_val_f1):
				best_val_f1 = val_rel_f1 if config.common["relation_aware"] else val_ent_f1
				best_epoch = epoch

				if train_config["save_model"]:
					save_path = os.path.join(train_config["train_dir"], 'checkpoint_%s_%s.pth.tar' % (run_name, config.common["run_id"]))
					with open(save_path, 'wb') as fout:
						torch.save(model, fout)
						print("Saving model to %s" % save_path)
				else:
					print("*"*12, " Model unsaved ", "*"*12)

			epoch_time = time.time() - start_time
			print("Epoch " + str(epoch) + " of " + str(train_config["num_epochs"]) + " took " + str(epoch_time) + "s")
			print("  training loss:                 " + str(train_loss))
			print("  validation loss:               " + str(val_loss))
			print("  validation triplet f1:         " + str(val_rel_f1))
			print("  validation entity f1:  " + str(val_ent_f1))
			print("  best epoch:                    " + str(best_epoch))
			print("  best validation f1:    " + str(best_val_f1))
				

			if logger is not None:
				logger.log({
					"train_loss": train_loss,
					"validation_loss": val_loss,
					"validation_f1": val_rel_f1,
					"validation_entity_f1": val_ent_f1,
					"best validation f1": best_val_f1,
					"learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
				})

	else:	# inference
		eval_config = config.eval_config
		model_path = os.path.join(eval_config["saved_model_dir"], 'checkpoint_%s_%s.pth.tar' % (eval_config["ent_model_name"], eval_config["ent_run_id"]))
		if os.path.exists(model_path):
			print("Loading model from %s" % model_path)
			model = torch.load(model_path)
		else:
			raise RuntimeError("No such checkpoint")

		model.to(device)
		val_loss, val_f1, val_ent_f1 = evaluate_entity(
			model=model, data=data["dev"],
			ent_spans=ent_spans["dev"], ent_widths=ent_widths["dev"],
			ent_scores=ent_scores["dev"],
			ent_labels=ent_labels["dev"], loss_masks=loss_masks["dev"],
			ent_counts=ent_counts["dev"], rel_tuples=rel_tuples["dev"],
			sample_indices=sample_indices["dev"], rel_types=rel_types["dev"],
			batch_size=eval_config["batch_size"],
			device=device, tokenizer=tokenizer, dataset=config.common["exp_name"],
			relation_aware=config.common["relation_aware"]
		)
		print("dev_set triple f1: {}".format(val_f1))
		print("dev_set entity f1:  {}".format(val_ent_f1))
		exit(0)