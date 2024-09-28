import config
import torch
from torch import optim
import numpy as np
import time
import random
from transformers import AutoTokenizer
import os
from relation_models import RelationModel
from utils import get_lr_schedular, init_logger
from load_data import load_relation_data
from scorer import evaluate_relation
import argparse

seed = config.train_config["seed"]
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train a relation classifier')
	parser.add_argument('--dataset', type=str, help='DrugVar, DrugProt or BC5CDR')
	parser.add_argument('--name', type=str, default='', help='Create a name for the model file')

	args = parser.parse_args()
	config.common['exp_name'] = args.dataset
	config.common['do_train'] = True
	config.common['run_name'] = args.name

	device = config.common["device"]

	# Initialize logger
	run_name = "Rel Clsf" if config.common["run_name"] is None else config.common["run_name"]
	logger = init_logger(run_name, "relation_classifier")

	tokenizer = AutoTokenizer.from_pretrained(config.bert_config["bert_path"], do_lower_case=False)
	data, rel_labels = load_relation_data(data_base_path=config.data["data_base_dir"], 
										  tokenizer=tokenizer, dataset=config.common["exp_name"])
	
	if config.common["do_train"] is True:	# training
		train_config = config.train_config
		
		if train_config["is_pretrain"] is False:
			print("Created model with fresh parameters.")
			model_config = config.model["relation_classifier"]
			model_config["rel_num"] = len(rel_labels["train"][0])
			model = RelationModel(model_config)

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

		for epoch in range(1, train_config["num_epochs"] + 1):
			start_time = time.time()
			st, ed, batch_num = 0, 0, 0
			losses = []
			while ed < len(data["train"]):
				batch_num += 1
				st_time = time.time()
				st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data["train"])) else len(data["train"])
				batched_input_ids = torch.tensor(input_ids[st:ed]).to(device)
				batched_attn_mask = torch.tensor(attn_mask[st:ed]).to(device)
				batched_rel_label = torch.tensor(rel_labels["train"][st:ed]).to(device)

				optimizer.zero_grad()
				outputs = model(input_ids=batched_input_ids, attn_mask=batched_attn_mask, rel_labels=batched_rel_label)
				loss, logits = outputs["loss"], outputs["logits"]
				loss.backward()
				optimizer.step()
				lr_schedular.step()
				losses.append(loss.tolist())

				if (batch_num) % 10 == 0:
					print("Epoch %d Batch %d, train loss %f" % (epoch, batch_num, np.mean(losses[-100:])))

			train_loss = np.mean(losses)

			val_loss, val_f1, predictions = evaluate_relation(model=model, data=data["dev"], 
												 rel_labels=rel_labels["dev"], 
												 batch_size=batch_size, 
												 device=device,
												 tokenizer=tokenizer)
			if val_f1 == np.nan:	val_f1 = 0
			if val_f1 > best_val_f1:
				best_val_f1 = val_f1
				best_epoch = epoch

				save_path = os.path.join(train_config["train_dir"], 'checkpoint_%s_%s.pth.tar' % (run_name, config.common["run_id"]))
				if train_config["save_model"] is True:
					with open(save_path, 'wb') as fout:
						torch.save(model, fout)
						print("Saving model to %s" % save_path)
				else:
					print("*** Metrics updated. Model unsaved. ***")

			epoch_time = time.time() - start_time
			print("Epoch " + str(epoch) + " of " + str(train_config["num_epochs"]) + " took " + str(epoch_time) + "s")
			print("  training loss:                 " + str(train_loss))
			print("  validation loss:               " + str(val_loss))
			print("  validation f1:         " + str(val_f1))
			print("  best epoch:                    " + str(best_epoch))
			print("  best validation f1:    " + str(best_val_f1))
				

			if logger is not None:
				logger.log({
					"train_loss": train_loss,
					"validation_loss": val_loss,
					"validation_f1": val_f1,
					"best validation f1": best_val_f1,
					"learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
				})

	else:	# inference
		eval_config = config.eval_config
		model_path = os.path.join(eval_config["saved_model_dir"], 'checkpoint_%s_%s.pth.tar' % (eval_config["rel_model_name"], eval_config["rel_run_id"]))
		if os.path.exists(model_path):
			print("Loading model from %s" % model_path)
			model = torch.load(model_path)
		else:
			raise RuntimeError("No such checkpoint")

		model.to(device)
		val_loss, val_f1, predictions = evaluate_relation(model=model, data=data["dev"], 
											 rel_labels=rel_labels["dev"], 
											 batch_size=eval_config["batch_size"], 
											 device=device,
											 tokenizer=tokenizer)
		print("dev_set f1: {}".format(val_f1))