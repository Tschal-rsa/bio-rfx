import config
import torch
from transformers import AutoTokenizer
import os
from load_data import load_count_data, load_entity_data_blind
from scorer import evaluate_count, evaluate_entity_blind
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluate NER')
	parser.add_argument('--dataset', type=str, help='DrugVar, DrugProt, BC5CDR or CRAFT')
	parser.add_argument('--ent_name', type=str, default='', help='The name of the entity detector model')
	parser.add_argument('--ent_id', type=str, help='The run ID of the entity detector model')
	parser.add_argument('--cnt_name', type=str, default='', help='The name of the number predictor model')
	parser.add_argument('--cnt_id', type=str, help='The run ID of the number predictor model')

	args = parser.parse_args()
	config.common['exp_name'] = args.dataset

	eval_config = config.eval_config
	eval_config['ent_model_name'] = args.ent_name
	eval_config['ent_run_id'] = args.ent_id
	eval_config['cnt_model_name'] = args.cnt_name
	eval_config['cnt_run_id'] = args.cnt_id

	device = config.common["device"]

	cnt_model_path = os.path.join(
		eval_config["saved_model_dir"], 
		'checkpoint_%s_%s.pth.tar' % (eval_config["cnt_model_name"], eval_config["cnt_run_id"])
	)
	ent_model_path = os.path.join(
		eval_config["saved_model_dir"], 
		'checkpoint_%s_%s.pth.tar' % (eval_config["ent_model_name"], eval_config["ent_run_id"])
	)

	if not os.path.exists(cnt_model_path):
		raise RuntimeError("No such count checkpoint: %s. Please check `config.py`." % cnt_model_path)
	if not os.path.exists(ent_model_path):
		raise RuntimeError("No such entity checkpoint: %s. Please check `config.py`." % ent_model_path)
	
	tokenizer = AutoTokenizer.from_pretrained(config.bert_config["bert_path"], do_lower_case=False)

	data = load_count_data(
		data_base_path=config.data["data_base_dir"], 
		tokenizer=tokenizer,
		dataset=config.common["exp_name"],
		relation_aware=False,
		field_list=["dev"]
	)

	print("Loading count model from %s" % cnt_model_path)
	cnt_model = torch.load(cnt_model_path)

	cnt_model.to(device)
	cnt_loss, mse_score, pred_counts = evaluate_count(
		model=cnt_model,
		data=data["dev"], 
		batch_size=eval_config["batch_size"],
		device=device,
		tokenizer=tokenizer
	)
	
	data, ent_spans, ent_widths, ent_scores, ent_labels, \
		loss_masks, ent_counts, rel_tuples, sample_indices, rel_types = load_entity_data_blind(
		data_base_path=config.data["data_base_dir"], 
		tokenizer=tokenizer, 
		dataset=config.common["exp_name"],
		field_list = ["dev"],
		maxlen=512
	)

	print("Loading entity model from %s" % ent_model_path)
	ent_model = torch.load(ent_model_path)

	ent_model.to(device)
	evaluate_entity_blind(
		model=ent_model,
		data=data["dev"],
		ent_spans=ent_spans["dev"],
		ent_widths=ent_widths["dev"],
		ent_scores=ent_scores["dev"],
		ent_labels=ent_labels["dev"],
		loss_masks=loss_masks["dev"],
		ent_counts=pred_counts,
		sample_indices=sample_indices["dev"],
		batch_size=eval_config["batch_size"],
		device=device,
		tokenizer=tokenizer,
		dataset=config.common["exp_name"]
	)
