import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from torch import optim
import config
from copy import deepcopy
import wandb
from transformers import  get_linear_schedule_with_warmup
from preprocess.const import task_umls_ent_labels, task_umls_rels, task_umls_rel_dicts, task_rel_question

def generate_quesiton(rel_type, use_umls, umls_entities=None):
	dataset = config.common["exp_name"]
	rel_str = task_umls_rels[dataset][rel_type]
	if dataset not in task_rel_question.keys():
		base_quesion = "What is the chemical that acts as " + rel_str + " of the gene?"
	else:
		base_quesion = task_rel_question[dataset][rel_type]
	if use_umls == True:
		umls_dict = {}
		details = " "
		for umls_ent_type in task_umls_ent_labels[dataset]:
			umls_dict[umls_ent_type] = [i["mention"] for i in umls_entities if i["sem_type"] == umls_ent_type]
			if len(umls_dict[umls_ent_type]) > 0:
				details += f"Typical {umls_ent_type.lower()} includes {', '.join(umls_dict[umls_ent_type])}. "
		details += f"A{rel_str[1:]} means {task_umls_rel_dicts[dataset][rel_type]}"
		return base_quesion + details
	else:
		return base_quesion

def generate_quesiton_and_context(rel_type, use_umls, natural_questions, umls_entities=None):
	dataset = config.common["exp_name"]
	rel_str = task_umls_rels[dataset][rel_type]
	# quesion = "What is the chemical that acts as " + rel_str + " of the gene?" if natural_questions else rel_type.lower().replace("_", " ").replace("-", " ").strip()
	quesion = (f"What is the chemical that acts as {rel_str} of the gene?" if "DrugProt" in dataset else task_rel_question[dataset][rel_type]) if natural_questions else rel_type
	if use_umls == True:
		umls_dict = {}
		context = " "
		for umls_ent_type in task_umls_ent_labels[dataset]:
			umls_dict[umls_ent_type] = [i["mention"] for i in umls_entities if i["sem_type"] == umls_ent_type]
			if len(umls_dict[umls_ent_type]) > 0:
				context += f"Typical {umls_ent_type.lower()} includes {', '.join(umls_dict[umls_ent_type])}. "
		context += f"A{rel_str[1:]} means {task_umls_rel_dicts[dataset][rel_type]}"
		return quesion, context
	else:
		return quesion, None

def compute_f1_and_auc(y_prob, y_true):
		"""
		:param y_prob: must be a numpy array or a torch tensor, shapes [n_samples, ].
		:param y_true: must be a numpy array or a torch tensor, shapes [n_samples, ], each value should be either 0 or 1.
		"""
		y_true = y_true.reshape(-1)
		y_prob = y_prob.reshape(-1)
		np.seterr(invalid='ignore')
		p, r, t = precision_recall_curve(y_true, y_prob)
		p = np.array(p)
		r = np.array(r)
		eps = 1e-8
		if True in np.isnan(p) or True in np.isnan(r):
			assert(1 not in y_true)
			f1 = None
			auc_score = None 
			threshold = 1
		else:
			f1 = np.max(2 * (p * r) / (p + r + eps))
			auc_score = auc(r, p)
			threshold = t[np.argmax(2 * (p * r) / (p + r + eps))]
		return f1, auc_score, threshold

def get_lr_schedular(train_config, optimizer, num_training_steps):
	if train_config["lr_scheduler"]["name"] == "const":
		lr_schedular = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
							gamma=1)
	elif train_config["lr_scheduler"]["name"] == "exp":
		lr_schedular = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
						gamma=train_config["lr_scheduler"]["exp_gamma"])
	elif train_config["lr_scheduler"]["name"] == "cos":
		lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
						T_max=train_config["lr_scheduler"]["cos_tmax"],
						eta_min=train_config["lr_scheduler"]["cos_eta_min"])
	elif train_config["lr_scheduler"]["name"] == "multi-step":
		lr_schedular = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
						milestones=train_config["lr_scheduler"]["milestones"], 
						gamma=train_config["lr_scheduler"]["multi_step_gamma"])
	elif train_config["lr_scheduler"]["name"] == "linear":
		lr_schedular = get_linear_schedule_with_warmup(optimizer,
						int(num_training_steps * 0.1),
						num_training_steps)
	else:
		raise NotImplementedError
	return lr_schedular

def init_logger(run_name, model_key, factor=None):
	logger_config = deepcopy(config.model[model_key])
	logger_config["add_umls_marker"] = config.common["add_umls_marker"]
	logger_config["add_umls_details"] = config.common["add_umls_details"]
	logger_config["dataset"] = config.common["exp_name"]
	if factor is not None:
		logger_config["factor"] = factor
	if config.common["do_train"] is True:
		logger_config.update(config.train_config)
	else:
		logger_config.update(config.eval_config)

	if "entity" in model_key:
		tags = ["Ent"] if config.common["relation_aware"] else ["Ent-w/o-Rel"]
	elif "relation" in model_key:
		tags = ["Rel"]
	elif "count" in model_key:
		tags = ["Cnt"] if config.common["relation_aware"] else ["Cnt-w/o-Rel"]
	else:
		raise NotImplementedError

	if config.common["logger"] == "wandb":
		wandb.init(project = "BioRE", 
				name = run_name + "_" + config.common["run_id"],
				config = logger_config,
				tags=tags
				)
		logger = wandb
	else:
		logger = None
	return logger
