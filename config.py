import time

run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

common = {
    # "exp_name": 		"DrugProt",
	# "do_train": 		False,		# True: training, False: inference.
	"run_id":			run_id,
	# "run_name": 		"Ent_DrugProt", # Default: Ent Dtct, Count Prdtc or Rel Clsf, for others please specify if
	"device": 			"cuda",
	"logger": 			None, #"wandb",
  	"add_umls_marker": 	False,
	"add_umls_details": False,
	"natural_questions": True,
	# "relation_aware" : True,		# set to False if you only want to extract entities
}


model = {
	"relation_classifier":{
		"initializer_range": 	0.02,
		"rel_num": 				None,
		"hidden_size": 			768,
		"activation_function": 	"gelu",
		"classifier_dropout": 	0.1,
	},
    "entity_detector":{
		"initializer_range": 0.02,
		"width_embed_num": 8,
		"width_embed_dim": 150,
		# for DrugProt, "simple" for 3 classes; "complex" for 7 classes
		"ent_cls_mode": "simple",
        "ent_cls_num": None,
		"hidden_size": 768,
		# "lstm_hidden_size": 1024,
		"head_hidden_size": 256,
		"head_dropout": 0.2,
		"detector_dropout": 0.1
    },
	"count_predictor":{
		"use_cnn": False,
		"initializer_range": 0.02,
		"hidden_size": 768,
		"activation_function": "gelu",
		"predictor_dropout": 0.1,
	}
}

data = {
	"data_base_dir": "./data",
}


train_config = {
	"is_pretrain": False,
	"pretrain_dir": "./train",			# Pre-Training directory for loading pretrained model.
	"pretrain_model": "ent_sci_bb_w0.1_pseuque_20221224_092022",			# Name of pre-trained model.	
	"batch_size": 8,
	"learning_rate": 1e-5, #5e-6,
	"head_learning_rate": 5e-4,
	"weight_decay": 1e-2, 
	"lr_scheduler": {
		"name": "linear", # "exp", "cos", "multi-step", "const", "linear"
		"exp_gamma": 0.999,
		"cos_tmax": 100,
		"cos_eta_min": 5e-7,
		"milestones": [1000],
		"multi_step_gamma": 0.5,
	},
	"num_epochs": 100,
	"save_model": True,
	"train_dir": "./train",		# Directory to save checkpoints during training.
	"seed": 2333,
}

eval_config = {
	"saved_model_dir": "./train",
	# "rel_model_name": "Rel_DrugProt",	# Name of relation model under test.
    # "rel_run_id": "20230116_124836", #"20230116_021811_epoch73", 
	# "cnt_model_name": "Cnt_DrugProt",
	# "cnt_run_id": "20230117_014051",#"20230110_230211",
	# "ent_model_name": "Ent_DrugProt_weighted-sum-repr_7cls",
	# "ent_run_id": "epoch83",
	"model_state_dict_dir": "./wandb", # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    "batch_size": 8,
    
    # results
    "save_res": False,
    "save_res_dir": "./results",
    
    # score: set true only if test set is tagged
    "score": True,
}

bert_config = {
    "bert_path": "allenai/scibert_scivocab_cased",
	"cls_token": 101,
    "sep_token": 102,
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
