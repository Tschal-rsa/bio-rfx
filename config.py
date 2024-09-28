import time

run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

common = {
	"run_id":			run_id,
	"device": 			"cuda",
	"logger": 			None,
  	"add_umls_marker": 	False,
	"add_umls_details": False,
	"natural_questions": True,
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
		"ent_cls_mode": "simple",
        "ent_cls_num": None,
		"hidden_size": 768,
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
	"pretrain_dir": "./train",
	"pretrain_model": "ent_sci_bb_w0.1_pseuque_20221224_092022",
	"batch_size": 8,
	"learning_rate": 1e-5,
	"head_learning_rate": 5e-4,
	"weight_decay": 1e-2, 
	"lr_scheduler": {
		"name": "linear",
		"exp_gamma": 0.999,
		"cos_tmax": 100,
		"cos_eta_min": 5e-7,
		"milestones": [1000],
		"multi_step_gamma": 0.5,
	},
	"num_epochs": 100,
	"save_model": True,
	"train_dir": "./train",	
	"seed": 2333,
}

eval_config = {
	"saved_model_dir": "./train",
	"model_state_dict_dir": "./wandb",
    "batch_size": 8,
    "save_res": False,
    "save_res_dir": "./results",
    "score": True,
}

bert_config = {
    "bert_path": "allenai/scibert_scivocab_cased",
	"cls_token": 101,
    "sep_token": 102,
}

cawr_scheduler = {
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
