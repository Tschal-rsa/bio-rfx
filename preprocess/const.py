task_ent_labels = {
	'DrugProt': ["CHEMICAL", "GENE"],
	'DrugProt-500': ["CHEMICAL", "GENE"],
	'DrugProt-200': ["CHEMICAL", "GENE"],
    'DrugVar': ["drug", "variant"],
    'DrugVar-500': ["drug", "variant"],
    'DrugVar-200': ["drug", "variant"]
}

task_cnt_question = {
	'DrugProt': "chemicals and genes",
	'DrugProt-500': "chemicals and genes",
    'DrugProt-200': "chemicals and genes",
    'DrugVar': "drugs and variants",
    'DrugVar-500': "drugs and variants",
    'DrugVar-200': "drugs and variants"
}

task_ent_to_id = {
	'DrugProt': {"CHEMICAL": 0, "GENE": 1},
	'DrugProt-500': {"CHEMICAL": 0, "GENE": 1},
    'DrugProt-200': {"CHEMICAL": 0, "GENE": 1},
    'DrugVar': {"drug": 0, "variant": 1},
    'DrugVar-500': {"drug": 0, "variant": 1},
    'DrugVar-200': {"drug": 0, "variant": 1}
}

task_rel_labels = {
	'DrugProt': ['product or substrate', 'activator', 'agonist or antagonist', 'regulator', 'part of', 'inhibitor'],
	'DrugProt-500': ['product or substrate', 'activator', 'agonist or antagonist', 'regulator', 'part of', 'inhibitor'],
    'DrugProt-200': ['product or substrate', 'activator', 'agonist or antagonist', 'regulator', 'part of', 'inhibitor'],
    'DrugVar': ['resistance', 'resistance or non-response', 'response', 'sensitivity'],
    'DrugVar-500': ['resistance', 'resistance or non-response', 'response', 'sensitivity'],
    'DrugVar-200': ['resistance', 'resistance or non-response', 'response', 'sensitivity']
}

DrugProt_tree = {
	"ACTIVATOR": "activator", 
	"AGONIST-ACTIVATOR": "activator", 
	"AGONIST": "agonist or antagonist",
	"ANTAGONIST": "agonist or antagonist", 
	"DIRECT-REGULATOR": "regulator", 
	"INDIRECT-DOWNREGULATOR": "regulator", 
	"INDIRECT-UPREGULATOR": "regulator", 
	"INHIBITOR": "inhibitor", 
	"AGONIST-INHIBITOR": "inhibitor", 
	"PART-OF": "part of", 
	"PRODUCT-OF": "product or substrate", 
	"SUBSTRATE": "product or substrate", 
	"SUBSTRATE_PRODUCT-OF": "product or substrate"
}

task_tup_limits = {
    'DrugVar': {
        "resistance": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "resistance or non-response": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "response": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "sensitivity":{
			'sbj_targets': [2],
			'obj_targets': [1]
		},
    },
    'DrugVar-500': {
        "resistance": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "resistance or non-response": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "response": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "sensitivity":{
			'sbj_targets': [2],
			'obj_targets': [1]
		},
    },
	'DrugVar-200': {
        "resistance": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "resistance or non-response": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "response": {
			'sbj_targets': [2],
			'obj_targets': [1]
		},
        "sensitivity":{
			'sbj_targets': [2],
			'obj_targets': [1]
		},
    },
    'DrugProt': {
        "product or substrate": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "activator": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "agonist or antagonist": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "regulator":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "part of":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
		"inhibitor":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
    },
    'DrugProt-500': {
        "product or substrate": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "activator": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "agonist or antagonist": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "regulator":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "part of":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
		"inhibitor":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
    },
    'DrugProt-200': {
        "product or substrate": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "activator": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "agonist or antagonist": {
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "regulator":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
        "part of":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
		"inhibitor":{
			'sbj_targets': [1],
			'obj_targets': [2]
		},
    }
}

task_umls_ent_labels = {
	'DrugProt': ['Organic Chemical', 'Amino Acid, Peptide, or Protein', 'Gene or Genome', 'Pharmacologic Substance'],
}

task_rel_question = {
    'DrugVar':
    {
        "resistance": "What is the gene variant that acts as the resistance to the drug?",
        "resistance or non-response": "What is the gene variant that does not response to the drug?", 
        "response": "What is the gene variant that responses to the drug?", 
        "sensitivity": "What is the gene variant that is sensitive to the drug?",
    },
    'DrugVar-500':
    {
        "resistance": "What is the gene variant that acts as the resistance to the drug?",
        "resistance or non-response": "What is the gene variant that does not response to the drug?", 
        "response": "What is the gene variant that responses to the drug?", 
        "sensitivity": "What is the gene variant that is sensitive to the drug?",
    },
    'DrugVar-200':
    {
        "resistance": "What is the gene variant that acts as the resistance to the drug?",
        "resistance or non-response": "What is the gene variant that does not response to the drug?", 
        "response": "What is the gene variant that responses to the drug?", 
        "sensitivity": "What is the gene variant that is sensitive to the drug?",
	}
}

task_umls_rels = {
	'DrugProt':
	{
		'product or substrate': 'a product or substrate', 
		'activator': 'an activator', 
		'agonist or antagonist': 'an agonist or antagonist', 
		'regulator': 'a regulator', 
		'part of': 'a part', 
		'inhibitor': 'an inhibitor'
	},
	'DrugProt-500':
	{
		'product or substrate': 'a product or substrate', 
		'activator': 'an activator', 
		'agonist or antagonist': 'an agonist or antagonist', 
		'regulator': 'a regulator', 
		'part of': 'a part', 
		'inhibitor': 'an inhibitor'
	},
    'DrugProt-200':
	{
		'product or substrate': 'a product or substrate', 
		'activator': 'an activator', 
		'agonist or antagonist': 'an agonist or antagonist', 
		'regulator': 'a regulator', 
		'part of': 'a part', 
		'inhibitor': 'an inhibitor'
	},
    'DrugVar':
    {
        "resistance": "resistance",
        "resistance or non-response": "resistance or non-response", 
        "response": "response", 
        "sensitivity": "sensitivity",
	},
	'DrugVar-500':
    {
        "resistance": "resistance",
        "resistance or non-response": "resistance or non-response", 
        "response": "response", 
        "sensitivity": "sensitivity",
	},
    'DrugVar-200':
    {
        "resistance": "resistance",
        "resistance or non-response": "resistance or non-response", 
        "response": "response", 
        "sensitivity": "sensitivity",
    }
}

task_umls_rel_dicts = {
	'DrugProt':
	{
		"ACTIVATOR": "a substance that makes another substance active or reactive, induces a chemical reaction, or combines with an enzyme to increase its catalytic activity.",
		"AGONIST": "a substance that can combine with a receptor on a cell to initiate signal transduction.",
		"AGONIST-ACTIVATOR": "a substance that makes an agonist active or reactive, induces a chemical reaction, or combines with an enzyme to increase its catalytic activity.",
		"AGONIST-INHIBITOR": "a substance that reduces or suppresses the activity of an agonist.",
		"ANTAGONIST": "a type of receptor ligand or drug that blocks or dampens a biological response by binding to and blocking a receptor.",
		"DIRECT-REGULATOR": "a substance that directly affects the amount of product or the progress of a biochemical reaction or process.", 
		"INDIRECT-DOWNREGULATOR": "a substance that indirectly inhibits or suppresses the normal response of an organ or system.", 
		"INDIRECT-UPREGULATOR": "a substance that indirectly increases the responsiveness of a cell or organ to a stimulus.", 
		"INHIBITOR": "a substance that reduces or suppresses the activity of another substance (such as an enzyme).", 
		"PART-OF": "a substance that composes, with one or more other physical units, some larger whole. This includes component of, division of, portion of, fragment of, section of, and layer of.", 
		"PRODUCT-OF": "a substance that is brought forth, generated or created. This includes yields, secretes, emits, biosynthesizes, generates, releases, discharges, and creates.", 
		"SUBSTRATE": "a chemical species being observed in a chemical reaction, or a surface on which other chemical reactions or microscopy are performed.", 
		"SUBSTRATE_PRODUCT-OF": "the product of a surface on which other chemical reactions or microscopy are performed.",
	},

}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

def get_shifted_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
