def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

task_ent_labels = {
	'DrugProt': ["CHEMICAL", "GENE"],
	'DrugProt-500': ["CHEMICAL", "GENE"],
	'DrugProt-200': ["CHEMICAL", "GENE"],
    'DrugVar': ["drug", "variant"],
    'DrugVar-500': ["drug", "variant"],
    'DrugVar-200': ["drug", "variant"],
	'BC5CDR': ['Chemical', 'Disease'],
	'CRAFT': ['gene', 'allele', 'genome', 'polypeptide_domain', 'exon', 'QTL', 'primer', 'transcript', 'base_pair', 'transgene', 'sequence_variant', 'vector_replicon', 'promoter', 'plasmid', 'intron', 'gene_cassette', 'loxP_site', 'orthologous_region', 'single', 'double', 'pseudogene', 'flanked', 'targeting_vector', 'floxed', 'homologous', 'BAC', 'homologous_region', 'PCR_product', 'three_prime_UTR', 'SNP', 'EST', 'reverse', 'binding_site', 'forward', 'consensus', 'reverse_primer', 'forward_primer', 'stop_codon', 'start_codon', 'internal_ribosome_entry_site', 'siRNA', 'paralogous_region', 'antisense', 'haplotype', 'five_prime_UTR', 'nuclear_localization_signal', 'coiled_coil', 'origin_of_replication', 'assembly', 'inframe', 'H3K9_trimethylation_site', 'enhancer', 'insertion', 'exon_region', 'splice_site', 'insertion_site', 'FRT_site', 'cDNA_clone', 'rRNA_18S', 'circular', 'alternatively_spliced_transcript', 'ORF', 'propeptide', 'PAC', 'restriction_enzyme_binding_site', 'stop_gained', 'polyA_signal_sequence', 'polyA_sequence', 'regulatory_region', 'shRNA', 'polypeptide_catalytic_motif', 'gene_component_region', 'coding_exon', 'orthologous', 'five_prime_flanking_region', 'contig', 'TSS', 'genomic_clone', 'syntenic', 'plasmid_vector', 'restriction_fragment', 'paralogous', 'protein_coding', 'TF_binding_site', 'gap', 'exon_junction', 'syntenic_region', 'UTR', 'fragment_assembly', 'CDS_region', 'codon', 'orphan', 'AFLP_fragment', 'deletion_breakpoint', 'peptide_helix', 'reading_frame', 'predicted_gene', 'chromosome_breakpoint', 'three_prime_flanking_region', 'transmembrane_polypeptide_region', 'chromosome_arm', 'flanking_region', 'H3K9_dimethylation_site', 'mitochondrial_DNA', 'non_synonymous', 'inversion_site', 'five_prime_noncoding_exon', 'lysosomal_localization_signal', 'consensus_region', 'floxed_gene', 'pre_edited_mRNA', 'coding_region_of_exon', 'terminator', 'polyA_site', 'splice_junction', 'sterol_regulatory_element', 'intron_domain', 'nuclear_gene', 'genetic_marker', 'endosomal_localization_signal', 'simple_sequence_length_variation', 'gene_member_region', 'transcript_region', 'linkage_group', 'processed_pseudogene', 'UTR_region', 'alpha_helix', 'gene_fragment', 'mt_gene', 'STS', 'primer_binding_site', 'repeat_region', 'silencer', 'CDS_fragment', 'ds_oligo', 'proximal_promoter_element', 'predicted_transcript', 'T_to_G_transversion', 'insertion_breakpoint', 'cryptic', 'coding_start', 'match', 'linear', 'RFLP_fragment', 'dicistronic', 'protein_binding_site', 'stem_loop', 'primary_transcript', 'five_prime_coding_exon', 'dicistronic_transcript', 'fingerprint_map', 'DNA_binding_site', 'noncoding_exon', 'mating_type_region', 'expressed_sequence_assembly', 'rRNA_28S', 'open_chromatin_region', 'read', 'DNA_chromosome', 'polyadenylated_mRNA', 'CAAT_signal', 'cosmid', 'nuclear_export_signal', 'catalytic_residue', 'chimeric_cDNA_clone', 'recombination_signal_sequence', 'intergenic_region', 'transmembrane_helix', 'BAC_end', 'transcription_regulatory_region', 'polypeptide_binding_motif', 'morpholino_backbone', 'sequence_feature', 'synthetic_sequence', 'five_prime_intron', 'D_loop', 'transcription_end_site', 'lambda_vector', 'transcript_fusion', 'G_box', 'deletion_junction', 'intramembrane_polypeptide_region', 'transposable_element', 'antisense_RNA', 'dicistronic_mRNA', 'cap', 'mini_gene', 'overlapping_feature_set', 'transversion', 'cloned_cDNA_insert', 'FRT_flanked', 'mobile_genetic_element', 'five_prime_coding_exon_noncoding_region', 'branch_site', 'polypyrimidine_tract', 'bidirectional_promoter', 'T7_RNA_Polymerase_Promoter', 'three_prime_coding_exon_noncoding_region'],
}

task_cnt_question = {
	'DrugProt': "chemicals and genes",
	'DrugProt-500': "chemicals and genes",
    'DrugProt-200': "chemicals and genes",
    'DrugVar': "drugs and variants",
    'DrugVar-500': "drugs and variants",
    'DrugVar-200': "drugs and variants",
    'BC5CDR': 'chemicals and diseases',
    'CRAFT': 'genetic elements, post-translational modifications (PTMs), structural features and other biomedical entities',
}

task_ent_to_id = {
	'DrugProt': {"CHEMICAL": 0, "GENE": 1},
	'DrugProt-500': {"CHEMICAL": 0, "GENE": 1},
    'DrugProt-200': {"CHEMICAL": 0, "GENE": 1},
    'DrugVar': {"drug": 0, "variant": 1},
    'DrugVar-500': {"drug": 0, "variant": 1},
    'DrugVar-200': {"drug": 0, "variant": 1},
    'BC5CDR': {'Chemical': 0, 'Disease': 1},
    'CRAFT': get_labelmap(task_ent_labels['CRAFT'])[0],
}

task_rel_labels = {
	'DrugProt': ['product or substrate', 'activator', 'agonist or antagonist', 'regulator', 'part of', 'inhibitor'],
	'DrugProt-500': ['product or substrate', 'activator', 'agonist or antagonist', 'regulator', 'part of', 'inhibitor'],
    'DrugProt-200': ['product or substrate', 'activator', 'agonist or antagonist', 'regulator', 'part of', 'inhibitor'],
    'DrugVar': ['resistance', 'resistance or non-response', 'response', 'sensitivity'],
    'DrugVar-500': ['resistance', 'resistance or non-response', 'response', 'sensitivity'],
    'DrugVar-200': ['resistance', 'resistance or non-response', 'response', 'sensitivity'],
    'BC5CDR': ['chemical-induced disease'],
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
    },
    'BC5CDR': {
		"chemical-induced disease":{
			'sbj_targets': [1],
			'obj_targets': [2],
		},
	},
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
	},
    'BC5CDR': {
		"chemical-induced disease": "What is the disease that is induced by the chemical?"
	},
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
    },
    'BC5CDR': {
		'chemical-induced disease': 'chemical-induced disease',
	},
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

def get_shifted_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
