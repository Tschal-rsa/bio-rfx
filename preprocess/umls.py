from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=False)

def append_marker_tokens(tokenizer, umls_ent_labels):
	'''
	Append marker tokens to tokenizer
	'''
	new_tokens = ['<START>', '<END>']
	for label in umls_ent_labels:
		new_tokens.append('<START=%s>'%label)
		new_tokens.append('<END=%s>'%label)
	tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
	print('# vocab after adding markers: %d'%len(tokenizer))
	
	marker_tok_2_id = {}
	for new_tok in new_tokens:
		marker_tok_2_id[new_tok] = tokenizer.encode_plus(text=new_tok)['input_ids'][1]
	return marker_tok_2_id
