from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
import time
from Models.GPT2 import GPT2
from Configs.train_config import config

def tokenize_function(examples):
	print(examples)
	return tokenizer(examples["text"])

def group_texts(examples):
	# Concatenate all texts.
	concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
	total_length = len(concatenated_examples[list(examples.keys())[0]])
	# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
		# customize this part to your needs.
	total_length = (total_length // block_size) * block_size
	# Split by chunks of max_len.
	result = {
		k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
		for k, t in concatenated_examples.items()
	}
	result["labels"] = result["input_ids"].copy()
	return result

BASE_PATH = '/disk1/dokhyam/Style2Text/'
path_model = BASE_PATH +  '/Refined_gpt_models/Epoch_19_iteration_50.pt'
path_prefixes = BASE_PATH + 'prefix_examples.txt'
sentences_data_path =  BASE_PATH + 'sentences.txt'
gpt2 = GPT2()
gpt2.load_model(path_model)
gpt2.model = gpt2.model.eval().cuda()
tested_model = gpt2
max = 20
eof = '<|endoftext|>'
tokenizer = tested_model.tokenizer
pad_token_id = tokenizer('[PAD]')['input_ids'][0]
block_size = 128

from datasets import load_dataset
datasets = load_dataset("text", data_files={"train":sentences_data_path , "test": path_prefixes})
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(
	group_texts,
	batched=True,
	batch_size=2,
	num_proc=4,
)
# Test
for idx,entry in enumerate(lm_datasets['test']):
	inputs = torch.LongTensor(entry['input_ids']).cuda()
	outputs = tested_model.generate(inputs[0], num_beams=2, no_repeat_ngram_size=2, max_length=max+1, pad_token_id=pad_token_id)
	textual_data = tested_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(textual_data)







