from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
import time
from Models.GPT2 import GPT2
from Configs.train_config import config

BASE_PATH = '/disk1/dokhyam/Style2Text/'
path_model = '/Refined_gpt_models/Epoch_19_iteration_50.pt'
gpt2 = GPT()
gpt2.load_model(path_model)
gpt2.model = gpt2.model.eval().cuda()
tested_model = gpt2
max = 20
eof = '<|endoftext|>'
pad_token_id = tested_model.tokenizer('[PAD]')['input_ids'][0]
# Test

textual_data = tested_model.tokenizer.decode(token_ids = '', skip_special_tokens=False).split('<comment>')
target_text = textual_data[1].split(eof)[0]
# targets.append(target_text)
input_text = textual_data[0] 
# inputs.append(input_text)

comment_idx = list(proccessed_data[i]).index(dataset.comment_encoding) + 1
input_encoding = proccessed_data[i][:comment_idx].unsqueeze(0).cuda()

input_encodings.append(input_encoding)

results = []
with torch.no_grad():
    outputs = tested_model.model.generate(input_encodings, num_beams=2, no_repeat_ngram_size=2, max_length=max+1, pad_token_id=pad_token_id)
    output_text = tested_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(output_text)


