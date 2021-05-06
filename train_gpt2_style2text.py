# pip install datasets
import datasets
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
d_data_path = '/disk1/dokhyam/Style2Text/directions/'
sentences_data_path = '/disk1/dokhyam/Style2Text/sentences.txt'
val_sentences_data_path = '/disk1/dokhyam/Style2Text/sentences.txt'

saved_models_path = BASE_PATH + 'Models/'

# Training and optimization configs 
gpt2 = GPT2()
model = gpt2.model.train()
tokenizer = gpt2.tokenizer
max_length = 20
eof = '<|endoftext|>'

# dataloaders
from datasets import load_dataset
datasets = load_dataset("text", data_files={"train":sentences_data_path , "validation": val_sentences_data_path})
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=64,
    num_proc=4,
)

optimizer = AdamW(model.parameters(), lr= config['lr'])
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=5000, num_training_steps=-1
)

loss = 0
pad_token_id = tokenizer('[PAD]')['input_ids'][0]
epochs = 20
batch_size=64

# Train/Validation loops
for epoch in range(epochs):
    with tqdm(total=len(dataset) / 2) as pbar:
        for idx,entry in enumerate(train_dataloader):

            if idx % 2000 == 0 and idx != 0:
              for i in range(batch_size):
                with torch.no_grad():
                    outputs = model.generate(validation_input_encodings[i], num_beams=2, no_repeat_ngram_size=2, max_length=max_length+1, pad_token_id=pad_token_id)
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                wandb.log({f"output_text {i}": wandb.Html(f'<p>{output_text}</p>')})
            
            if idx % 50000 == 0:
              torch.save(model.state_dict(), f'{saved_models_path}{idx}_{time.time()}_{int(loss)}.bin')

            model.zero_grad()

            inputs = entry[0].cuda()
            attn_masks = entry[1].cuda()
            labels = entry[2].cuda()
            outputs = model(inputs, labels=labels, attention_mask = attn_masks)

            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(2)
