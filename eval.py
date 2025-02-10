import pathlib as pl
import torch

from transformers import RobertaTokenizerFast

from model import GPTConfig, GPT

device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"

root_dir = pl.Path.cwd()
out_dir = root_dir / 'out'
ckpt_path = out_dir / 'ckpt_1600.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
model_args = {k: checkpoint_model_args[k] for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size')}

# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
state_dict = {k.removeprefix(unwanted_prefix): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']

roberta = RobertaTokenizerFast.from_pretrained('model/byte_level_tokenizer', add_prefix_space=True)

model.eval()

t = "<mask> har en båd med en motor."
t2 = " Jeg har en båd med <mask> motor."
d = roberta(t, return_tensors='pt', return_attention_mask=True, padding='max_length', max_length=gptconf.block_size)
d2 = roberta(t2, return_tensors='pt', return_attention_mask=True, padding='max_length', max_length=gptconf.block_size)
ids = d['input_ids']
print(roberta.decode(ids.squeeze()))
ids2 = d2['input_ids']
print(roberta.decode(ids2.squeeze()))
mask = d['attention_mask']
logits, _ = model(ids, mask)
pids = torch.argmax(logits, -1).squeeze()
pt = roberta.decode(pids)
print(pt)