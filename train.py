"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import itertools
import os
import time
import math
import pathlib as pl
import datetime as dt
import pickle as pkl

from contextlib import nullcontext

from sklearn.model_selection import train_test_split

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader

from transformers import RobertaTokenizerFast

from model import GPTConfig, GPT

from masking_dataset import MaskingDataset

from tqdm import tqdm

today = dt.datetime.today().strftime("%Y%m%d-%H%M%S")

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
root_dir = pl.Path.cwd()
out_dir = root_dir / 'out'
eval_interval = 200
log_interval = 1
eval_iters = 20
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
#wandb_project = 'owt'
#wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
#dataset = 'openwebtext'
gradient_accumulation_steps = 5  # used to simulate larger batch sizes
batch_size = 4  # if gradient_accumulation_steps > 1, this is the micro-batch size
# block_size = 1024
block_size = 256
# model
# n_layer = 12
n_layer = 6
# n_head = 12
n_head = 6
# n_embd = 768
n_embd = 384
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 6000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
#device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = "mps" if torch.backends.mps.is_available() else "cpu"
#device = torch.device(device)
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# various inits, derived attributes, I/O setup
ddp = int(os.getenv('RANK') or -1) != -1   # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8  # simulate 8 gpus

if master_process:
    out_dir.mkdir(exist_ok=True, parents=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

roberta = RobertaTokenizerFast.from_pretrained('model/byte_level_tokenizer')

data_dir = pl.Path.cwd() / "data"
oscar_dir = data_dir / "oscar"
oscar_20230630_dir = oscar_dir / "20230630"
paths = list(oscar_20230630_dir.glob('**/clean_samples_*.txt'))

samples_pkl_file = oscar_20230630_dir / "samples.pkl"
if samples_pkl_file.is_file():
    with open(samples_pkl_file, 'rb') as f:
        samples = pkl.load(f)
else:
    samples = []
    for file_path in paths:
        with open(file_path, 'r') as f:
            lines = [line.rstrip() for line in f]
            samples.append(lines)
    with open(samples_pkl_file, 'wb') as f:
        pkl.dump(samples, f)

training_data, test_data = train_test_split(samples, test_size=0.20, random_state=42, shuffle=False)

training_dataset = MaskingDataset(training_data, roberta, block_size)
validation_dataset = MaskingDataset(test_data, roberta, block_size)

tg = torch.Generator()
tg.manual_seed(42)

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, generator=tg)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
vocab_size = len(roberta.get_vocab())

# model init
model_args = dict(n_layer=n_layer,
                  n_head=n_head,
                  n_embd=n_embd,
                  block_size=block_size,
                  bias=bias,
                  vocab_size=vocab_size,
                  dropout=dropout)  # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {str(out_dir)}")
    # resume training from a checkpoint.
    ckpt_path = out_dir / 'ckpt.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    lossesv = torch.zeros(eval_iters)
    ii = train_dataloader
    vi = val_dataloader

    for k, ((X,M,Y),(Xv,Mv,Yv)) in enumerate(zip(itertools.islice(ii,eval_iters), itertools.islice(vi, eval_iters))):
        X, M, Y, Xv, Mv, Yv = (t.to(device) for t in (X,M,Y,Xv,Mv, Yv))
        with ctx:
            _, loss = model(X, M, Y)
            losses[k] = loss.item()
            _, loss = model(Xv, Mv, Yv)
            lossesv[k] = loss.item()
    out['train'] = losses.mean()
    out['val'] = lossesv.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project="emil", name="emil_" + today, config=config)

import plotly.express as px
import pandas as pd

plot_dir = root_dir / "plots"
plot_dir.mkdir(exist_ok=True, parents=True)

# training loop
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

all_losses = []

epochs = 4
iter_num2 = 0
for epoch in range(epochs):
    for X, M, Y in tqdm(train_dataloader):
        X, M, Y = (t.to(device) for t in (X,M,Y))
    #for iter_num2 in tqdm(range(iter_num, iter_num+max_iters)):

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num2) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num2 % eval_interval == 0 and master_process:
            fig = px.line(pd.DataFrame(all_losses, columns=["loss"]))
            fig.write_html(plot_dir / today / f"roberta_loss_{len(all_losses)}.html")
            losses = estimate_loss()
            print(f"step {iter_num2}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num2,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num2 > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num2,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, out_dir / today / f'ckpt_{iter_num2}.pt')
        if iter_num2 == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, M, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad()

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num2 % log_interval == 0 and master_process:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num2}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            all_losses.append(lossf)
        local_iter_num += 1
        iter_num2 += 1

if ddp:
    destroy_process_group()
