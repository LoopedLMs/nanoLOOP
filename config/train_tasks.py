# Train a looped transformer on procedurally generated reasoning tasks.
# Example: python train.py config/train_tasks.py

# tasks — see tasks/__init__.py for available tasks and level ranges
task_mix = "addition:1-5 subtraction:1-5 multiplication:1-3 sat:2-5 grid:2-5 graph:3-6 maze:2-5"

# model — small, with deep looping for iterative reasoning
n_prelude = 2
n_block = 2
n_coda = 2
n_loop = 8
n_head = 4
n_embd = 128
block_size = 256
dropout = 0.0
bias = False
bptt_k = 4
input_injection = "inject"

# training
batch_size = 64
gradient_accumulation_steps = 1
max_iters = 50000
eval_interval = 500
eval_iters = 100

# optimizer
learning_rate = 3e-4
warmup_iters = 500
lr_decay_iters = 50000
min_lr = 3e-5

# logging
wandb_log = False
wandb_run_name = "tasks-looped"
out_dir = "out-tasks"
