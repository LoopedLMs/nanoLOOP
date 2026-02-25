# train a looped character-level shakespeare model
# demonstrates the prelude/recur/coda architecture with input injection

out_dir = "out-shakespeare-char-looped"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "shakespeare-char"
wandb_run_name = "mini-gpt-looped"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# looped model: 2 prelude + 2 block (x4 loops) + 2 coda = 12 effective layers
n_prelude = 2
n_block = 2
n_coda = 2
n_loop = 4
n_head = 6
n_embd = 384
dropout = 0.2
input_injection = "inject"
bptt_k = 2  # backprop through last 2 recurrences

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
