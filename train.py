import torch
import numpy as np

from tokenizer import truncated_cl100k
from model.gpt import GPTLanguageModel


# hyperparameters
num_merges = 16000 # number of merges for the tokenizer
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = n_embd // 64
n_layer = 8
dropout = 0.1
# ------------

torch.manual_seed(1337)

# train and test data
train_data = np.memmap(f"data/train.bin", dtype=np.uint16, mode="r+")
val_data = np.memmap(f"data/val.bin", dtype=np.uint16, mode="r+")

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    arr = np.asarray(data)
    L = arr.shape[0]
    idx = torch.randint(low=0, high=L - block_size - 1, size=(batch_size,))
    x = torch.stack([torch.from_numpy(arr[i:i+block_size]) for i in idx])
    y = torch.stack([torch.from_numpy(arr[i+1:i+block_size+1]) for i in idx])
    x = x.to(device=device, dtype=torch.long, non_blocking=True)
    y = y.to(device=device, dtype=torch.long, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

tokenizer = truncated_cl100k(16000)
vocab_size = tokenizer.n_vocab

model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)

m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save the model
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
}, "model.pt")