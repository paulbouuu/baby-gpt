
# baby-gpt

Original code created in the video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7) in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT.

## What I implemented

This project extends the original GPT by scaling both the dataset and the model. I also re-engineered the tokenization and data pipeline in order to handle a larger dataset.

### Tokenization
- **Custom BPE tokenizer** : I implemented a BPE algorithm from scratch (documentation in [`tokenizer/documentation.md`](tokenizer/documentation.md)) to understand how tokenization works. The implementation is intentionally simple and educational and is not optimized for speed. Therefore I use `tiktoken` to handle the tokenisation for later large-scale experiments.
- **GPT-4–based tokenizer** : for large-scale runs, I built a tokenizer based on OpenAI’s `cl100k_base` (the GPT-4 tokenizer). I added the ability to truncate it to a specific number of merges (set to 16k) to control vocabulary size. It allows to evaluate the trade-off between compression and model complexity

### Data pipeline
- replaced the original single-file, character-level dataset (~1 MB) with a 1 GB+ corpus (OpenWebText corpus).
- implemented a preprocessing stage using `np.memmap` to store all token IDs in binary format (`train.bin` and `val.bin`), allowing the model to stream data directly from disk with minimal RAM usage
- rewrote the `get_batch` function to sample training blocks from the memory-mapped data

### Model scaling

Scaled the model from 10M to 42M parameters:
  - increased vocab size from 65 to 16,256
  - increased embedding size, number of heads, and context length
  - implemented positional embeddings from scratch: **sinusoidal embedding** and **RoPE** (Rotary Positional Embeddings)
  - trained with the new tokenizer and dataset to validate scalability and performance improvements
  - rule of thumb ~20 training tokens per parameter: we had 724.3M training tokens so ~17.3 training tokens/parameter

I kept the lightweight GPT architecture developed by Karpathy to maintain clarity and readability.

### Usage

1. Data downloading and preprocessing: `python data/prepare.py` (tokenization and storage)

NB: since the data is hosted on Hugging Face, you will need to login to Hugging Face

2. Training job: `python train.py`.

I did my experiments with the following hyperparameters:

```python
num_merges = 16000 # number of merges for the tokenizer (+256 base tokens)
batch_size = 32 # how many independent sequences we process in parallel
block_size = 512 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 512
n_head = n_embd // 64
n_layer = 8
dropout = 0.1
```

## What's next?

1. Add special tokens (e.g. end-of-text, beginning-of-sentence, padding, unknown) to the tokenizer and retrain the model
2. Instruction fine-tuning on a smaller, high-quality dataset, selection (here)[https://huggingface.co/collections/librarian-bots/top-10-instruction-tuning-datasets]