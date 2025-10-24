import os
import numpy as np

from tqdm import tqdm
from random import random

from datasets import load_dataset
from ..tokenizer import truncated_cl100k

# parameters
TRAIN_TEST_SPLIT = 0.9
OUT_DIR = "data"
# ------------

ds = load_dataset(
    "parquet",
    data_files="hf://datasets/Skylion007/openwebtext@refs/pr/19/data/train-0000[0-2]-of-00080.parquet",
    split="train",
    streaming=True
)

tokenizer = truncated_cl100k()

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

train_data = open(os.path.join(OUT_DIR, "train.bin"), "wb")
val_data = open(os.path.join(OUT_DIR, "val.bin"), "wb")

n_train_tokens = 0
n_val_tokens = 0

for sample in tqdm(ds):
    text = sample['text']
    ids = np.asarray((tokenizer.encode(text + "\n\n\n")), dtype=np.uint16)

    if random() < TRAIN_TEST_SPLIT:
        ids.tofile(train_data)
        n_train_tokens += len(ids)
    else:
        ids.tofile(val_data)
        n_val_tokens += len(ids)

print(f"Number of training tokens: {n_train_tokens}")
print(f"Number of validation tokens: {n_val_tokens}")