# Custom BPE tokenizer

A minimal implementation of the Byte Pair Encoding (BPE) algorithm used to tokenize text at the byte level and merge frequent token pairs.

## Usage

```python
from tokenizer import Tokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

data_path = "data/input.txt"
num_merges = 50

tokenizer = Tokenizer(data_path, GPT4_SPLIT_PATTERN, num_merges=num_merges)

ids = tokenizer.encode("Hello world!")
text = tokenizer.decode(ids)

```

This trains a BPE tokenizer from scratch starting from 256 byte tokens and performing `num_merges = 50` merge operations.

The tokenizer has an attribute `itob` mapping the token IDs to bytes.