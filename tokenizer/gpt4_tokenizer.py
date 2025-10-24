
import tiktoken

def truncated_cl100k(num_merges=16000):
    # load tokenizer
    base_tokenizer = tiktoken.get_encoding("cl100k_base")
    mergeable_ranks = base_tokenizer._mergeable_ranks  # {bytes: rank}

    base_byte_count = 256

    saved_ranks = {
        token: rank
        for token, rank in mergeable_ranks.items()
        if rank < num_merges + base_byte_count
    }

    # new tokenizer
    truncated_tokenizer = tiktoken.Encoding(
        name=f"cl100k_base_{num_merges}_merges",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=saved_ranks,
        special_tokens={}  # no special tokens
    )

    return truncated_tokenizer