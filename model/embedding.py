import torch

from math import log


def sinusoidal_embedding(max_len, d_model):
    """ implement sinusoidal positional embeddings """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) *
        (-log(10000.0) / d_model)
    )  # (d_model/2,)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)