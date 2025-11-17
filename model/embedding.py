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


def build_rope_cache(max_len, head_dim):
    """ build RoPE cos and sin cache """
    assert head_dim % 2 == 0
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    inverse_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # (head_dim/2,)
    frequencies = position * inverse_freq.unsqueeze(0)  # (T, head_dim/2)

    cos = torch.repeat_interleave(torch.cos(frequencies), 2, dim=-1)  # (T, head_dim)
    sin = torch.repeat_interleave(torch.sin(frequencies), 2, dim=-1)  # (T, head_dim)
    return cos, sin


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


def apply_rope(q, k, cos, sin, T):
    """ apply RoPE to q and k """
    cos_t = cos[:T, :].unsqueeze(0)  # (1, T, D)
    sin_t = sin[:T, :].unsqueeze(0)  # (1, T, D)

    q_rot = (q * cos_t) + (rotate_half(q) * sin_t) # (B, T, D)
    k_rot = (k * cos_t) + (rotate_half(k) * sin_t) # (B, T, D)
    return q_rot, k_rot