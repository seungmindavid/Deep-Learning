

import os
import pathlib

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

# General
from tqdm import tqdm
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


import math

class InputEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    # nn.Embedding is a dictionary kind of a layer that just maps number to the vector every time and this vector is learned by the model
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    # According to "Attention is all you need", in the embedding layers, we need to multiply those weights by sqrt(emb_dim)
    return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, seq_len):
    super().__init__()

    pe = torch.zeros(seq_len, d_model)

    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

    pe[:, 0::2] = torch.sin(pos*div)
    pe[:, 1::2] = torch.cos(pos*div)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
    #x = x + self.pe.requires_grad_(False)
    return x

class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.d_model = config['d_model']
    self.d_ff = self.d_model * 4

    self.ffn = nn.Sequential(
        nn.Linear(self.d_model, self.d_ff),
        nn.ReLU(),
        nn.Linear(self.d_ff, self.d_model),
        nn.Dropout(config['dropout'])
    )

  def forward(self, x):
    return self.ffn(x)

"""## Multi-Head Attention"""

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super(MultiHeadAttention, self).__init__()

    self.d_model = config['d_model']
    self.h = config['h']

    assert self.d_model % self.h == 0, "d_model must be divisible by h"


    self.d_k = self.d_model // self.h
    # Query, key, value
    self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
    self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
    self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

    # Last Layer
    self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

    # dropout
    self.dropout = nn.Dropout(config['dropout'])

  def forward(self, q, k, v, mask):
    # input size: torch.Size([B, seq_len, d_model])
    B, Seq_len, d_model = q.size()

    h = self.h
    d_k = self.d_k
    d_model = self.d_model

    query = self.W_q(q)
    key = self.W_k(k)
    value = self.W_v(v)

    # size: (Batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_model // h)
    query = query.view(B, Seq_len, h, d_k)
    key = key.view(B, Seq_len, h, d_k)
    value = value.view(B, Seq_len, h, d_k)

    # (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
    # (Batch, h, Seq_len, d_k) -> (Batch * h, Seq_len, d_k)
    query = query.transpose(1,2).contiguous().view(B * h, Seq_len, d_k)
    key = key.transpose(1,2).contiguous().view(B * h, Seq_len, d_k)
    value = value.transpose(1,2).contiguous().view(B * h, Seq_len, d_k)

    # Attention: W
    # paying attention to each sequences, therefore size should be (Batch *h, Seq_len, Seq_len)
    W = query @ key.transpose(1,2)
    W = W / (d_model ** 0.5)

    if mask is not None:
      mask = mask.view(B * h, Seq_len, Seq_len)
      W = W.masked_fill(mask == 0, float("-inf"))

    W = W.softmax(dim = -1)
    # drop out
    W = self.dropout(W)

    out = W @ value # (B * h, seq_len, d_k)
    out = out.view(B, h, Seq_len, d_k)
    out = out.transpose(1,2).contiguous().view(B, Seq_len, h * d_k)
    return self.W_o(out)

"""### Encoder Block"""

class EncoderBlock(nn.Module):
  def __init__(self, config):
    super(EncoderBlock, self).__init__()

    self.d_model = config['d_model']

    self.MultiHeadAttention = MultiHeadAttention(config)
    self.ln_1 = nn.LayerNorm(self.d_model)
    self.ln_2 = nn.LayerNorm(self.d_model)
    self.FeedForward = FeedForward(config)

  def forward(self, x, mask):
    x = x + self.MultiHeadAttention(x, x, x, mask)
    x = self.ln_1(x)
    x = x + self.FeedForward(x)
    x = self.ln_2(x)
    return x

"""# Encoder only Transformer"""

class ToxicDetector(nn.Module):
  def __init__(self, config, vocab_size):
    super(ToxicDetector, self).__init__()
    self.d_model = config['d_model']
    self.seq_len = config['seq_len']
    self.num_classes = config['num_classes']
    self.depth = config['depth']

    # Input Embedding, Positional Embedding
    self.emb = InputEmbedding(vocab_size, self.d_model)
    self.pe = PositionalEmbedding(self.d_model, self.seq_len)

    # Encoder: blocks of encoder blocks
    self.Encoder = nn.ModuleList([
        EncoderBlock(config) for _ in range(self.depth)
    ])
    self.Encoder = nn.Sequential(*self.Encoder)
    self.mlp = nn.Linear(self.d_model, self.num_classes)

  def forward(self, x, mask):
    x = self.pe(self.emb(x))

    for block in self.Encoder:
      x = block(x, mask)
    x = x.mean(dim=1)
    x = self.mlp(x)
    # return x
    return torch.log_softmax(x, dim=1)


