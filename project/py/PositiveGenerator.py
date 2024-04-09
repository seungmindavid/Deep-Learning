import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    # nn.Embedding is a dictionary kind of a layer that just maps number to the vector every time and this vector is learned by the model
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    # According to "Attention is all you need", in the embedding layers, we need to multiply those weights by sqrt(d_model)
    return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, seq_len):
    super().__init__()

    pe = torch.zeros(seq_len, d_model)

    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

    pe[:,0::2] = torch.sin(pos*div)
    pe[:,1::2] = torch.cos(pos*div)
    # pe[:,:, 0::2] = torch.sin(pos*div)
    # pe[:,:, 1::2] = torch.cos(pos*div)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x = x + self.pe.requires_grad_(False)
    x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
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

"""## Multi-head Attention"""

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

    h = self.h
    d_k = self.d_k
    d_model = self.d_model

    query = self.W_q(q)
    key = self.W_k(k)
    value = self.W_v(v)

    # It's necessary since decode process will be (target, src_output, src_output, mask)
    # which means, q_seq_len and k_seq_len will be different.
    q_B, q_seq_len, _ = query.size()
    k_B, k_seq_len, _ = key.size()
    v_B, v_seq_len, _ = value.size()
    # size: (Batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_model // h)
    query = query.view(q_B, q_seq_len, h, d_k)
    key = key.view(k_B, k_seq_len, h, d_k)
    value = value.view(v_B, v_seq_len, h, d_k)

    # (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
    # (Batch, h, Seq_len, d_k) -> (Batch * h, Seq_len, d_k)
    query = query.transpose(1,2).contiguous().view(q_B * h, q_seq_len, d_k)
    key = key.transpose(1,2).contiguous().view(k_B * h, k_seq_len, d_k)
    value = value.transpose(1,2).contiguous().view(v_B * h, v_seq_len, d_k)

    # Attention: W
    # paying attention to each sequences, therefore size should be (Batch *h, Seq_len, Seq_len)
    W = query @ key.transpose(1,2)
    W = W / (d_model ** 0.5)

    # If there is a mask, make masked spots -INF
    # seq_len must be equal to query's sequence length.
    if mask is not None:
      mask = mask.view(k_B * h, k_seq_len, k_seq_len) # (B, h, Seq_len, Seq_len) => (B * h, Seq_len, Seq_len)
      # It's for when we are generating new positive-toned manner text from the empty decoder_input
      if q_seq_len != k_seq_len:
        mask = mask[:,:q_seq_len,:]
      W = W.masked_fill_(mask == 0, float('-inf'))

    W = W.softmax(dim = -1)
    # drop out
    W = self.dropout(W)

    out = W @ value # (B * h, seq_len, d_k)
    B, Seq_len, d_k = out.size()
    B = B // h
    out = out.view(B, h, Seq_len, d_k)
    out = out.transpose(1,2).contiguous().view(B, Seq_len, h * d_k)
    return self.W_o(out)

"""## Encoder & Decoder"""

class EncoderBlock(nn.Module):
  def __init__(self, config):
    super(EncoderBlock, self).__init__()

    self.d_model = config['d_model']

    self.MultiHeadAttention = MultiHeadAttention(config)

    self.ln_1 = nn.LayerNorm(self.d_model)
    self.ln_2 = nn.LayerNorm(self.d_model)
    self.FeedForward = FeedForward(config)

  def forward(self, x, src_mask):
    x = x + self.MultiHeadAttention(x, x, x, src_mask)
    x = self.ln_1(x)
    x = x + self.FeedForward(x)
    x = self.ln_2(x)
    return x

class Encoder(nn.Module):
  def __init__(self, config):
    super(Encoder, self).__init__()

    self.depth = config['depth']
    # Encoder: blocks of encoder blocks
    self.blocks = nn.ModuleList([
        EncoderBlock(config) for _ in range(self.depth)
    ])
    self.blocks = nn.Sequential(*self.blocks)

  def forward(self, x, src_mask):
    for block in self.blocks:
      x = block(x, src_mask)
    return x

class DecoderBlock(nn.Module):
  def __init__(self, config):
    super(DecoderBlock, self).__init__()

    self.d_model = config['d_model']

    self.SelfHeadAttention = MultiHeadAttention(config)
    self.CrossHeadAttention = MultiHeadAttention(config)

    self.ln_1 = nn.LayerNorm(self.d_model)
    self.ln_2 = nn.LayerNorm(self.d_model)
    self.ln_3 = nn.LayerNorm(self.d_model)

    self.FeedForward = FeedForward(config)

  def forward(self, x, encoder_out, src_mask, tgt_mask):
    # x: target, in our case positively-toned comment
    x = x + self.SelfHeadAttention(x, x, x, tgt_mask)
    x = self.ln_1(x)
    x = x + self.CrossHeadAttention(x, encoder_out, encoder_out, src_mask)
    x = self.ln_2(x)
    x = x + self.FeedForward(x)
    x = self.ln_3(x)
    return x

class Decoder(nn.Module):
  def __init__(self, config):
    super(Decoder, self).__init__()

    self.depth = config['depth']

    self.blocks = nn.ModuleList([
        DecoderBlock(config) for _ in range(self.depth)
    ])
    self.blocks = nn.Sequential(*self.blocks)

  def forward(self, x, encoder_out, src_mask, tgt_mask):
    for block in self.blocks:
      x = block(x, encoder_out, src_mask, tgt_mask)
    return x

"""# Transformer"""

class PositiveGenerator(nn.Module):
  def __init__(self, config, vocab_size):
    super(PositiveGenerator, self).__init__()

    self.encoder = Encoder(config)
    self.decoder = Decoder(config)

    self.d_model = config['d_model']
    self.seq_len = config['seq_len']

    # Input Embedding for source and target
    self.src_embedding = InputEmbedding(vocab_size, self.d_model)
    self.tgt_embedding = InputEmbedding(vocab_size, self.d_model)

    # Positional Embedding for source and target
    self.src_pos_embedding = PositionalEmbedding(self.d_model, self.seq_len)
    self.tgt_pos_embedding = PositionalEmbedding(self.d_model, self.seq_len)

    self.norm = nn.LayerNorm(self.d_model)
    self.projection = nn.Linear(self.d_model, vocab_size)

  def encode(self, source, src_mask):
    source = self.src_embedding(source)
    source = self.src_pos_embedding(source)
    return self.encoder(source, src_mask)

  def decode(self, target, encoder_out, src_mask, tgt_mask):
    target = self.tgt_embedding(target)
    target = self.tgt_pos_embedding(target)
    return self.decoder(target, encoder_out, src_mask, tgt_mask)

  def forward(self, decoder_out):
    #out = self.norm(decoder_out)
    out = self.projection(decoder_out)
    return torch.log_softmax(out, dim=-1)
