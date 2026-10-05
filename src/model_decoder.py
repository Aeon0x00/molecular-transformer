import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_model * 4, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.out(x)

def build_decoder_transformer(vocab_size, d_model=512, n_layers=6, heads=8, max_len=128, dropout=0.1):
    return DecoderTransformer(vocab_size, d_model, n_layers, heads, max_len, dropout)