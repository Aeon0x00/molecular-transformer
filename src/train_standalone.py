import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import json
import re
import math
import pandas as pd
import requests
from datetime import datetime

class SMILES_tokenizator:
    def __init__(self, smiles_list):
        self.pattern = r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
        self.regex = re.compile(self.pattern)
        tokens = set()
        for smiles in smiles_list:
            tokens.update(self.regex.findall(smiles))
        self.vocab = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': 3}
        for i, token in enumerate(sorted(tokens), start=4):
            self.vocab[token] = i
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, smiles):
        return self.regex.findall(smiles)

class MOSESDataset:
    def __init__(self, data_dir='./moses_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
        self.test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')

    def download(self, url, path):
        if os.path.exists(path):
            return
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    def load_train(self):
        self.download(self.train_url, self.train_path)
        return pd.read_csv(self.train_path)['SMILES'].tolist()

    def load_test(self):
        self.download(self.test_url, self.test_path)
        return pd.read_csv(self.test_path)['SMILES'].tolist()

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SOS = tokenizer.vocab['[SOS]']
        self.EOS = tokenizer.vocab['[EOS]']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer.tokenize(smiles)
        encoded = [self.tokenizer.vocab.get(t, self.tokenizer.vocab['[UNK]']) for t in tokens]
        if len(encoded) > self.max_len - 2:
            encoded = encoded[:self.max_len-2]
        src = [self.SOS] + encoded + [self.EOS]
        tgt = encoded + [self.EOS]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=0)
    mask = (src_padded != 0).unsqueeze(1).unsqueeze(2)
    seq_len = src_padded.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask & causal_mask
    return src_padded, tgt_padded, mask

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

d_model = 512
n_layers = 8
n_heads = 8
batch_size = 64
lr = 3e-4
epochs = 10
data_fraction = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_dir = './training_logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'training_{timestamp}.json')

training_log = {
    'start_time': timestamp,
    'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads, 'batch_size': batch_size, 'lr': lr, 'epochs': epochs, 'data_fraction': data_fraction},
    'device': str(device),
    'epochs': []
}

for key, value in training_log['config'].items():

moses = MOSESDataset()
train_smiles = moses.load_train()
test_smiles = moses.load_test()

train_size = int(len(train_smiles) * data_fraction)
test_size = int(len(test_smiles) * data_fraction)
train_smiles = train_smiles[:train_size]
test_smiles = test_smiles[:test_size]

training_log['dataset'] = {'train_size': len(train_smiles), 'test_size': len(test_smiles)}

tokenizer = SMILES_tokenizator(train_smiles)
vocab_size = len(tokenizer.vocab)
training_log['vocab_size'] = vocab_size

train_dataset = SMILESDataset(train_smiles, tokenizer)
test_dataset = SMILESDataset(test_smiles, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = DecoderTransformer(vocab_size, d_model, n_layers, n_heads, max_len=128).to(device)
total_params = sum(p.numel() for p in model.parameters())
training_log['model_params'] = total_params

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['[PAD]'])

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    with torch.set_grad_enabled(train):
        for src, tgt, mask in tqdm(loader, desc='Train' if train else 'Val'):
            src, tgt, mask = src.to(device), tgt.to(device), mask.to(device)
            logits = model(src, mask)
            logits = logits[:, :tgt.size(1), :]
            loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

best_val_loss = float('inf')
for epoch in range(epochs):
    epoch_start = datetime.now()
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(test_loader, train=False)
    epoch_time = (datetime.now() - epoch_start).total_seconds()

    epoch_log = {
        'epoch': epoch + 1,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_ppl': float(torch.exp(torch.tensor(train_loss)).item()),
        'val_ppl': float(torch.exp(torch.tensor(val_loss)).item()),
        'overfit_gap': float(val_loss - train_loss),
        'time_seconds': float(epoch_time),
        'is_best': val_loss < best_val_loss
    }
    training_log['epochs'].append(epoch_log)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'model': model.state_dict(),
            'tokenizer': tokenizer.vocab,
            'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads, 'vocab_size': vocab_size},
            'training_log': training_log
        }
        torch.save(checkpoint, 'best_model_logged.pt')

    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

training_log['end_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
training_log['best_val_loss'] = float(best_val_loss)

with open(log_file, 'w') as f:
    json.dump(training_log, f, indent=2)