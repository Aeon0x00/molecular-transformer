import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Tokenizer import SMILES_tokenizator
from dataset import MOSESDataset, SMILESDataset, collate_fn
from model_decoder import DecoderTransformer

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
    'config': {
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'batch_size': batch_size,
        'lr': lr,
        'epochs': epochs,
        'data_fraction': data_fraction
    },
    'device': str(device),
    'epochs': []
}

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
training_log['model_params'] = sum(p.numel() for p in model.parameters())

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
    t0 = datetime.now()
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(test_loader, train=False)
    elapsed = (datetime.now() - t0).total_seconds()

    epoch_log = {
        'epoch': epoch + 1,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_ppl': float(torch.exp(torch.tensor(train_loss)).item()),
        'val_ppl': float(torch.exp(torch.tensor(val_loss)).item()),
        'overfit_gap': float(val_loss - train_loss),
        'time_seconds': float(elapsed),
        'is_best': val_loss < best_val_loss
    }
    training_log['epochs'].append(epoch_log)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'tokenizer': tokenizer.vocab,
            'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads, 'vocab_size': vocab_size},
            'training_log': training_log
        }, 'best_model_logged.pt')

    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

training_log['end_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
training_log['best_val_loss'] = float(best_val_loss)

with open(log_file, 'w') as f:
    json.dump(training_log, f, indent=2)