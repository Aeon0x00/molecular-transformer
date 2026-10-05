import os
import pandas as pd
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MOSESDataset:
    def __init__(self, data_dir='./moses_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
        self.test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'

    def download(self, url, path):
        if os.path.exists(path):
            return
        r = requests.get(url, stream=True)
        total = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def load_train(self):
        self.download(self.train_url, self.train_path)
        return pd.read_csv(self.train_path)['SMILES'].dropna().tolist()

    def load_test(self):
        self.download(self.test_url, self.test_path)
        return pd.read_csv(self.test_path)['SMILES'].dropna().tolist()

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos = tokenizer.vocab['[SOS]']
        self.eos = tokenizer.vocab['[EOS]']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer.encode(smiles)
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        src = [self.sos] + tokens + [self.eos]
        tgt = tokens + [self.eos]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=0)
    pad_mask = (src_padded != 0).unsqueeze(1).unsqueeze(2)
    seq_len = src_padded.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=src_padded.device)).bool()
    mask = pad_mask & causal_mask
    return src_padded, tgt_padded, mask