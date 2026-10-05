import os
import pandas as pd
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MOSESDataset:
    TRAIN_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv"
    TEST_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv"

    def __init__(self, data_dir='./moses_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')

    def _download(self, url, path):
        if os.path.exists(path):
            return
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {os.path.basename(path)}') as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def load_train(self):
        self._download(self.TRAIN_URL, self.train_path)
        return pd.read_csv(self.train_path)['SMILES'].dropna().tolist()

    def load_test(self):
        self._download(self.TEST_URL, self.test_path)
        return pd.read_csv(self.test_path)['SMILES'].dropna().tolist()

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_idx = tokenizer.char2idx['[PAD]']
        self.sos_idx = tokenizer.char2idx['[SOS]']
        self.eos_idx = tokenizer.char2idx['[EOS]']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoded = self.tokenizer.encode(smiles)

        if len(encoded) > self.max_len - 2:
            encoded = encoded[:self.max_len - 2]

        src = [self.sos_idx] + encoded + [self.eos_idx]
        tgt = encoded + [self.eos_idx]

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    # Causal triangular and padding mask
    pad_mask = (src_padded != 0).unsqueeze(1).unsqueeze(2)
    seq_len = src_padded.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=src_padded.device)).bool()
    mask = pad_mask & causal_mask

    return src_padded, tgt_padded, mask

if __name__ == '__main__':
    dataset = MOSESDataset()
    train = dataset.load_train()
    test = dataset.load_test()