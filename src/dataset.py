import os
import pandas as pd
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MOSESDataset:
    MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"

    def __init__(self, data_dir='./moses_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')

        if not self._check_files_exist():
            self._download_and_prepare()

    def _check_files_exist(self):
        return os.path.exists(self.train_path) and os.path.exists(self.test_path)

    def _download_and_prepare(self):
        full_csv_path = os.path.join(self.data_dir, 'dataset_v1.csv')

        if not os.path.exists(full_csv_path):
            response = requests.get(self.MOSES_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(full_csv_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        df = pd.read_csv(full_csv_path)
        df[df['SPLIT'] == 'train'][['SMILES']].to_csv(self.train_path, index=False)
        df[df['SPLIT'] == 'test'][['SMILES']].to_csv(self.test_path, index=False)
        os.remove(full_csv_path)

    def load_train(self):
        return pd.read_csv(self.train_path)['SMILES'].tolist()

    def load_test(self):
        return pd.read_csv(self.test_path)['SMILES'].tolist()

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

        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

if __name__ == '__main__':
    dataset = MOSESDataset()
    train = dataset.load_train()
    test = dataset.load_test()