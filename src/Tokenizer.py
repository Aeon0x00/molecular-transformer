import re

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

class SMILES_tokenizator:
    def __init__(self, train):
        all_tokens = set()
        for smiles in train:
            tokens = re.findall(SMI_REGEX_PATTERN, smiles)
            all_tokens.update(tokens)

        special = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
        vocab = special + sorted(list(all_tokens))

        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(vocab)

    def encode(self, smiles):
        tokens = re.findall(SMI_REGEX_PATTERN, smiles)
        return [self.char2idx.get(t, self.char2idx['[UNK]']) for t in tokens]

    def decode(self, indices):
        chars = []
        for i in indices:
            char = self.idx2char.get(i, '[UNK]')
            if char in ['[PAD]', '[SOS]', '[EOS]']:
                break
            chars.append(char)
        return ''.join(chars)