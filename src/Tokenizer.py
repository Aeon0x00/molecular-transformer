import re

pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

class SMILES_tokenizator:
    def __init__(self, train=None, vocab=None):
        self.regex = re.compile(pattern)
        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            tokens = set()
            for s in (train or []):
                tokens.update(self.regex.findall(s))
            special = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
            all_tokens = special + sorted(list(tokens))
            self.vocab = {c: i for i, c in enumerate(all_tokens)}

        self.char2idx = self.vocab
        self.idx2char = {i: c for c, i in self.vocab.items()}
        self.reverse_vocab = self.idx2char
        self.vocab_size = len(self.vocab)

    def tokenize(self, smiles):
        return self.regex.findall(smiles)

    def encode(self, smiles):
        tokens = self.tokenize(smiles)
        unk = self.vocab.get('[UNK]', 3)
        return [self.vocab.get(t, unk) for t in tokens]

    def decode(self, indices):
        chars = []
        for i in indices:
            c = self.idx2char.get(int(i), '')
            if c == '[EOS]':
                break
            if c in ['[PAD]', '[SOS]', '[UNK]']:
                continue
            chars.append(c)
        return ''.join(chars)