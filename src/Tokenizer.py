import re

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

class SMILES_tokenizator:
    def __init__(self, train=None, vocab=None):
        self.pattern = SMI_REGEX_PATTERN
        self.regex = re.compile(self.pattern)

        if vocab is not None:
            self.char2idx = dict(vocab)
        else:
            all_tokens = set()
            for smiles in (train or []):
                tokens = self.regex.findall(smiles)
                all_tokens.update(tokens)

            special = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
            vocab_list = special + sorted(list(all_tokens))
            self.char2idx = {c: i for i, c in enumerate(vocab_list)}

        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab = self.char2idx
        self.reverse_vocab = self.idx2char
        self.vocab_size = len(self.char2idx)

    def tokenize(self, smiles):
        return self.regex.findall(smiles)

    def encode(self, smiles):
        tokens = self.tokenize(smiles)
        return [self.char2idx.get(t, self.char2idx['[UNK]']) for t in tokens]

    def decode(self, indices):
        chars = []
        for i in indices:
            char = self.idx2char.get(int(i), '[UNK]')
            if char == '[EOS]':
                break
            if char in ['[PAD]', '[SOS]', '[UNK]']:
                continue
            chars.append(char)
        return ''.join(chars)