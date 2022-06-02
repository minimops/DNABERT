from itertools import product

def create_vocab(path, k):
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    bases = ["A", "T", "C", "G"]
    tokens.extend([''.join(comb) for comb in product(bases, repeat=k)])
    with open(path + "/vocab.txt", "w+") as f:
        f.write('\n'.join(tokens))

#create_vocab("src/transformers/dnabert-config/bert-config-small-8", 8)