import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")


def _generate_synthetic_text(n_chars=300_000):
    rng = np.random.default_rng(42)

    words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
        "with", "as", "was", "on", "are", "be", "by", "this", "which", "or",
        "from", "but", "not", "have", "had", "his", "her", "she", "he", "they",
        "at", "one", "all", "would", "there", "their", "we", "him", "been", "has",
        "when", "who", "will", "more", "no", "if", "out", "so", "what", "up",
        "king", "lord", "thou", "thee", "thy", "shall", "hath", "doth", "said",
        "come", "good", "well", "now", "then", "here", "upon", "them", "our",
        "love", "death", "life", "man", "time", "heart", "soul", "world", "day",
        "night", "though", "yet", "my", "me", "I", "you", "your", "we", "us",
    ]

    lines = []
    total = 0
    while total < n_chars:
        n_words = rng.integers(5, 18)
        line = " ".join(rng.choice(words, n_words))
        if rng.random() < 0.15:
            line = line.capitalize() + "."
        elif rng.random() < 0.1:
            line = line.capitalize() + "?"
        else:
            line = line + ","
        lines.append(line)
        total += len(line) + 1

    return "\n".join(lines)


def load_text():
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 10_000:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()

    try:
        print("Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text) > 10_000:
            print(f"Downloaded {len(text):,} chars")
            return text
    except Exception as e:
        print(f"Download failed ({e}), using synthetic text")

    print("Generating synthetic text corpus...")
    text = _generate_synthetic_text()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Generated {len(text):,} chars")
    return text


class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars         = sorted(set(text))
        self.vocab    = chars
        self.stoi     = {c: i for i, c in enumerate(chars)}
        self.itos     = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def make_splits(block_size=256, val_frac=0.1):
    text   = load_text()
    n_val  = int(len(text) * val_frac)
    train_ds = CharDataset(text[:-n_val], block_size)
    val_ds   = CharDataset(text[-n_val:],  block_size)
    print(f"Vocab size   : {train_ds.vocab_size}")
    print(f"Train tokens : {len(train_ds):,}")
    print(f"Val tokens   : {len(val_ds):,}")
    return train_ds, val_ds
