import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, root, seq_len=512):
        self.files = glob.glob(os.path.join(root, "*.npy"))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        if len(x) < self.seq_len + 1:
            pad = self.seq_len + 1 - len(x)
            x = np.concatenate([x, np.zeros(pad, dtype=np.int32)])
        x = x[:self.seq_len+1]
        inp = torch.tensor(x[:-1], dtype=torch.long)
        tgt = torch.tensor(x[1:], dtype=torch.long)
        return inp, tgt

def sample_sequence(model, vocab, steps=512, temperature=1.0, cond_instruments=None, device="cpu"):
    import torch.nn.functional as F
    model.eval()
    seq = []
    if cond_instruments:
        for i in cond_instruments:
            seq.append(vocab.stoi.get(f"INST_{i}", 0))
    x = torch.tensor(seq[-128:] if seq else [0], dtype=torch.long, device=device).unsqueeze(0)
    state = None
    for _ in range(steps):
        logits, state = model(x[:,-1:], state)
        logits = logits[:, -1, :] / max(1e-5, temperature)
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, tok], dim=1)
    return x.squeeze(0).tolist()
