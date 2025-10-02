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
        # after loading x = np.load(...)
        # small chance to drop the first BAR (simulating different phrase alignment)
        import numpy as np
        if np.random.rand() < 0.3:
            # find first BAR id in x and shift
            # (be defensive: if not found, skip)
            pass  # implement if desired based on vocab.stoi["BAR"]

        x = np.load(self.files[idx])
        if len(x) < self.seq_len + 1:
            pad = self.seq_len + 1 - len(x)
            x = np.concatenate([x, np.zeros(pad, dtype=np.int32)])
        x = x[:self.seq_len+1]
        inp = torch.tensor(x[:-1], dtype=torch.long)
        tgt = torch.tensor(x[1:], dtype=torch.long)
        return inp, tgt

def sample_sequence(model, vocab, steps=512, temperature=1.0, top_p=0.9, cond_instruments=None, device="cpu"):
    import torch
    import torch.nn.functional as F

    model.eval()
    seq = []
    if cond_instruments:
        for i in cond_instruments:
            tok = vocab.stoi.get(f"INST_{i}", None)
            if tok is not None: seq.append(tok)
    # prime with BAR + BEAT_1 for stability
    for t in ["BAR", "BEAT_1"]:
        if t in vocab.stoi: seq.append(vocab.stoi[t])

    x = torch.tensor(seq[-1024:] if seq else [0], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(steps):
        logits = model(x)[:, -1, :] / max(1e-5, temperature)
        probs = F.softmax(logits, dim=-1)

        # nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumsum > top_p).float().argmax(dim=-1)
        for b in range(sorted_probs.size(0)):
            k = int(cutoff[b].item())
            mask = torch.ones_like(sorted_probs[b])
            mask[k+1:] = 0.0
            sorted_probs[b] *= mask
            sorted_probs[b] /= sorted_probs[b].sum()

        next_tok_sorted = torch.multinomial(sorted_probs, 1)
        next_tok = sorted_idx.gather(-1, next_tok_sorted)
        x = torch.cat([x, next_tok], dim=1)
    return x.squeeze(0).tolist()

