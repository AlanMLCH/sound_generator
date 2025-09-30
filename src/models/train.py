import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.dataset import TokenDataset
from src.models.architecture import MusicLSTM
from src.common.constants import PROCESSED_DIR, MODELS_DIR
from src.common.utils import seed_everything, device
from src.common.vocab import EventVocab

def run(batch_size=32, epochs=2, lr=1e-3, seq_len=512):
    seed_everything(42)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ds = TokenDataset(os.path.join(PROCESSED_DIR, "tokens"), seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    dev = device()
    vocab = EventVocab()
    model = MusicLSTM(len(vocab), 256, 512, 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for inp, tgt in pbar:
            inp, tgt = inp.to(dev), tgt.to(dev)
            logits, _ = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=loss.item())
    ckpt = os.path.join(MODELS_DIR, "music_lstm.pt")
    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)

if __name__ == "__main__":
    run()
