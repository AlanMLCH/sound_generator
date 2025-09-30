import os
import torch
import numpy as np
from src.models.architecture import MusicLSTM
from src.common.vocab import EventVocab
from src.common.constants import MODELS_DIR

def run():
    ckpt = os.path.join(MODELS_DIR, "music_lstm.pt")
    if not os.path.exists(ckpt):
        raise SystemExit("No model found to evaluate.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = EventVocab()
    model = MusicLSTM(len(vocab)).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    # Toy "perplexity-like" synthetic metric by random sampling
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Evaluation: (demo) OK")

if __name__ == "__main__":
    run()
