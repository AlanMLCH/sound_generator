from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
import time
from src.models.architecture import MusicLSTM
from src.common.vocab import EventVocab, INSTRUMENT_MAP
from src.common.utils import seed_everything
from src.models.dataset import sample_sequence
import pretty_midi

ARTIF_DIR = "artifacts/models"
os.makedirs(ARTIF_DIR, exist_ok=True)

app = FastAPI(title="MusicGen API")

class GenerateRequest(BaseModel):
    instruments: List[str] = ["piano", "drums"]
    steps: int = 512
    temperature: float = 1.0
    seed: int = 42
    tempo: Optional[int] = 120

def load_latest_checkpoint():
    # find latest .pt in artifacts/models
    cks = [os.path.join(ARTIF_DIR, f) for f in os.listdir(ARTIF_DIR) if f.endswith(".pt")]
    if not cks:
        return None
    return max(cks, key=os.path.getmtime)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    seed_everything(req.seed)
    ckpt = load_latest_checkpoint()
    if ckpt is None:
        raise HTTPException(400, "No trained model checkpoint found in artifacts/models")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = EventVocab()
    model = MusicLSTM(
        vocab_size=len(vocab),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Build conditioning vector from selected instruments
    instr_ids = [INSTRUMENT_MAP.get(i.lower(), INSTRUMENT_MAP["piano"]) for i in req.instruments]
    with torch.no_grad():
        tokens = sample_sequence(model, vocab, steps=req.steps, temperature=req.temperature, cond_instruments=instr_ids, device=device)

    # Convert tokens to MIDI
    pm = vocab.tokens_to_pretty_midi(tokens, tempo=req.tempo or 120)
    out_name = f"artifacts/runs/gen_{int(time.time())}.mid"
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    pm.write(out_name)
    return FileResponse(out_name, media_type="audio/midi", filename=os.path.basename(out_name))
