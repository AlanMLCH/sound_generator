# services/api/app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
import time

# Import BOTH model classes
from src.common.vocab import EventVocab, INSTRUMENT_MAP
from src.common.utils import seed_everything
from src.models.dataset import sample_sequence  # dual-mode sampler (LSTM/Transformer)

# These may or may not exist depending on your repo version; handle gracefully
try:
    from src.models.architecture import MusicTransformer
except Exception:
    MusicTransformer = None

try:
    from src.models.architecture import MusicLSTM
except Exception:
    MusicLSTM = None

ARTIF_DIR = "artifacts/models"
os.makedirs(ARTIF_DIR, exist_ok=True)

app = FastAPI(title="MusicGen API")

class GenerateRequest(BaseModel):
    instruments: List[str] = ["piano", "drums"]
    steps: int = 512
    temperature: float = 1.0
    seed: int = 42
    tempo: Optional[int] = 120
    top_p: float = 0.9  # nucleus sampling


def _all_checkpoints():
    return [os.path.join(ARTIF_DIR, f) for f in os.listdir(ARTIF_DIR) if f.endswith(".pt")]


def _detect_ckpt_kind(ckpt_path: str) -> str:
    """
    Return 'lstm' or 'transformer' by inspecting state_dict keys.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    keys = list(state.keys())
    kset = set(keys)
    if any(k.startswith("lstm.") for k in keys) or "embed.weight" in kset or "fc.weight" in kset:
        return "lstm"
    if "tok_emb.weight" in kset or any(k.startswith("dec.layers.") for k in keys):
        return "transformer"
    # fallback by filename hint
    name = os.path.basename(ckpt_path).lower()
    if "lstm" in name: return "lstm"
    if "transformer" in name: return "transformer"
    return "lstm"  # default safe guess


def _pick_checkpoint() -> Optional[str]:
    """
    Prefer a transformer ckpt if present; else latest by mtime.
    """
    cks = _all_checkpoints()
    if not cks:
        return None
    # prefer by name
    tr = [p for p in cks if "transformer" in os.path.basename(p).lower()]
    if tr:
        return max(tr, key=os.path.getmtime)
    # else latest any
    return max(cks, key=os.path.getmtime)


def _build_model(kind: str, vocab_size: int, device: str):
    if kind == "transformer":
        if MusicTransformer is None:
            raise HTTPException(500, "Transformer class not available in this build.")
        return MusicTransformer(
            vocab_size=vocab_size,
            d_model=256, nhead=4, num_layers=4, dim_feedforward=768,
            dropout=0.1, max_len=1536
        ).to(device)
    # lstm
    if MusicLSTM is None:
        raise HTTPException(500, "LSTM class not available in this build.")
    try:
        return MusicLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2).to(device)
    except TypeError:
        return MusicLSTM(vocab_size).to(device)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    seed_everything(req.seed)
    ckpt = _pick_checkpoint()
    if ckpt is None:
        raise HTTPException(400, "No trained model checkpoint found in artifacts/models")

    kind = _detect_ckpt_kind(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = EventVocab()
    model = _build_model(kind, len(vocab), device)

    # load matching weights
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    model.eval()

    instr_ids = [INSTRUMENT_MAP.get(i.lower(), INSTRUMENT_MAP["piano"]) for i in req.instruments]
    tokens = sample_sequence(
        model, vocab,
        steps=req.steps,
        temperature=req.temperature,
        top_p=req.top_p,
        cond_instruments=instr_ids,
        device=device
    )

    pm = vocab.tokens_to_pretty_midi(tokens, tempo=req.tempo or 120)
    out_name = f"artifacts/runs/gen_{int(time.time())}.mid"
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    pm.write(out_name)
    return FileResponse(out_name, media_type="audio/midi", filename=os.path.basename(out_name))
