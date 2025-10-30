from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import os
import time
import re

from src.common.vocab import EventVocab, INSTRUMENT_MAP
from src.common.utils import seed_everything
from src.models.dataset import sample_sequence  # dual-mode sampler
import pretty_midi  # noqa: F401

# Try to import both model classes; either/both may exist depending on your code
try:
    from src.models.architecture import MusicTransformer
except Exception:
    MusicTransformer = None  # type: ignore

try:
    from src.models.architecture import MusicLSTM
except Exception:
    MusicLSTM = None  # type: ignore

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


def _all_checkpoints() -> List[str]:
    return [os.path.join(ARTIF_DIR, f) for f in os.listdir(ARTIF_DIR) if f.endswith(".pt")]


def _pick_checkpoint() -> Optional[str]:
    """Prefer a *transformer*-looking filename; else latest by mtime."""
    cks = _all_checkpoints()
    if not cks:
        return None
    tr = [p for p in cks if "transformer" in os.path.basename(p).lower()]
    return max(tr, key=os.path.getmtime) if tr else max(cks, key=os.path.getmtime)


def _load_state(ckpt_path: str):
    st = torch.load(ckpt_path, map_location="cpu")
    return st["state_dict"] if isinstance(st, dict) and "state_dict" in st else st


def _infer_kind_and_hparams(state: Dict[str, torch.Tensor]):
    kset = set(state.keys())
    # LSTM?
    if any(k.startswith("lstm.") for k in kset) or "embed.weight" in kset or "fc.weight" in kset:
        return {"kind": "lstm"}

    enc_pref = any(k.startswith("enc.layers.") for k in kset)
    dec_pref = any(k.startswith("dec.layers.") for k in kset)

    d_model = state.get("tok_emb.weight", torch.empty(1, 256)).shape[1]  # default 256
    max_len = state.get("pos_emb.weight", torch.empty(1536, 1)).shape[0]  # default 1536

    def count_layers(prefix: str) -> int:
        import re
        pat = re.compile(rf"^{prefix}\.(\d+)\.")
        idx = set()
        for k in kset:
            m = pat.search(k)
            if m:
                idx.add(int(m.group(1)))
        return (max(idx) + 1) if idx else 4

    if enc_pref:
        num_layers = count_layers("enc.layers")
        blocks = "enc"
        # infer ff_dim from any encoder layer’s linear1/linear2
        ff_dim = None
        for i in range(num_layers):
            w1 = state.get(f"enc.layers.{i}.linear1.weight", None)
            if w1 is not None:
                ff_dim = int(w1.shape[0])  # (ff_dim, d_model)
                break
            b1 = state.get(f"enc.layers.{i}.linear1.bias", None)
            if b1 is not None:
                ff_dim = int(b1.shape[0])
                break
            w2 = state.get(f"enc.layers.{i}.linear2.weight", None)
            if w2 is not None:
                ff_dim = int(w2.shape[1])  # (d_model, ff_dim)
                break
    elif dec_pref:
        num_layers = count_layers("dec.layers")
        blocks = "dec"
        ff_dim = None
        for i in range(num_layers):
            w1 = state.get(f"dec.layers.{i}.linear1.weight", None)
            if w1 is not None:
                ff_dim = int(w1.shape[0])
                break
            b1 = state.get(f"dec.layers.{i}.linear1.bias", None)
            if b1 is not None:
                ff_dim = int(b1.shape[0])
                break
            w2 = state.get(f"dec.layers.{i}.linear2.weight", None)
            if w2 is not None:
                ff_dim = int(w2.shape[1])
                break
    else:
        num_layers = 4
        blocks = "enc"
        ff_dim = None

    # nhead: pick a divisor of d_model
    nhead = next((nh for nh in (8, 6, 4, 2, 1) if d_model % nh == 0), 4)

    if ff_dim is None:
        ff_dim = max(4 * int(d_model), 512)  # sensible default

    return {
        "kind": "transformer",
        "d_model": int(d_model),
        "max_len": int(max_len),
        "num_layers": int(num_layers),
        "nhead": int(nhead),
        "ff_dim": int(ff_dim),
        "blocks": blocks,  # 'enc' or 'dec'
    }



def _remap_blocks_if_needed(state: Dict[str, torch.Tensor], want_blocks: str):
    """
    If checkpoint uses 'dec.layers.*' but model uses 'enc.layers.*' (or vice versa),
    rename keys accordingly.
    """
    def swap_prefix(sd, frm, to):
        out = {}
        for k, v in sd.items():
            if k.startswith(frm + ".layers."):
                out[k.replace(frm + ".", to + ".")] = v
            else:
                out[k] = v
        return out

    kset = set(state.keys())
    has_enc = any(k.startswith("enc.layers.") for k in kset)
    has_dec = any(k.startswith("dec.layers.") for k in kset)
    if want_blocks == "enc" and has_dec and not has_enc:
        return swap_prefix(state, "dec", "enc")
    if want_blocks == "dec" and has_enc and not has_dec:
        return swap_prefix(state, "enc", "dec")
    return state


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    seed_everything(req.seed)

    ckpt = _pick_checkpoint()
    if ckpt is None:
        raise HTTPException(400, "No trained model checkpoint found in artifacts/models")

    state = _load_state(ckpt)
    meta = _infer_kind_and_hparams(state)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = EventVocab()

    # Build a model that matches the checkpoint
    if meta["kind"] == "lstm":
        if MusicLSTM is None:
            raise HTTPException(500, "LSTM class not available in this build.")
        try:
            model = MusicLSTM(vocab_size=len(vocab), embedding_dim=256, hidden_dim=512, num_layers=2).to(device)
        except TypeError:
            model = MusicLSTM(len(vocab)).to(device)
    else:
        if MusicTransformer is None:
            raise HTTPException(500, "Transformer class not available in this build.")
        model = MusicTransformer(
            vocab_size=len(vocab),
            d_model=meta["d_model"],
            nhead=meta["nhead"],
            num_layers=meta["num_layers"],
            dim_feedforward=meta["ff_dim"],   # <-- use inferred FFN size
            dropout=0.1,
            max_len=meta["max_len"],
        ).to(device)

        # Our implementation uses 'enc.layers.*'. If the ckpt has 'dec.layers.*', remap.
        state = _remap_blocks_if_needed(state, want_blocks="enc")

    # Load weights (strict)
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
