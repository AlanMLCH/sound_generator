import os
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

API_URL = os.environ.get("API_URL", "http://localhost:8000")
app = FastAPI()
templates = Jinja2Templates(directory="templates")

INSTRUMENTS = ["piano", "drums", "bass", "guitar", "strings", "organ", "synth"]

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "instruments": INSTRUMENTS})

@app.post("/generate")
def generate(
    request: Request,
    instruments: list[str] = Form(default=["piano", "drums"]),
    steps: int = Form(default=512),
    temperature: float = Form(default=1.0),
    seed: int = Form(default=42),
    tempo: int = Form(default=120),
):
    payload = {
        "instruments": instruments,
        "steps": steps,
        "temperature": temperature,
        "seed": seed,
        "tempo": tempo
    }
    r = requests.post(f"{API_URL}/generate", json=payload)
    if r.status_code != 200:
        return templates.TemplateResponse("index.html", {"request": request, "instruments": INSTRUMENTS, "error": r.text})
    # Save the MIDI locally for download
    out_path = "last.mid"
    with open(out_path, "wb") as f:
        f.write(r.content)
    return FileResponse(out_path, media_type="audio/midi", filename="generated.mid")
