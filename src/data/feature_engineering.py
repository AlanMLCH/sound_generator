import os, json, glob
import numpy as np
import pretty_midi
from src.common.constants import INTERIM_DIR, PROCESSED_DIR
from src.common.vocab import EventVocab

os.makedirs(PROCESSED_DIR, exist_ok=True)

def midi_to_tokens(pm: pretty_midi.PrettyMIDI, vocab: EventVocab, max_len=1024):
    events = []
    # Very naive: add an instrument token per track, then notes with coarse timing
    for inst in pm.instruments:
        inst_id = 128 if inst.is_drum else inst.program
        events.append(f"INST_{inst_id}")
        # sort notes by start
        for note in sorted(inst.notes, key=lambda n: n.start):
            dt = max(1, int((note.start)*100))  # to centiseconds bins
            events.append(f"TIME_{min(dt,100)}")
            events.append(f"VEL_{min(32, max(1, note.velocity//4))}")
            events.append(f"NOTE_ON_{note.pitch}")
    ids = vocab.encode(events)[:max_len]
    return np.array(ids, dtype=np.int32)

def run():
    vocab = EventVocab()
    out_dir = os.path.join(PROCESSED_DIR, "tokens")
    os.makedirs(out_dir, exist_ok=True)
    for line in open(os.path.join(INTERIM_DIR, "scan.csv"), "r", encoding="utf-8").read().splitlines()[1:]:
        path = line.split(",")[0]
        if not path or not os.path.exists(path):
            continue
        try:
            pm = pretty_midi.PrettyMIDI(path)
            ids = midi_to_tokens(pm, vocab)
            base = os.path.splitext(os.path.basename(path))[0]
            np.save(os.path.join(out_dir, base + ".npy"), ids)
        except Exception as e:
            print("Failed tokens for:", path, e)
    print("Tokens written to", out_dir)

if __name__ == "__main__":
    run()
