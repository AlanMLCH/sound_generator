import os, json, glob
import numpy as np
import pretty_midi
from src.common.constants import INTERIM_DIR, PROCESSED_DIR
from src.common.vocab import EventVocab

os.makedirs(PROCESSED_DIR, exist_ok=True)

def midi_to_tokens(pm: pretty_midi.PrettyMIDI, vocab: EventVocab,
                   beats_per_bar=4, grid_hz=4, max_len=1024):
    """
    Quantize to a bar/beat grid, emit BAR/BEAT/TEMPO + INST/NOTE_ON/DUR tokens.
    """
    import numpy as np

    tempo = int(np.clip(pm.estimate_tempo() if pm.estimate_tempo() > 0 else 120, 40, 200))
    sec_per_step = 60.0 / (tempo * grid_hz)

    # Build an event list by iterating bars->beats->steps; collect notes that start near each step.
    # For simplicity: flatten all instruments but keep INST tokens before their notes.
    events = []
    events.append(f"TEMPO_{tempo - (tempo % 5)}")  # bin tempo to multiples of 5

    # Prepare note events: (start_sec, pitch, dur_steps, inst_id)
    note_events = []
    for inst in pm.instruments:
        inst_id = 128 if inst.is_drum else inst.program
        for n in inst.notes:
            start = max(0.0, n.start)
            dur_steps = max(1, int(round((n.end - n.start) / sec_per_step)))
            note_events.append((start, n.pitch, min(32, dur_steps), inst_id))
    note_events.sort(key=lambda x: x[0])

    # Calculate grid coverage from last note
    total_time = max([ne[0] for ne in note_events], default=0.0) + 8 * sec_per_step
    total_steps = int(total_time / sec_per_step) + 1
    steps_per_bar = beats_per_bar * grid_hz

    idx = 0
    for step in range(total_steps):
        if step % steps_per_bar == 0:
            events.append("BAR")
        beat_idx = (step // grid_hz) % beats_per_bar
        if step % grid_hz == 0:
            events.append(f"BEAT_{beat_idx+1}")
        # Emit all notes starting at this step
        cur_time = step * sec_per_step
        # flush notes whose start is within half-step window
        while idx < len(note_events) and abs(note_events[idx][0] - cur_time) <= (sec_per_step/2):
            _, pitch, d_steps, inst_id = note_events[idx]
            events.append(f"INST_{inst_id}")
            events.append(f"NOTE_ON_{pitch}")
            events.append(f"DUR_{max(1, min(32, d_steps))}")
            idx += 1
        if len(events) > max_len: break

    return np.array(vocab.encode(events[:max_len]), dtype=np.int32)


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
