import pretty_midi
from typing import List

# Simple instrument map stays
INSTRUMENT_MAP = {
    "piano": 0, "bright_acoustic": 1, "electric_grand": 2,
    "organ": 16, "guitar": 24, "bass": 32, "strings": 48,
    "synth": 80, "drums": 128
}

class EventVocab:
    """
    REMI-ish vocab:
      - BAR
      - BEAT_1..BEAT_4   (assuming 4/4 grid; adjust as needed)
      - TEMPO_40..TEMPO_200 (coarse bins)
      - INST_0..INST_128 (128=drums)
      - NOTE_ON_0..127
      - DUR_1..DUR_32   (duration bins in grid steps)
    """
    def __init__(self, beats_per_bar=4, min_tempo=40, max_tempo=200, tempo_step=5, max_dur=32):
        bars = ["BAR"]
        beats = [f"BEAT_{i}" for i in range(1, beats_per_bar + 1)]
        tempos = [f"TEMPO_{t}" for t in range(min_tempo, max_tempo + 1, tempo_step)]
        instruments = [f"INST_{i}" for i in range(0, 129)]
        notes = [f"NOTE_ON_{p}" for p in range(128)]
        durs = [f"DUR_{d}" for d in range(1, max_dur + 1)]
        self.tokens = bars + beats + tempos + instruments + notes + durs
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}

    def __len__(self): return len(self.tokens)
    def encode(self, events: List[str]): return [self.stoi[e] for e in events if e in self.stoi]
    def decode(self, ids: List[int]):   return [self.itos[i] for i in ids]

    def tokens_to_pretty_midi(self, ids: List[int], tempo: int = 120, grid_hz: int = 4):
        """
        Reconstruct MIDI from tokens assuming BAR/BEAT grid and DUR bins.
        grid_hz=4 means 4 time-steps per beat (16th notes in 4/4).
        """
        evs = self.decode(ids)
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        cur_time = 0.0
        sec_per_step = 60.0 / (tempo * grid_hz)
        cur_inst = 0
        tracks = {}  # (prog,is_drum)->Instrument

        def get_track(program, is_drum):
            key = (program, is_drum)
            if key not in tracks:
                tracks[key] = pretty_midi.Instrument(program=program if not is_drum else 0, is_drum=is_drum)
            return tracks[key]

        beat_pos = 0
        i = 0
        while i < len(evs):
            e = evs[i]
            if e == "BAR":
                beat_pos = 0
            elif e.startswith("BEAT_"):
                beat_pos = (int(e.split("_")[1]) - 1)
            elif e.startswith("TEMPO_"):
                try:
                    tempo = int(e.split("_")[1]); sec_per_step = 60.0 / (tempo * grid_hz)
                except: pass
            elif e.startswith("INST_"):
                cur_inst = int(e.split("_")[1])
            elif e.startswith("NOTE_ON_"):
                pitch = int(e.split("_")[2])
                dur_steps = 4  # default duration if not followed by DUR
                if i + 1 < len(evs) and evs[i+1].startswith("DUR_"):
                    dur_steps = int(evs[i+1].split("_")[1]); i += 1
                is_drum = (cur_inst == 128)
                program = 0 if is_drum else cur_inst
                tr = get_track(program, is_drum)
                note = pretty_midi.Note(
                    velocity=90, pitch=pitch,
                    start=cur_time, end=cur_time + dur_steps * sec_per_step
                )
                tr.notes.append(note)
            # time advances one step at every BEAT boundary only when you move through grid.
            # We’ll advance time uniformly per token group in tokenizer, not here.
            i += 1

        for ins in tracks.values():
            pm.instruments.append(ins)
        return pm
