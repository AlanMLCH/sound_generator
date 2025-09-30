import pretty_midi
from typing import List

# Simple map for UI/API
INSTRUMENT_MAP = {
    "piano": 0, "bright_acoustic": 1, "electric_grand": 2,
    "organ": 16, "guitar": 24, "bass": 32, "strings": 48,
    "synth": 80, "drums": 128  # 128 -> flag for drums
}

# Token scheme (very simplified demo): NOTE_ON_p, NOTE_OFF_p, TIME_SHIFT_t, VELOCITY_v, INSTRUMENT_i
# In practice, use a robust vocabulary (e.g., REMI, MuMIDI)
class EventVocab:
    def __init__(self):
        self.tokens = []
        # Build ranges
        self.note_on = [f"NOTE_ON_{p}" for p in range(128)]
        self.note_off = [f"NOTE_OFF_{p}" for p in range(128)]
        self.time = [f"TIME_{t}" for t in range(1, 101)]  # coarse time shift
        self.velocity = [f"VEL_{v}" for v in range(1, 33)] # 32 bins
        self.instrument = [f"INST_{i}" for i in range(0,129)] # 0..128 (128=drums)

        for seg in [self.note_on, self.note_off, self.time, self.velocity, self.instrument]:
            self.tokens.extend(seg)
        self.stoi = {t:i for i,t in enumerate(self.tokens)}
        self.itos = {i:t for t,i in self.stoi.items()}

    def __len__(self):
        return len(self.tokens)

    def encode(self, events: List[str]):
        return [self.stoi[e] for e in events if e in self.stoi]

    def decode(self, ids: List[int]):
        return [self.itos[i] for i in ids]

    def tokens_to_pretty_midi(self, ids: List[int], tempo: int = 120):
        evs = self.decode(ids)
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst_map = {}  # channel by instrument token
        current_time = 0.0
        vel = 80
        current_inst = 0

        for e in evs:
            if e.startswith("TIME_"):
                dt = int(e.split("_")[1]) * 0.01
                current_time += dt
            elif e.startswith("VEL_"):
                vel = int(e.split("_")[1]) * 4 + 20
            elif e.startswith("INST_"):
                current_inst = int(e.split("_")[1])
                prog = 0 if current_inst==128 else current_inst
                is_drum = current_inst==128
                key = (prog, is_drum)
                if key not in inst_map:
                    p = pretty_midi.Instrument(program=prog if not is_drum else 0, is_drum=is_drum)
                    inst_map[key] = p
            elif e.startswith("NOTE_ON_"):
                pitch = int(e.split("_")[2])
                key = (0 if current_inst==128 else current_inst, current_inst==128)
                if key not in inst_map:
                    p = pretty_midi.Instrument(program=key[0], is_drum=key[1])
                    inst_map[key] = p
                note = pretty_midi.Note(velocity=min(127, vel), pitch=pitch, start=current_time, end=current_time+0.1)
                inst_map[key].notes.append(note)
            # NOTE_OFF ignored in this simplified conversion (short notes)

        for ins in inst_map.values():
            pm.instruments.append(ins)
        return pm
