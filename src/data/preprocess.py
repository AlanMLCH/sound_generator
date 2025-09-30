import os, glob
import pretty_midi
import pandas as pd
from src.common.constants import RAW_DIR, INTERIM_DIR
from src.common.vocab import INSTRUMENT_MAP

os.makedirs(INTERIM_DIR, exist_ok=True)

def extract_instruments(pm: pretty_midi.PrettyMIDI):
    result = []
    for inst in pm.instruments:
        if inst.is_drum:
            result.append(128)  # drums
        else:
            result.append(inst.program)
    return sorted(set(result))

def run():
    rows = []
    for fp in glob.glob(os.path.join(RAW_DIR, "**/*.mid*"), recursive=True):
        try:
            pm = pretty_midi.PrettyMIDI(fp)
            instruments = extract_instruments(pm)
            rows.append({"path": fp, "instruments": instruments, "n_tracks": len(pm.instruments), "n_notes": sum(len(i.notes) for i in pm.instruments)})
        except Exception as e:
            print("Failed:", fp, e)
    df = pd.DataFrame(rows)
    outp = os.path.join(INTERIM_DIR, "scan.csv")
    df.to_csv(outp, index=False)
    print("Wrote", outp)

if __name__ == "__main__":
    run()
