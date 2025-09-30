import argparse
from src.data.feature_engineering import run as fe_run
from src.data.preprocess import run as scan_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preprocess","features"], required=True)
    args = ap.parse_args()
    if args.stage == "preprocess":
        scan_run()
    else:
        fe_run()

if __name__ == "__main__":
    main()
