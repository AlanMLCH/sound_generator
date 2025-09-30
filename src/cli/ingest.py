from src.data.download import run as download_run
from src.data.preprocess import run as preprocess_scan

def main():
    download_run()
    preprocess_scan()

if __name__ == "__main__":
    main()
