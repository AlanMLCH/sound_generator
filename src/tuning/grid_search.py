import itertools
from src.models.train import run as train_run

def run():
    grid = {
        "lr": [1e-3, 5e-4],
        "batch_size": [16, 32],
        "seq_len": [256, 512]
    }
    best = None
    for lr, bs, sl in itertools.product(grid["lr"], grid["batch_size"], grid["seq_len"]):
        print(f"Training with lr={lr}, batch_size={bs}, seq_len={sl}")
        train_run(batch_size=bs, epochs=1, lr=lr, seq_len=sl)  # short runs
        score = 0.0  # placeholder; you could compute validation loss here
        if best is None or score > best[0]:
            best = (score, {"lr": lr, "batch_size": bs, "seq_len": sl})
    print("Best config:", best)

if __name__ == "__main__":
    run()
