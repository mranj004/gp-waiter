import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import TModel


def transform(line_gen: str) -> np.ndarray:
    values = line_gen.strip("\n").split(",")[1:]
    return np.array([float(x) for x in values], dtype=np.float32).reshape(112, 597)


def log_training_info(logfile: Path, epoch: int, epoch_loss: float, corr_train: float, corr_test: float) -> None:
    with logfile.open("a", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "epoch_loss": float(epoch_loss),
                "corr_train": float(corr_train),
                "corr_test": float(corr_test),
            },
            f,
        )
        f.write("\n")


def train(phe_s: str, root_path: Path, divide: float, num_epochs: int, batch_size: int, lr: float) -> tuple[float, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    phe = pd.read_csv(root_path / f"demo.phenotype.{phe_s}.csv", index_col=0).to_numpy(dtype=np.float32) / divide
    site = pd.read_csv(root_path / f"demo.weighted.{phe_s}.csv")["c2"].to_numpy(dtype=np.float32)
    weights = torch.tensor(np.log10(1 / site).reshape(112, 597), dtype=torch.float32, device=device)

    g_array = []
    with (root_path / f"demo.genotype.{phe_s}.txt").open("r", encoding="utf-8") as f:
        for line in f:
            g_array.append(transform(line))
    X = np.array(g_array, dtype=np.float32)

    xtrain, xtest, ytrain, ytest = train_test_split(X, phe, test_size=0.2, random_state=100)

    params_path = Path("parameters") / phe_s
    params_path.mkdir(parents=True, exist_ok=True)
    logfile = params_path / f"{phe_s}_training_log.json"
    logfile.write_text("", encoding="utf-8")

    param = [
        {"embed_size1": 198, "embed_size2": 150, "num_heads": 9},
        {"embed_size1": 150, "embed_size2": 100, "num_heads": 10},
        {"embed_size1": 100, "embed_size2": 20, "num_heads": 5},
    ]
    model = TModel(embed_size=20, w=weights, param=param, num_layers=3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(xtrain), torch.tensor(ytrain).squeeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(xtest), torch.tensor(ytest).squeeze(-1)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_corr = -np.inf
    best_epoch = -1
    best_outputs, best_labels = None, None

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            train_preds.append(pred.detach().cpu().numpy())
            train_labels.append(yb.detach().cpu().numpy())

        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        corr_train = float(np.corrcoef(train_preds, train_labels)[0, 1])

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = model(xb)
                test_preds.append(pred.cpu().numpy())
                test_labels.append(yb.numpy())
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        corr_test = float(np.corrcoef(test_preds, test_labels)[0, 1])

        epoch_loss = running_loss / len(train_loader.dataset)
        log_training_info(logfile, epoch, epoch_loss, corr_train, corr_test)
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss:.6f} train_corr={corr_train:.4f} test_corr={corr_test:.4f}")

        if corr_test > best_corr:
            best_corr, best_epoch = corr_test, epoch
            best_outputs, best_labels = test_preds.copy(), test_labels.copy()
            torch.save(model.state_dict(), params_path / f"best_{phe_s}.params")

    best_df = pd.DataFrame({"Predictions": best_outputs, "True_Labels": best_labels})
    best_df.to_csv(params_path / f"best_{phe_s}_predictions.csv", index=False)
    return best_corr, best_epoch


if __name__ == "__main__":
    phe_list = ["O"]
    summary_logfile = Path("best_results_summary.json")
    root_path = Path("./demo_data")

    with summary_logfile.open("a", encoding="utf-8") as f:
        f.write("--- Model running started ---\n")

    for phe_s in phe_list:
        best_corr, best_epoch = train(phe_s, root_path, divide=100, num_epochs=20, batch_size=32, lr=0.001)
        with summary_logfile.open("a", encoding="utf-8") as f:
            json.dump({"phe_s": phe_s, "best_test_corr": float(best_corr), "best_epoch": int(best_epoch)}, f)
            f.write("\n")
