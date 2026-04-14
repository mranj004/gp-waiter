import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch import nn

from model import TModel


def transform(line_gen: str, rows: int, cols: int) -> np.ndarray:
    values = line_gen.strip("\n").split(",")[1:]
    g = np.array([float(v) for v in values], dtype=np.float32).reshape(rows, cols)
    return g


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0, path: str = "best_model.params"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    phe = pd.read_csv(args.phenotype_csv)
    y = phe[args.phenotype_column].to_numpy(dtype=np.float32) / args.phenotype_divisor

    site = pd.read_csv(args.weight_csv)
    weights = site[args.weight_column].to_numpy(dtype=np.float32).reshape(args.rows, args.cols)

    g_array = []
    with open(args.genotype_txt, "r", encoding="utf-8") as f:
        for line in f:
            g_array.append(transform(line, args.rows, args.cols))
    X = np.array(g_array, dtype=np.float32)

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    params_path = Path(args.output_dir)
    params_path.mkdir(parents=True, exist_ok=True)

    param = [
        {"embed_size1": 198, "embed_size2": 150, "num_heads": 9},
        {"embed_size1": 150, "embed_size2": 100, "num_heads": 10},
        {"embed_size1": 100, "embed_size2": 20, "num_heads": 5},
    ]
    model = TModel(embed_size=20, w=torch.tensor(weights, dtype=torch.float32, device=device), param=param, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(xtrain), torch.tensor(ytrain)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(xtest), torch.tensor(ytest)),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    early_stopping = EarlyStopping(patience=args.patience, path=str(params_path / "best_model.params"))

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for train_data, train_label in train_loader:
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            pred = model(train_data)
            loss = criterion(pred, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_data.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for test_data, test_label in test_loader:
                test_data = test_data.to(device)
                test_label = test_label.to(device)
                pred = model(test_data)
                loss = criterion(pred, test_label)
                val_loss += loss.item() * test_data.size(0)
        val_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/test GP-WAITER.")
    p.add_argument("--genotype-txt", required=True)
    p.add_argument("--phenotype-csv", required=True)
    p.add_argument("--weight-csv", required=True)
    p.add_argument("--phenotype-column", required=True)
    p.add_argument("--weight-column", default="c2")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--output-dir", default="parameters/run")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--phenotype-divisor", type=float, default=100.0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
