import sys
import os
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, preprocess, get_Xy


# ── GPU setup ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Model architecture ────────────────────────────────────────────────────────
class BRCANet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


# ── Train/val loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out.argmax(dim=1).cpu())
            labels.append(y_batch)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return accuracy_score(labels, preds), preds, labels


# ── Hyperparameter search ─────────────────────────────────────────────────────
def search_lr(X_train, X_val, y_train, y_val, le, scaler):
    lrs = [0.0001, 0.0005, 0.001, 0.005]
    results = []
    
    for lr in tqdm(lrs, desc="LR search", unit="lr"):
        model = BRCANet(X_train.shape[1], len(le.classes_)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(scaler.fit_transform(X_train)),
                torch.LongTensor(le.transform(y_train))
            ),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(scaler.transform(X_val)),
                torch.LongTensor(le.transform(y_val))
            ),
            batch_size=32
        )
        
        for _ in range(20):
            train_epoch(model, train_loader, criterion, optimizer, device)
        
        acc, _, _ = eval_model(model, val_loader, device)
        results.append([lr, acc])
    
    return pd.DataFrame(results, columns=["learning_rate", "Accuracy"])


def search_dropout(X_train, X_val, y_train, y_val, le, scaler, best_lr):
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for drop in tqdm(dropouts, desc="Dropout search", unit="drop"):
        model = BRCANet(X_train.shape[1], len(le.classes_), dropout=drop).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr)
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(scaler.fit_transform(X_train)),
                torch.LongTensor(le.transform(y_train))
            ),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(scaler.transform(X_val)),
                torch.LongTensor(le.transform(y_val))
            ),
            batch_size=32
        )
        
        for _ in range(20):
            train_epoch(model, train_loader, criterion, optimizer, device)
        
        acc, _, _ = eval_model(model, val_loader, device)
        results.append([drop, acc])
    
    return pd.DataFrame(results, columns=["dropout", "Accuracy"])


if __name__ == "__main__":
    data, clinic = load_data()
    df = preprocess(data, clinic)
    X, y = get_Xy(df)
    
    le = LabelEncoder()
    le.fit(y)
    
    scaler = StandardScaler()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # ── Search ────────────────────────────────────────────────────────────────
    print("\n[1/2] LR search...")
    lr_df = search_lr(X_train, X_val, y_train, y_val, le, scaler)
    best_lr = lr_df.loc[lr_df["Accuracy"].idxmax(), "learning_rate"]
    print(f"Best LR: {best_lr}")
    
    print("\n[2/2] Dropout search...")
    dropout_df = search_dropout(X_train, X_val, y_train, y_val, le, scaler, best_lr)
    best_dropout = dropout_df.loc[dropout_df["Accuracy"].idxmax(), "dropout"]
    print(f"Best dropout: {best_dropout}")
    
    # ── Final model ───────────────────────────────────────────────────────────
    print(f"\nTraining final model (lr={best_lr}, dropout={best_dropout})...")

    model = BRCANet(X_train.shape[1], len(le.classes_), dropout=best_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(scaler.fit_transform(X_train_full)),
            torch.LongTensor(le.transform(y_train_full))
        ),
        batch_size=32, shuffle=True, drop_last=True  # ← ADD THIS
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(scaler.transform(X_test)),
            torch.LongTensor(le.transform(y_test))
        ),
        batch_size=32
    )

    for epoch in tqdm(range(100), desc="Training", unit="epoch"):
        train_epoch(model, train_loader, criterion, optimizer, device)

    # Eval needs model.eval() to disable dropout/batchnorm training mode
    acc, preds, labels = eval_model(model, test_loader, device)
    y_pred = le.inverse_transform(preds)
    conf_mat = confusion_matrix(y_test, y_pred, labels=le.classes_)
    
    results = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "label_encoder": le,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "classes": le.classes_,
        "accuracy": acc,
        "conf_mat": conf_mat,
        "lr_df": lr_df,
        "dropout_df": dropout_df,
        "best_lr": best_lr,
        "best_dropout": best_dropout,
        "input_dim": X_train.shape[1],
        "num_classes": len(le.classes_),
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/nn_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nNeural Network Accuracy: {acc:.4f}")
    print("Saved → models/nn_results.pkl")