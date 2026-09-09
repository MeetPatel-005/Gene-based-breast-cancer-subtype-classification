"""
train_rnn.py
Bidirectional GRU + Attention model for BRCA subtype classification.

Architecture:
  Gene expression vector (19k+ features)
    → Reshape into sequence of 64-gene chunks (~300 steps)
    → 2-layer Bidirectional GRU (hidden_dim=256)
    → Attention-weighted pooling over sequence
    → Classification head → 5 subtypes

Gene driver explainability is built in:
  - Attention weights show which gene groups matter
  - Gradient × input attribution gives per-gene directional importance
"""

import sys
import os
import math
import pickle

import numpy as np
import pandas as pd
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


# ── Constants ─────────────────────────────────────────────────────────────────
TOP_GENES = 2000         # Select top-N most variable genes (60k → 2k)
CHUNK_SIZE = 64          # genes per sequence step
HIDDEN_DIM = 256         # GRU hidden size
NUM_LAYERS = 2           # GRU layers
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 1e-3      # L2 regularisation
LABEL_SMOOTHING = 0.1    # Smooth one-hot targets to reduce overconfidence
NOISE_STD = 0.1          # Gaussian noise augmentation during training


# ── Model architecture ───────────────────────────────────────────────────────
class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over sequence outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # BiGRU → hidden_dim*2
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, gru_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim*2)
        Returns:
            context:   (batch, hidden_dim*2) — attention-weighted sum
            weights:   (batch, seq_len)      — attention weights (sum to 1)
        """
        scores = self.attn(gru_output).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)               # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), gru_output).squeeze(1)  # (batch, hidden*2)
        return context, weights


class BRCAGenomicRNN(nn.Module):
    """
    Bidirectional GRU + Attention for gene expression classification.

    Input:  (batch, n_genes) — flat gene expression vector
    Output: (batch, num_classes) — class logits
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        chunk_size: int = CHUNK_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Pad input to make it divisible by chunk_size
        self.seq_len = math.ceil(input_dim / chunk_size)
        self.padded_dim = self.seq_len * chunk_size

        # Per-chunk projection: chunk_size → hidden_dim
        # This learns a dense embedding for each group of genes
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding (learnable) so the RNN knows genomic position
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.02)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling
        self.attention = Attention(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Store last attention weights for interpretability
        self._last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — raw gene expression values
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Pad and reshape into sequence of gene chunks
        if self.padded_dim > self.input_dim:
            padding = torch.zeros(batch_size, self.padded_dim - self.input_dim, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x

        # (batch, seq_len, chunk_size)
        x_seq = x_padded.view(batch_size, self.seq_len, self.chunk_size)

        # Encode each chunk
        # (batch, seq_len, hidden_dim)
        x_encoded = self.chunk_encoder(x_seq)

        # Add positional encoding
        x_encoded = x_encoded + self.pos_embedding

        # BiGRU
        # gru_out: (batch, seq_len, hidden_dim * 2)
        gru_out, _ = self.gru(x_encoded)

        # Attention pooling
        context, attn_weights = self.attention(gru_out)
        self._last_attn_weights = attn_weights.detach()

        # Classify
        logits = self.classifier(context)
        return logits

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return attention weights from the last forward pass."""
        return self._last_attn_weights


# ── Gene driver explainability ────────────────────────────────────────────────
def compute_gene_drivers_rnn(
    model: BRCAGenomicRNN,
    sample: np.ndarray,
    feature_names: list[str],
    scaler: StandardScaler,
    le: LabelEncoder,
    n_top: int = 10,
) -> list[dict]:
    """
    Compute per-gene importance for a single sample using attention × gradient.

    Method:
      1. Forward pass to get prediction + attention weights
      2. Backprop w.r.t. predicted class to get input gradients
      3. Compute attribution = gradient × input (signed)
      4. Modulate by attention weights (per-chunk)
      5. Return top-N genes with directional importance

    Returns list of dicts compatible with the webapp's gene driver format.
    """
    # cuDNN RNN backward requires training mode, but we want eval-like behaviour.
    # Solution: keep model in train mode (so cuDNN allows backward) but use
    # torch.no_grad() selectively and disable cuDNN for the backward pass.
    was_training = model.training
    model.eval()  # disable dropout/batchnorm training behaviour

    # Prepare input
    sample_scaled = scaler.transform(sample.reshape(1, -1))
    dev = next(model.parameters()).device
    x = torch.FloatTensor(sample_scaled).to(dev)
    x.requires_grad_(True)

    # Forward pass — must be in train mode for cuDNN RNN backward
    model.train()
    logits = model(x)
    pred_class = logits.argmax(dim=1).item()

    # Backward pass w.r.t. predicted class
    model.zero_grad()
    logits[0, pred_class].backward()

    # Restore original mode
    if not was_training:
        model.eval()

    # Input gradients: (1, n_genes)
    grad = x.grad.detach().cpu().numpy().flatten()
    input_vals = x.detach().cpu().numpy().flatten()

    # Gradient × Input attribution (signed — preserves directionality)
    gxi = grad * input_vals  # (n_genes,)

    # Get attention weights: (seq_len,)
    attn = model.get_attention_weights()
    if attn is not None:
        attn = attn.cpu().numpy().flatten()  # (seq_len,)
        # Expand attention from per-chunk to per-gene
        chunk_size = model.chunk_size
        attn_per_gene = np.repeat(attn, chunk_size)[: len(gxi)]
        # Modulate: attribution × attention
        attribution = gxi * attn_per_gene
    else:
        attribution = gxi

    # Rank by absolute attribution
    abs_attr = np.abs(attribution)
    top_indices = np.argsort(abs_attr)[::-1][:n_top]

    # Build result list
    max_attr = abs_attr[top_indices[0]] if len(top_indices) > 0 else 1.0
    raw_expression = sample.flatten()

    # Compute population-baseline z-scores (using scaler's stored mean/std)
    pop_mean = scaler.mean_
    pop_std = scaler.scale_
    z_scores = (raw_expression - pop_mean) / np.where(pop_std > 0, pop_std, 1.0)

    drivers = []
    for rank, idx in enumerate(top_indices, 1):
        if idx >= len(feature_names):
            continue
        gene_id = feature_names[idx]
        attr_val = float(attribution[idx])
        expr_val = float(raw_expression[idx])
        z_val = float(z_scores[idx])

        drivers.append({
            "rank": rank,
            "ensembl_id": gene_id,
            "gene_symbol": gene_id.split(".")[0],   # placeholder, resolved by webapp
            "gene_name": "",                         # resolved by webapp
            "importance_score": float(abs_attr[idx]),
            "importance_pct": float(abs_attr[idx] / max_attr) if max_attr > 0 else 0.0,
            "attribution": round(attr_val, 6),       # signed: +ve = over-expressed driver
            "direction": "Upregulated" if attr_val > 0 else "Downregulated",
            "expression_value": round(expr_val, 2),
            "expression_level": (
                "High" if z_val > 1.5 else
                "Low" if z_val < -1.5 else
                "Normal"
            ),
            "z_score": round(z_val, 3),
            "attention_weight": float(attn_per_gene[idx]) if attn is not None and idx < len(attn_per_gene) else None,
        })

    return drivers


# ── Feature selection ─────────────────────────────────────────────────────────
def select_top_genes(X: pd.DataFrame, n_top: int = TOP_GENES) -> list[str]:
    """Select the top-N highest-variance genes across all samples.

    High-variance genes carry the most discriminative signal between subtypes.
    This reduces dimensionality from ~60k to ~2k, making the RNN feasible.
    """
    variances = X.var(axis=0)
    top_cols = variances.nlargest(n_top).index.tolist()
    return top_cols


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, noise_std=NOISE_STD):
    """Train one epoch with optional Gaussian noise augmentation."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Gaussian noise augmentation: simulate biological variation
        if noise_std > 0:
            X_batch = X_batch + torch.randn_like(X_batch) * noise_std
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


def train_with_early_stopping(
    model, train_loader, val_loader, criterion, optimizer,
    device, epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, desc="Training",
    scheduler=None,
):
    """Train with early stopping + optional LR scheduler. Returns best state."""
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "lr": []}

    pbar = tqdm(range(epochs), desc=desc, unit="epoch")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = eval_model(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        pbar.set_postfix({
            "loss": f"{train_loss:.4f}", "val_acc": f"{val_acc:.4f}",
            "best": f"{best_val_acc:.4f}", "lr": f"{current_lr:.1e}",
        })

        # Step scheduler (ReduceLROnPlateau uses val metric)
        if scheduler is not None:
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1} (best val_acc={best_val_acc:.4f})")
                break

    return best_state, best_val_acc, history


# ── Hyperparameter search ────────────────────────────────────────────────────
def make_loaders(X_train, X_val, y_train, y_val, scaler, le, batch_size=BATCH_SIZE):
    """Create train/val DataLoaders with scaling and encoding."""
    X_tr_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_tr_scaled),
            torch.LongTensor(le.transform(y_train)),
        ),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.LongTensor(le.transform(y_val)),
        ),
        batch_size=batch_size,
    )
    return train_loader, val_loader


def search_lr(X_train, X_val, y_train, y_val, le, scaler, input_dim, num_classes):
    """Search over learning rates using short training runs."""
    lrs = [0.0001, 0.0005, 0.001, 0.005]
    results = []

    for lr in tqdm(lrs, desc="LR search", unit="lr"):
        model = BRCAGenomicRNN(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, scaler, le)

        # Short training run (20 epochs, no early stopping)
        for _ in range(20):
            train_epoch(model, train_loader, criterion, optimizer, device)

        acc, _, _ = eval_model(model, val_loader, device)
        results.append([lr, acc])
        print(f"  LR={lr:.4f}  Val Accuracy={acc:.4f}")

    return pd.DataFrame(results, columns=["learning_rate", "Accuracy"])


def search_dropout(X_train, X_val, y_train, y_val, le, scaler, input_dim, num_classes, best_lr):
    """Search over dropout rates."""
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for drop in tqdm(dropouts, desc="Dropout search", unit="drop"):
        model = BRCAGenomicRNN(input_dim, num_classes, dropout=drop).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=1e-5)

        train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, scaler, le)

        for _ in range(20):
            train_epoch(model, train_loader, criterion, optimizer, device)

        acc, _, _ = eval_model(model, val_loader, device)
        results.append([drop, acc])
        print(f"  Dropout={drop:.1f}  Val Accuracy={acc:.4f}")

    return pd.DataFrame(results, columns=["dropout", "Accuracy"])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Data loading (same as all other training scripts) ─────────────────────
    data, clinic = load_data()
    df = preprocess(data, clinic)
    X, y = get_Xy(df)

    print(f"\nRaw dataset: {X.shape[0]} samples × {X.shape[1]} genes")

    # ── Feature selection (critical for deep learning on genomic data) ────────
    # 60k genes × 962 samples is too extreme for RNN — select top 2000 by variance
    selected_genes = select_top_genes(X, n_top=TOP_GENES)
    X = X[selected_genes]
    feature_names = list(X.columns)  # Ensembl IDs for gene mapping
    input_dim = X.shape[1]

    le = LabelEncoder()
    le.fit(y)
    num_classes = len(le.classes_)

    scaler = StandardScaler()

    print(f"After feature selection: {X.shape[0]} samples × {X.shape[1]} genes → {num_classes} classes")
    print(f"Classes: {list(le.classes_)}")
    print(f"Sequence: {math.ceil(input_dim / CHUNK_SIZE)} steps × {CHUNK_SIZE} genes/step\n")

    # ── Train / Val / Test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}\n")

    # ── Step 1: LR search ─────────────────────────────────────────────────────
    print("[1/3] Learning rate search...")
    lr_df = search_lr(X_train, X_val, y_train, y_val, le, scaler, input_dim, num_classes)
    best_lr = lr_df.loc[lr_df["Accuracy"].idxmax(), "learning_rate"]
    print(f"  → Best LR: {best_lr}\n")

    # ── Step 2: Dropout search ────────────────────────────────────────────────
    print("[2/3] Dropout search...")
    dropout_df = search_dropout(
        X_train, X_val, y_train, y_val, le, scaler, input_dim, num_classes, best_lr,
    )
    best_dropout = dropout_df.loc[dropout_df["Accuracy"].idxmax(), "dropout"]
    print(f"  → Best Dropout: {best_dropout}\n")

    # ── Step 3: Final training with early stopping ────────────────────────────
    print(f"[3/3] Training final model (lr={best_lr}, dropout={best_dropout}, "
          f"epochs={EPOCHS}, patience={EARLY_STOP_PATIENCE})...")

    # Keep val set separate for honest early stopping (DO NOT merge into train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(le.transform(y_train)),
        ),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.LongTensor(le.transform(y_val)),
        ),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.LongTensor(le.transform(y_test)),
        ),
        batch_size=BATCH_SIZE,
    )

    model = BRCAGenomicRNN(
        input_dim, num_classes, dropout=best_dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    best_state, best_val_acc, history = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        device, epochs=EPOCHS, patience=EARLY_STOP_PATIENCE,
        desc="Final training", scheduler=scheduler,
    )

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    acc, preds, labels = eval_model(model, test_loader, device)
    y_pred = le.inverse_transform(preds)
    conf_mat = confusion_matrix(y_test, y_pred, labels=le.classes_)

    print(f"\n{'=' * 50}")
    print(f"RNN Test Accuracy: {acc:.4f}")
    print(f"{'=' * 50}")
    print(f"\nConfusion Matrix:")
    print(conf_mat)

    # ── Gene driver demo (first test sample) ──────────────────────────────────
    print("\n── Gene Driver Demo (first test sample) ──")
    sample_raw = X_test.iloc[0].values
    drivers = compute_gene_drivers_rnn(model, sample_raw, feature_names, scaler, le, n_top=10)
    for d in drivers:
        print(f"  #{d['rank']:2d}  {d['ensembl_id']:<20s}  "
              f"attr={d['attribution']:+.4f}  {d['direction']:<14s}  "
              f"expr={d['expression_value']:.1f} TPM  ({d['expression_level']})")

    # ── Save results ──────────────────────────────────────────────────────────
    model_config = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "chunk_size": CHUNK_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": best_dropout,
    }

    results = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
        "selected_genes": selected_genes,    # which genes were selected (for inference)
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
        "input_dim": input_dim,
        "num_classes": num_classes,
        "training_history": history,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/rnn_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nRNN Accuracy: {acc:.4f}")
    print("Saved → models/rnn_results.pkl")
