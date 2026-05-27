"""
Speech-Only Emotion Recognition Pipeline
Architecture:
  Preprocessing  : resample to 16kHz, trim silence, pad/truncate to fixed length
  Feature Extraction : Log-Mel Spectrogram (time_steps x n_mels)
  Temporal Modelling : 2-layer Bidirectional LSTM
  Classifier         : FC layers -> softmax (7 emotions)
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "data_csv": "data/tess_metadata.csv",
    "sample_rate": 16000,
    "max_duration": 3.0,  # seconds — clips longer are truncated
    "n_mels": 64,
    "n_fft": 512,
    "hop_length": 160,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 13,
    "lr": 1e-3,
    "seed": 42,
    "save_dir": "Results",
    "model_path": "models/speech_pipeline/speech_model.pt",
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# PREPROCESSING + FEATURE EXTRACTION
# ─────────────────────────────────────────────
def load_and_preprocess(path: str, cfg: dict) -> np.ndarray:
    """
    1. Load audio at target sample rate
    2. Trim leading/trailing silence
    3. Pad or truncate to fixed length
    4. Extract Log-Mel Spectrogram  shape: (n_mels, time_steps)
    """
    sr = cfg["sample_rate"]
    max_samples = int(sr * cfg["max_duration"])

    # load & resample
    y, _ = librosa.load(path, sr=sr, mono=True)

    # trim silence (top_db=20 is a standard threshold)
    y, _ = librosa.effects.trim(y, top_db=20)

    # pad or truncate
    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))
    else:
        y = y[:max_samples]

    # log-mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)

    # normalise to zero mean unit variance per sample
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    return log_mel.T  # → (T, n_mels)  i.e. (time_steps, features)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TESSAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature = load_and_preprocess(row["path"], self.cfg)  # (T, n_mels)
        label = LABEL2IDX[row["emotion"]]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class SpeechEmotionModel(nn.Module):
    """
    Feature Extraction  : handled outside (log-mel)
    Temporal Modelling  : Bidirectional LSTM
    Classifier          : FC -> ReLU -> Dropout -> FC -> softmax
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        out, (hn, _) = self.lstm(x)
        # take mean over time steps as the sequence representation
        pooled = out.mean(dim=1)  # (batch, hidden*2)
        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits

    def get_representation(self, x):
        """Returns temporal representation before classifier (for visualisation)."""
        out, _ = self.lstm(x)
        return out.mean(dim=1)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False, desc="  train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/speech_training_curves.png", dpi=150)
    plt.close()
    print(f"  Saved training curves → {save_dir}/speech_training_curves.png")


def plot_confusion(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Speech Model — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/speech_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {save_dir}/speech_confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load metadata ──
    csv_path = Path(CFG["data_csv"])
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run data/explore.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # keep only known emotions
    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)
    print(f"Total samples: {len(df)}")
    print(df["emotion"].value_counts().to_string())

    # ── train / val / test split (70 / 15 / 15) ──
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["emotion"], random_state=CFG["seed"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["emotion"], random_state=CFG["seed"]
    )
    print(f"Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # save test split for test.py
    test_df.to_csv("data/speech_test_split.csv", index=False)

    # ── datasets & loaders ──
    train_ds = TESSAudioDataset(train_df, CFG)
    val_ds = TESSAudioDataset(val_df, CFG)
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── compute time steps ──
    sample_feat = load_and_preprocess(train_df.iloc[0]["path"], CFG)
    T, n_mels = sample_feat.shape
    print(f"Feature shape per sample: ({T}, {n_mels})")

    # ── model ──
    model = SpeechEmotionModel(
        input_size=n_mels,
        hidden_size=CFG["lstm_hidden"],
        num_layers=CFG["lstm_layers"],
        num_classes=len(EMOTIONS),
        dropout=CFG["dropout"],
    ).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # ── training ──
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    save_dir = Path(CFG["save_dir"]) / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        vl_loss, vl_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{CFG['epochs']}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CFG["model_path"])
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")

    # ── final report ──
    print("\n── Validation Classification Report ──")
    print(classification_report(labels, preds, target_names=EMOTIONS))

    plot_curves(history, save_dir)
    plot_confusion(labels, preds, save_dir)

    # save history
    with open(f"{CFG['save_dir']}/speech_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("Speech pipeline training complete.")


if __name__ == "__main__":
    main()
