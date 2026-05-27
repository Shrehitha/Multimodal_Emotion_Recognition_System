# models/fusion_pipeline/train.py

"""
Multimodal Fusion Emotion Recognition
Fusion:
    Speech Encoder (BiLSTM)
    +
    Text Encoder (DistilBERT)
    ↓
    Concatenation
    ↓
    Fusion Classifier
"""

import os
import json
import random
import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertTokenizer, DistilBertModel

from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "data_csv": "data/tess_metadata.csv",
    # speech
    "sample_rate": 16000,
    "max_duration": 3.0,
    "n_mels": 64,
    "n_fft": 512,
    "hop_length": 160,
    # text
    "model_name": "distilbert-base-uncased",
    "max_length": 16,
    # fusion
    "batch_size": 16,
    "epochs": 10,
    "lr": 1e-4,
    "dropout": 0.3,
    "seed": 42,
    "model_path": "models/fusion_pipeline/fusion_model.pt",
    "save_dir": "Results",
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]

LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}


# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────
# SPEECH FEATURE EXTRACTION
# ─────────────────────────────────────────────
def load_audio(path, cfg):

    sr = cfg["sample_rate"]

    max_samples = int(sr * cfg["max_duration"])

    y, _ = librosa.load(path, sr=sr, mono=True)

    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))
    else:
        y = y[:max_samples]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    return log_mel.T


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FusionDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):

        self.df = df.reset_index(drop=True)

        self.tokenizer = tokenizer

        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        speech = load_audio(row["path"], self.cfg)

        text = str(row["transcript"]).lower()

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg["max_length"],
            return_tensors="pt",
        )

        label = LABEL2IDX[row["emotion"]]

        return {
            "speech": torch.tensor(speech, dtype=torch.float32),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# SPEECH ENCODER
# ─────────────────────────────────────────────
class SpeechEncoder(nn.Module):
    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        pooled = out.mean(dim=1)

        return pooled


# ─────────────────────────────────────────────
# TEXT ENCODER
# ─────────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, model_name):

        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0]

        return cls_embedding


# ─────────────────────────────────────────────
# FUSION MODEL
# ─────────────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.speech_encoder = SpeechEncoder()

        self.text_encoder = TextEncoder(cfg["model_name"])

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, len(EMOTIONS)),
        )

    def forward(self, speech, input_ids, attention_mask):

        speech_repr = self.speech_encoder(speech)

        text_repr = self.text_encoder(input_ids, attention_mask)

        fusion = torch.cat([speech_repr, text_repr], dim=1)

        logits = self.classifier(fusion)

        return logits


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, leave=False, desc="train"):
        speech = batch["speech"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(speech, input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(labels)

        preds = logits.argmax(1)

        correct += (preds == labels).sum().item()

        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for batch in loader:
        speech = batch["speech"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        logits = model(speech, input_ids, attention_mask)

        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)

        preds = logits.argmax(1)

        correct += (preds == labels).sum().item()

        total += len(labels)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_curves(history, save_dir):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Fusion Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Fusion Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()

    out_path = f"{save_dir}/fusion_training_curves.png"

    plt.savefig(out_path, dpi=150)

    plt.close()

    print(f"Saved → {out_path}")


def plot_confusion(labels, preds, save_dir):

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Fusion Model Confusion Matrix")

    plt.tight_layout()

    out_path = f"{save_dir}/fusion_confusion_matrix.png"

    plt.savefig(out_path, dpi=150)

    plt.close()

    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    set_seed(CFG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    df = pd.read_csv(CFG["data_csv"])

    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG["model_name"])

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["emotion"], random_state=CFG["seed"]
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["emotion"], random_state=CFG["seed"]
    )

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    test_df.to_csv("data/fusion_test_split.csv", index=False)

    train_ds = FusionDataset(train_df, tokenizer, CFG)

    val_ds = FusionDataset(val_df, tokenizer, CFG)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)

    model = FusionModel(CFG).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])

    best_val_acc = 0.0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    save_dir = Path(CFG["save_dir"]) / "plots"

    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        vl_loss, vl_acc, preds, labels = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{CFG['epochs']} "
            f"train_loss={tr_loss:.4f} "
            f"train_acc={tr_acc:.4f} "
            f"val_loss={vl_loss:.4f} "
            f"val_acc={vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc

            torch.save(model.state_dict(), CFG["model_path"])

            print(f"✓ Best model saved ({best_val_acc:.4f})")

    plot_curves(history, save_dir)

    plot_confusion(labels, preds, save_dir)

    with open(f"{CFG['save_dir']}/fusion_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n── Validation Classification Report ──")

    print(classification_report(labels, preds, target_names=EMOTIONS))

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    print("Fusion pipeline training complete.")


if __name__ == "__main__":
    main()
