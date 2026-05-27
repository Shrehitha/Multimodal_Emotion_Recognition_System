# models/text_pipeline/train.py

"""
Text-Only Emotion Recognition Pipeline
Architecture:
  Preprocessing        : tokenization using DistilBERT tokenizer
  Feature Extraction   : DistilBERT embeddings
  Contextual Modelling : DistilBERT transformer
  Classifier           : FC -> ReLU -> Dropout -> Softmax
"""

import os
import json
import random
import numpy as np
import pandas as pd
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
    "model_name": "distilbert-base-uncased",
    "max_length": 16,
    "batch_size": 16,
    "epochs": 6,
    "lr": 2e-5,
    "dropout": 0.3,
    "seed": 42,
    "model_path": "models/text_pipeline/text_model.pt",
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TESSTextDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["transcript"]).lower()
        label = LABEL2IDX[row["emotion"]]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg["max_length"],
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class TextEmotionModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding)

        return logits

    def get_representation(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, leave=False, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)

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
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()

    out_path = f"{save_dir}/text_training_curves.png"

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

    plt.tight_layout()

    out_path = f"{save_dir}/text_confusion_matrix.png"

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

    print(f"Total samples: {len(df)}")

    tokenizer = DistilBertTokenizer.from_pretrained(CFG["model_name"])

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["emotion"], random_state=CFG["seed"]
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["emotion"], random_state=CFG["seed"]
    )

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    test_df.to_csv("data/text_test_split.csv", index=False)

    train_ds = TESSTextDataset(train_df, tokenizer, CFG)
    val_ds = TESSTextDataset(val_df, tokenizer, CFG)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)

    model = TextEmotionModel(
        model_name=CFG["model_name"], num_classes=len(EMOTIONS), dropout=CFG["dropout"]
    ).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()

    # optimizer = AdamW(model.parameters(), lr=CFG["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = 0.0

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

            print(f"✓ Best model saved (val_acc={best_val_acc:.4f})")

    print("\n── Validation Classification Report ──")

    print(classification_report(labels, preds, target_names=EMOTIONS))

    plot_curves(history, save_dir)

    plot_confusion(labels, preds, save_dir)

    with open(f"{CFG['save_dir']}/text_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    print("Text pipeline training complete.")


if __name__ == "__main__":
    main()
