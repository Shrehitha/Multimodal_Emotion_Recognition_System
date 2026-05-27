# models/text_pipeline/test.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm

from transformers import DistilBertTokenizer

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from train import CFG, EMOTIONS, LABEL2IDX, IDX2LABEL, TESSTextDataset, TextEmotionModel

from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)

        preds = logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def plot_confusion(labels, preds):
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

    out_path = "Results/plots/text_test_confusion.png"

    plt.savefig(out_path, dpi=150)

    plt.close()

    print(f"Saved → {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    df = pd.read_csv("data/text_test_split.csv")

    print(f"Test samples: {len(df)}")

    tokenizer = DistilBertTokenizer.from_pretrained(CFG["model_name"])

    test_ds = TESSTextDataset(df, tokenizer, CFG)

    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False)

    model = TextEmotionModel(
        model_name=CFG["model_name"], num_classes=len(EMOTIONS), dropout=CFG["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(CFG["model_path"], map_location=device))

    print(f"Model loaded from {CFG['model_path']}")

    preds, labels = evaluate(model, test_loader, device)

    print("\n── Test Classification Report ──")

    report = classification_report(
        labels, preds, target_names=EMOTIONS, output_dict=True
    )

    print(classification_report(labels, preds, target_names=EMOTIONS))

    accuracy = report["accuracy"]

    rows = []

    for emotion in EMOTIONS:
        rows.append(
            {
                "Emotion": emotion,
                "Precision": round(report[emotion]["precision"], 3),
                "Recall": round(report[emotion]["recall"], 3),
                "F1-Score": round(report[emotion]["f1-score"], 3),
                "Support": int(report[emotion]["support"]),
            }
        )

    table_df = pd.DataFrame(rows)

    out_csv = "Results/text_accuracy_table.csv"

    table_df.to_csv(out_csv, index=False)

    print(f"\nAccuracy table saved → {out_csv}")

    print(table_df.to_string(index=False))

    print(f"\nOverall Test Accuracy: {accuracy:.4f}")

    plot_confusion(labels, preds)

    print("\nText pipeline evaluation complete.")


if __name__ == "__main__":
    main()
