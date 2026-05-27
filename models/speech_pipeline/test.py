"""
Speech-Only Emotion Recognition — Test / Inference Script
Loads the best saved model and evaluates on the held-out test split.
Also generates:
  - Classification report
  - Confusion matrix
  - t-SNE plot of temporal representations
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# reuse everything defined in train.py
sys.path.append(str(Path(__file__).parent))
from train import (
    CFG, EMOTIONS, LABEL2IDX, IDX2LABEL,
    TESSAudioDataset, SpeechEmotionModel, load_and_preprocess
)

RESULTS_DIR = Path("Results/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path: str, input_size: int, device):
    model = SpeechEmotionModel(
        input_size  = input_size,
        hidden_size = CFG["lstm_hidden"],
        num_layers  = CFG["lstm_layers"],
        num_classes = len(EMOTIONS),
        dropout     = CFG["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels, all_reprs = [], [], []
    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        reprs  = model.get_representation(x)   # (batch, hidden*2)
        logits = model.classifier(reprs)
        preds  = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_reprs.extend(reprs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_reprs)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Speech Model — Test Confusion Matrix")
    plt.tight_layout()
    out = RESULTS_DIR / "speech_test_confusion.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved → {out}")


def plot_tsne(reprs, labels):
    print("Running t-SNE on temporal representations...")
    tsne   = TSNE(n_components=2, random_state=CFG["seed"], perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(reprs)

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(EMOTIONS))
    for i, emotion in enumerate(EMOTIONS):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=emotion, color=palette[i], alpha=0.6, s=25)
    ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("t-SNE — Speech Temporal Representations")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.tight_layout()
    out = RESULTS_DIR / "speech_tsne.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved → {out}")


def save_accuracy_table(labels, preds):
    report = classification_report(labels, preds, target_names=EMOTIONS, output_dict=True)
    rows   = []
    for emotion in EMOTIONS:
        r = report[emotion]
        rows.append({
            "Emotion"  : emotion,
            "Precision": round(r["precision"], 3),
            "Recall"   : round(r["recall"], 3),
            "F1-Score" : round(r["f1-score"], 3),
            "Support"  : int(r["support"]),
        })
    df = pd.DataFrame(rows)
    out_csv = Path("Results") / "speech_accuracy_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAccuracy table saved → {out_csv}")
    print(df.to_string(index=False))

    overall = accuracy_score(labels, preds)
    print(f"\nOverall Test Accuracy: {overall:.4f}")
    return overall


# ─────────────────────────────────────────────
# FAILURE CASE ANALYSIS
# ─────────────────────────────────────────────
def analyse_failures(df_test, preds, labels, n=5):
    print(f"\n── Top {n} Failure Cases ──")
    failures = []
    for i, (pred, true) in enumerate(zip(preds, labels)):
        if pred != true:
            failures.append({
                "file"     : Path(df_test.iloc[i]["path"]).name,
                "transcript": df_test.iloc[i]["transcript"],
                "true"     : IDX2LABEL[true],
                "predicted": IDX2LABEL[pred],
            })
    failures_df = pd.DataFrame(failures)
    if not failures_df.empty:
        print(failures_df.head(n).to_string(index=False))
        failures_df.to_csv("Results/speech_failures.csv", index=False)
        print("All failures saved → Results/speech_failures.csv")
    else:
        print("No failures found!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load test split ──
    test_csv = Path("data/speech_test_split.csv")
    if not test_csv.exists():
        print("ERROR: data/speech_test_split.csv not found. Run train.py first.")
        sys.exit(1)

    df_test = pd.read_csv(test_csv)
    df_test = df_test[df_test["emotion"].isin(EMOTIONS)].reset_index(drop=True)
    print(f"Test samples: {len(df_test)}")

    # ── dataset & loader ──
    test_ds     = TESSAudioDataset(df_test, CFG)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                             shuffle=False, num_workers=2)

    # ── infer input size ──
    sample = load_and_preprocess(df_test.iloc[0]["path"], CFG)
    _, n_mels = sample.shape

    # ── load model ──
    model_path = CFG["model_path"]
    if not Path(model_path).exists():
        print(f"ERROR: model not found at {model_path}. Run train.py first.")
        sys.exit(1)
    model = load_model(model_path, n_mels, device)

    # ── inference ──
    preds, labels, reprs = run_inference(model, test_loader, device)

    # ── reports & plots ──
    print("\n── Test Classification Report ──")
    print(classification_report(labels, preds, target_names=EMOTIONS))

    overall = save_accuracy_table(labels, preds)
    plot_confusion(labels, preds)
    plot_tsne(reprs, labels)
    analyse_failures(df_test, preds, labels, n=5)

    print("\nSpeech pipeline evaluation complete.")
    return overall


if __name__ == "__main__":
    main()
