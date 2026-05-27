"""
Data exploration script for TESS dataset.
Run this after downloading TESS from Kaggle into data/TESS/
Expected structure:
  data/TESS/
    OAF_angry/   OAF_disgust/  OAF_Fear/  OAF_happy/
    OAF_neutral/ OAF_Pleasant_surprise/ OAF_Sad/
    YAF_angry/   YAF_disgust/  YAF_fear/  YAF_happy/
    YAF_neutral/ YAF_pleasant_surprised/ YAF_Sad/

Label extraction: emotion is the last part of the folder name
  e.g. OAF_angry -> angry, YAF_happy -> happy
"""

import os
import sys
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "TESS"


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    records = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        # emotion = last segment after underscore, lowercased
        emotion = folder.name.split("_")[-1].lower()
        # normalize label

        if emotion in ["pleasant_surprise", "pleasant_surprised", "ps", "surprised"]:
            emotion = "surprise"
        for wav_file in sorted(folder.glob("*.wav")):
            # transcript = middle part of filename e.g. OAF_back_angry.wav -> "back"
            parts = wav_file.stem.split("_")
            transcript = parts[1] if len(parts) >= 3 else ""
            records.append(
                {
                    "path": str(wav_file),
                    "transcript": transcript,
                    "emotion": emotion,
                    "speaker": folder.name.split("_")[0],  # OAF or YAF
                }
            )
    return pd.DataFrame(records)


def analyse(df: pd.DataFrame):
    print("=" * 50)
    print(f"Total samples : {len(df)}")
    print(f"Emotions      : {sorted(df['emotion'].unique())}")
    print(f"Speakers      : {sorted(df['speaker'].unique())}")
    print("\nClass distribution:")
    print(df["emotion"].value_counts().to_string())

    # check audio durations on a small sample
    print("\nChecking audio durations (sample of 20)...")
    durations = []
    for path in df["path"].sample(min(20, len(df)), random_state=42):
        try:
            y, sr = librosa.load(path, sr=None)
            durations.append(len(y) / sr)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    if durations:
        print(f"  Min  : {min(durations):.2f}s")
        print(f"  Max  : {max(durations):.2f}s")
        print(f"  Mean : {np.mean(durations):.2f}s")


def plot_distribution(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # class bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["emotion"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette="Set2")
    axes[0].set_title("Emotion Distribution")
    axes[0].set_xlabel("Emotion")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    # speaker breakdown
    pivot = df.groupby(["speaker", "emotion"]).size().unstack(fill_value=0)
    pivot.T.plot(kind="bar", ax=axes[1], colormap="Set1")
    axes[1].set_title("Emotion by Speaker")
    axes[1].set_xlabel("Emotion")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Speaker")

    plt.tight_layout()
    out_path = out_dir / "data_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close()


def save_csv(df: pd.DataFrame):
    out = Path(__file__).parent / "tess_metadata.csv"
    df.to_csv(out, index=False)
    print(f"Metadata CSV saved → {out}")


if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"ERROR: TESS data not found at {DATA_DIR}")
        print("Please download from Kaggle and extract into data/TESS/")
        sys.exit(1)

    df = build_dataframe(DATA_DIR)
    analyse(df)
    plot_distribution(df, out_dir=Path(__file__).parent.parent / "Results" / "plots")
    save_csv(df)
    print("\nExploration complete.")
