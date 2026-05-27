# Multimodal Emotion Recognition System

> Recognizing emotions from **Speech**, **Text**, and **Multimodal Fusion** using the TESS dataset.

---

## Overview

This project implements three emotion recognition pipelines:

| Pipeline | Input | Model | Test Accuracy |
|---|---|---|---|
| Speech-Only | Audio (.wav) | Log-Mel Spectrogram + BiLSTM | **99.76%** |
| Text-Only | Transcript | DistilBERT | ~10% |
| Multimodal Fusion | Audio + Text | BiLSTM + DistilBERT + Concatenation | ~98% |

**Emotions:** `angry` `disgust` `fear` `happy` `neutral` `surprise` `sad`

---

## Project Structure

```
emotion_label/
├── data/
│   ├── explore.py              ← data exploration & metadata generation
│   └── tess_metadata.csv       ← auto-generated after running explore.py
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py            ← train speech-only model
│   │   └── test.py             ← evaluate speech-only model
│   ├── text_pipeline/
│   │   ├── train.py            ← train text-only model
│   │   └── test.py             ← evaluate text-only model
│   └── fusion_pipeline/
│       ├── train.py            ← train multimodal fusion model
│       └── test.py             ← evaluate fusion model
├── Results/
│   ├── plots/                  ← all generated plots and visualizations
│   ├── speech_accuracy_table.csv
│   └── speech_failures.csv
├── report/
│   └── Multimodal_Emotion_Recognition_Report.docx
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Shrehitha/Multimodal_Emotion_Recognition_System.git
cd Multimodal_Emotion_Recognition_System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> Recommended: Python 3.9+ and a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Download the TESS Dataset
- Go to: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- Download and extract the dataset
- Place it inside the `data/` folder so the structure looks like:

```
data/
└── TESS/
    ├── OAF_angry/
    ├── OAF_disgust/
    ├── OAF_Fear/
    ├── OAF_happy/
    ├── OAF_neutral/
    ├── OAF_Pleasant_surprise/
    ├── OAF_Sad/
    ├── YAF_angry/
    ├── YAF_disgust/
    ├── YAF_fear/
    ├── YAF_happy/
    ├── YAF_neutral/
    ├── YAF_pleasant_surprised/
    └── YAF_Sad/
```

---

## How to Run

### Step 1: Explore the Dataset
```bash
python data/explore.py
```
This generates:
- `data/tess_metadata.csv` — metadata file used by all pipelines
- `Results/plots/data_distribution.png` — class distribution plot

---

### Step 2: Train the Speech-Only Model
```bash
python models/speech_pipeline/train.py
```
Generates:
- `models/speech_pipeline/speech_model.pt` — saved model weights
- `Results/plots/speech_training_curves.png`
- `Results/plots/speech_confusion_matrix.png`
- `data/speech_test_split.csv`

### Step 3: Evaluate the Speech-Only Model
```bash
python models/speech_pipeline/test.py
```
Generates:
- `Results/plots/speech_test_confusion.png`
- `Results/plots/speech_tsne.png`
- `Results/speech_accuracy_table.csv`
- `Results/speech_failures.csv`

---

### Step 4: Train the Text-Only Model
```bash
python models/text_pipeline/train.py
```
Generates:
- `models/text_pipeline/text_model.pt`
- `Results/plots/text_training_curves.png`
- `Results/plots/text_confusion_matrix.png`

### Step 5: Evaluate the Text-Only Model
```bash
python models/text_pipeline/test.py
```

---

### Step 6: Train the Fusion Model
```bash
python models/fusion_pipeline/train.py
```
Generates:
- `models/fusion_pipeline/fusion_model.pt`
- `Results/plots/fusion_training_curves.png`
- `Results/plots/fusion_confusion_matrix.png`

### Step 7: Evaluate the Fusion Model
```bash
python models/fusion_pipeline/test.py
```

---

## Results Summary

### Accuracy Comparison

| Model | Test Accuracy |
|---|---|
| Speech-Only (BiLSTM) | **99.76%** |
| Multimodal Fusion | ~98% |
| Text-Only (DistilBERT) | ~10% |

### Speech Pipeline — Per Emotion Performance

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| Angry | 1.000 | 1.000 | 1.000 |
| Disgust | 1.000 | 1.000 | 1.000 |
| Fear | 1.000 | 1.000 | 1.000 |
| Happy | 0.984 | 1.000 | 0.992 |
| Neutral | 1.000 | 1.000 | 1.000 |
| Surprise | 1.000 | 0.983 | 0.992 |
| Sad | 1.000 | 1.000 | 1.000 |

---

## Key Findings

- **Speech dominates** — acoustic prosody (pitch, energy, tempo) is the primary carrier of emotion in TESS
- **Text fails** — isolated neutral words (door, chair, page) carry no semantic emotional meaning
- **Fusion ≈ Speech** — adding a weak modality introduces minor noise; fusion didn't outperform speech alone
- **Hardest pair** — Happy vs. Surprise (similar prosodic patterns); only 1 test error observed

---

## Technologies Used

| Category | Library |
|---|---|
| Deep Learning | PyTorch |
| NLP | HuggingFace Transformers (DistilBERT) |
| Audio Processing | Librosa |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Scikit-learn |

---

## Note on Model Weights

Model `.pt` files are not included in this repository due to GitHub's file size limits.
To reproduce them, run the training scripts in order as described above.

---

## Author

**Shrehitha Sureddy**  
GitHub: https://github.com/Shrehitha/Multimodal_Emotion_Recognition_System
