# Multimodal Emotion Recognition using Speech, Text, and Fusion

This project performs **emotion recognition** using:

- Speech-only features
- Text-only features
- Multimodal fusion of speech and text

The system is trained and evaluated on the **TESS (Toronto Emotional Speech Set)** dataset.

---

# Emotions Recognized

- angry
- disgust
- fear
- happy
- neutral
- surprise
- sad

---

# Project Overview

The project compares three different emotion recognition pipelines:

| Pipeline | Description |
|---|---|
| Speech-Only | Uses acoustic speech features and BiLSTM |
| Text-Only | Uses DistilBERT on transcripts |
| Fusion | Combines speech and text representations |

The goal is to analyze how different modalities contribute to emotion recognition performance.

---

# Dataset

Dataset used:
- TESS (Toronto Emotional Speech Set)

Download from Kaggle:

https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Place extracted dataset inside:

```bash
data/TESS/
```

---

# Project Structure

```text
project/
├── data/
│   ├── TESS/
│   ├── explore.py
│   └── tess_metadata.csv
│
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   │
│   ├── text_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── fusion_pipeline/
│       ├── train.py
│       └── test.py
│
├── Results/
│   ├── plots/
│   ├── speech_accuracy_table.csv
│   ├── speech_failures.csv
│   ├── speech_history.json
│   ├── text_history.json
│   └── fusion_history.json
│
├── requirements.txt
└── README.md
```

---

# Speech Pipeline

## Architecture

Speech preprocessing:
- Resampling to 16kHz
- Silence trimming
- Padding/truncation

Feature extraction:
- Log-Mel Spectrograms

Temporal modelling:
- Bidirectional LSTM (BiLSTM)

Classifier:
- Fully connected neural network with softmax output

## Generated Outputs

- Training curves
- Confusion matrix
- t-SNE visualization
- Accuracy table
- Failure analysis

---

# Text Pipeline

## Architecture

Text preprocessing:
- Transcript tokenization

Feature extraction:
- DistilBERT embeddings

Contextual modelling:
- Transformer-based encoder

Classifier:
- Fully connected neural network

## Important Observation

The text-only model performs poorly because TESS transcripts contain isolated neutral words such as:

```text
door
page
chair
back
```

These words contain very little semantic emotional information.

Emotion is primarily encoded in speech prosody rather than text semantics.

---

# Fusion Pipeline

## Architecture

Fusion combines:
- Speech embeddings from BiLSTM
- Text embeddings from DistilBERT

Fusion strategy:
- Feature concatenation

Classifier:
- Multi-layer fully connected network

---

# Results

| Model | Accuracy |
|---|---|
| Speech-Only | 99.76% |
| Text-Only | ~10% |
| Fusion | ~98% |

---

# Key Observations

- Speech modality significantly outperformed text modality.
- Emotional information in TESS is primarily encoded in acoustic speech patterns.
- Text transcripts are emotionally weak because they contain isolated neutral words.
- Fusion achieved strong performance but did not significantly outperform speech-only classification.

---

# Visualizations

Generated plots include:

- Emotion distribution
- Training curves
- Confusion matrices
- t-SNE visualization of learned embeddings

Plots are saved inside:

```bash
Results/plots/
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional libraries:

```bash
pip install transformers datasets sentencepiece accelerate
```

---

# Usage

## 1. Explore Dataset

```bash
python data/explore.py
```

---

## 2. Train Speech Model

```bash
python models/speech_pipeline/train.py
```

---

## 3. Evaluate Speech Model

```bash
python models/speech_pipeline/test.py
```

---

## 4. Train Text Model

```bash
python models/text_pipeline/train.py
```

---

## 5. Evaluate Text Model

```bash
python models/text_pipeline/test.py
```

---

## 6. Train Fusion Model

```bash
python models/fusion_pipeline/train.py
```

---

## 7. Evaluate Fusion Model

```bash
python models/fusion_pipeline/test.py
```

---

# Technologies Used

- Python
- PyTorch
- Transformers (HuggingFace)
- Librosa
- Scikit-learn
- Matplotlib
- Seaborn

---

# Future Improvements

- Attention-based fusion
- Transformer-based speech encoders
- Real-time emotion recognition
- Sentence-level multimodal datasets
- Cross-speaker generalization

---

# Author

Shrehitha Sureddy