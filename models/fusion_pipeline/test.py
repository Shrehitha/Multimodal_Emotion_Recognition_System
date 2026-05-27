# models/fusion_pipeline/test.py

import pandas as pd
import torch

from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from train import CFG, EMOTIONS, FusionDataset, FusionModel

from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        speech = batch["speech"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        logits = model(speech, input_ids, attention_mask)

        preds = logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    df = pd.read_csv("data/fusion_test_split.csv")

    tokenizer = DistilBertTokenizer.from_pretrained(CFG["model_name"])

    test_ds = FusionDataset(df, tokenizer, CFG)

    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False)

    model = FusionModel(CFG).to(device)

    model.load_state_dict(torch.load(CFG["model_path"], map_location=device))

    print(f"Loaded model from {CFG['model_path']}")

    preds, labels = evaluate(model, test_loader, device)

    print("\n── Fusion Test Classification Report ──")

    print(classification_report(labels, preds, target_names=EMOTIONS))


if __name__ == "__main__":
    main()
