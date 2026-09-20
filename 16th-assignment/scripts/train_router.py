"""Train the 11-class intent router and export it to ONNX for serving.

This is the W14 support-routing model turned into a production artifact. The
serving container never imports torch: it loads models/router/model.onnx with
onnxruntime (see app/router.py).

    # fast CPU baseline, ~20 s, no torch (default)
    python scripts/train_router.py --backend tfidf

    # the W14 DistilBERT encoder, GPU recommended
    python scripts/train_router.py --backend distilbert --epochs 2

Both write: models/router/model.onnx, labels.json, metrics.json
(+ tokenizer.json for the transformer backend).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "router"

DATA_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
LABELS = [
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]
SEED = 0
MAX_LEN = 128


def load_split():
    """Same source and 80/10/10 stratified split as the W14 notebook."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    rows = load_dataset(DATA_ID, split="train")
    texts, labels = [], []
    for r in rows:
        if r["category"] in LABELS:
            texts.append(r["instruction"])
            labels.append(LABELS.index(r["category"]))

    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    x_va, x_te, y_va, y_te = train_test_split(
        x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )
    print(f"train={len(x_tr)} val={len(x_va)} test={len(x_te)}")
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def report(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
    }


def train_tfidf(train, val, test) -> dict:
    """TF-IDF + logistic regression, vectoriser baked into the ONNX graph."""
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    import numpy as np

    (x_tr, y_tr), _, (x_te, y_te) = train, val, test
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=1000, C=4.0, n_jobs=-1),
    )
    pipe.fit(x_tr, y_tr)
    metrics = report(y_te, pipe.predict(x_te))

    onx = to_onnx(
        pipe,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={id(pipe.steps[-1][1]): {"zipmap": True}},
        target_opset=15,
    )
    (OUT / "model.onnx").write_bytes(onx.SerializeToString())
    return metrics


def train_distilbert(train, val, test, epochs: int, model_id: str) -> dict:
    """Fine-tune the W14 encoder, then export the graph with torch.onnx."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = train, val, test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(LABELS)
    ).to(device)

    class DS(Dataset):
        def __init__(self, xs, ys):
            self.enc = tok(xs, truncation=True, padding="max_length", max_length=MAX_LEN)
            self.ys = ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[i])
            return item

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loader = DataLoader(DS(x_tr, y_tr), batch_size=32, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in DataLoader(DS(x_te, y_te), batch_size=64):
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            preds += model(**batch).logits.argmax(-1).cpu().tolist()
    metrics = report(y_te, preds)

    model.to("cpu")
    dummy = tok("export", truncation=True, padding="max_length", max_length=MAX_LEN,
                return_tensors="pt")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    tok.backend_tokenizer.save(str(OUT / "tokenizer.json"))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tfidf", "distilbert"], default="tfidf")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--model-id", default="distilbert-base-uncased")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = load_split()

    started = time.time()
    if args.backend == "tfidf":
        metrics = train_tfidf(train, val, test)
    else:
        metrics = train_distilbert(train, val, test, args.epochs, args.model_id)
    metrics |= {"backend": args.backend, "train_seconds": round(time.time() - started, 1)}

    (OUT / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
