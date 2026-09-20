"""Intent router -- the W14 transformer, exported to ONNX and served by
onnxruntime (no torch in the serving image).

Two graph shapes are supported, both produced by scripts/train_router.py:
  * transformer  -> inputs input_ids / attention_mask, needs the HF tokenizer
  * tfidf+linear -> a single string input, preprocessing baked into the graph

If no artifact is present the router returns None and the pipeline falls back to
letting the LLM classify the intent itself (it emits `intent` in its JSON anyway).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from .config import settings


def _softmax(xs: list[float]) -> list[float]:
    hi = max(xs)
    exps = [math.exp(x - hi) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


@lru_cache
def _load():
    """Returns (session, labels, tokenizer|None) or None when no model is on disk."""
    d: Path = settings().router_path
    model_path = d / "model.onnx"
    if not model_path.exists():
        return None

    import onnxruntime as ort  # heavy import, only when an artifact exists

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))

    tokenizer = None
    if (d / "tokenizer.json").exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=128)
        tokenizer.enable_padding(length=128)
    return session, labels, tokenizer


def classify(text: str) -> tuple[str, float] | None:
    loaded = _load()
    if loaded is None:
        return None
    session, labels, tokenizer = loaded

    if tokenizer is not None:
        import numpy as np

        enc = tokenizer.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        feed = {i.name: feed[i.name] for i in session.get_inputs() if i.name in feed}
        logits = session.run(None, feed)[0][0].tolist()
    else:
        import numpy as np

        name = session.get_inputs()[0].name
        out = session.run(None, {name: np.array([[text]])})
        # skl2onnx classifiers return (label, [{class_index: probability, ...}])
        if len(out) > 1 and isinstance(out[1], list) and isinstance(out[1][0], dict):
            scores = out[1][0]
            best = max(scores, key=scores.get)
            label = labels[int(best)] if str(best).lstrip("-").isdigit() else str(best)
            return _floor(label, float(scores[best]))
        logits = list(out[-1][0])

    probs = _softmax([float(x) for x in logits])
    idx = max(range(len(probs)), key=probs.__getitem__)
    return _floor(labels[idx], probs[idx])


def _floor(label: str, confidence: float) -> tuple[str, float] | None:
    """Below the floor the router is guessing; say nothing rather than hand the
    LLM a misleading prior."""
    if confidence < settings().router_min_confidence:
        return None
    return label, round(confidence, 4)


def available() -> bool:
    return _load() is not None
