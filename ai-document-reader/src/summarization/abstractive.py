"""Optional abstractive summary with a PRETRAINED model (English only).

Model: sshleifer/distilbart-cnn-12-6 (Hugging Face), a DistilBART model
fine-tuned on CNN/DailyMail news summarisation. We do NOT train it.

Requires `transformers` + `torch` and an internet connection on first use
(~1.2 GB download). If anything fails, the caller falls back to the clearly
labelled extractive summary.
"""

from __future__ import annotations

from functools import lru_cache

from .extractive import SummaryResult, split_sentences

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
SUPPORTED_LANGUAGES = {"english"}
MAX_INPUT_WORDS = 700  # model input limit is 1024 tokens


class AbstractiveUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise AbstractiveUnavailableError("`transformers` is not installed.") from exc
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME), AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    except Exception as exc:  # network errors, missing torch, etc.
        raise AbstractiveUnavailableError(f"Could not load {MODEL_NAME}: {exc}") from exc


def abstractive_summary(text: str, language: str, max_length: int = 150) -> SummaryResult:
    if language not in SUPPORTED_LANGUAGES:
        raise AbstractiveUnavailableError(
            f"{MODEL_NAME} only supports English; detected language is {language}."
        )
    words = text.split()
    if len(words) < 40:
        raise AbstractiveUnavailableError("Document is too short for abstractive summarisation.")
    import torch

    tokenizer, model = _load_model()
    inputs = tokenizer(" ".join(words[:MAX_INPUT_WORDS]), return_tensors="pt",
                       truncation=True, max_length=1024)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=max_length, min_length=min(40, max_length // 2),
                             num_beams=4, do_sample=False, early_stopping=True)
    summary = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return SummaryResult(summary, split_sentences(summary) or [summary], "abstractive", MODEL_NAME, [])
