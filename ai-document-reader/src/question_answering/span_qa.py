"""Extractive span QA with a PRETRAINED model (not trained in this project).

Model: deepset/xlm-roberta-base-squad2 - multilingual XLM-RoBERTa fine-tuned on
SQuAD 2.0. Given a question and a passage, it predicts the start and end of the
answer span inside the passage, or "no answer". The answer is always an exact
substring of the document.

Requires `transformers`, `torch` and `sentencepiece` (model download ~1.1 GB on
first use). This model is used only for question answering; the language
detection task still uses the TF-IDF baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MODEL_NAME = "deepset/xlm-roberta-base-squad2"
MAX_ANSWER_TOKENS = 30


class SpanQAUnavailableError(RuntimeError):
    pass


@dataclass
class Span:
    text: str
    score: float       # start_logit + end_logit of the best span
    null_score: float  # score of "no answer"
    context: str


@lru_cache(maxsize=1)
def _load():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:
        raise SpanQAUnavailableError("Exact answers need `transformers`, `torch` and `sentencepiece`.") from exc
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        model.eval()
        return AutoTokenizer.from_pretrained(MODEL_NAME), model
    except Exception as exc:
        raise SpanQAUnavailableError(f"Could not load {MODEL_NAME}: {exc}") from exc


def best_span(question: str, context: str) -> Span | None:
    import torch

    tokenizer, model = _load()
    enc = tokenizer(question, context, truncation="only_second", max_length=384, stride=96,
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        out = model(**enc)

    best = None
    for i in range(len(offsets)):
        start_logits, end_logits = out.start_logits[i], out.end_logits[i]
        seq_ids = enc.sequence_ids(i)
        null = float(start_logits[0] + end_logits[0])
        starts = start_logits.argsort(descending=True)[:20].tolist()
        ends = end_logits.argsort(descending=True)[:20].tolist()
        for s in starts:
            for e in ends:
                if seq_ids[s] != 1 or seq_ids[e] != 1 or e < s or e - s + 1 > MAX_ANSWER_TOKENS:
                    continue
                score = float(start_logits[s] + end_logits[e])
                if best is None or score > best.score:
                    a, b = int(offsets[i][s][0]), int(offsets[i][e][1])
                    # null score of the same window the span comes from
                    best = Span(context[a:b].strip(), score, null, context)
    return best if best and best.text else None
