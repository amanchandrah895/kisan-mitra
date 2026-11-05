"""Groundedness checking: is each answer sentence supported by the retrieved evidence?

NLIGrounder uses an MNLI/FEVER/ANLI DeBERTa model: a sentence is 'supported' if the max
entailment probability over evidence chunks exceeds a threshold. LexicalGrounder is a
CPU-only fallback (content-word overlap) used in tests."""
from __future__ import annotations

import re

from .lang import split_sentences
from .retriever import tokenize

_SKIP = re.compile(r"^(safety:|please call|i do not know|i could not find|i am kisan mitra)", re.I)


def answer_sentences(answer: str) -> list[str]:
    return [s for s in split_sentences(answer) if len(tokenize(s)) >= 3 and not _SKIP.search(s)]


class LexicalGrounder:
    name = "lexical"

    def __init__(self, min_overlap: float = 0.6):
        self.min_overlap = min_overlap

    def sentence_scores(self, sentences: list[str], contexts: list[str]) -> list[float]:
        ctx = set(tokenize(" ".join(contexts)))
        out = []
        for s in sentences:
            toks = tokenize(s)
            out.append(sum(t in ctx for t in toks) / max(1, len(toks)))
        return out

    def score(self, answer: str, contexts: list[str], threshold: float | None = None) -> dict:
        thr = self.min_overlap if threshold is None else threshold
        sents = answer_sentences(answer)
        if not sents or not contexts:
            return {"n_sentences": len(sents), "supported": 0, "ratio": None, "per_sentence": []}
        sc = self.sentence_scores(sents, contexts)
        sup = sum(x >= thr for x in sc)
        return {"n_sentences": len(sents), "supported": sup, "ratio": sup / len(sents),
                "per_sentence": [round(x, 3) for x in sc]}


class NLIGrounder(LexicalGrounder):
    name = "nli"

    def __init__(self, model_name: str, device: str = "cuda", threshold: float = 0.5):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in labels.items() if "entail" in l)
        self.min_overlap = threshold

    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self

    def sentence_scores(self, sentences, contexts):
        torch = self.torch
        pairs = [(c, s) for s in sentences for c in contexts]
        probs = []
        for b in range(0, len(pairs), 16):
            batch = pairs[b:b + 16]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            probs += torch.softmax(logits, -1)[:, self.entail_idx].tolist()
        n = len(contexts)
        return [max(probs[i * n:(i + 1) * n]) for i in range(len(sentences))]
