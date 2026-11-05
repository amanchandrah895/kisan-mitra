"""Evaluation metrics: key-fact accuracy, tool numeric accuracy, retrieval, abstention,
translation (chrF++), speech recognition (WER/CER) and number preservation."""
from __future__ import annotations

import re
import unicodedata

from .lang import to_ascii_digits

# ------------------------------------------------------------------ normalisation

def normalize(text: str) -> str:
    t = to_ascii_digits(unicodedata.normalize("NFKC", text or "")).lower()
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"(?<=\d),(?=\d{2,3}\b)", "", t)          # 50,000 / 1,50,000 -> 50000
    t = re.sub(r"\s+", " ", t)
    return t


def _digits_joined(t: str) -> str:
    return re.sub(r"(?<=\d)[\s-]+(?=\d)", "", t)          # 1800-180-1551 -> 18001801551


def fact_hit(answer: str, alternatives: list[str]) -> bool:
    a = normalize(answer)
    variants = (a, _digits_joined(a))
    for alt in alternatives:
        alt_n = normalize(alt)
        pat = re.escape(alt_n)
        if alt_n[:1].isdigit():
            pat = r"(?<![\d.])" + pat      # '2 g' must not match inside '12 g'
        if alt_n[-1:].isdigit():
            pat = pat + r"(?![\d])"        # '60' must not match inside '600'
        if any(re.search(pat, v) for v in variants):
            return True
    return False


def key_fact_scores(answer: str, key_facts: list[list[str]]) -> dict:
    if not key_facts:
        return {"fact_recall": None, "all_facts": None, "facts_hit": []}
    hits = [fact_hit(answer, alts) for alts in key_facts]
    return {"fact_recall": sum(hits) / len(hits), "all_facts": float(all(hits)), "facts_hit": hits}


# ------------------------------------------------------------------ tool / numbers

_NUM = re.compile(r"\d+(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(normalize(text))]


def within(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-9)


def tool_scores(pred: dict, expected: dict, rel_tol: float = 0.05) -> dict:
    """Agent: the tool must be called with the right crop and area, and return the right kg.
    Baselines (no tool): the urea quantity must appear in the free-text answer (+/-5%)."""
    tr = pred.get("tool_result")
    if tr:
        crop_ok = tr["crop_key"] == expected["crop_key"]
        area_ok = within(tr["area_ha"], expected["area_ha"], 0.01)
        qty_ok = all(within(tr[k], expected[k], 0.01) for k in ("urea_kg", "ssp_kg", "mop_kg"))
        return {"tool_called": 1.0, "tool_crop_ok": float(crop_ok), "tool_area_ok": float(area_ok),
                "quantity_correct": float(crop_ok and area_ok and qty_ok)}
    nums = extract_numbers(pred.get("answer_en", ""))
    ok = any(within(n, expected["urea_kg"], rel_tol) for n in nums)
    return {"tool_called": 0.0, "tool_crop_ok": None, "tool_area_ok": None, "quantity_correct": float(ok)}


def number_preservation(answer_en: str, answer_local: str) -> float | None:
    """Share of numbers in the English answer that survive back-translation."""
    src = set(extract_numbers(answer_en))
    if not src:
        return None
    tgt = set(extract_numbers(answer_local))
    return sum(n in tgt for n in src) / len(src)


# ------------------------------------------------------------------ retrieval

def retrieval_scores(retrieved_doc_ids: list[str], gold_docs: list[str], ks=(1, 3, 5)) -> dict:
    if not gold_docs:
        return {}
    out = {f"hit@{k}": float(any(d in gold_docs for d in retrieved_doc_ids[:k])) for k in ks}
    rr = 0.0
    for i, d in enumerate(retrieved_doc_ids):
        if d in gold_docs:
            rr = 1.0 / (i + 1)
            break
    out["mrr"] = rr
    return out


# ------------------------------------------------------------------ abstention

def binary_scores(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(1, len(gold))}


# ------------------------------------------------------------------ text similarity

def chrf_pp(hyps: list[str], refs: list[str]) -> float | None:
    if not hyps:
        return None
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).corpus_score(hyps, [refs]).score


def sentence_chrf_pp(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).sentence_score(hyp, [ref]).score


# ------------------------------------------------------------------ ASR

_PUNCT = re.compile(r"[।॥.,!?;:\"'()\[\]{}\-–—/]")


def asr_normalize(text: str) -> str:
    t = to_ascii_digits(unicodedata.normalize("NFC", text or "")).lower()
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    import jiwer
    r, h = asr_normalize(ref), asr_normalize(hyp)
    if not r:
        return (0.0, 0.0)
    return float(jiwer.wer(r, h)), float(jiwer.cer(r, h))
