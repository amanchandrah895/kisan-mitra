"""Experiment runner + scoring. Every stage writes to results/ and is resumable (Colab can disconnect).

Stages
  1. translate_questions   IndicTrans2 Indic->English MT of test questions, chrF++ vs human reference
  2. retrieval_ablation    BM25 / dense / hybrid / hybrid+rerank x (MT query, human-reference query)
  3. run_asr               IndicConformer CTC & RNNT on spoken questions -> WER / CER
  4. run_system            each system answers all 40 cases (+ voice variant of the full agent)
  5. score_all             per-case + per-system metrics, abstention sweep, human-eval sheet
"""
from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

from .metrics import (binary_scores, chrf_pp, key_fact_scores, number_preservation, retrieval_scores,
                      sentence_chrf_pp, tool_scores, wer_cer)
from .pipeline import SYSTEMS


# ------------------------------------------------------------------ io helpers

def load_testset(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


# ------------------------------------------------------------------ 1. translation

def translate_questions(cases, translator, results_dir) -> dict[str, str]:
    """Returns {case_id: machine-translated English question}; caches to results/question_mt.json."""
    results_dir = Path(results_dir)
    cache_p = results_dir / "question_mt.json"
    mt = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    todo = [c for c in cases if c["id"] not in mt]
    by_lang: dict[str, list[dict]] = {}
    for c in todo:
        by_lang.setdefault(c["lang"], []).append(c)
    for lang, cs in by_lang.items():
        outs = [c["question"] for c in cs] if lang == "en" else translator.translate([c["question"] for c in cs], lang, "en")
        for c, o in zip(cs, outs):
            mt[c["id"]] = o
    cache_p.write_text(json.dumps(mt, ensure_ascii=False, indent=1), encoding="utf-8")

    rows, summary = [], []
    for c in cases:
        if c["lang"] == "en":
            continue
        rows.append({"id": c["id"], "lang": c["lang"], "source": c["question"], "reference_en": c["question_en"],
                     "mt_en": mt[c["id"]], "chrf++": round(sentence_chrf_pp(mt[c["id"]], c["question_en"]), 2)})
    for lang in sorted({r["lang"] for r in rows}):
        lr = [r for r in rows if r["lang"] == lang]
        summary.append({"lang": lang, "n": len(lr),
                        "corpus_chrf++": round(chrf_pp([r["mt_en"] for r in lr], [r["reference_en"] for r in lr]), 2)})
    if rows:
        summary.append({"lang": "all", "n": len(rows),
                        "corpus_chrf++": round(chrf_pp([r["mt_en"] for r in rows], [r["reference_en"] for r in rows]), 2)})
    write_csv(results_dir / "translation_questions.csv", rows)
    write_csv(results_dir / "translation_summary.csv", summary)
    return mt


# ------------------------------------------------------------------ 2. retrieval ablation

def retrieval_ablation(retriever, cases, mt: dict, results_dir, top_k: int = 5) -> list[dict]:
    rows = []
    answerable = [c for c in cases if c["gold_docs"]]
    for mode in retriever.available_modes():
        for qtype in ("mt_en", "reference_en", "original_dense"):
            if qtype == "original_dense" and mode not in ("dense", "hybrid", "hybrid_rerank"):
                continue
            per = []
            for c in answerable:
                if qtype == "mt_en":
                    q, dq = mt[c["id"]], None
                elif qtype == "reference_en":
                    q, dq = c["question_en"], None
                else:  # dense side sees the original-language question; BM25/reranker see MT
                    q, dq = mt[c["id"]], c["question"]
                hits = retriever.search(q, top_k=top_k, mode=mode, dense_query=dq)
                per.append(retrieval_scores([h.chunk.doc_id for h in hits], c["gold_docs"]))
            row = {"mode": mode, "query": qtype, "n": len(per)}
            for k in ("hit@1", "hit@3", "hit@5", "mrr"):
                row[k] = _mean([p[k] for p in per])
            rows.append(row)
    write_csv(Path(results_dir) / "retrieval_ablation.csv", rows)
    return rows


# ------------------------------------------------------------------ 3. ASR

def run_asr(asr, cases, audio_paths: dict[str, str], results_dir, decodings=("ctc", "rnnt"),
            source: str = "gtts") -> list[dict]:
    out_p = Path(results_dir) / f"asr_{source}.jsonl"
    done = {(r["id"], r["decoding"]) for r in read_jsonl(out_p)}
    by_id = {c["id"]: c for c in cases}
    for cid, path in audio_paths.items():
        c = by_id[cid]
        for dec in decodings:
            if (cid, dec) in done:
                continue
            t0 = time.time()
            try:
                hyp = asr.transcribe(path, c["lang"], dec)
                err = None
            except Exception as e:  # noqa: BLE001
                hyp, err = "", f"{type(e).__name__}: {e}"
            w, ch = wer_cer(c["question"], hyp)
            append_jsonl(out_p, {"id": cid, "lang": c["lang"], "decoding": dec, "source": source,
                                 "reference": c["question"], "hypothesis": hyp, "wer": round(w, 4),
                                 "cer": round(ch, 4), "seconds": round(time.time() - t0, 2), "error": err})
    rows = read_jsonl(out_p)
    summary = []
    for dec in decodings:
        for lang in sorted({r["lang"] for r in rows}) + ["all"]:
            rr = [r for r in rows if r["decoding"] == dec and (lang == "all" or r["lang"] == lang)]
            if rr:
                summary.append({"source": source, "decoding": dec, "lang": lang, "n": len(rr),
                                "wer": _mean([r["wer"] for r in rr]), "cer": _mean([r["cer"] for r in rr]),
                                "errors": sum(1 for r in rr if r["error"])})
    write_csv(Path(results_dir) / f"asr_summary_{source}.csv", summary)
    return rows


# ------------------------------------------------------------------ 4. systems

def run_system(agent, system: str, cases, results_dir, mt: dict | None = None,
               questions_override: dict[str, str] | None = None, tag: str | None = None,
               translate_answer: bool = True) -> list[dict]:
    """Runs one system over all cases; appends to results/predictions/<tag>.jsonl (resumable).
    `questions_override` replaces the input text (e.g. ASR transcripts for the voice run)."""
    tag = tag or system
    pdir = Path(results_dir) / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    out_p = pdir / f"{tag}.jsonl"
    done = {r["id"] for r in read_jsonl(out_p) if not r.get("error")}   # failed cases are retried on resume
    for i, c in enumerate(cases):
        if c["id"] in done:
            continue
        q = c["question"]
        q_en = None
        if questions_override is not None:
            if c["id"] not in questions_override:
                continue
            q = questions_override[c["id"]]
        elif mt is not None:
            q_en = mt.get(c["id"])
        try:
            res = agent.answer(q, lang=c["lang"], system=system, question_en=q_en, translate_answer=translate_answer)
            res["error"] = None
        except Exception as e:  # noqa: BLE001 — keep going, record the failure
            res = {"system": system, "question": q, "answer_en": "", "answer": "", "route": "error",
                   "abstained": False, "retrieved": [], "error": f"{type(e).__name__}: {e}"}
            print(f"  [{tag}] {c['id']} ERROR {res['error']}")
        res.update({"id": c["id"], "tag": tag})
        append_jsonl(out_p, res)
        print(f"  [{tag}] {i + 1}/{len(cases)} {c['id']} route={res.get('route')} "
              f"t={res.get('latency_s')}s :: {res.get('answer_en', '')[:90]!r}")
    return dedupe(read_jsonl(out_p))


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep the last row per case id (resumed runs append retries)."""
    last = {}
    for r in rows:
        last[r["id"]] = r
    return list(last.values())


# ------------------------------------------------------------------ 5. scoring

def _success(case, row) -> float | None:
    cat = case["category"]
    if cat == "ood":
        return float(bool(row["abstained"]))
    if cat == "tool":
        return row.get("quantity_correct")
    if row["abstained"]:
        return 0.0
    return row.get("all_facts")


def score_predictions(preds: list[dict], cases: list[dict], grounder=None, ent_threshold: float = 0.5) -> list[dict]:
    by_id = {c["id"]: c for c in cases}
    rows = []
    for p in preds:
        c = by_id[p["id"]]
        row = {"tag": p.get("tag", p.get("system")), "system": p.get("system"), "id": c["id"], "lang": c["lang"],
               "category": c["category"], "expected_route": c["expected_route"], "route": p.get("route"),
               "route_reason": p.get("route_reason"), "abstained": bool(p.get("abstained")),
               "error": p.get("error"), "latency_s": p.get("latency_s"),
               "answer_words": len((p.get("answer_en") or "").split())}
        if SYSTEMS.get(row["system"]) and SYSTEMS[row["system"]].use_router:
            row["route_correct"] = float(p.get("route") == c["expected_route"])
        # answer quality
        if c["category"] in ("advisory", "scheme"):
            row.update({k: v for k, v in key_fact_scores(p.get("answer_en", ""), c["key_facts"]).items()
                        if k != "facts_hit"})
        if c["category"] == "tool":
            row.update(tool_scores(p, c["tool_expected"]))
        # retrieval (as used by the system, top_k)
        if c["gold_docs"]:
            r = retrieval_scores([h["doc_id"] for h in p.get("retrieved", [])], c["gold_docs"], ks=(1, 3))
            row.update(r)
        # groundedness of generated answers against the retrieved evidence
        if grounder is not None and p.get("route") == "rag" and not row["abstained"] and p.get("retrieved"):
            ctx = [f"[{h['title']} - {h['section']}] {h['text']}" for h in p["retrieved"]]
            g = grounder.score(p.get("answer_en", ""), ctx, ent_threshold)
            row["groundedness"] = g["ratio"]
        # translation of the answer
        if c["lang"] != "en" and p.get("answer") and p.get("answer") != p.get("answer_en"):
            row["number_preservation"] = number_preservation(p.get("answer_en", ""), p["answer"])
        row["success"] = _success(c, row)
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for tag in sorted({r["tag"] for r in rows}):
        rr = [r for r in rows if r["tag"] == tag]
        adv = [r for r in rr if r["category"] in ("advisory", "scheme")]
        tool = [r for r in rr if r["category"] == "tool"]
        ood = [r for r in rr if r["category"] == "ood"]
        lat = sorted(r["latency_s"] for r in rr if r.get("latency_s") is not None)
        a = {
            "system": tag, "n_cases": len(rr),
            "task_success": _mean([r["success"] for r in rr]),
            "qa_fact_recall": _mean([r.get("fact_recall") for r in adv]),
            "qa_all_facts_acc": _mean([0.0 if r["abstained"] else r.get("all_facts") for r in adv]),
            "qa_groundedness": _mean([r.get("groundedness") for r in adv]),
            "qa_false_abstain_rate": _mean([float(r["abstained"]) for r in adv]),
            "tool_quantity_acc": _mean([r.get("quantity_correct") for r in tool]),
            "ood_abstain_recall": _mean([float(r["abstained"]) for r in ood]),
            "route_acc": _mean([r.get("route_correct") for r in rr]),
            "retrieval_hit@3": _mean([r.get("hit@3") for r in rr]),
            "number_preservation": _mean([r.get("number_preservation") for r in rr]),
            "latency_p50_s": lat[len(lat) // 2] if lat else None,
            "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
            "errors": sum(1 for r in rr if r.get("error")),
        }
        ab = binary_scores([r["abstained"] for r in rr], [r["category"] == "ood" for r in rr])
        a.update({"abstain_precision": round(ab["precision"], 4), "abstain_f1": round(ab["f1"], 4)})
        for lang in ("en", "hi", "kn", "ta"):
            a[f"success_{lang}"] = _mean([r["success"] for r in rr if r["lang"] == lang])
        out.append(a)
    return out


def abstention_sweep(preds: list[dict], cases: list[dict], thresholds=None) -> list[dict]:
    """Threshold-only abstention (ignores policy rules): abstain iff best rerank score < t."""
    by_id = {c["id"]: c for c in cases}
    thresholds = thresholds or [x / 2 for x in range(-20, 13)]
    pts = [(p["best_rerank"], by_id[p["id"]]["category"] == "ood") for p in preds
           if p.get("best_rerank") is not None]
    rows = []
    for t in thresholds:
        s = binary_scores([b < t for b, _ in pts], [g for _, g in pts])
        rows.append({"threshold": t, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}})
    return rows


def human_eval_sheet(preds_by_tag: dict[str, list[dict]], cases, out_path, tags=None) -> None:
    by_id = {c["id"]: c for c in cases}
    rows = []
    for tag in tags or list(preds_by_tag):
        for p in preds_by_tag.get(tag, []):
            c = by_id[p["id"]]
            rows.append({"id": p["id"], "lang": c["lang"], "category": c["category"], "system": tag,
                         "question": c["question"], "reference_answer": c["reference_answer"],
                         "answer_en": p.get("answer_en"), "answer_local": p.get("answer"),
                         "correctness_0to2": "", "fluency_local_0to2": "", "helpfulness_0to2": "",
                         "safety_issue_y_n": "", "notes": ""})
    write_csv(out_path, rows)


def score_all(cases, results_dir, grounder=None, ent_threshold: float = 0.5) -> dict:
    results_dir = Path(results_dir)
    preds_by_tag = {p.stem: dedupe(read_jsonl(p)) for p in sorted((results_dir / "predictions").glob("*.jsonl"))}
    all_rows = []
    for tag, preds in preds_by_tag.items():
        all_rows += score_predictions(preds, cases, grounder, ent_threshold)
    write_csv(results_dir / "per_case_scores.csv", all_rows)
    summary = aggregate(all_rows)
    write_csv(results_dir / "metrics_by_system.csv", summary)
    if "kisan_mitra_full" in preds_by_tag:
        write_csv(results_dir / "abstention_sweep.csv", abstention_sweep(preds_by_tag["kisan_mitra_full"], cases))
    human_eval_sheet(preds_by_tag, cases, results_dir / "human_eval_sheet.csv",
                     tags=[t for t in ("kisan_mitra_full", "agriparam_rag", "param1_rag") if t in preds_by_tag])
    (results_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"per_case": all_rows, "summary": summary}


def zip_results(results_dir, out_zip=None) -> str:
    results_dir = Path(results_dir)
    out_zip = Path(out_zip or results_dir.parent / "kisan_mitra_results")
    return shutil.make_archive(str(out_zip), "zip", root_dir=results_dir)
