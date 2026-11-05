"""Re-score saved predictions on CPU (no models needed).

Recomputes every metric from results/predictions/*.jsonl with the current scoring code. The
'abstained' flag is re-derived from the answer text, and NLI groundedness is carried over from the
previous per_case_scores.csv (it needs the GPU NLI model). Used after fixing the abstention
detector, which missed phrasings such as "I don't have the information".

    PYTHONPATH=src python scripts/rescore.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kisan_mitra import evaluate as ev  # noqa: E402
from kisan_mitra.pipeline import looks_like_abstention  # noqa: E402

R = ROOT / "results"
cases = ev.load_testset(ROOT / "data" / "eval" / "testset.jsonl")
old = {(r["tag"], r["id"]): r for r in csv.DictReader(open(R / "per_case_scores.csv", encoding="utf-8"))}
shutil.copy(R / "per_case_scores.csv", R / "per_case_scores_before_rescore.csv")
shutil.copy(R / "metrics_by_system.csv", R / "metrics_by_system_before_rescore.csv")

rows, changed = [], []
for pfile in sorted((R / "predictions").glob("*.jsonl")):
    preds = ev.dedupe(ev.read_jsonl(pfile))
    for p in preds:
        new_flag = p.get("route") == "abstain" or looks_like_abstention(p.get("answer_en", ""))
        if new_flag != bool(p.get("abstained")):
            changed.append((pfile.stem, p["id"], p.get("answer_en", "")[:80]))
        p["abstained"] = new_flag
    for r in ev.score_predictions(preds, cases, grounder=None):
        g = old.get((r["tag"], r["id"]), {}).get("groundedness", "")
        r["groundedness"] = float(g) if g not in ("", None) else None
        rows.append(r)

ev.write_csv(R / "per_case_scores.csv", rows)
summary = ev.aggregate(rows)
ev.write_csv(R / "metrics_by_system.csv", summary)
(R / "metrics_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(f"abstention flag changed for {len(changed)} predictions:")
for c in changed:
    print("  ", c)
