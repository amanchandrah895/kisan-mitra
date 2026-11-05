"""End-to-end experiment orchestration used by notebooks/01_run_all_experiments.ipynb and
`python -m kisan_mitra.experiment`. `dry_run=True` swaps every neural model for a CPU stand-in so
the whole pipeline (and the results format) can be tested without a GPU or model downloads."""
from __future__ import annotations

import argparse
import json
import time

from . import evaluate as ev
from .config import load_config
from .factory import (build_chunks, build_grounder, build_retriever, build_translator, load_asr, load_llm,
                      offload_aux_to_cpu)
from .hf_utils import env_report, free_memory
from .llm import ExtractiveLLM
from .pipeline import KisanMitra
from .translate import LookupTranslator


class _FakeASR:
    """Dry-run ASR: returns the reference with one word dropped (so WER is non-zero)."""

    def __init__(self, cases):
        self.ref = {c["question"]: c["question"] for c in cases}
        self.by_path: dict[str, str] = {}

    def transcribe(self, audio, lang, decoding="ctc"):
        text = self.by_path.get(str(audio), "")
        words = text.split()
        return " ".join(words[:-1]) if len(words) > 3 else text


def build_agent(cfg, cases, dry_run: bool = False) -> KisanMitra:
    docs, chunks = build_chunks(cfg)
    if dry_run:
        table = {c["question"]: c["question_en"] for c in cases}
        translator = LookupTranslator(table)
        retriever = build_retriever(cfg, chunks, neural=False)
        grounder = build_grounder(cfg, real=False)
        cfg["retrieval"]["mode"] = "bm25"
    else:
        translator = build_translator(cfg)
        retriever = build_retriever(cfg, chunks)
        grounder = build_grounder(cfg)
    return KisanMitra(cfg, translator, retriever, llms={"extractive": ExtractiveLLM()}, grounder=grounder)


def run_all(config_path=None, dry_run: bool = False, results_dir=None, skip_asr: bool = False,
            systems=None, max_cases: int | None = None) -> dict:
    cfg = load_config(config_path)
    if results_dir:
        cfg["paths"]["results_dir"] = str(results_dir)
    rdir = cfg.path("results_dir")
    rdir.mkdir(parents=True, exist_ok=True)
    cases = ev.load_testset(cfg.path("testset"))[:max_cases]
    info = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_run": dry_run, "env": env_report(),
            "config": dict(cfg), "n_cases": len(cases)}
    t_all = time.time()

    agent = build_agent(cfg, cases, dry_run)

    print("== 1. question translation")
    mt = ev.translate_questions(cases, agent.translator, rdir)

    print("== 2. retrieval ablation")
    ev.retrieval_ablation(agent.retriever, cases, mt, rdir)

    transcripts = {}
    if not skip_asr:
        print("== 3. ASR")
        from .tts import find_user_recordings, synthesize_questions
        if dry_run:
            asr = _FakeASR(cases)
            audio = {c["id"]: f"dry/{c['id']}.wav" for c in cases if c["lang"] != "en"}
            asr.by_path = {p: next(c["question"] for c in cases if c["id"] == cid) for cid, p in audio.items()}
        else:
            audio = synthesize_questions(cases, cfg.path("audio_dir"))
            asr = load_asr(cfg)
        decs = tuple(cfg["asr"]["decoding"])
        rows = ev.run_asr(asr, cases, audio, rdir, decs, source="gtts")
        user = find_user_recordings(cases, cfg.path("audio_dir").parent / "audio_user")
        if user:
            ev.run_asr(asr, cases, user, rdir, decs, source="user")
        transcripts = {r["id"]: r["hypothesis"] for r in rows if r["decoding"] == decs[0] and r["hypothesis"]}
        if not dry_run:
            del asr
            free_memory()

    systems = systems or ["extractive_rag", "agriparam_closedbook", "agriparam_rag", "kisan_mitra_full",
                          "kisan_mitra_full@voice", "param1_rag"]
    llm_needed = {"agriparam": [s for s in systems if s.split("@")[0] in
                                ("agriparam_closedbook", "agriparam_rag", "kisan_mitra_full")],
                  "param1": [s for s in systems if s.startswith("param1")]}

    print("== 4. systems")
    if "extractive_rag" in systems:
        ev.run_system(agent, "extractive_rag", cases, rdir, mt)
    for llm_name, sys_list in llm_needed.items():
        if not sys_list:
            continue
        if dry_run:
            agent.llms[llm_name] = ExtractiveLLM()
        else:
            print(f"   loading {llm_name} ...")
            agent.llms[llm_name] = load_llm(cfg, llm_name, before_fp32_reload=lambda: offload_aux_to_cpu(agent))
            info[f"{llm_name}_dtype"] = str(agent.llms[llm_name].dtype)
        for s in sys_list:
            base, _, variant = s.partition("@")
            if variant == "voice":
                if transcripts:
                    ev.run_system(agent, base, cases, rdir, questions_override=transcripts, tag=s.replace("@", "_"))
            else:
                ev.run_system(agent, base, cases, rdir, mt)
        if not dry_run:
            info[f"{llm_name}_use_cache"] = getattr(agent.llms[llm_name], "use_cache", None)
            agent.llms[llm_name].unload()
            agent.llms.pop(llm_name)
            free_memory()

    print("== 5. scoring")
    out = ev.score_all(cases, rdir, agent.grounder, cfg["grounding"]["entail_threshold"])
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["total_minutes"] = round((time.time() - t_all) / 60, 1)
    (rdir / "run_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    for row in out["summary"]:
        print({k: row[k] for k in ("system", "task_success", "qa_all_facts_acc", "tool_quantity_acc",
                                   "ood_abstain_recall", "qa_groundedness")})
    return out


def main():
    ap = argparse.ArgumentParser(description="Run all Kisan Mitra experiments")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="CPU stand-ins for every model")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None)
    a = ap.parse_args()
    run_all(a.config, a.dry_run, a.results_dir, a.skip_asr, max_cases=a.max_cases)


if __name__ == "__main__":
    main()
