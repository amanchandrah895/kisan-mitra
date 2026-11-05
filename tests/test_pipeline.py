"""CPU-only tests (no model downloads): corpus, retrieval, router, tool, metrics and the full
agent loop with stand-in models.   Run:  pytest -q"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kisan_mitra.config import load_config  # noqa: E402
from kisan_mitra.corpus import chunk_documents, load_corpus  # noqa: E402
from kisan_mitra.evaluate import load_testset, score_predictions  # noqa: E402
from kisan_mitra.experiment import build_agent  # noqa: E402
from kisan_mitra.lang import detect_language, to_ascii_digits  # noqa: E402
from kisan_mitra.llm import ExtractiveLLM, clean_answer  # noqa: E402
from kisan_mitra.metrics import fact_hit, key_fact_scores, tool_scores, wer_cer  # noqa: E402
from kisan_mitra.router import policy_route  # noqa: E402
from kisan_mitra.tools import fertilizer_calculator, parse_fertilizer_query  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cases(cfg):
    return load_testset(cfg.path("testset"))


@pytest.fixture(scope="module")
def agent(cfg, cases):
    a = build_agent(cfg, cases, dry_run=True)
    a.llms["agriparam"] = ExtractiveLLM()
    return a


def test_corpus_and_chunks(cfg):
    docs = load_corpus(cfg.path("corpus_dir"))
    assert len(docs) >= 15
    assert all(d.source_url.startswith("http") for d in docs)
    chunks = chunk_documents(docs, 160, 30)
    assert len(chunks) > len(docs)
    assert all(len(c.text.split()) <= 200 for c in chunks)
    assert not any("Source: http" in c.text for c in chunks)


def test_testset_integrity(cfg, cases):
    assert 30 <= len(cases) <= 50
    doc_ids = {d.doc_id for d in load_corpus(cfg.path("corpus_dir"))}
    langs = {c["lang"] for c in cases}
    assert langs == {"en", "hi", "kn", "ta"}
    for c in cases:
        assert set(c["gold_docs"]) <= doc_ids, c["id"]
        if c["category"] in ("advisory", "scheme"):
            # every key fact must be supported by the gold documents
            text = " ".join(open(cfg.path("corpus_dir") / f"{d}.md", encoding="utf-8").read() for d in c["gold_docs"])
            for alts in c["key_facts"]:
                assert fact_hit(text, alts), (c["id"], alts)


def test_language_detection():
    assert detect_language("धान की नर्सरी") == "hi"
    assert detect_language("ರಾಗಿ ಬೆಳೆ") == "kn"
    assert detect_language("நெல் வயல்") == "ta"
    assert detect_language("How much urea?") == "en"
    assert to_ascii_digits("७२ घंटे") == "72 घंटे"


def test_fertilizer_tool(cases):
    for c in [c for c in cases if c["category"] == "tool"]:
        args = parse_fertilizer_query(c["question_en"])
        assert args is not None, c["id"]
        r = fertilizer_calculator(**args)
        exp = c["tool_expected"]
        assert r["crop_key"] == exp["crop_key"], c["id"]
        assert abs(r["urea_kg"] - exp["urea_kg"]) < 1.0, c["id"]
        assert tool_scores({"tool_result": r}, exp)["quantity_correct"] == 1.0
    # questions without an explicit area must NOT go to the tool
    assert parse_fertilizer_query("What NPK dose is recommended for timely sown irrigated wheat?") is None
    assert parse_fertilizer_query("What is the recommended fertilizer dose per hectare for ragi?") is None


def test_policy_router():
    assert policy_route("My cow has a fever, what medicine can I give?").reason == "veterinary"
    assert policy_route("What will be the price of onions next week?").reason == "market_price"
    assert policy_route("How much urea for 2 acres of wheat?").route == "tool"
    assert policy_route("When should I irrigate wheat?") is None


def test_metrics():
    assert fact_hit("Call 1800 180 1551 now", ["18001801551"])
    assert fact_hit("Apply 50,000 per acre", ["50000"])
    assert not fact_hit("Use 12 g per litre", ["2 g"])
    assert not fact_hit("about 600 kg", ["60"])
    assert key_fact_scores("120:60:30 kg", [["120"], ["60"], ["30"]])["all_facts"] == 1.0
    w, c = wer_cer("धान की नर्सरी में", "धान की नर्सरी")
    assert 0 < w <= 0.5 and 0 < c < 0.5
    assert clean_answer("Apply 20 kg. Apply 20 kg. Then <user> junk") == "Apply 20 kg."


def test_agent_end_to_end(agent, cases):
    by_id = {c["id"]: c for c in cases}
    r = agent.answer(by_id["en03"]["question"], system="kisan_mitra_full")
    assert r["route"] == "rag" and r["citations"]
    assert r["retrieved"][0]["doc_id"] == "cotton_pink_bollworm"
    assert fact_hit(r["answer_en"], ["8 moths"])
    r = agent.answer(by_id["tool02"]["question"], lang="hi", system="kisan_mitra_full",
                     question_en=by_id["tool02"]["question_en"])
    assert r["route"] == "tool" and abs(r["tool_result"]["urea_kg"] - 659.8) < 1
    r = agent.answer(by_id["ood04"]["question"], lang="ta", system="kisan_mitra_full",
                     question_en=by_id["ood04"]["question_en"])
    assert r["route"] == "abstain" and r["abstained"]
    # pesticide answers get a safety note
    r = agent.answer("Which insecticide spray controls whitefly in tomato leaf curl?", system="kisan_mitra_full")
    assert r["safety_note"]


def test_scoring_shapes(agent, cases):
    preds = []
    for c in cases[:6] + [c for c in cases if c["category"] in ("tool", "ood")]:
        p = agent.answer(c["question"], lang=c["lang"], system="kisan_mitra_full", question_en=c["question_en"])
        p.update({"id": c["id"], "tag": "kisan_mitra_full"})
        preds.append(json.loads(json.dumps(p, default=str)))
    rows = score_predictions(preds, cases)
    assert all("success" in r for r in rows)
    assert all(r["route_correct"] == 1.0 for r in rows if r["category"] == "tool")
