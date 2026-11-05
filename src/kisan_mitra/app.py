"""Gradio front-end for Kisan Mitra.

    python -m kisan_mitra.app --share            # real models (GPU)
    python -m kisan_mitra.app --dry-run          # CPU stand-ins, for UI development
"""
from __future__ import annotations

import argparse
import html
import json
import tempfile

from .lang import LANG_NAMES

LANG_CHOICES = ["Auto-detect", "Hindi", "Kannada", "Tamil", "English"]
_NAME_TO_CODE = {v: k for k, v in LANG_NAMES.items()}

EXAMPLES = [
    ["धान की नर्सरी में बीज उपचार के लिए कार्बेन्डाजिम की कितनी मात्रा प्रयोग करें?", "Auto-detect"],
    ["5 एकड़ धान के लिए कितना यूरिया, एसएसपी और पोटाश चाहिए?", "Auto-detect"],
    ["ಹತ್ತಿಯಲ್ಲಿ ಗುಲಾಬಿ ಕಾಯಿಕೊರಕದ ನಿಗಾಕ್ಕೆ ಹೆಕ್ಟೇರಿಗೆ ಎಷ್ಟು ಫೆರೋಮೋನ್ ಬಲೆಗಳನ್ನು ಹಾಕಬೇಕು?", "Auto-detect"],
    ["தக்காளியில் முன்பருவ கருகல் நோய்க்கு காரணம் என்ன, அதன் அறிகுறிகள் என்ன?", "Auto-detect"],
    ["What is the economic threshold level for pink bollworm using pheromone traps?", "Auto-detect"],
    ["My cow has a fever, what medicine can I give?", "Auto-detect"],
]

ROUTE_BADGE = {"rag": "Answered from verified sources", "tool": "Calculated with the fertilizer tool",
               "abstain": "Declined safely (out of scope / no evidence)"}


def _sources_md(res: dict) -> str:
    if not res["citations"]:
        return "_No sources (the assistant did not use the knowledge base for this reply)._"
    lines = [f"{i + 1}. [{c['title']}]({c['url']})" for i, c in enumerate(res["citations"])]
    return "\n".join(lines)


def _trace_md(res: dict) -> str:
    rows = ["| step | time (ms) | details |", "|---|---:|---|"]
    for t in res["trace"]:
        det = {k: v for k, v in t.items() if k not in ("step", "ms")}
        rows.append(f"| {t['step']} | {t['ms']} | {html.escape(json.dumps(det, ensure_ascii=False)[:220])} |")
    g = res.get("grounding_check")
    if g and g.get("ratio") is not None:
        rows.append(f"\n**Grounding check:** {g['supported']}/{g['n_sentences']} answer sentences supported by the evidence.")
    if res.get("best_rerank") is not None:
        rows.append(f"\n**Best evidence relevance (cross-encoder):** {res['best_rerank']:.2f}")
    return "\n".join(rows)


def _evidence_md(res: dict) -> str:
    out = []
    for h in res["retrieved"]:
        score = h["rerank_score"] if h["rerank_score"] is not None else h["score"]
        out.append(f"**{h['title']} – {h['section']}** (score {score:.2f})\n\n> {h['text'][:600]}")
    return "\n\n".join(out) or "_none_"


def build_app(agent, speak_default: bool = False):
    import gradio as gr

    def respond(text, audio, lang_choice, speak):
        lang = _NAME_TO_CODE.get(lang_choice)
        transcript = ""
        if audio:
            asr_lang = lang if lang and lang != "en" else "hi"
            try:
                transcript = agent.transcribe(audio, asr_lang)
                text = transcript
                lang = asr_lang
            except Exception as e:  # noqa: BLE001
                return (f"Speech recognition failed: {e}", "", "", "", "", None, "")
        if not (text or "").strip():
            return ("Please type or speak a question.", "", "", "", "", None, "")
        res = agent.answer(text, lang=lang)
        badge = ROUTE_BADGE.get(res["route"], res["route"])
        header = f"**{badge}** · language: {LANG_NAMES.get(res['lang'], res['lang'])} · {res['latency_s']} s"
        audio_out = None
        if speak and res["lang"] in ("hi", "kn", "ta", "en"):
            try:
                from gtts import gTTS
                f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                gTTS(res["answer"], lang=res["lang"]).save(f.name)
                audio_out = f.name
            except Exception:
                audio_out = None
        return (res["answer"], res["answer_en"] if res["lang"] != "en" else "", header, _sources_md(res),
                _trace_md(res), audio_out, transcript or res["question_en"])

    with gr.Blocks(title="Kisan Mitra – farmer advisory assistant") as demo:
        gr.Markdown("# 🌾 Kisan Mitra\nAsk farming questions in **Hindi, Kannada, Tamil or English** — "
                    "by typing or speaking. Answers come from curated ICAR / TNAU / Government of India sources, "
                    "with citations. Powered by BharatGen **AgriParam** and AI4Bharat **IndicTrans2** + "
                    "**IndicConformer**.")
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="Your question", lines=3, placeholder="e.g. गेहूं की बुवाई कब करें?")
                mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="…or speak / upload audio")
                with gr.Row():
                    lang = gr.Dropdown(LANG_CHOICES, value="Auto-detect", label="Language")
                    speak = gr.Checkbox(value=speak_default, label="Read the answer aloud")
                btn = gr.Button("Ask", variant="primary")
                gr.Examples(EXAMPLES, inputs=[q, lang])
            with gr.Column(scale=4):
                status = gr.Markdown()
                ans = gr.Textbox(label="Answer", lines=6)
                ans_en = gr.Textbox(label="English (internal answer)", lines=4)
                voice = gr.Audio(label="Spoken answer", type="filepath")
                understood = gr.Textbox(label="What the assistant understood (ASR / translation)", lines=2)
                with gr.Accordion("Sources", open=True):
                    src = gr.Markdown()
                with gr.Accordion("Agent trace (how this answer was produced)", open=False):
                    trace = gr.Markdown()
        gr.Markdown("_Advisory only. For serious crop or animal problems call the Kisan Call Centre "
                    "1800-180-1551 (6 am–10 pm) or visit your nearest KVK._")
        btn.click(respond, [q, mic, lang, speak], [ans, ans_en, status, src, trace, voice, understood])
        q.submit(respond, [q, mic, lang, speak], [ans, ans_en, status, src, trace, voice, understood])
    return demo


def build_agent_for_app(config_path=None, dry_run: bool = False, with_asr: bool = True):
    from .config import load_config
    from .evaluate import load_testset
    from .experiment import build_agent
    from .factory import load_asr, load_llm, offload_aux_to_cpu
    from .llm import ExtractiveLLM
    cfg = load_config(config_path)
    cases = load_testset(cfg.path("testset"))
    agent = build_agent(cfg, cases, dry_run)
    if dry_run:
        agent.llms["agriparam"] = ExtractiveLLM()
    else:
        agent.llms["agriparam"] = load_llm(cfg, "agriparam", before_fp32_reload=lambda: offload_aux_to_cpu(agent))
        if with_asr:
            try:
                agent.asr = load_asr(cfg)
            except Exception as e:  # noqa: BLE001
                print("ASR not available:", e)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args()
    agent = build_agent_for_app(a.config, a.dry_run, not a.no_asr)
    build_app(agent).launch(share=a.share)


if __name__ == "__main__":
    main()
