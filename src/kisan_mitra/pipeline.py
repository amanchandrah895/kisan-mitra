"""The Kisan Mitra agent: language ID -> translation -> routing (tool / abstain / RAG) ->
retrieval -> grounded generation -> grounding check -> back-translation, with a full trace.

The same class runs every evaluated system (baselines are the agent with features switched off)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .lang import LANG_NAMES, detect_language
from .router import (ABSTAIN_MESSAGES, SAFETY_NOTE, RouteDecision, needs_safety_note, policy_route,
                     relevance_route)
from .tools import fertilizer_calculator, render_tool_answer

GROUNDING_CAVEAT = ("Note: part of this answer could not be verified against my sources; please confirm it "
                    "with your nearest KVK or the Kisan Call Centre (1800-180-1551).")

_ABSTAIN_RX = re.compile(r"(i (do not|don't|cannot|can't) (know|answer|find)|not (mentioned|provided|available|given)"
                         r" in the (context|information)|no information|unable to (answer|find)|"
                         r"i could not find|i can only (advise|answer))", re.I)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    llm: str                 # agriparam | param1 | extractive
    use_retrieval: bool = True
    use_router: bool = False
    grounding_caveat: bool = False
    description: str = ""


SYSTEMS = {
    "extractive_rag": SystemSpec("extractive_rag", "extractive", True, False, False,
                                 "Hybrid retrieval + extract best sentences (no LLM)"),
    "agriparam_closedbook": SystemSpec("agriparam_closedbook", "agriparam", False, False, False,
                                       "AgriParam alone, no retrieval"),
    "agriparam_rag": SystemSpec("agriparam_rag", "agriparam", True, False, False,
                                "AgriParam + hybrid retrieval (no router/tools/guardrails)"),
    "param1_rag": SystemSpec("param1_rag", "param1", True, False, False,
                             "General Param-1 Instruct + same retrieval"),
    "kisan_mitra_full": SystemSpec("kisan_mitra_full", "agriparam", True, True, True,
                                   "Full agent: router + fertilizer tool + abstention + RAG + grounding check"),
}


def looks_like_abstention(answer_en: str) -> bool:
    return bool(_ABSTAIN_RX.search(answer_en or ""))


class KisanMitra:
    def __init__(self, cfg, translator, retriever, llms: dict | None = None, grounder=None, asr=None):
        self.cfg = cfg
        self.translator = translator
        self.retriever = retriever
        self.llms = llms or {}
        self.grounder = grounder
        self.asr = asr

    # ------------------------------------------------------------------ helpers
    def _t(self, trace, step, t0, **detail):
        trace.append({"step": step, "ms": round((time.time() - t0) * 1000), **detail})

    def transcribe(self, audio_path: str, lang: str, decoding: str = "ctc") -> str:
        if self.asr is None:
            raise RuntimeError("ASR model not loaded")
        return self.asr.transcribe(audio_path, lang, decoding)

    # ------------------------------------------------------------------ main entry
    def answer(self, question: str, lang: str | None = None, system: str = "kisan_mitra_full",
               question_en: str | None = None, translate_answer: bool = True) -> dict:
        spec = SYSTEMS[system]
        rcfg, gcfg = self.cfg["router"], self.cfg["grounding"]
        trace: list[dict] = []
        t_start = time.time()

        # 1. language
        t0 = time.time()
        lang = lang or detect_language(question)
        self._t(trace, "detect_language", t0, lang=lang, name=LANG_NAMES.get(lang, lang))

        # 2. translate question to English (pivot language for retrieval + LLM)
        t0 = time.time()
        if question_en is None:
            question_en = question if lang == "en" else self.translator(question, lang, "en")
        self._t(trace, "translate_question", t0, question_en=question_en)

        # 3. policy routing (tool / out-of-scope) before retrieval
        decision: RouteDecision | None = None
        if spec.use_router:
            t0 = time.time()
            decision = policy_route(question_en, rcfg.get("enable_tool", True), rcfg.get("enable_abstain", True))
            self._t(trace, "policy_router", t0, decision=None if decision is None else decision.route,
                    reason=None if decision is None else decision.reason)

        # 4. retrieval (always computed for logging/evaluation; only fed to the LLM if spec.use_retrieval)
        t0 = time.time()
        rmode = self.cfg["retrieval"]["mode"]
        hits = self.retriever.search(question_en, top_k=self.cfg["retrieval"]["top_k"], mode=rmode)
        best = self.retriever.relevance(question_en, hits)
        self._t(trace, "retrieve", t0, mode=rmode, best_rerank=best,
                docs=[h.chunk.doc_id for h in hits])
        contexts = [h.chunk.as_context() for h in hits]

        # 5. relevance-based abstention
        if spec.use_router and decision is None:
            decision = relevance_route(best, rcfg.get("abstain_threshold", 0.0), rcfg.get("enable_abstain", True))
            self._t(trace, "relevance_router", time.time(), decision=decision.route, reason=decision.reason)
        if decision is None:
            decision = RouteDecision("rag", "baseline")

        # 6. act
        t0 = time.time()
        tool_result = None
        gen_stats = {}
        if decision.route == "tool":
            tool_result = fertilizer_calculator(**decision.tool_args)
            answer_en = render_tool_answer(tool_result)
            self._t(trace, "tool:fertilizer_calculator", t0, args=decision.tool_args,
                    urea_kg=tool_result["urea_kg"], ssp_kg=tool_result["ssp_kg"], mop_kg=tool_result["mop_kg"])
        elif decision.route == "abstain":
            answer_en = ABSTAIN_MESSAGES.get(decision.reason, ABSTAIN_MESSAGES["low_relevance"])
            self._t(trace, "abstain", t0, reason=decision.reason)
        else:
            llm = self.llms.get(spec.llm)
            if llm is None:
                raise RuntimeError(f"LLM '{spec.llm}' is not loaded for system '{system}'")
            answer_en = llm.generate(question_en, contexts if spec.use_retrieval else None)
            gen_stats = getattr(llm, "last_stats", {}) or {}
            self._t(trace, f"generate:{spec.llm}", t0, with_context=spec.use_retrieval, **gen_stats)
            if not answer_en.strip():
                answer_en = "I do not know."

        # 7. guardrails: grounding check + safety note (full system only)
        grounding = None
        if spec.grounding_caveat and decision.route == "rag" and self.grounder is not None:
            t0 = time.time()
            grounding = self.grounder.score(answer_en, contexts, gcfg.get("entail_threshold"))
            if grounding["ratio"] is not None and grounding["ratio"] < gcfg.get("caveat_below", 0.5):
                answer_en = f"{answer_en} {GROUNDING_CAVEAT}"
            self._t(trace, "grounding_check", t0, ratio=grounding["ratio"])
        safety = False
        if spec.use_router and rcfg.get("enable_safety_note", True) and decision.route == "rag" \
                and needs_safety_note(question_en, answer_en):
            answer_en = f"{answer_en} {SAFETY_NOTE}"
            safety = True

        # 8. back-translate
        t0 = time.time()
        answer_local = answer_en
        if translate_answer and lang != "en":
            answer_local = self.translator.translate_long(answer_en, "en", lang)
        self._t(trace, "translate_answer", t0)

        # 9. citations
        cites = []
        if decision.route == "tool":
            doc_ids = [tool_result["source_doc"], "fertilizer_basics"]
            by_id = {ch.doc_id: ch for ch in self.retriever.chunks}
            for d in doc_ids:
                if d in by_id:
                    cites.append({"doc_id": d, "title": by_id[d].title, "url": by_id[d].source_url})
        elif decision.route == "rag" and spec.use_retrieval:
            seen = set()
            for h in hits:
                if h.chunk.doc_id not in seen:
                    seen.add(h.chunk.doc_id)
                    cites.append({"doc_id": h.chunk.doc_id, "title": h.chunk.title, "url": h.chunk.source_url})

        abstained = decision.route == "abstain" or looks_like_abstention(answer_en)
        return {
            "system": system, "lang": lang, "question": question, "question_en": question_en,
            "route": decision.route, "route_reason": decision.reason, "abstained": abstained,
            "answer_en": answer_en, "answer": answer_local, "tool_result": tool_result,
            "safety_note": safety, "grounding_check": grounding, "best_rerank": best,
            "retrieved": [h.to_dict() for h in hits], "citations": cites, "gen_stats": gen_stats,
            "latency_s": round(time.time() - t_start, 2), "trace": trace,
        }
