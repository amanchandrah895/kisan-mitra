"""Agent router / guardrails.

Decides, for an English (translated) question, which action the agent takes:
  tool    -> fertilizer calculator (deterministic arithmetic)
  abstain -> safe refusal + referral (out of scope, or no relevant evidence in the corpus)
  rag     -> retrieve + generate with AgriParam
and whether to attach a pesticide-safety note to the answer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tools import parse_fertilizer_query

HELPLINE = ("Please call the Kisan Call Centre on 1800-180-1551 (toll-free, 6 am to 10 pm, all days) "
            "or contact your nearest Krishi Vigyan Kendra (KVK).")

ABSTAIN_MESSAGES = {
    "veterinary": "I can only advise on crops and farmer schemes. For a sick animal, please contact a "
                  "veterinary doctor or the nearest veterinary hospital immediately. " + HELPLINE,
    "market_price": "I do not have live or forecast market price information in my verified sources. " + HELPLINE,
    "not_agri": "I am Kisan Mitra, a farm advisory assistant, so I can only answer questions about crops, "
                "farming practices and farmer schemes.",
    "low_relevance": "I could not find verified information about this in my agricultural sources, so I do not "
                     "want to guess. " + HELPLINE,
}

# Policy rules (scope of a crop-advisory assistant), applied before retrieval.
_VET_ANIMAL = re.compile(r"\b(cows?|buffalo(es)?|goats?|cattle|calf|calves|ox|oxen|bullocks?|sheep|poultry|hens?|"
                         r"chickens?|livestock|dog|pig)\b", re.I)
_VET_ISSUE = re.compile(r"\b(fever|medicine|sick|ill|illness|disease|treatment|drug|injection|vaccin\w*|"
                        r"not eating|diarrh\w*|wound)\b", re.I)
_MARKET = re.compile(r"\b(price|prices|mandi rates?|market rates?|selling rate)\b", re.I)
_NOT_AGRI = re.compile(r"\b(poem|song|story|joke|essay|lyrics)\b", re.I)  # creative-writing requests

_PESTICIDE = re.compile(
    r"\b(spray\w*|insecticides?|pesticides?|fungicides?|herbicides?|weedicides?|chemicals?|imidacloprid|"
    r"dimethoate|thiamethoxam|mancozeb|carbendazim|chlorpyri\w*|quinalphos|azoxystrobin|cyazofamid|zineb|"
    r"copper oxychloride|thiodicarb|pyrethroids?|2,4-d|pendimethalin|metsulfuron|oxyfluorfen|"
    r"tricyclazole|pyroquilon|methyl demeton|clodinafop|sulfosulfuron|metribuzin|fipronil|carboxin)\b", re.I)

SAFETY_NOTE = ("Safety: use only pesticides that have an approved label claim for this crop, at the label dose. "
               "Wear gloves, a mask and full clothing while spraying, do not spray against the wind, and keep "
               "chemicals away from children, food and water sources.")


@dataclass
class RouteDecision:
    route: str                     # rag | tool | abstain
    reason: str = ""
    tool_args: dict | None = None
    notes: list[str] = field(default_factory=list)


def policy_route(q_en: str, enable_tool: bool = True, enable_abstain: bool = True) -> RouteDecision | None:
    """Pre-retrieval decision. Returns None when retrieval is needed to decide."""
    if enable_abstain:
        if _VET_ANIMAL.search(q_en) and _VET_ISSUE.search(q_en):
            return RouteDecision("abstain", "veterinary")
        if _MARKET.search(q_en):
            return RouteDecision("abstain", "market_price")
        if _NOT_AGRI.search(q_en):
            return RouteDecision("abstain", "not_agri")
    if enable_tool:
        args = parse_fertilizer_query(q_en)
        if args:
            return RouteDecision("tool", "fertilizer_quantity", tool_args=args)
    return None


def relevance_route(best_rerank_score: float | None, threshold: float, enable_abstain: bool = True) -> RouteDecision:
    if enable_abstain and best_rerank_score is not None and best_rerank_score < threshold:
        return RouteDecision("abstain", "low_relevance")
    return RouteDecision("rag", "relevant_evidence")


def needs_safety_note(*texts: str) -> bool:
    return any(_PESTICIDE.search(t or "") for t in texts)
