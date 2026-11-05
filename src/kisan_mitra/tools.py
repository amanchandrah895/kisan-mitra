"""Deterministic agent tool: fertilizer calculator.

Small LLMs are unreliable at arithmetic, so when a farmer asks "how much urea for N acres of
<crop>", the agent routes the question to this tool instead of the LLM. Crop N-P2O5-K2O
recommendations come from the curated corpus (each entry cites its corpus document)."""
from __future__ import annotations

import re

HA_PER_ACRE = 0.404686
UREA_N, SSP_P, MOP_K = 46.0, 16.0, 60.0          # % nutrient (see corpus: fertilizer_basics)
UREA_BAG_KG, SSP_BAG_KG, MOP_BAG_KG = 45.0, 50.0, 50.0

CROP_RECS = {
    # key: (N, P2O5, K2O kg/ha, label, corpus doc)
    "wheat_timely_irrigated": (120, 60, 30, "timely sown irrigated wheat", "wheat_central_india"),
    "wheat_late_sown": (100, 50, 25, "late sown wheat", "wheat_central_india"),
    "wheat_restricted_irrigation": (80, 40, 20, "wheat with restricted irrigation", "wheat_central_india"),
    "wheat_durum": (140, 70, 35, "durum wheat", "wheat_central_india"),
    "rice_medium_long": (150, 50, 50, "rice (medium/long duration, wet season blanket dose)", "rice_nutrient_management"),
    "rice_short_dry": (120, 40, 40, "short-duration rice in the dry season", "rice_nutrient_management"),
    "rice_hybrid": (175, 60, 60, "hybrid rice", "rice_nutrient_management"),
    "ragi_irrigated": (60, 30, 30, "irrigated ragi (finger millet)", "ragi_finger_millet"),
    "ragi_rainfed": (40, 20, 20, "rainfed ragi (finger millet)", "ragi_finger_millet"),
}

_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "half": 0.5, "a": 1, "an": 1}
_FERT = re.compile(r"\b(fertili[sz]er\w*|urea|dap|ssp|super ?phosphate|mop|muriate|potash|npk|nutrients?)\b", re.I)
_AREA = re.compile(r"\b(\d+(?:\.\d+)?|" + "|".join(_WORDNUM) + r")\s*-?\s*(acres?|hectares?|ha)\b", re.I)


def detect_crop(q: str) -> str | None:
    q = q.lower()
    if re.search(r"\b(rice|paddy|dhan)\b", q):
        if "hybrid" in q:
            return "rice_hybrid"
        if "short" in q:
            return "rice_short_dry"
        return "rice_medium_long"
    if re.search(r"\bwheat\b", q):
        if "durum" in q:
            return "wheat_durum"
        if re.search(r"\b(late|delayed|december|january)\b", q):
            return "wheat_late_sown"
        if re.search(r"\b(restricted|limited|one irrigation|rainfed)\b", q):
            return "wheat_restricted_irrigation"
        return "wheat_timely_irrigated"
    if re.search(r"\b(ragi|finger millet|nachni|mandua)\b", q):
        return "ragi_rainfed" if "rainfed" in q else "ragi_irrigated"
    return None


def parse_area_ha(q: str) -> float | None:
    m = _AREA.search(q)
    if not m:
        return None
    raw, unit = m.group(1).lower(), m.group(2).lower()
    val = float(_WORDNUM[raw]) if raw in _WORDNUM else float(raw)
    return round(val * HA_PER_ACRE, 4) if unit.startswith("acre") else val


def parse_fertilizer_query(q_en: str) -> dict | None:
    """Returns tool arguments if this is a 'how much fertilizer for X area of crop' question."""
    if not _FERT.search(q_en):
        return None
    area = parse_area_ha(q_en)
    crop = detect_crop(q_en)
    if area is None or crop is None:
        return None
    return {"crop_key": crop, "area_ha": area}


def fertilizer_calculator(crop_key: str, area_ha: float) -> dict:
    n, p, k, label, doc = CROP_RECS[crop_key]
    urea = n * 100 / UREA_N * area_ha
    ssp = p * 100 / SSP_P * area_ha
    mop = k * 100 / MOP_K * area_ha
    return {
        "crop_key": crop_key, "crop_label": label, "source_doc": doc, "npk_per_ha": [n, p, k],
        "area_ha": round(area_ha, 4), "area_acres": round(area_ha / HA_PER_ACRE, 2),
        "urea_kg": round(urea, 1), "ssp_kg": round(ssp, 1), "mop_kg": round(mop, 1),
        "urea_bags_45kg": round(urea / UREA_BAG_KG, 1), "ssp_bags_50kg": round(ssp / SSP_BAG_KG, 1),
        "mop_bags_50kg": round(mop / MOP_BAG_KG, 1),
    }


def render_tool_answer(r: dict) -> str:
    n, p, k = r["npk_per_ha"]
    return (
        f"For {r['crop_label']} the recommended dose is {n}:{p}:{k} kg N:P2O5:K2O per hectare. "
        f"For your {r['area_acres']} acres ({r['area_ha']} ha) you need about {r['urea_kg']:.0f} kg urea "
        f"({r['urea_bags_45kg']} bags of 45 kg), {r['ssp_kg']:.0f} kg single super phosphate "
        f"({r['ssp_bags_50kg']} bags of 50 kg) and {r['mop_kg']:.0f} kg muriate of potash "
        f"({r['mop_bags_50kg']} bags of 50 kg). "
        "Apply phosphorus and potash as recommended for the crop and split the nitrogen across growth stages. "
        "If you have a Soil Health Card, follow its soil-test based dose instead of this blanket dose."
    )
