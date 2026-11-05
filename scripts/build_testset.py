"""Builds data/eval/testset.jsonl — the 40-case evaluation set for Kisan Mitra.

Each case:
  id, lang, question (in `lang`), question_en (human reference English),
  category: advisory | scheme | tool | ood
  expected_route: rag | tool | abstain
  gold_docs: corpus doc ids that contain the answer
  key_facts: list of facts; each fact is a list of acceptable surface forms
             (matched case-insensitively on the English answer after normalisation)
  reference_answer: short English gold answer written from the corpus
  tool_expected: for tool cases, the fertilizer quantities (kg) the tool must return

Run:  python scripts/build_testset.py
"""
import json
from pathlib import Path

HA_PER_ACRE = 0.404686


def fert(n, p, k, area_ha):
    return {
        "urea_kg": round(n * 100 / 46 * area_ha, 1),
        "ssp_kg": round(p * 100 / 16 * area_ha, 1),
        "mop_kg": round(k * 100 / 60 * area_ha, 1),
    }


CASES = [
    # ---------------- English (8) ----------------
    dict(id="en01", lang="en", category="advisory",
         question="What seed rate should I use for hybrid rice to plant one hectare?",
         question_en="What seed rate should I use for hybrid rice to plant one hectare?",
         gold_docs=["rice_nursery"], key_facts=[["20 kg"]],
         reference_answer="For hybrid rice use 20 kg of seed per hectare."),
    dict(id="en02", lang="en", category="advisory",
         question="In SRI rice, what spacing and seedling age are recommended for transplanting?",
         question_en="In SRI rice, what spacing and seedling age are recommended for transplanting?",
         gold_docs=["rice_sri"], key_facts=[["22.5"], ["15 day", "15-day", "15 days"]],
         reference_answer="Transplant 15-day-old seedlings at 22.5 x 22.5 cm square spacing."),
    dict(id="en03", lang="en", category="advisory",
         question="What is the economic threshold level for pink bollworm using pheromone traps?",
         question_en="What is the economic threshold level for pink bollworm using pheromone traps?",
         gold_docs=["cotton_pink_bollworm"],
         key_facts=[["8 moths", "eight moths"], ["3 consecutive", "three consecutive"]],
         reference_answer="8 moths per trap for 3 consecutive days."),
    dict(id="en04", lang="en", category="advisory",
         question="How long does vermicompost take to be ready and which earthworm is preferred?",
         question_en="How long does vermicompost take to be ready and which earthworm is preferred?",
         gold_docs=["vermicompost"], key_facts=[["45"], ["eisenia", "red earthworm"]],
         reference_answer="It is ready in 45 to 50 days; the red earthworm Eisenia foetida is preferred."),
    dict(id="en05", lang="en", category="scheme",
         question="What premium does a farmer pay for kharif crops under PMFBY?",
         question_en="What premium does a farmer pay for kharif crops under PMFBY?",
         gold_docs=["scheme_pmfby"], key_facts=[["2%", "2 %", "2 percent", "two percent"]],
         reference_answer="A maximum of 2% of the sum insured for kharif crops."),
    dict(id="en06", lang="en", category="scheme",
         question="What are the working hours of the Kisan Call Centre?",
         question_en="What are the working hours of the Kisan Call Centre?",
         gold_docs=["kisan_call_centre"],
         key_facts=[["6 am", "6:00 am", "6.00 am", "6 a.m", "6:00 a.m"],
                    ["10 pm", "10:00 pm", "10.00 pm", "10 p.m", "10:00 p.m"]],
         reference_answer="6:00 am to 10:00 pm on all seven days."),
    dict(id="en07", lang="en", category="advisory",
         question="What weather conditions favour late blight in tomato?",
         question_en="What weather conditions favour late blight in tomato?",
         gold_docs=["tomato_diseases"],
         key_facts=[["cool"], ["wet", "rain", "fog", "humid", "moist"]],
         reference_answer="Cool nights, warm days and long wet periods from rain or fog; sporulation is best at 18-22 C."),
    dict(id="en08", lang="en", category="advisory",
         question="What is the seed rate for timely sown bread wheat?",
         question_en="What is the seed rate for timely sown bread wheat?",
         gold_docs=["wheat_central_india"], key_facts=[["100 kg"]],
         reference_answer="100 kg per hectare (125 kg for durum)."),

    # ---------------- Hindi (10) ----------------
    dict(id="hi01", lang="hi", category="advisory",
         question="धान की नर्सरी में बीज उपचार के लिए कार्बेन्डाजिम की कितनी मात्रा प्रयोग करें?",
         question_en="How much carbendazim should be used for seed treatment in a rice nursery?",
         gold_docs=["rice_nursery"], key_facts=[["2 g", "2g", "2 gram", "2 gm", "two grams"]],
         reference_answer="2 g per litre of water for each kg of seed, soaked for 10 hours."),
    dict(id="hi02", lang="hi", category="scheme",
         question="पीएम किसान योजना में किसान परिवार को साल में कितने रुपये मिलते हैं?",
         question_en="How much money does a farmer family get per year under PM-KISAN?",
         gold_docs=["scheme_pm_kisan"], key_facts=[["6000", "six thousand"], ["2000", "two thousand"]],
         reference_answer="Rs 6,000 per year in three instalments of Rs 2,000."),
    dict(id="hi03", lang="hi", category="advisory",
         question="समय पर बोए गए सिंचित गेहूं के लिए नाइट्रोजन, फास्फोरस और पोटाश की कितनी मात्रा चाहिए?",
         question_en="What NPK dose is recommended for timely sown irrigated wheat?",
         gold_docs=["wheat_central_india"], key_facts=[["120"], ["60"], ["30"]],
         reference_answer="120:60:30 kg N:P:K per hectare."),
    dict(id="hi04", lang="hi", category="advisory",
         question="मक्का में फॉल आर्मीवर्म के नियंत्रण के लिए कौन से परजीवी कीट छोड़ने चाहिए?",
         question_en="Which parasitoids should be released to control fall armyworm in maize?",
         gold_docs=["maize_fall_armyworm"], key_facts=[["trichogramma"], ["50000", "50 000", "fifty thousand"]],
         reference_answer="Trichogramma pretiosum and Telenomus remus at 50,000 per acre weekly."),
    dict(id="hi05", lang="hi", category="advisory",
         question="कपास में गुलाबी सुंडी के लिए नीम का छिड़काव कब करना चाहिए?",
         question_en="When should neem be sprayed against pink bollworm in cotton?",
         gold_docs=["cotton_pink_bollworm"],
         key_facts=[["50"], ["neem seed kernel", "nske", "5%", "5 %"]],
         reference_answer="Spray NSKE 5% plus neem oil at 50-60 days after sowing."),
    dict(id="hi06", lang="hi", category="advisory",
         question="टमाटर में पत्ती मोड़क रोग किस कीट से फैलता है?",
         question_en="Which insect spreads tomato leaf curl disease?",
         gold_docs=["tomato_diseases"], key_facts=[["whitefly", "white fly", "bemisia"]],
         reference_answer="The whitefly Bemisia tabaci."),
    dict(id="hi07", lang="hi", category="scheme",
         question="मृदा स्वास्थ्य कार्ड में मिट्टी के कितने मापदंड जांचे जाते हैं?",
         question_en="How many soil parameters are tested for the Soil Health Card?",
         gold_docs=["scheme_soil_health_card"], key_facts=[["12", "twelve"]],
         reference_answer="12 parameters: N, P, K, S, Zn, Fe, Cu, Mn, B, pH, EC and OC."),
    dict(id="hi08", lang="hi", category="scheme",
         question="फसल बीमा योजना में फसल नुकसान की सूचना कितने घंटे के अंदर देनी होती है?",
         question_en="Within how many hours must crop loss be reported under the crop insurance scheme?",
         gold_docs=["scheme_pmfby"], key_facts=[["72", "seventy-two", "seventy two"]],
         reference_answer="Within 72 hours."),
    dict(id="hi09", lang="hi", category="scheme",
         question="किसान क्रेडिट कार्ड का कर्ज समय पर चुकाने पर ब्याज दर कितनी लगती है?",
         question_en="What is the effective interest rate on a Kisan Credit Card loan if it is repaid on time?",
         gold_docs=["scheme_kcc_credit"], key_facts=[["4%", "4 %", "4 percent", "four percent"]],
         reference_answer="7% minus 3% prompt repayment incentive = 4% per year."),
    dict(id="hi10", lang="hi", category="scheme",
         question="ड्रिप सिंचाई लगाने पर छोटे और सीमांत किसानों को कितनी सब्सिडी मिलती है?",
         question_en="How much subsidy do small and marginal farmers get for installing drip irrigation?",
         gold_docs=["scheme_pmksy_pdmc"], key_facts=[["55"]],
         reference_answer="55% of the indicative unit cost (45% for other farmers)."),

    # ---------------- Kannada (6) ----------------
    dict(id="kn01", lang="kn", category="advisory",
         question="ನೀರಾವರಿ ರಾಗಿ ಬೆಳೆಗೆ ಶಿಫಾರಸು ಮಾಡಿದ ರಸಗೊಬ್ಬರ ಪ್ರಮಾಣ ಎಷ್ಟು?",
         question_en="What is the recommended fertilizer dose for irrigated ragi?",
         gold_docs=["ragi_finger_millet"], key_facts=[["60"], ["30"]],
         reference_answer="60 kg N, 30 kg P2O5 and 30 kg K2O per hectare."),
    dict(id="kn02", lang="kn", category="advisory",
         question="ರಾಗಿ ಸಸಿಗಳನ್ನು ಎಷ್ಟು ದಿನಗಳ ವಯಸ್ಸಿನಲ್ಲಿ ನಾಟಿ ಮಾಡಬೇಕು?",
         question_en="At what age should ragi seedlings be transplanted?",
         gold_docs=["ragi_finger_millet"], key_facts=[["18"]],
         reference_answer="When they are 18 to 20 days old."),
    dict(id="kn03", lang="kn", category="advisory",
         question="ಹತ್ತಿಯಲ್ಲಿ ಗುಲಾಬಿ ಕಾಯಿಕೊರಕದ ನಿಗಾಕ್ಕೆ ಹೆಕ್ಟೇರಿಗೆ ಎಷ್ಟು ಫೆರೋಮೋನ್ ಬಲೆಗಳನ್ನು ಹಾಕಬೇಕು?",
         question_en="How many pheromone traps per hectare should be installed to monitor pink bollworm in cotton?",
         gold_docs=["cotton_pink_bollworm"],
         key_facts=[["5 traps", "five traps", "5 pheromone", "5 per hectare", "5 per ha", "5/ha", "5 traps per"]],
         reference_answer="5 pheromone traps per hectare, from 45 days after sowing."),
    dict(id="kn04", lang="kn", category="advisory",
         question="ಟೊಮೇಟೊ ಎಲೆ ಮುರುಟು ರೋಗ ನಿಯಂತ್ರಣಕ್ಕೆ ಎಕರೆಗೆ ಎಷ್ಟು ಹಳದಿ ಅಂಟು ಬಲೆಗಳು ಬೇಕು?",
         question_en="How many yellow sticky traps per acre are needed to manage tomato leaf curl?",
         gold_docs=["tomato_diseases"],
         key_facts=[["5 per acre", "5 yellow", "five yellow", "5 sticky", "5 traps", "5/acre", "five traps"]],
         reference_answer="5 yellow sticky traps per acre."),
    dict(id="kn05", lang="kn", category="scheme",
         question="ಪಿಎಂ ಕಿಸಾನ್ ಯೋಜನೆಗೆ ಯಾರು ಅರ್ಹರಲ್ಲ?",
         question_en="Who is not eligible for PM-KISAN?",
         gold_docs=["scheme_pm_kisan"],
         key_facts=[["income tax"], ["pension"], ["institutional"],
                    ["doctor", "engineer", "lawyer", "professional"]],
         reference_answer="Institutional landholders, constitutional post holders, government employees, pensioners with Rs 10,000+ pension, income tax payers and practising professionals."),
    dict(id="kn06", lang="kn", category="advisory",
         question="ಎರೆಹುಳು ಗೊಬ್ಬರವನ್ನು ಹೊಲದ ಬೆಳೆಗಳಿಗೆ ಹೆಕ್ಟೇರಿಗೆ ಎಷ್ಟು ಹಾಕಬೇಕು?",
         question_en="How much vermicompost should be applied per hectare to field crops?",
         gold_docs=["vermicompost"], key_facts=[["5-6", "5 to 6", "5 - 6", "five to six"], ["tonne", "ton", "t/ha"]],
         reference_answer="5 to 6 tonnes per hectare."),

    # ---------------- Tamil (6) ----------------
    dict(id="ta01", lang="ta", category="advisory",
         question="ஒரு ஹெக்டேர் நடவு நெல்லுக்கு எவ்வளவு நாற்றங்கால் பரப்பு தேவை?",
         question_en="How much nursery area is needed for one hectare of transplanted rice?",
         gold_docs=["rice_nursery"], key_facts=[["20 cent", "800"]],
         reference_answer="20 cents (800 square metres)."),
    dict(id="ta02", lang="ta", category="advisory",
         question="நெல் நடவு செய்த பிறகு வயலில் எவ்வளவு ஆழம் தண்ணீர் வைக்க வேண்டும்?",
         question_en="What water depth should be maintained in the rice field after transplanting?",
         gold_docs=["rice_water_management"], key_facts=[["2 cm"], ["5 cm"]],
         reference_answer="2 cm up to 7 days after transplanting, then cyclic 5 cm submergence."),
    dict(id="ta03", lang="ta", category="advisory",
         question="நெல்லில் மாற்று நனைத்தல் உலர்த்தல் முறைக்கு பயன்படும் நீர் குழாயின் அளவு என்ன?",
         question_en="What are the dimensions of the field water tube used for alternate wetting and drying in rice?",
         gold_docs=["rice_water_management"], key_facts=[["40 cm"], ["15 cm"]],
         reference_answer="A plastic pipe 40 cm long and 15 cm in diameter, perforated except the top 15 cm."),
    dict(id="ta04", lang="ta", category="advisory",
         question="தக்காளியில் முன்பருவ கருகல் நோய்க்கு காரணம் என்ன, அதன் அறிகுறிகள் என்ன?",
         question_en="What causes early blight in tomato and what are its symptoms?",
         gold_docs=["tomato_diseases"], key_facts=[["alternaria"], ["concentric", "ring"]],
         reference_answer="The fungus Alternaria solani; brown spots with concentric rings and yellow margins."),
    dict(id="ta05", lang="ta", category="scheme",
         question="கிசான் அழைப்பு மையத்தின் இலவச தொலைபேசி எண் என்ன?",
         question_en="What is the toll-free number of the Kisan Call Centre?",
         gold_docs=["kisan_call_centre"], key_facts=[["18001801551"]],
         reference_answer="1800-180-1551."),
    dict(id="ta06", lang="ta", category="scheme",
         question="ரபி பருவ உணவுப் பயிர்களுக்கு பயிர் காப்பீட்டில் விவசாயி செலுத்த வேண்டிய பிரீமியம் எவ்வளவு?",
         question_en="What premium does a farmer pay for rabi food crops under crop insurance?",
         gold_docs=["scheme_pmfby"], key_facts=[["1.5"]],
         reference_answer="A maximum of 1.5% of the sum insured."),

    # ---------------- Fertilizer calculator tool (5) ----------------
    dict(id="tool01", lang="en", category="tool",
         question="How much urea, SSP and MOP do I need for 2 hectares of timely sown irrigated wheat?",
         question_en="How much urea, SSP and MOP do I need for 2 hectares of timely sown irrigated wheat?",
         gold_docs=["wheat_central_india", "fertilizer_basics"], key_facts=[],
         reference_answer="Urea ~522 kg, SSP 750 kg, MOP 100 kg.",
         tool_expected=dict(crop_key="wheat_timely_irrigated", area_ha=2.0, **fert(120, 60, 30, 2.0))),
    dict(id="tool02", lang="hi", category="tool",
         question="5 एकड़ धान के लिए कितना यूरिया, एसएसपी और पोटाश चाहिए?",
         question_en="How much urea, SSP and potash are needed for 5 acres of rice?",
         gold_docs=["rice_nutrient_management", "fertilizer_basics"], key_facts=[],
         reference_answer="For 150:50:50 on 5 acres: urea ~660 kg, SSP ~632 kg, MOP ~169 kg.",
         tool_expected=dict(crop_key="rice_medium_long", area_ha=round(5 * HA_PER_ACRE, 4),
                            **fert(150, 50, 50, 5 * HA_PER_ACRE))),
    dict(id="tool03", lang="kn", category="tool",
         question="3 ಎಕರೆ ರಾಗಿ ಬೆಳೆಗೆ ಎಷ್ಟು ಯೂರಿಯಾ ಬೇಕು?",
         question_en="How much urea is needed for 3 acres of ragi?",
         gold_docs=["ragi_finger_millet", "fertilizer_basics"], key_facts=[],
         reference_answer="For 60:30:30 on 3 acres: urea ~158 kg.",
         tool_expected=dict(crop_key="ragi_irrigated", area_ha=round(3 * HA_PER_ACRE, 4),
                            **fert(60, 30, 30, 3 * HA_PER_ACRE))),
    dict(id="tool04", lang="ta", category="tool",
         question="2 ஹெக்டேர் வீரிய ஒட்டு நெல்லுக்கு எவ்வளவு யூரியா தேவை?",
         question_en="How much urea is needed for 2 hectares of hybrid rice?",
         gold_docs=["rice_nutrient_management", "fertilizer_basics"], key_facts=[],
         reference_answer="For 175:60:60 on 2 ha: urea ~761 kg.",
         tool_expected=dict(crop_key="rice_hybrid", area_ha=2.0, **fert(175, 60, 60, 2.0))),
    dict(id="tool05", lang="en", category="tool",
         question="I am sowing wheat late in December on 4 acres. How much fertilizer should I apply?",
         question_en="I am sowing wheat late in December on 4 acres. How much fertilizer should I apply?",
         gold_docs=["wheat_central_india", "fertilizer_basics"], key_facts=[],
         reference_answer="For late sown 100:50:25 on 4 acres: urea ~352 kg, SSP ~506 kg, MOP ~67 kg.",
         tool_expected=dict(crop_key="wheat_late_sown", area_ha=round(4 * HA_PER_ACRE, 4),
                            **fert(100, 50, 25, 4 * HA_PER_ACRE))),

    # ---------------- Out-of-scope / unanswerable (5) ----------------
    dict(id="ood01", lang="en", category="ood",
         question="What will be the price of onions in Nashik mandi next week?",
         question_en="What will be the price of onions in Nashik mandi next week?",
         gold_docs=[], key_facts=[], reference_answer="ABSTAIN: market price forecasts are not in the knowledge base."),
    dict(id="ood02", lang="hi", category="ood",
         question="क्रिकेट विश्व कप 2011 किसने जीता था?",
         question_en="Who won the 2011 Cricket World Cup?",
         gold_docs=[], key_facts=[], reference_answer="ABSTAIN: not an agriculture question."),
    dict(id="ood03", lang="kn", category="ood",
         question="ತೆಂಗಿನ ಮರದಲ್ಲಿ ಕಪ್ಪು ತಲೆ ಹುಳುವಿನ ನಿರ್ವಹಣೆ ಹೇಗೆ?",
         question_en="How to manage black-headed caterpillar in coconut trees?",
         gold_docs=[], key_facts=[], reference_answer="ABSTAIN: coconut pests are not covered by the knowledge base."),
    dict(id="ood04", lang="ta", category="ood",
         question="என் மாட்டுக்கு காய்ச்சல் இருக்கிறது, என்ன மருந்து கொடுக்கலாம்?",
         question_en="My cow has a fever, what medicine can I give?",
         gold_docs=[], key_facts=[], reference_answer="ABSTAIN: veterinary treatment is out of scope; contact a veterinarian."),
    dict(id="ood05", lang="en", category="ood",
         question="Write me a short poem about the monsoon.",
         question_en="Write me a short poem about the monsoon.",
         gold_docs=[], key_facts=[], reference_answer="ABSTAIN: not an agricultural advisory request."),
]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "testset.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = set()
    with out.open("w", encoding="utf-8") as f:
        for c in CASES:
            assert c["id"] not in ids, c["id"]
            ids.add(c["id"])
            c.setdefault("tool_expected", None)
            c["answerable"] = c["category"] != "ood"
            c["expected_route"] = {"tool": "tool", "ood": "abstain"}.get(c["category"], "rag")
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} cases -> {out}")


if __name__ == "__main__":
    main()
