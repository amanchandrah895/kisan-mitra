# Data card

## Knowledge base (`data/corpus/`)
17 markdown documents, each with YAML front matter (`id`, `title`, `crop`, `topic`, `source_org`, `source_url`, `accessed`).
Content is **paraphrased factual notes** from public Indian government / ICAR / state agricultural university sources,
keeping every number (doses, timings, thresholds, amounts) exactly as published. Notes were compiled on 2026-11-05.
Text marked *editorial note* is ours, not from the source. Where a source was ambiguous (e.g. Soil Health Card issue
cycle: 2 vs 3 years; an OCR-garbled pheromone-trap count in the PPQS fall-armyworm advisory) the ambiguity is stated or
the number omitted rather than guessed.

| id | title | source | url | words |
|---|---|---|---|---:|
| `cotton_pink_bollworm` | Pink bollworm management in cotton | ICAR-Central Institute for Cotton Research (CICR), Nagpur - Pink bollworm management strategies (Kharif 2019) | [link](https://icar.org.in/sites/default/files/Circulars/PBW-Management-strategies-2019.pdf) | 346 |
| `fertilizer_basics` | Converting N-P-K recommendations into urea, SSP and MOP quantities | Derived from the nutrient contents used in ICAR/Krishisewa wheat fertilizer tables (urea 46% N, SSP 16% P2O5, MOP 60% K2O) | [link](https://krishisewa.com/improved-wheat-cultivation-practices-in-central-india) | 252 |
| `kisan_call_centre` | Kisan Call Centre (farmer helpline 1800-180-1551) | mKisan portal (Department of Agriculture & Farmers Welfare) and ICAR | [link](https://mkisan.gov.in/Home/AboutKCC) | 179 |
| `maize_fall_armyworm` | Fall armyworm (Spodoptera frugiperda) management in maize | Directorate of Plant Protection, Quarantine & Storage (PPQS), Govt. of India - advisory dated 12.07.2024 | [link](https://ppqs.gov.in/sites/default/files/faw_advisory.pdf) | 298 |
| `ragi_finger_millet` | Ragi (finger millet) crop production | Tamil Nadu Agricultural University (TNAU) Agritech Portal | [link](https://agritech.tnau.ac.in/agriculture/millets_ragi.html) | 453 |
| `rice_nursery` | Rice wet nursery and seed treatment (transplanted puddled lowland rice) | Tamil Nadu Agricultural University (TNAU) Agritech Portal | [link](https://agritech.tnau.ac.in/agriculture/agri_cropproduction_cereals_rice_tranpudlow_wet_nursery.html) | 415 |
| `rice_nutrient_management` | Rice main-field nutrient management (transplanted rice) | Tamil Nadu Agricultural University (TNAU) Agritech Portal | [link](https://agritech.tnau.ac.in/agriculture/agri_cropproduction_cereals_rice_tranpudlow_mainfield_nutrient_mgmt_inorganic.html) | 385 |
| `rice_sri` | System of Rice Intensification (SRI) | Tamil Nadu Agricultural University (TNAU) Agritech Portal | [link](https://agritech.tnau.ac.in/agriculture/agri_cereals_rice_sri.html) | 369 |
| `rice_water_management` | Rice main-field water management and alternate wetting and drying | Tamil Nadu Agricultural University (TNAU) Agritech Portal | [link](https://agritech.tnau.ac.in/agriculture/agri_cropproduction_cereals_rice_tranpudlow_mainfield_water_mgmt.html) | 425 |
| `scheme_kcc_credit` | Kisan Credit Card (KCC) and Modified Interest Subvention Scheme | Union Cabinet decision of 28 May 2025 on MISS (as reported by Vision IAS and Angel One) and Union Budget 2025-26 announcement (as reported by A2Z Taxcorp) | [link](https://visionias.in/current-affairs/news-today/2025-05-29/schemes-in-news/union-cabinet-approves-continuation-of-modified-interest-subvention-scheme-for-fy-2025-26) | 202 |
| `scheme_pm_kisan` | PM-KISAN (Pradhan Mantri Kisan Samman Nidhi) income support | PM-KISAN official FAQ, Department of Agriculture & Farmers Welfare | [link](https://fw.pmkisan.gov.in/Documents/FAQPMKISAN.pdf) | 311 |
| `scheme_pmfby` | Pradhan Mantri Fasal Bima Yojana (PMFBY) crop insurance | Lok Sabha Unstarred Question 3866 (21 December 2021) and PIB note on revised PMFBY operational guidelines | [link](https://eparlib.nic.in/bitstream/123456789/977813/1/AU3866.pdf) | 253 |
| `scheme_pmksy_pdmc` | PMKSY Per Drop More Crop - micro irrigation (drip and sprinkler) subsidy | Press Information Bureau - Ministry of Agriculture & Farmers Welfare (4 February 2022) | [link](https://pib.gov.in/PressReleasePage.aspx?PRID=1795499) | 159 |
| `scheme_soil_health_card` | Soil Health Card scheme | Press Information Bureau (PIB) factsheet on the Soil Health Card scheme (January 2022) | [link](https://www.pib.gov.in/FactsheetDetails.aspx?Id=148602) | 303 |
| `tomato_diseases` | Major diseases of tomato and their management | Tamil Nadu Agricultural University (TNAU) Agritech Portal - Crop Protection (updated 2023) | [link](https://agritech.tnau.ac.in/crop_protection/tomato_diseases_6.html) | 508 |
| `vermicompost` | Vermicompost production and use | ICAR Research Complex for NEH Region, Umiam - Vermicompost: Production and Practices | [link](https://static.vikaspedia.in/media/files_en/agriculture/farm-based-enterprises/vermicompost-production-and-practices.pdf) | 397 |
| `wheat_central_india` | Improved wheat cultivation practices for Central India | Krishisewa (compilation of ICAR wheat recommendations for the Central Zone) | [link](https://krishisewa.com/improved-wheat-cultivation-practices-in-central-india) | 411 |

Chunking: split on `##` sections, then 160-word windows with 30-word overlap; inline `Source:` URLs are removed from the
evidence text and shown as citations instead.

## Test set (`data/eval/testset.jsonl`, generated by `scripts/build_testset.py`)
40 cases: 12 English, 12 Hindi, 8 Kannada, 8 Tamil. 20 crop-advisory, 10 scheme, 5 fertilizer-quantity (tool), 5 out-of-scope.
Fields: `question` (in `lang`), `question_en` (human reference translation), `gold_docs`, `key_facts` (each fact is a
list of accepted surface forms), `reference_answer`, `tool_expected` (exact kg for tool cases), `expected_route`.
Questions were written by the project team from the corpus; Indic questions are hand-written (not machine translated).
A unit test verifies every key fact occurs in its gold documents.

Known biases: questions are short and well-formed; real farmer queries are noisier, code-mixed and dialectal. Speech
tests use synthetic gTTS audio.
