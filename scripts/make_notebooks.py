"""Generates the Colab notebooks in notebooks/ (kept as code so they stay in sync with the package)."""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def notebook(cells, name):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": name},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}}}
    (NB_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", NB_DIR / name)


SETUP_DRIVE = '''
# 2) Get the project: from Google Drive if you uploaded it there, otherwise clone it from GitHub
import os, sys, subprocess
REPO_URL = 'https://github.com/amanchandrah895/kisan-mitra.git'   # used only if the Drive folder is not found
DRIVE_DIR = '/content/drive/MyDrive/kisan-mitra'

try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
except Exception as e:
    print('Drive not mounted:', e)

if os.path.exists(f'{DRIVE_DIR}/configs/default.yaml'):
    PROJECT_DIR = DRIVE_DIR                      # results persist on Drive
else:
    PROJECT_DIR = '/content/kisan-mitra'
    if not os.path.exists(f'{PROJECT_DIR}/configs/default.yaml'):
        subprocess.run(['git', 'clone', '-q', REPO_URL, PROJECT_DIR], check=True)
os.chdir(PROJECT_DIR)
if f'{PROJECT_DIR}/src' not in sys.path:
    sys.path.insert(0, f'{PROJECT_DIR}/src')
# results go to Drive whenever it is mounted (survives Colab disconnects)
if PROJECT_DIR == DRIVE_DIR:
    RESULTS_DIR = f'{DRIVE_DIR}/results'
elif os.path.exists('/content/drive/MyDrive'):
    RESULTS_DIR = '/content/drive/MyDrive/kisan-mitra-results/results'
else:
    RESULTS_DIR = RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
print('Project folder:', PROJECT_DIR, '| results ->', RESULTS_DIR)
'''

INSTALL = '''
# 3) Install dependencies (~2-3 min). Colab already has torch / torchaudio.
!pip install -q -r requirements.txt
!pip install -q -r requirements-asr.txt || pip install -q onnxruntime-gpu onnx
!apt-get -qq install -y ffmpeg > /dev/null
print('\\nDone. If pip printed "you must restart the runtime": Runtime > Restart session, then run cell 2 again and continue from cell 4.')
'''

TOKEN_CHECK = '''
# 4) Hugging Face token + model access check
#    - Add your token: left sidebar "key" icon (Secrets) > Add new secret > Name: HF_TOKEN, Value: hf_xxx > enable "Notebook access"
#    - Accept the terms (one click each, while logged in) on:
#        https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
#        https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
#        https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
from kisan_mitra.config import load_config
from kisan_mitra.hf_utils import hf_token, check_model_access, env_report

cfg = load_config()
print('HF token found' if hf_token() else 'NO HF_TOKEN FOUND - add it in Colab Secrets (see above)')
status = check_model_access(list(cfg['models'].values()))
for m, s in status.items():
    print(('OK   ' if s == 'ok' else 'FAIL ') + m + ('' if s == 'ok' else '  ->  ' + s))
print(env_report())
bad = [m for m, s in status.items() if s != 'ok']
assert not bad, 'Fix model access for: ' + ', '.join(bad)
'''

nb1 = [
    md('''
# Kisan Mitra — run all experiments (Colab, T4 GPU)

This notebook runs the **complete evaluation** of the Kisan Mitra farmer advisory assistant and packs every result into
`kisan_mitra_results.zip`.

**Before you start**
1. *Runtime → Change runtime type → **T4 GPU*** (free tier is enough).
2. Upload the whole `kisan-mitra` folder to **My Drive** (so the path is `MyDrive/kisan-mitra`).
3. Add your Hugging Face token as a Colab secret named `HF_TOKEN`, and accept the terms of the 3 gated AI4Bharat models (links in cell 4).
4. *Runtime → Run all*. Expected time: **~60–90 min** (most of it is downloading ~20 GB of model weights and generating ~200 answers).

**If Colab disconnects:** reconnect, run cells 2, 3, 4 and then cell 6 again — finished work is saved on Drive and is skipped.

When it finishes, the zip downloads automatically (and is also saved in your Drive folder). Bring it back to your laptop.
'''),
    code('''
# 1) Check the GPU
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
'''),
    code(SETUP_DRIVE),
    code(INSTALL),
    code(TOKEN_CHECK),
    md('''
### 5) Smoke test (≈2 min)
Loads IndicTrans2 and translates one sentence each way, so problems show up before the long run.
'''),
    code('''
from kisan_mitra.factory import build_translator
from kisan_mitra.hf_utils import free_memory
tr = build_translator(cfg)
print(tr('धान की नर्सरी में बीज उपचार के लिए कार्बेन्डाजिम की कितनी मात्रा प्रयोग करें?', 'hi', 'en'))
print(tr('ರಾಗಿ ಸಸಿಗಳನ್ನು ಎಷ್ಟು ದಿನಗಳ ವಯಸ್ಸಿನಲ್ಲಿ ನಾಟಿ ಮಾಡಬೇಕು?', 'kn', 'en'))
print(tr.translate_long('Use 20 kg of seed per hectare. Transplant 25-day-old seedlings.', 'en', 'ta'))
del tr; free_memory()
'''),
    md('''
### 6) Run everything
Stages: question translation → retrieval ablation → speech recognition on spoken questions → 5 systems + voice run
(extractive RAG, AgriParam closed-book, AgriParam RAG, **Kisan Mitra full agent**, Param-1 RAG) → scoring.
Progress is printed per question. Safe to re-run: completed work is skipped.
'''),
    code('''
from kisan_mitra.experiment import run_all
out = run_all(results_dir=RESULTS_DIR)
'''),
    md('### 7) Quick look at the results'),
    code('''
import pandas as pd
pd.set_option('display.max_columns', 50); pd.set_option('display.width', 250)
R = RESULTS_DIR
show = ['system','task_success','qa_all_facts_acc','qa_fact_recall','qa_groundedness','tool_quantity_acc',
        'ood_abstain_recall','qa_false_abstain_rate','route_acc','success_en','success_hi','success_kn','success_ta',
        'latency_p50_s','errors']
display(pd.read_csv(f'{R}/metrics_by_system.csv')[show])
display(pd.read_csv(f'{R}/retrieval_ablation.csv'))
display(pd.read_csv(f'{R}/translation_summary.csv'))
import glob
for f in glob.glob(f'{R}/asr_summary_*.csv'):
    display(pd.read_csv(f))
'''),
    md('### 8) Zip and download the results'),
    code('''
from kisan_mitra.evaluate import zip_results
z = zip_results(RESULTS_DIR, os.path.join(os.path.dirname(RESULTS_DIR), 'kisan_mitra_results'))
print('Saved to Drive:', z)
from google.colab import files
files.download(z)
'''),
]

nb2 = [
    md('''
# Kisan Mitra — live demo (Gradio)

Launches the web app with a **public share link** so you can demo it from any device.

1. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
2. Open the `https://xxxx.gradio.live` link printed at the bottom.
3. Try the example questions (Hindi / Kannada / Tamil / English), the microphone, and "Read the answer aloud".
4. For the submission: take screenshots (one per language + the fertilizer tool + a refusal) and record a
   2–3 minute screen video. Save them into `docs/screenshots/` of the project.

The link stays alive while this notebook is running (max ~72 h, usually until Colab idles out).
'''),
    code('''
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    code(SETUP_DRIVE),
    code(INSTALL),
    code(TOKEN_CHECK),
    code('''
# 5) Load the models (AgriParam + IndicTrans2 + retriever + grounding checker + IndicConformer ASR), ~5-8 min first time
from kisan_mitra.app import build_agent_for_app, build_app
agent = build_agent_for_app()
r = agent.answer('5 एकड़ धान के लिए कितना यूरिया चाहिए?')
print(r['route'], '|', r['answer'])
'''),
    code('''
# 6) Launch - click the public URL
demo = build_app(agent)
demo.launch(share=True, debug=True)
'''),
]

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    notebook(nb1, "01_run_all_experiments.ipynb")
    notebook(nb2, "02_demo_app.ipynb")
