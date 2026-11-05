# Run guide (what you do on Colab — about 15 minutes of clicking, ~1–1.5 h of waiting)

## One-time setup (5 min)
1. **Hugging Face account** → https://huggingface.co/settings/tokens → *Create new token* → type **Read** → copy it (`hf_…`).
2. While logged in, open each page and click **“Agree and access repository”**:
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
3. **Google Drive** → upload the whole `kisan-mitra` folder into *My Drive* (drag the folder in). The path must be
   `My Drive/kisan-mitra/` (or edit `PROJECT_DIR` in cell 2).

## Run the experiments (notebook 01)
1. In Drive, double-click `kisan-mitra/notebooks/01_run_all_experiments.ipynb` → *Open with Google Colaboratory*.
2. *Runtime → Change runtime type → **T4 GPU** → Save*.
3. Left sidebar 🔑 **Secrets** → *Add new secret* → Name `HF_TOKEN`, Value = your token → toggle *Notebook access* on.
4. *Runtime → Run all*. Allow Drive access when asked.
5. Keep the tab open. Cell 4 must print `OK` for all 8 models; if one says `FAIL … GATED`, do step 2 above for it and re-run cell 4.
6. When finished, `kisan_mitra_results.zip` downloads (a copy is also in your Drive folder).

**If something breaks:** reconnect, run cells 2 and 4, then cell 6 again — completed parts are skipped.
If pip asks to restart: *Runtime → Restart session*, then run cell 2, then continue from cell 4.

## Bring the results back
Put `kisan_mitra_results.zip` into the `LLM Lab` folder on your laptop (or unzip it into `kisan-mitra/results/`).
Claude then writes the report, failure analysis, figures and final README numbers from the real results.

## Demo (notebook 02) — for screenshots / video
Open `notebooks/02_demo_app.ipynb` the same way → T4 GPU → *Run all* → click the `https://….gradio.live` link.
Capture: one Hindi, one Kannada, one Tamil question, the fertilizer calculator (e.g. “5 एकड़ धान के लिए कितना यूरिया…”),
a refusal (“My cow has a fever…”), and the *Agent trace* panel. Save screenshots to `docs/screenshots/`.
