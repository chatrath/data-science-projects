# AI evolution dashboard

Open this uvdemo folder in VS Code. Select .venv/Scripts/python.exe as the Python interpreter.

Install or synchronize dependencies:

```powershell
uv sync
```

Run from this folder without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Press Ctrl+C in the terminal to stop.

Dependencies are managed by pyproject.toml and uv.lock. requirements.txt is provided for pip-based installations.

## Timeline dataset

Edit `timeline_data.csv` beside `app.py` to update milestones, then rerun or refresh the app. Keep the headers `date,tool,title,kind,text,source`. Dates use `YYYY-MM-DD`; kind is `Closed` or `Open-weight`. The app sorts rows by date and derives tool columns from the CSV. Save as UTF-8 CSV.

## Model comparison

The comparison panel appears below the AI Insights Generator. Its model choices come from the full timeline CSV, independently of the timeline filters. Select two different releases within Closed or Open-weight. Grouped releases require checking the exact variant's specifications and terms.

`model_comparison.py` renders the panel. Click **Compare Models** to fetch the selected releases' public documentation using `live_model_docs.py`. Each click performs a fresh fetch; later cost edits reuse that result, and changing the model selection hides it until the next click. Only factors with evidence for both models and differing canonical facts appear. The Model Documentation factor contains Link 1 to the official README/model card (or technical documentation when no README is published) and Link 2 to the official model website. Every current timeline release has two configured official sources. Shared factors and factors without enough evidence are omitted; failed downloads are reported without substituting static model descriptions. These are conservative extracted documentation facts, not measured benchmark scores or security ratings. Internet access is required; no model API key or additional package is needed.

`model_comparison_data.json` holds documentation routes and selected verified standard API prices, checked on 11 October 2026. Profile keys match timeline titles exactly. `documentation_urls` can specify up to two official documentation or model-card URLs; otherwise the app uses the model URL and the timeline's source. New timeline rows appear automatically. Some historical announcements contain few comparable specifications, and grouped releases can describe multiple variants. Missing prices appear as blank message/reply price fields and must be entered before calculating. Update the JSON and checked date when re-verifying prices. The optional pricing fields `threshold`, `input_multiplier` and `output_multiplier` apply to input token counts above that threshold.

For open-weight models, Cost scenario appears inside the Cost calculator. Cost estimates cover either billed API tokens or a user-entered monthly infrastructure and operations estimate. Enter the workload and any required rates, then click **Compare Costs** to calculate both models using the current values. Changing models, usage, prices or deployment estimates hides old results until you click the button again. Published hosted prices are prefilled and editable; provider reference and confirmation fields are omitted for this POC. Self-hosted estimates show only monthly usage, server/equipment cost and maintenance/support cost.

Each comparison field has a short help definition; the information icons beside table factors open their definitions when clicked. Estimates exclude extra fees and do not prove a self-hosted deployment has sufficient capacity. No external model API is called and no API key is needed. The comparison CSV includes the fetched differing factors, documentation links and fetch time, calculated monthly costs and workload assumptions; costs remain blank until a valid cost comparison is calculated. Restart Streamlit after changing the comparison module if an existing session retains the old display.

## Local AI adoption insights

The summary and question-answer features run locally with Ollama and Qwen3 1.7B. No API key, paid model account or cloud inference is required. The small model suits this computer's approximately 16 GB RAM. Runtime binaries, model weights and logs are stored in `.local-ai` on D: and excluded from Git.

After restarting Windows, open a PowerShell terminal in the project and run:

```powershell
.\start_local_ai.ps1
```

This starts Ollama in the background at `http://127.0.0.1:11434` and downloads the model only if it is missing. Then start Streamlit as usual. The first generation can take longer while the model loads. Set `OLLAMA_MODEL` before starting the app to use another installed local model.

Generate Summary creates the executive paragraph from the active survey view. AI Insights Generator is an independent generative chat with session history and follow-up questions. It sends only the current question and recent conversation to the local model, with no survey context or reference retrieval. Chat replies aim for 100–250 words and never exceed 1,500 words. New conversation clears the history. Executive summaries retain the under-500-word limit and do not display source links. Changing the survey view hides previous executive summaries until generated again; chat history stays available. Compare Models and Cost Calculator are unchanged.

Survey observations, dates, sample sizes and methodological notes are sent to the local model. Relevant official AWS security and Hugging Face deployment references can also be fetched and supplied as context. This is targeted document retrieval rather than unrestricted web search; links are added only for supplied surveys and successfully fetched references. Reference downloads require internet; local inference itself does not. Prompts distinguish hypotheses from measured causes and forbid treating unlike survey metrics as a time trend or inventing closed-sector adoption rates.

The portable Ollama runtime is CPU-only. Small local models can be less accurate than larger models; check generated claims against the linked survey. The installed default is Qwen3 1.7B Q4_K_M. Old OpenAI secrets are not read or required by this local backend.

## Setup after cloning from GitHub

The repository excludes the virtual environment, private credentials, Ollama runtime, and model weights. Run `uv sync` inside this folder to create the Python environment. Before using the local AI features, place the portable Ollama runtime (including `ollama.exe` and its supporting files) in `.local-ai/`, then run `.\start_local_ai.ps1`. The script downloads `qwen3:1.7b` if needed. The application itself does not require an OpenAI API key.
