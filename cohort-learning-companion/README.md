# Cohort Learning Companion

Local RAG chat for course handouts, slide decks and live-session transcripts.
Uses PostgreSQL/pgvector, BM25 hybrid retrieval, local cross-encoder reranking,
Ollama and a streaming Chainlit interface.

## Local setup

Use Python 3.13 for the Chainlit UI. With uv installed, run in PowerShell:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Configure `DATABASE_URL` for a PostgreSQL database with pgvector installed.
Project-local database helper scripts require a separately installed runtime
under `.local-db/runtime`; that runtime is deliberately not in this repository.
See [RETRIEVAL.md](RETRIEVAL.md) for database, OCR and provider details.

Create these local folders and add your own course materials:

```text
corpus/
  pdfs/
  slides/
  transcripts/
```

For scanned slides, install English Tesseract language data as described in
RETRIEVAL.md. Then ingest and launch:

```powershell
.\.venv\Scripts\python.exe -m src.retriever ingest
.\start_app.ps1
```

The first reranking request downloads the local cross-encoder model. The UI
launcher manages its separate Python 3.13 environment. Ollama and PostgreSQL
must be running.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## Repository contents

Only application code, tests, configuration templates and documentation are
intended for GitHub. `corpus/` (including rendered slides, JSONL exports and
parse caches), `.env`, `.files/`, `.local-db/`, virtual environments and logs
are excluded. Never add these ignored directories with `git add --force`.
