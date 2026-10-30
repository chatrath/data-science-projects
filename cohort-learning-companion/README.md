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
ollama pull qwen3:4b-instruct
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


### Optional corrective RAG agent

Set `ENABLE_AGENT_GRAPH=true` in `.env` and restart Chainlit to use the local
LangGraph agent. The default (`false`) keeps the existing RAG pipeline. The
agent decomposes the question, retrieves and reranks evidence, and proceeds
straight to generation when any nonblank reranked chunks remain. Grading is a
local evidence check with no extra Ollama call. Empty retrieval retries once;
no evidence after two attempts returns the standard refusal. Planning and
retrieval run off the UI event loop; only
final-answer tokens are streamed. Relevant validated PDF previews remain image
elements, without source text cards. The text-only model does not inspect images.

The compiled graph also supports direct asynchronous invocation:

```python
from src.agent_graph import graph

state = await graph.ainvoke({"question": "What is context engineering?", "loop_step": 0})
answer = state["generation"]
image_paths = state["slide_images"]
```


Retrieval enforces a minimum cross-encoder raw logit of **0.5** before selecting
context chunks (this score is not cosine similarity or a probability). A higher
`--min-rerank-score` may tighten CLI searches; lower values cannot weaken the
floor. `--no-rerank` remains an explicit diagnostic bypass. No second LLM grader
is used. Context blocks retain source filenames, page metadata and citation IDs.
If the context cannot support the complete answer, generation responds:
`I could not find this in the course materials.`


The local generation model defaults to `qwen3:4b-instruct`. Set
`OLLAMA_CHAT_MODEL` in `.env` to override it for both agent query planning and
answer generation. Changing the chat model does not require re-embedding the
corpus; embeddings continue to use `nomic-embed-text`.


### Incremental PDF updates

Drop PDFs into `corpus/pdfs/` (handouts) or `corpus/slides/` (decks), then run:

```powershell
python update_corpus.py
```

The scan also accepts PDFs directly under `corpus/` or other raw subfolders.
Generated renders and hidden cache folders are excluded. File hashes identify
new/modified PDFs; unchanged indexed files are skipped. Existing PDFs indexed
before this updater are registered without re-embedding on its first scan.

Handouts retain logical sections and page metadata. Text is split into windows
of 800 reference-tokenizer tokens with 150-token overlap by default; slide
windows never cross page boundaries. Section tails can be shorter. A conservative
UTF-8 byte limit also protects embedding inputs. To customize the scan:

```powershell
python update_corpus.py --corpus corpus --chunk-tokens 800 --overlap-tokens 150
```

Embeddings use local `nomic-embed-text` with 768 dimensions and F16 model weights.
New vector values are rounded to F16 precision; the existing PostgreSQL
`vector(768)` column stores those values as float32, avoiding a schema migration.
Embedding calls remain batched. No chat model or API key is needed.

The updater merges complete JSONL snapshots to preserve earlier handouts,
slides, and transcripts. It refuses incomplete snapshots rather than deleting
older indexed data. Checkpoints are written only after successful DB ingestion.
BM25 uses the same unified table and is rebuilt on every search, so active
Chainlit sessions see new vocabulary without restarting the app. The script
also constructs a fresh BM25 index after ingestion to validate the refresh.
All processed snapshots, hashes, renders and manifests stay under ignored
`corpus/` paths. Rerun the scan whenever you add PDFs; `--reprocess` reparses all
PDFs while reusing unchanged embeddings. Files removed from disk are retained
in the index by this incremental updater.

## Local security controls

See [SECURITY_HARDENING.md](SECURITY_HARDENING.md) for trusted local HTTP/WebSocket origins, separate read-only chat and ingestion database roles, escaped context boundaries, code/quote citation isolation, and bounded concurrent requests. The normal database launcher provisions the restricted roles without rebuilding the corpus.

## Multi-part question coverage

With `ENABLE_AGENT_GRAPH=true`, explicit questions separated by `?` or coordinated interrogatives are preserved as focused searches. Polite prefixes and enumeration wording no longer change the requested topic. Questions with unresolved pronouns or quoted syntax retain model-based planning.

Single-question retrieval remains top-3. Multi-part retrieval defaults to up to two reranker-approved chunks per part and six distinct chunks overall (`RAG_MULTI_CHUNKS_PER_QUERY=2`, `RAG_MULTI_MAX_CHUNKS=6`). Deduplication merges question-part provenance for shared evidence. A part's best qualifying topic-heading match is considered before generic material; the relevance floor remains a raw cross-encoder logit of 0.5.

Context packing reserves a whole representative chunk for each part before adding extras. Multi-part requests can use a 10,000-byte target (`RAG_MULTI_CONTEXT_BYTES`) and 384 output tokens (`RAG_MULTI_NUM_PREDICT`), subject to the configured Ollama context capacity. Ordinary requests retain their existing limits. Source headers identify their question parts. If packing loses every retrieved excerpt for a part, generation refuses instead of silently attempting a partial answer. An entirely missing part also fails this coverage check.

Selected excerpts are never cut mid-code or mid-sentence. The corpus and index are not deleted or re-chunked by this change. A bounded prompt cannot include all corpus text: extra chunks can still be omitted when the context window is full. Agent result timings report retrieved/packed counts, omitted chunks, bytes, and unpacked parts. Increase the documented limits carefully if a legitimate question needs more evidence; higher limits can increase CPU latency. Preview eligibility remains independently gated by cosine >= 0.75, valid local paths, and cited-source alignment, so not every text chunk needs an image.

## Complete previews for cited PDF pages

After citation validation, the answer finalizer fills missing previews for exact, single-page PDF chunks that passed the cross-encoder relevance floor. This route verifies the cited source, page bounds, canonical render path, PDF type, and chunk identity. It repairs missing/corrupt renders and deduplicates pages in citation order. The UI independently verifies this provenance before accepting a `cited_page` element.

This is separate from approximate visual matching: the query cosine threshold remains 0.75 for that route, and original cosine scores are never boosted or overwritten. A directly cited page may therefore display even when its full-page text has lower query cosine; the displayed frame is the exact PDF page being cited, rather than another slide found by broad topic overlap. Unknown/rejected reranker scores, uncited chunks, ambiguous multi-page sections, invalid files, and unsafe paths cannot add previews through this route. PDF/image work runs outside the UI event loop, and render failure leaves the supported text answer intact. Generation remains text-only and no native text source cards are added.
