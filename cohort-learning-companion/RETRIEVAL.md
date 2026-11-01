# PostgreSQL hybrid retrieval

## Ingest all raw course sources

    python -m src.retriever ingest

This command discovers and parses corpus/pdfs/*.pdf (handouts), PDF/Markdown/text
slide decks under corpus/slides/, and TXT/VTT/SRT transcripts under
corpus/transcripts/ recursively. It rebuilds slides.jsonl and materials.jsonl,
then embeds all nonempty chunks into the same companion_chunks table. Native
PDF text is preferred; image-only PDF pages use local PyMuPDF/Tesseract OCR.
Slide images are rendered and their paths are retained for Chainlit previews.
Transcript speakers, timestamps, and code-block context remain in metadata.

The ingestion summary lists input files, searchable chunks, empty records, OCR
pages, and cached files for each kind (handout, slide, transcript), plus unified
stored counts. Source hashes invalidate cached parses whenever a raw file changes.
Removed raw files disappear from snapshots and are pruned from those datasets on
synchronization. Unchanged parses/OCR and embeddings are reused.

For the current corpus: 68 handout chunks, 46 slide chunks, 56 transcript chunks.
All three participate in per-category BM25/dense candidate pools and RRF. Each
qualifying source gets one reserved slot when top_k allows it; remaining slots
follow the global ranking. See stratified retrieval below for relevance gates.

To load existing JSONL without reprocessing raw sources:

    python -m src.retriever ingest --processed-only

Supplying explicit JSONL paths also bypasses raw parsing. --corpus selects an
alternative corpus directory containing pdfs/, slides/, and transcripts/.

### Local OCR language data

English OCR data is installed at .local-db/tessdata/eng.traineddata, which is
ignored by Git. PyMuPDF's integrated Tesseract engine uses it locally; no inference
API or separate OCR executable is needed. On another machine, install English
Tesseract language data in that directory or set TESSDATA_PREFIX to its folder.
The official model is available at:
https://github.com/tesseract-ocr/tessdata_fast/raw/refs/heads/main/eng.traineddata

OCR recovers text labels, not diagram meaning, and may contain recognition errors.
Always consult the attached slide preview for complex visual layouts. Pages that
remain truly blank are counted explicitly and omitted from text embeddings.


## Project-local PostgreSQL on Windows

PostgreSQL 16 and pgvector are installed under .local-db/runtime using
conda-forge packages. No system-wide service or Docker installation is required.
Database files, binaries, logs and generated credentials are local only;
.local-db/ and .env are ignored by Git. DATABASE_URL is configured automatically
for 127.0.0.1:5432/cohort_db with SCRAM authentication and a generated password.

Start the database after reboot or whenever it is stopped:

    .\start_database.ps1

Stop it when finished:

    .\stop_database.ps1

Equivalent commands that do not require PowerShell script execution permission:

    python -m src.local_database start
    python -m src.local_database stop

The scripts initialize a new local cluster only if none exists and reuse existing
data on subsequent starts. They preserve a remote DATABASE_URL rather than
overwriting it. Logs are in .local-db/postgres.log. Ollama remains a separate
server and must also be running for local embeddings or generation.

## Local Ollama (active project configuration)

The project .env selects LLM_PROVIDER=ollama. No OpenAI, Nebius, or LangSmith
API keys are needed for this mode. PostgreSQL/pgvector is still required; set
DATABASE_URL to your local database for completely local execution.

Install Ollama from https://ollama.com/download and ensure its server is running.
If the desktop app is not serving already, run `ollama serve` in another terminal.
Download models once (internet required for installation/model downloads):

    ollama pull llama3.1:8b
    ollama pull nomic-embed-text

Environment settings:

    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_CHAT_MODEL=llama3.1:8b
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text
    OLLAMA_EMBEDDING_DIMENSIONS=768

To use Qwen instead, pull qwen2.5:7b and set OLLAMA_CHAT_MODEL=qwen2.5:7b.
The Ollama URL is the native server URL, without /v1. The implementation uses
langchain_ollama.ChatOllama and langchain_ollama.OllamaEmbeddings directly.
Nomic document inputs use search_document: and queries use search_query:.
The embedder requests num_ctx=8192; the conservative 8191-byte input guard includes
these prefixes. Returned vectors must match 768 dimensions. No hosted-provider
fallback occurs if the local server or model is unavailable.

From the activated project environment:

    python -m src.retriever ingest
    python -m src.retriever search "Explain context engineering" --top-k 5

Use a fresh PostgreSQL database when switching embedding providers: any prior
Nebius/OpenAI collection has incompatible vectors. Existing data is preserved
and a configuration mismatch is rejected instead of resizing or deleting it.
To switch back, change LLM_PROVIDER explicitly; saved hosted credentials remain.
Generation uses the same generate_text helper shown below; max_tokens is mapped
to Ollama's num_predict. Set LANGSMITH_TRACING=false to disable tracing if you
previously enabled it globally. After setup, model inference needs no internet.

References:
- https://docs.langchain.com/oss/python/integrations/embeddings/ollama
- https://docs.langchain.com/oss/python/integrations/chat/ollama
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5


Set DATABASE_URL and your provider credentials in the project .env (see
.env.example). Nebius is selected by LLM_PROVIDER=nebius, or automatically when
NEBIUS_API_KEY is present and LLM_PROVIDER is unset. Local mode uses the explicit
LLM_PROVIDER=ollama setting above. Explicit provider selection
always wins; credentials are never silently substituted across providers.

Nebius settings:

    LLM_PROVIDER=nebius
    NEBIUS_API_KEY=<your key>
    NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
    NEBIUS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
    NEBIUS_EMBEDDING_DIMENSIONS=4096
    NEBIUS_CHAT_MODEL=<exact generation model ID activated in your account>

NEBIUS_BASE_URL is optional; missing or blank uses the endpoint above. Paste a
plain URL, without Markdown link syntax. Existing .env values are preserved.
Use the model catalog to confirm model availability after account activation.
For OpenAI, set LLM_PROVIDER=openai and OPENAI_API_KEY. Optional OPENAI_BASE_URL,
OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIMENSIONS and OPENAI_CHAT_MODEL use the
same shared client configuration. OpenAI embeddings default to
text-embedding-3-small with 1536 dimensions.

Example database URL:
postgresql://username:password@localhost:5432/cohort_learning
The server must have pgvector installed. The ingestion role needs permission to
create the vector extension and the companion_chunks / companion_embedding_config
tables. A DBA can enable the extension first if the role cannot create it.

From the project directory, with the virtual environment activated:

    python -m src.retriever ingest
    python -m src.retriever search "How does hybrid retrieval work?" --top-k 5

Or specify processed files explicitly:

    python -m src.retriever ingest corpus/slides.jsonl corpus/materials.jsonl

Ingestion sends nonempty text to the selected provider and embedding model. Metadata, including page references, sections, speaker/time blocks,
and slide render paths, is kept in PostgreSQL JSONB. Images are not embedded.
The default OpenAI embedding input limit is 8191 cl100k tokens. Nebius uses a
conservative 8191 UTF-8 byte budget instead of applying an OpenAI tokenizer to
Qwen; the provider still enforces its own model-specific token limit. Oversized
records are rejected rather than silently truncated. Rechunk them upstream.
Alternative embedding models must support the dimensions parameter and return
the configured output size; output vector dimensions are validated.

Each supplied JSONL file is synchronized atomically. Reruns reuse unchanged text
embeddings, update metadata, and remove obsolete records only for those files.
Do not ingest a partial JSONL file as a replacement for a full dataset. Input
absolute paths identify datasets: moving a processed file creates a new dataset.

BM25 uses consistent Unicode/code-identifier tokenization with k1=1.5, b=0.75,
and positive Robertson IDF. It loads text from PostgreSQL for each query, so there
is no stale separate keyword index. This is intended for a small cohort corpus.
Dense retrieval is exact cosine search in pgvector. Each channel retrieves 25
candidates by default; weighted reciprocal rank fusion (k=60) returns the top 5.
Use --candidates to expand the pool and --bm25-weight between 0 and 1 to tune
keyword versus semantic emphasis. The default gives both equal weight.

Python use:

    from src.retriever import HybridRetriever
    retriever = HybridRetriever()
    hits = retriever.search("Explain context engineering", top_k=5)

Each hit contains text, original metadata, fused score, channel ranks and raw
channel scores. Scores express relevance, not calibrated confidence or evidence
sufficiency; refusal decisions belong in the answering chain.

Offline verification (no API calls or database required):

    python -m unittest discover -s tests -v

## Generation using the same provider

The grounded retrieval/answering chain is implemented in src/chain.py:

    python -m src.chain ask "What is context engineering?"
    python -m src.chain ask "Explain evidence grounding" --top-k 3 --max-tokens 256

It retrieves hybrid-search hits, packs whole chunks into a context block, and
uses a LangChain prompt -> ChatOllama -> string-parser pipeline. It defaults to
the configured llama3.1:8b model; OLLAMA_CHAT_MODEL=qwen2.5:7b is also supported.
This chain requires LLM_PROVIDER=ollama and uses no hosted inference API.
PostgreSQL and Ollama must be running, and corpus ingestion must have completed.

Answers cite excerpt IDs such as [S1]. The CLI lists cited filenames and available
page ranges, section titles or transcript time ranges. The system prompt requires
answers grounded only in retrieved excerpts and refuses insufficient evidence.
Empty context bypasses generation. Missing or invalid citations also produce a
refusal. Citation validation is structural, not a factual-entailment guarantee;
retrieval relevance and semantic grounding still depend on the model.

The context budget defaults to 5000 UTF-8 bytes, retaining whole code/demo chunks and
skipping oversized chunks instead of cutting them mid-block. Questions are
limited to 1024 UTF-8 bytes; the Ollama context defaults to 8192 tokens. Generation defaults
to 256 output tokens (configurable with --max-tokens, between 1 and 1024).
--candidates and --bm25-weight match the retriever CLI. No RRF confidence threshold
is used because reciprocal-rank scores are not calibrated relevance confidence.

Python use:

    from src.chain import RAGChain
    result = RAGChain().ask("What is context engineering?")
    print(result["answer"])

For generation without retrieval, the shared helper remains available:

    from src.provider import generate_text
    answer = generate_text([
        {"role": "user", "content": "Explain retrieval-augmented generation briefly."}
    ], max_tokens=512)

For hosted generation, set NEBIUS_CHAT_MODEL or OPENAI_CHAT_MODEL. Local generation
defaults to OLLAMA_CHAT_MODEL=llama3.1:8b.
This uses the same provider key and base URL as retrieval. Creating a client or
running offline tests does not send inference requests.

## Embedding compatibility

The database records provider, base URL, embedding model, and dimensions. Newly
created tables use the configured vector dimensions. Existing legacy tables are
recognized as OpenAI collections during the next ingestion schema update.
Changing any embedding setting cannot reuse the old vectors: use a separate
PostgreSQL database and ingest the processed JSONL files again. No existing
vectors are automatically deleted or resized. Run ingestion before search on a
legacy schema. Changing only the generation model does not require re-ingestion.

Nebius references:
- https://api.tokenfactory.nebius.com/docs
- https://github.com/nebius/token-factory-cookbook/blob/main/integrations/pixeltable/README.md

## Chainlit chat interface

With the virtual environment active, PostgreSQL started and Ollama running:

    .\start_app.ps1

Open http://localhost:8000 in your browser. Each chat session initializes its own
RAGChain. Retrieval runs in a worker thread and generation uses the native asynchronous
LangChain/Ollama stream. An empty Chainlit message receives each generated text
chunk through stream_token. The final message adds cited filenames, page/section
details or transcript timestamps after citation validation.
Refusals appear without a fabricated sources list. The interface retries chain
initialization on the next message if local configuration was unavailable at start.
Questions are independent; prior turns are not supplied as model context.

The UI launcher uses .venv-ui (Python 3.13). The original .venv stays available
for database and ingestion commands. Chainlit 2.12's nest_asyncio patch breaks
Python 3.14 event-loop detection, so do not launch Chainlit with that interpreter.
Direct equivalent, which does not require PowerShell script execution permission:

    .\.venv-ui\Scripts\python.exe -m chainlit run app.py --host 127.0.0.1

To recreate the UI environment on another machine:

    .\start_app.ps1 -Check

The app explicitly disables Chainlit chat-history persistence. DATABASE_URL
continues to be used by the RAG retriever; no asyncpg dependency or separate
Chainlit history schema is required. The async welcome and message handlers
catch initialization, retrieval, generation, and disconnected-client failures.

The launcher stores its Python 3.13 runtime under .local-db/python inside this
project. It does not depend on an AppData Python installation. If .venv-ui points
to an unavailable or external interpreter, the launcher rebuilds its configuration
with --allow-existing, preserving installed packages, then checks dependencies.
If the local runtime is absent, uv downloads it on demand (internet required).
Use .\start_app.ps1 -Check to repair/validate without launching the server.

## Rendered slide previews

Hybrid search now returns a resolved render_path for records that reference an
existing image in corpus/rendered. RAGChain.ask includes deduplicated slide_images
with filename/page labels. Chainlit attaches these frames as inline images beside
the answer and source citations, including when text evidence is insufficient for
an answer. CLI answers print the local image paths.

This uses the UI display route: llama3.1:8b / qwen2.5:7b continue to receive text,
not image inputs. The prompt explicitly prevents claiming to inspect diagrams.
Slide previews are labeled as retrieved material, separately from cited answer
sources. Missing images are skipped. Only local images within corpus/rendered
are accepted; old absolute paths can be rebased after moving the project.

No re-ingestion is needed for current slide records because render_path already
exists in their saved metadata. Newly rendered PDFs must be ingested normally.
Default raw ingestion OCRs image-only slides before embedding. Truly blank pages
remain excluded, with their count shown in the ingestion summary.

Local Ollama chat has a 600-second request timeout to allow CPU prompt processing.
Embedding requests retain their existing timeout.

## Local performance tuning

Generation now defaults to 3 retrieved chunks, a 5000-byte context budget,
256 output tokens and an 8192-token Ollama context window. Full chunks/code blocks
are retained or skipped; they are never cut mid-demo. The context budget also
reserves room for system instructions, the question, output and chat framing.
Use --top-k or --max-tokens for questions that need more evidence or longer code.
Smaller context/output limits can reduce detail, so increase them when needed.

Environment settings (loaded when the app/chain starts):

    EMBEDDING_BATCH_SIZE=32
    RAG_TOP_K=3
    RAG_CONTEXT_BYTES=5000
    OLLAMA_NUM_CTX=8192
    OLLAMA_NUM_PREDICT=256
    OLLAMA_KEEP_ALIVE=600
    OLLAMA_NUM_THREAD=6

The current computer has 6 physical CPU cores; its .env uses 6 threads. On other
machines, leave OLLAMA_NUM_THREAD blank for Ollama's automatic choice, or set it
to the physical core count and measure. No GPU is forced off. keep_alive retains
chat and embedding models for 600 seconds when memory allows, reducing reloads.
Generation options are sent in native nested Ollama options (including through
the standalone generate_text helper).

Chunk embeddings were already batched. Batches remain capped at 32 inputs for
compatible-provider request limits, and identical texts are now deduplicated
within each embedding call, then mapped back to every original chunk in order.
Unchanged stored embeddings continue to be reused. Ingestion reports unique
embedding inputs and actual batch count; cached ingestion requires zero calls.

A per-retriever 128-entry, 5-minute cache reuses exact query embeddings. Ranked
results are NOT cached: every query still reads the current database and searches
all three source kinds. Corpus ingestion/updates are therefore visible immediately.
The database and BM25 portion measured about 0.12 seconds, so their exact-search
behavior was preserved rather than trading retrieval quality for approximations.

Profile future queries with:

    python -m src.chain ask "What is context engineering?" --profile

This prints retrieval/generation wall times, context chunk count and byte size.
Python RAGChain.ask returns the same measurements in result['timings']. The chat
interface keeps those diagnostics out of its answer display.

## Live answer streaming

app.py now consumes RAGChain.astream(query). The chain emits token events as
LangChain/Ollama produces text, followed by one final event carrying the checked
answer, citations, slide previews and timings. The synchronous ask() method and
CLI retain their existing interface and share context preparation/final validation.

Each token is sent using await reply.stream_token(token). Tokens are provisional:
when generation finishes, grouped citations are normalized and the final answer
replaces the streamed text. Failed citation checks replace the draft with the
standard refusal, with no fabricated source list. Mid-stream failures replace
partial text with the existing friendly error message. Streams are explicitly
closed on cancellation/disconnection, and sources/images are attached at the end.

Streaming removes the wait for the complete answer; retrieval, model loading and
prompt processing still happen before the first token. The final result records
first_token_seconds in timings for generated responses.

## Stratified multi-source retrieval (default)

The source categories are metadata.kind = handout, slide, transcript. Slide PDFs
are distinguished from handout PDFs by this metadata, not just the PDF extension.

The keyword index still uses corpus-wide BM25 statistics so scores stay comparable.
It collects up to --candidates hits per category. Dense search uses one pgvector
query with row_number() partitioned by source category, also keeping that many
candidates per category. Both channels share one query embedding and merge into
a single global RRF ranking. A dominant source can no longer remove another
category from the candidate pool before balancing happens.

One best RRF-ranked representative per category is eligible for a reserved slot
when EITHER relevance gate is met:

- cosine similarity >= 0.60; OR
- BM25 score >= 2.0 AND at least 25% of meaningful query words occur in the chunk.

A small stopword list prevents matches only on words like 'what', 'is', and 'the'
from earning lexical reservations. These are explicit heuristics, not calibrated
confidence probabilities. RRF scores are never used as absolute relevance gates.

When top_k >= 3 and all three categories qualify, each appears at least once.
If top_k is smaller, the highest RRF-ranked qualifying category representatives
win; no source has fixed priority. Missing or weak categories receive no forced
slots. Remaining slots are backfilled by global RRF. Returned results preserve
RRF order, remain unique, and never exceed top_k. The thresholds gate reserved
slots only; global backfill retains the previous relevance-ranking behavior.
Unknown source kinds remain eligible for backfill but have no reserved quota.

Python API (used by RAG/Chainlit through the new defaults):

    hits = retriever.search("What is context engineering?", top_k=3)
    hits = retriever.search("Question", min_cosine=0.65, min_bm25=3.0,
                            min_keyword_coverage=0.5)

CLI controls:

    python -m src.retriever search "What is context engineering?" --top-k 3
    python -m src.retriever search "Question" --min-cosine 0.65 --min-bm25 3
    python -m src.retriever search "Question" --global-ranking

stratified=False in Python, or --global-ranking in the CLI, restores the original
global candidate cutoffs/ranking. Diagnostics include selection_reason,
baseline_relevance_met, keyword_coverage and existing raw/rank channel scores.
Rendering and source citation metadata are preserved. No database migration or
re-embedding is required. The chain's separate context-byte budget still applies.

Live verification on the 170-chunk local corpus:
- 'What is context engineering?' changed from slide/handout/handout to
  slide/handout/transcript.
- 'What is the gamified Foundation Certificate?' retained transcript-only results
  because other categories did not qualify for reservations.
- Warm stratified retrieval measured 0.128-0.137 seconds in these two checks.


Transcript context and ranking
------------------------------
Transcript chunks now target 4,500 characters with 900 characters of overlap,
rounded to whole speaker cues/code blocks. The size is a soft limit so code
blocks remain intact. Timestamps and block offsets survive in metadata.
Every chunk starts with `[Transcript Segment - Topic keywords: ...]`: these
are frequent content words extracted from that segment after removing speaker
names and conversational stopwords, not a generated factual summary. The header
is included in both embedding input and BM25 text.

Default stratified retrieval multiplies relevant transcript RRF scores by 1.10.
The existing cosine/lexical baseline gates this boost; raw relevance scores and
source reservation thresholds remain unchanged. `--transcript-boost 1` disables
the boost; `--global-ranking` retains the previous unboosted global ranking.
Results expose `rrf_score` and `source_boost` for inspection.

Run `python -m src.retriever ingest` to refresh the transcript parser cache and
replace old transcript vectors while reusing unchanged slides/handouts. The
standalone materials CLI also supports `--transcript-max-chars 4500` and
`--transcript-overlap-chars 900`. Oversized individual blocks still must fit
the embedding model input limit; they are deliberately not cut through code.


Slide preview relevance
-----------------------
Slide previews require the specific slide chunk to have raw dense cosine
similarity >= 0.75. RRF scores, BM25 matches, transcript boosts and source
reservations cannot qualify an image. Missing/non-finite similarity (including
keyword-only search) suppresses previews. The same check runs during retrieval,
RAG image collection and Chainlit rendering. Lower-confidence slide text can
still be retrieved and cited; its image is omitted. No re-ingestion is needed.


Local cross-encoder reranking
----------------------------
Search now builds a diverse pool of up to 15 hybrid candidates, then uses
`CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')` to score query/text pairs
in batches of eight on CPU. The process caches one model instance and serializes
inference to avoid concurrent CPU oversubscription. Final selection uses only
cross-encoder relevance, with no source quotas or transcript boost applied after
reranking. Default search and generator output is at most three chunks.

Install requirements.txt in both CLI/UI environments. First use downloads the
model into `.local-db/reranker`; subsequent inference is local without API keys.
Ingestion does not load the model, and existing vectors do not need rebuilding.
The model input is capped at 512 tokens per query/passage pair: long passages
are truncated by the tokenizer. This stage adds CPU latency before generation.

`python -m src.retriever search "Your question" --top-k 3 --rerank-pool 15`

`--min-rerank-score 0` is the default raw-logit cutoff. Scores are neither
probabilities nor cosine similarities; this cutoff is a tunable starting point,
not a calibrated confidence estimate. Fewer than three results (including zero)
can survive, in which case the existing chain can refuse rather than use weak
evidence. `--no-rerank` explicitly restores hybrid-only selection.
`--global-ranking` now controls candidate selection; combine it with
`--no-rerank --transcript-boost 1` for the previous global-only behavior.
Diagnostics keep hybrid `score`, raw `dense_score`, and separate `rerank_score`.
The 0.75 slide image cosine gate still applies independently, and only final
reranked hits can contribute previews.

Reference: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2


After cross-encoder filtering and top-N selection, context order is rank 1,
ranks 3 through N, then rank 2. This puts the strongest chunk first and the
runner-up last. For three chunks the order is 1, 3, 2; zero, one or two chunks
keep their existing order. Scores, metadata and selected membership are unchanged.
Hybrid-only (`--no-rerank`) selection retains its original order.


PDF preview rendering reliability
--------------------------------
PDF slide ingestion renders to `corpus/rendered/` by default before OCR.
Rasterization retries at lower resolution and writes atomically. Per-page
OCR/render errors are recorded and counted, allowing later pages to continue.
Blank text is not embedded; the page image and `needs_ocr` status remain in the
processed record. Failed page caches retry on the next ingestion. The standalone
CLI stages JSONL output and supports `--no-ocr`.

Caches require complete page numbering, correct source/page paths and decodable
images. Retrieval normalizes document/raw JSON metadata, reranks before rendering,
and repairs missing, corrupt or wrong-page previews for selected local PDF hits.
Both slide decks and strongly matched handout pages can provide previews. The
0.75 raw cosine threshold remains independent of cross-encoder scores; transcript
hits and low-confidence visuals cannot force pictures. Existing response/prompt
behavior is preserved, with no native text source cards.


Local semantic response cache
-----------------------------
`src/cache.py` stores successful, non-refused RAG result objects (answers,
citations and image metadata) in `.local-db/cache/semantic.sqlite3`. It uses
SQLite plus the CPU model `sentence-transformers/all-MiniLM-L6-v2`, downloaded
once to `.local-db/semantic-cache-model`. No inference API keys or Ollama calls
are needed for cache matching. These local files are ignored by Git.

Configuration in `.env` (read on every incoming message):

```dotenv
ENABLE_SEMANTIC_CACHE=true
SEMANTIC_CACHE_THRESHOLD=0.95
SEMANTIC_CACHE_TTL_SECONDS=86400
SEMANTIC_CACHE_MAX_ENTRIES=256
```

Set `ENABLE_SEMANTIC_CACHE=false` to immediately bypass all cache/model/SQLite
work for subsequent requests without restarting. Explicit `.env` values take
precedence over process values for these cache settings. When disabled, the
existing retrieval, reranking, streaming and preview behavior is unchanged.

Exact repeats use an indexed SQLite lookup without query embedding. Other
queries require cosine >= the configured threshold and matching numeric/
negation/quoted-code signatures. Long queries use exact matching only. This
is conservative approximate matching: some equivalent paraphrases can still
miss the threshold. The first semantic-model initialization is slower; warm
cache hits avoid retrieval and generation entirely. Refusals/errors are not
stored. Cache problems fail open to the normal streaming pipeline. Writes
run after the final answer is delivered; all SQLite/embedding work uses
worker threads, preserving async token streaming.

Cache namespaces invalidate on corpus files, processed snapshots, Python code,
`.env` and provider/model configuration changes. TTL and a bounded entry count
limit retained history. A change during generation prevents a stale write.
Cached image paths are checked before reuse; missing/corrupt previews cause a
miss so the normal rendering repair can run. Existing image relevance gates
and native image rendering stay in place; no text cards are introduced.

To clear cached answers without touching the corpus, models or vector table:

```powershell
python -m src.cache clear
```

After manually changing the PostgreSQL table outside the ingestion pipeline,
clear the response cache: such external edits do not update local snapshots.
The cache contains local course-derived responses; it should remain private.
