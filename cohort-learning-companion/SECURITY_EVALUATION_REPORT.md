# RAG Security and Evaluation Report — Post-hardening Review

Review date: 30 October 2026. Application: Cohort Learning Companion.

Scope: current application code/configuration, local role privileges, HTTP/WebSocket origin behavior, deterministic synthetic probes, retained evaluation artifacts, and the regression suite. No application code, configuration, credentials, or corpus files were changed during this review. Synthetic PDF/worker probes used temporary data only. This report supersedes the earlier pre-hardening assessment; its prior copy is retained under ignored `.local-db/security-review-history/`.

**Assessment:** the application has materially improved structural safeguards and is reasonable for a trusted, single-user localhost workflow. It does not prove factual grounding, complete answers, or resistance to semantic prompt injection. It is not an authenticated multi-user service. The previously confirmed wildcard-origin and runtime-superuser findings are resolved in the observed running configuration.

## 1. System Prompts & Precision

### Verified prompt contracts

`src/chain.py:21-58` defines the shared prompt and exact refusal. The system message requires context-only facts, no guessing or invented code, claim-level citations, preserved code indentation, and honest text-only treatment of displayed PDF images. It permits combining/paraphrasing supported facts and qualifies “better” comparisons instead of assuming universal superiority. It instructs the model to ignore instructions embedded in excerpts and prohibits user overrides of grounding.

The human template puts reference excerpts and the user question before a final reminder of citation, formatting, and whole-request refusal rules. System instructions have their own message role; source data is not turned into executable code or a new system-role message by the application.

For multi-part requests, the prompt requests numbered `###` sections, one or two short bullets per part, and at most 45 words per part. These are model instructions, not fully enforced output-schema constraints.

### Executable controls

- `src/chain.py:62-80`: metadata controls/newlines/header characters are escaped, known SOURCE CHUNK marker lines are escaped even with alternative line separators, and known model role-delimiter tokens are escaped. Ordinary multiline text/code survives. This is syntactic hardening, not a complete semantic injection filter.
- `src/chain.py:122-144`: bounded whole chunks retain source metadata, question-part provenance, chunk IDs, and raw ranking scores. Oversized chunks are skipped rather than cut through code.
- `src/query_planning.py:12-49`: explicit questions are preserved and enumeration wording is normalized. Pronouns/quoted syntax conservatively fall back to model planning.
- `src/agent_nodes.py:104-134`: model plans use structured JSON and validate query count, uniqueness, and size, with fallback to the original question. Explicit parsing supports up to five parts; model-based decomposition remains limited to two or three queries.
- `src/agent_nodes.py:137-246`: per-part retrieval deduplicates evidence and uses adaptive limits: up to two accepted chunks per part, six overall by default. Question-heading prioritization does not admit candidates rejected by the reranker.
- `src/agent_nodes.py:262-308`: context reserves representatives for planned parts before extras, keeps allocation diagnostics, and detects when an entire planned part lacks packed evidence, when part provenance is present.
- `src/chain.py:281-303`: active citation IDs are normalized, checked against supplied sources, and used to retain only cited sources/previews. Code/quotation masking is applied before ID recognition.
- `app.py:37-80`: existing headings get clean spacing; fenced code is preserved; source labels are escaped before the Markdown footer.

### Precision gaps confirmed by probes

The finalizer still accepts an unsupported claim if it carries a valid source ID. It also accepts an uncited invented claim followed by a separately cited sentence, and an answer missing required multi-part headings. It does not inspect whether a claim is entailed by the cited chunk, whether every factual claim is cited, or whether all requested facts have been answered.

The improved code/quote isolation rejects citations located only in fenced/inline code and ordinary single-line quoted examples. **Residual parser edges remain:** a citation inside a multiline straight-quoted string, and a Markdown blockquote lazy-continuation line without `>`, were accepted as active citations in synthetic checks. The implementation is a regex-based approximation, not a complete Markdown syntax tree.

The per-part coverage check means “a retrieved candidate associated with this planned query survived packing.” It does not mean that the text sufficiently answers that part. A broad/generic chunk can match multiple sub-queries; a model plan can also omit a requirement from the original question. Structural coverage therefore does not establish semantic completeness.

### Recommendations

Prioritize a labeled answerability/faithfulness evaluation set over assuming that more prompt wording provides a guarantee. Use a Markdown parser or an explicit structured claim/citation representation to identify active citations and render answer sections. Validate required section counts and per-part statuses where practical. Semantic support still needs evidence-aware evaluation, selective review, or human labels; deterministic citation syntax alone cannot establish factuality. Keep expensive evaluators offline or restricted to flagged cases to avoid restoring unnecessary latency and false refusals.

## 2. Refusal Prompt Strictness

### Enforced behavior

The canonical application refusal is exactly:

> I could not find this in the course materials.

`src/chain.py:281-303` returns this string for empty output, absent sources, missing active citations, unknown recognized source IDs, or a terminal standalone canonical refusal appended to a partial answer. Refused outputs have no sources or previews. The conventional chain skips generation when packed context is empty (`src/chain.py:305-323`).

`src/agent_nodes.py:249-259` uses a local nonblank-evidence gate. It is implemented and active, but is not an LLM sufficiency/faithfulness grader. `src/agent_nodes.py:311-328` refuses before generation when evidence is empty, the gate is negative, or context packing leaves a planned part uncovered. `src/agent_graph.py:17-40` retries empty retrieval once, then terminates. No post-answer factual correction or approval loop exists.

The cross-encoder floor remains 0.5 (`src/retriever.py:526-570`). This is a raw relevance logit, not an answerability probability. High relevance does not prove that a requested fact exists in the chunk.

### Remaining strictness gaps

- Out-of-scope detection is indirect: retrieval filtering plus the generator prompt. There is no independent out-of-scope or claim-entailment classifier.
- An unsupported answer with a valid active ID passes finalization.
- A paraphrased refusal with a valid active ID passes as an ordinary answer; the finalizer does not canonicalize arbitrary refusal meanings.
- The whole-request rule is conservative: a missing part can cause a full refusal even when other parts are answerable. Budget exhaustion and true absence of facts can reach the same user-visible fallback.
- Literal source IDs in multiline/lazy quote forms can still defeat the intended active-citation distinction.

Fresh synthetic checks confirmed rejection of canonical refusal, uncited output, unknown IDs, code-only IDs, and ordinary quote-only IDs. Unsupported-but-cited claims, paraphrased refusals, and structurally incomplete answers were accepted. These are validator probes, not a live claim that Qwen generated those fabricated answers.

### Streaming caveat

`src/chain.py:314-336`, `src/agent_graph.py:66-81`, and `app.py:220-240` stream provisional text before final validation. A later refusal replaces the visible completion, but a client can retain earlier tokens. The system cannot guarantee that unchecked or unsupported text is never displayed.

Recommendation: represent answerability/refusal as explicit status data and render the canonical string deterministically. Pair this with per-part evaluation; a status field alone does not prove support. If zero exposure of unchecked output is required, buffer or validate before release and accept the latency tradeoff. Preserve legitimate quoted/code examples rather than expanding refusal detection into indiscriminate text matching.

## 3. Application Security & Vulnerability Check

### Resolved/verified controls

| Control | Code location | Current verification |
|---|---|---|
| Trusted local origins for HTTP and WebSocket | `src/security.py:13-58`; `app.py:25`; `.chainlit/config.toml:24` | Trusted localhost/127.0.0.1 origins accepted; untrusted and null HTTP origins returned 403; untrusted WebSocket Origin rejected. |
| Restricted database connections | `src/local_database.py:42-79`; `src/retriever.py:574-600` | Runtime role: no superuser/create-db/create-role; SELECT true and INSERT/UPDATE/DELETE false on corpus table. Ingestion role: non-superuser with required writes. Effective default retriever connection also confirmed non-superuser. |
| Bounded active requests | `src/security.py:61-89`; `app.py:172-188` | Two active requests per process by current configuration; per-session overlap rejected; no application waiting queue. Tests cover saturation/error/cancellation cleanup. |
| SQL/process safety | `src/retriever.py:670-742`; `src/local_database.py:23-30,61-75` | Parameterized data queries, quoted role/database identifiers, validated numeric schema settings, and subprocess argument lists without `shell=True`. No model-output-to-shell/eval tool path found. |
| Context/header hardening | `src/chain.py:62-80` | Synthetic forged source marker escaped; metadata controls are normalized; code is retained. |
| Local preview containment | `src/retriever.py:262-310`; `src/previews.py:16-97` | Raster path containment, source folder/PDF/page validation, chunk/citation matching for exact-page route, corruption repair, duplicate removal, and fail-closed UI revalidation. |
| Browser/data exposure reduction | `.chainlit/config.toml`; `app.py:30-34,71-80` | Uploads, raw HTML, MCP, sharing, and automatic chat-history persistence disabled; source footer metadata escaped. |

Live HTTP checks accepted requests without Origin for CLI compatibility, accepted trusted local origins, and rejected an untrusted HTTPS origin and `null`. HTTP CORS echoed only the trusted tested origins. WebSocket checks independently confirmed trusted acceptance and untrusted rejection; the earlier framework-level WebSocket gap is now covered by the outer ASGI middleware.

### Residual findings and priorities

| Finding | Priority / condition | Evidence and recommendation |
|---|---|---|
| Semantic prompt injection remains possible | High priority for answer integrity when queries/corpus are adversarial | Escaping known delimiters does not isolate arbitrary natural-language instructions. Raw source content still enters the model; valid-ID checks do not prove compliance. Add adversarial corpus/query evaluations and evidence-aware review; do not claim injection-proof operation. |
| No user authentication/authorization | Required before shared/public exposure | `/auth/config` reports no password/header/OAuth login. Originless local clients can connect by design. Loopback origins are trusted across ports. Add authentication, document authorization and host/deployment controls before exposing beyond one trusted local user. Origin validation is not identity validation. |
| Cancellation can outlive admission control | Medium availability risk under hostile/repeated cancellation | A synthetic handler cancelled during `asyncio.to_thread` work released its request slot while the native worker continued. The limiter bounds handlers, not necessarily all outstanding CPU/native work. Use a bounded worker pool or retain capacity until real work finishes; add budgets/rate limits and worker isolation for untrusted workloads. |
| Preview/index/source versions are not bound together | Medium integrity risk if files change without synchronization | `src/previews.py:48-66` checks path/page/decodability, not source-content digest. A synthetic changed PDF reused an old cached frame; after removing that cached frame, a different frame regenerated under the same old chunk ID. Store/verify source revision hashes with chunks and previews, or serve immutable indexed snapshots. Re-ingestion remains necessary after changes. |
| Markdown links/images are not application-filtered | Conditional privacy/social-engineering risk | A synthetic remote Markdown image passed finalization. Raw HTML disabling is separate from Markdown URLs. No browser network request or XSS exploit was demonstrated. Add a URL/image policy and suitable CSP if stronger outbound restrictions are needed. |
| Process-level secrets remain accessible to the same OS user | Defense-in-depth gap | Bootstrap URL is in ignored `.local-db/bootstrap.env`; `.env` contains ingestion credentials. SQL least privilege is effective, but an OS-user/process compromise can read local files. Supply ingestion credentials only to ingestion processes and isolate bootstrap administration for stronger deployments. |
| Native parsing/model/dependency trust is not fully audited | Hardening gap | PDF/image parsers have no demonstrated hard file/page/pixel/time sandbox. Package ranges and model references are not an immutable lock. Add file/work caps, dependency/advisory scans, model revision pinning and reproducible installs. No dependency CVE-clean claim is made here. |

### Preview policy: important current distinction

Approximate visual matching still requires query cosine >= 0.75 (`src/retriever.py:63-74`). The new exact cited-page route **does not require that cosine threshold**: it requires the actual final cited source, a single identifiable PDF page, accepted reranker score, canonical path/page bounds, and matching chunk identity (`src/previews.py:16-97`; `app.py:98-108`). This is an intentional provenance alternative, not a fabricated cosine boost. The original cosine is retained. A model citation can still be semantically wrong; this route proves which page is displayed, not whether its contents entail the answer.

The UI's legacy string-path branches (`app.py:90-95,109-114`) depend on upstream trusted result construction and do not independently reconstruct every relevance/citation check. They are not currently direct user-controlled chat fields. For defense in depth, standardize typed preview records and validate provenance uniformly before introducing new result producers or re-enabling caches.

### Secrets, logs and local-only limits

Provider keys are excluded from configuration repr (`src/provider.py:22`). Base URLs reject embedded credentials/query strings, but allow administrator-configured HTTP(S) hosts (`src/provider.py:53-56`); selecting Ollama alone is not an enforcement that the endpoint is loopback. Enforce an endpoint allowlist/TLS policy if required. No user-input-controlled endpoint or SQL identifier path was found in the reviewed code.

Local database setup uses generated passwords, SCRAM authentication, and loopback binding. Bootstrap credentials are not loaded by chat dotenv configuration. Legacy single-URL and explicit constructor overrides remain available, so least privilege must also be maintained in configuration; it is not guaranteed for every possible caller-supplied URL.

A scan of reviewed application source/templates found zero exact matches for current configured secret values. Git ignores `.env`, corpus, local bootstrap/runtime/cache data, rendered assets and generated chat files. This does not cover all Git history, unknown secret patterns, filesystem ACLs, encryption, backups, or framework file-serving internals. Ignored files remain sensitive local data.

Application logs generally use exception classes rather than raw credential-bearing errors. Semantic caching is currently disabled; old local cache/evaluation artifacts may still exist. First model setup/initialization can make model-hub requests; no active hosted tracing was observed and no query-text exfiltration was demonstrated. Offline inference/privacy claims should distinguish local inference from model download/metadata network traffic.

## 4. Evaluation & Human-in-the-Loop Frameworks

### Current test evidence

A fresh run of `.venv-ui/Scripts/python.exe -m unittest discover -s tests -q` passed **232 tests in 7.367 seconds**, across **21 modules**. Earlier validation also passed the same 232-test suite in the CLI environment. No new application changes were made for this audit.

| Coverage area | Test files / counts | Evidence level |
|---|---|---|
| Prompts, plans, graph, refusal/citations | `test_chain.py` (12), `test_agent_nodes.py` (26), `test_agent_graph.py` (18) | Primarily unit/mocked generation and real graph routing |
| Adaptive multi-part coverage | `test_multipart_retrieval.py` (14) | Deterministic planning, quotas, deduplication, context packing and fail-closed omissions |
| Security controls | `test_security.py` (24) | ASGI origin checks, synthetic escaping/citations, role routing/provisioning mocks, admission limits |
| Preview provenance | `test_slide_images.py` (17), `test_cited_previews.py` (4), `test_exact_cited_previews.py` (13) | Synthetic real PDFs/renders plus path, score, wrong-page, repair and UI checks |
| UI/streaming/formatting | `test_app.py` (15), `test_streaming.py` (4), `test_response_markdown.py` (7) | Mocked UI streams and deterministic formatting behavior |
| Retrieval/providers/ingestion/cache | Remaining ten modules (78 tests) | Algorithm/provider mocks and synthetic/local parser/update checks |

No measured line/branch coverage percentage is available; the coverage package/report/config was not found. `pip-audit` was not installed, and no advisory audit was executed. Passing unit tests establish their asserted behavior, not comprehensive live-model factual accuracy or exploit resistance.

Current synthetic audit probes additionally exposed multiline/lazy-quote parsing, worker-cancellation admission limits, and source-version binding gaps not prevented by the passing suite. These should become regression tests when their corresponding behavior is intentionally hardened.

### Existing evaluations and observability

Retained `.local-db/benchmarks/` artifacts include a historical six-case Qwen evaluation, before/after multi-part retrieval records, and end-to-end streaming/citation/PNG checks. The historical `.local-db/evaluate_qwen_prompt.py` uses frozen retrieval states and keyword/refusal/source-presence criteria; it is not a broad fresh-retrieval or claim-entailment benchmark. These helpers/artifacts are ignored by Git and are not a reproducible shared evaluation framework without safe committed fixtures.

`grade_documents` is an active local evidence-presence gate, not a stub and not a factual grader. The graph retries empty evidence only. Context allocation now reports retrieved/packed counts, omitted chunks, byte budgets and unpacked parts (`src/agent_nodes.py:303-308`; `src/agent_graph.py:57`). The active agent still lacks the conventional chain's equivalent complete retrieval/generation/first-token timing instrumentation (`src/chain.py:268-276,325-335`), and there is no evaluation trace schema with structured refusal reasons.

No explicit configured Ragas, DeepEval, TruLens, Phoenix or comparable evaluation integration was found. No active LangSmith tracing flags or application-defined tracing callbacks were observed in the inspected project/process configuration. LangChain can be instrumented externally; absence of explicit code does not categorically prohibit external tracing.

### HITL status

No application-defined reviewer node, human approval gate, LangGraph interrupt, durable checkpoint, adjudication queue or feedback-to-evaluation-dataset workflow was found. `graph.compile()` is called without a checkpointer (`src/agent_graph.py:40`). Manual user checks and frontend feedback controls, if present, do not constitute an integrated approval workflow.

### Recommended evaluation/HITL framework

1. Create a versioned, private-data-safe gold set with question/part IDs, answerability labels, required facts, expected source/page IDs, allowed citations, and expected refusal/format behavior. Include paraphrases and compound questions about the same concept.
2. Include out-of-scope, mixed-supported, contradictory, OCR-noisy, injected-instruction, spoofed-marker, multiline quote, code-only citation, changed-PDF and cancelled-request cases. Keep adversarial fixtures out of the live corpus.
3. Measure retrieval recall@k/MRR or nDCG, per-part semantic evidence coverage, answer faithfulness, citation precision/coverage, refusal precision/recall, injection success rate, preview version/provenance correctness, latency and time to first token. Calibrate raw-logit/cosine thresholds with labeled data rather than presenting them as confidence probabilities.
4. Keep fast unit checks separate from opt-in live integration/evaluation jobs. Record model/revision, prompt hash, corpus version, settings, retrieved IDs/scores, validation/refusal reasons and timings. Redact content by default; make any hosted tracing an explicit privacy decision.
5. Add a human review queue for flagged answers with source evidence and clear supported/unsupported/incomplete labels. Promote reviewed cases into regression fixtures. If human approval must prevent exposure, place approval before publishing tokens; post-answer feedback cannot retract previously observed content.

Audit artifacts: `.local-db/benchmarks/security-audit-2026-10-30.json` contains current deterministic/network probe results. Temporary PDF/native-worker probes changed no real corpus data. This was an assessment only: no security fixes, model changes, commits or pushes were performed.
