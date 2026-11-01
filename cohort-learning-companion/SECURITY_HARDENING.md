# Local RAG security hardening

Implemented 29 October 2026. This note supersedes the corresponding pre-hardening findings in SECURITY_EVALUATION_REPORT.md.

## Trusted local browser origins

`app.py` installs `src/security.py` middleware before Chainlit starts, including when using `python -m chainlit run app.py` directly. The outer ASGI guard covers both HTTP and mounted Socket.IO/WebSocket routes. HTTP CORS uses the same local-host pattern; no site-packages files are modified.

Accepted browser origins are exactly `http://localhost` or `http://127.0.0.1`, optionally with ports 1-65535. Other hosts, subdomains, credentials, paths, null origins, duplicate Origin headers, and non-HTTP schemes are rejected. Browser requests from trusted local ports keep working. Requests without Origin remain supported for local CLI, health checks and existing Socket.IO clients. This is origin protection, not authentication: non-browser local programs can omit/spoof Origin. Keep the server bound to loopback; do not expose it publicly without authentication and deployment-level access control.

## Database credential separation

Run the normal `start_database.ps1` or `python -m src.local_database start`. Startup retains the initialized database and extension, then idempotently provisions:

- `DATABASE_URL`: `cohort_runtime`, a non-superuser with SELECT on the two corpus tables, no INSERT/UPDATE/DELETE access, and default read-only transactions.
- `DATABASE_INGEST_URL`: `cohort_ingest`, a non-superuser that owns only the application tables and can create application objects in public. The existing ingestion schema maintenance requires table ownership. This role has no create-database/create-role powers.
- `.local-db/bootstrap.env`: administrator URL used only by the local database helper. It is not loaded by the chat application's dotenv configuration and remains ignored by Git.

`HybridRetriever` keeps its existing constructor and `_connect()` interface. Ingestion selects its connection through context-local routing, so concurrent reads retain the runtime URL. Explicit `database_url=` overrides both roles as before. If `DATABASE_INGEST_URL` is blank, legacy single-URL behavior remains available; that fallback does not itself guarantee least privilege. Do not configure an administrator as the runtime URL.

Existing rows, stable IDs, embedding configuration and vectors are preserved. Existing corpus table ownership transfers to the ingestion role. Future tables created by that role receive runtime SELECT grants. New database initialization still runs extension installation with the bootstrap role. For a managed/remote database, have its administrator grant equivalent permissions and set the two URLs; the local helper continues to refuse overwriting a remote database configuration.

Bootstrap credentials and the private pre-change snapshot live under `.local-db/`; protect that folder and `.env` with OS permissions. The current single-user application process can still read local files, and `.env` contains ingestion credentials. This separation protects runtime SQL permissions; it is not a sandbox against total compromise of the OS user. Stronger deployments should supply ingestion credentials only to a separate ingestion process, and keep bootstrap credentials under a separate administrative identity.

## Context and citation handling

`src/chain.py` escapes reserved SOURCE CHUNK marker lines and known model role delimiters in retrieved text. Metadata is normalized to one line with control characters removed and header delimiters escaped. Normal multiline text and code blocks remain unchanged. A literal reserved marker inside code is escaped rather than treated as a new source; original stored chunks/metadata remain untouched.

Citation normalization and validity checks now ignore fenced code (backticks or tildes), inline code, indented code, Markdown blockquotes, and paired straight/curly quotation spans. Literal citation examples are preserved verbatim. An answer whose only citations occur in those spans cannot pass validation; active citations outside them still work. Quoted evidence should be cited after the closing quote.

These are structural safeguards. They do not prove claim entailment, enforce every formatting rule, or eliminate semantic prompt injection. Provisional streaming remains enabled and can display text before final validation.

## Request admission

`MAX_CONCURRENT_REQUESTS=2` defaults to two active chat requests per Python process; valid configured values are 1-8. Excess requests receive a short busy message immediately, with no waiting queue. Each chat session can run only one question at a time. Slots are released in finally on success, error and cancellation. The cap covers retrieval, on-demand preview repair, generation, and optional cache work in the handler. It does not throttle trusted standalone ingestion commands or impose a requests-per-minute quota.

Single-user requests do not wait on a new queue and tokens stream as before. For multi-worker deployments, use shared admission controls: this process-local counter is not a global service quota. Cancellation cannot forcibly terminate already-running Python thread/native work; use worker isolation if hostile resource-exhaustion resistance is required.

## Validation

The original 181 tests were retained without modification. New synthetic security tests cover HTTP/WebSocket rejection, trusted local origins, CORS installation, context marker/header escaping, code/quote citation isolation, role routing/restoration and provisioning, request saturation, session overlap, errors and cancellation.

Live checks verified SELECT-only runtime privileges, denied runtime UPDATE, ingestion access without superuser powers, and preservation/reuse of all 384 indexed chunks. Trusted local HTTP and WebSocket origins connected; untrusted and null origins were rejected. Final validation: **205/205 tests passed** in both `.venv-ui` (Python 3.13) and `.venv` (Python 3.14), including all original 181 tests plus 24 new security regressions. Live Qwen chat delivered 126 token events, separate Markdown answer sections, three cited sources, and three image elements whose URLs returned valid PNG content. Database startup was repeated successfully after migration; the incremental updater reported zero new/modified files and no index changes.

A private rollback snapshot was taken before changes under `.local-db/security-backups/`. No commit or push is performed automatically.
