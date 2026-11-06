# Multi-Agent NL-Ops MVP

A Python 3.10+ Streamlit helpdesk with two LangChain agents and a compiled
LangGraph workflow: input → intent/safety → SQL generation → SQLGlot validation → EXPLAIN plan inspection
→ human SQL approval → execution → response formatting → audit logging. Blocked and failed stages route directly
to the response. Results are formatted locally without sending rows to the LLM.

## Setup (PowerShell)

```powershell
cd "D:\Gen Academy\Projects\AI projects\nl-ops-mvp"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

This MVP uses local Ollama models with no OpenAI account or API key.
The UI has no sidebar; each request returns up to 50 rows by default.
Tracing, caching, and SMTP are configured outside the UI through `.env` and CLI utilities.
The installed model store is `D:\Ollama\models`. The existing Ollama service
serves these models at `http://127.0.0.1:11434`.

```powershell
ollama list
# Only if the Ollama service is not already running:
# $env:OLLAMA_MODELS = "D:\Ollama\models"
# ollama serve
python database.py
streamlit run app.py
```

The default `.env` configuration is:

```dotenv
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=nl-ops-db-mvp
```

Installed chat alternatives are `llama3.1:8b` and `llama3:latest`.
`nomic-embed-text:latest` is an embedding model and cannot serve these chat agents.
Set `OLLAMA_MODEL` to switch models, then restart the app. Agents use Ollama
JSON-schema structured output and a 120-second timeout for local model loading.
`python-dotenv` loads `.env`; existing environment variables take precedence.
SQLite is built into Python and has no pip requirement.
Replace the LangSmith key placeholder to enable trace uploads. Without a real
key the app continues with tracing inactive. Set `LANGCHAIN_TRACING_V2=false`
to explicitly disable tracing.

## Viewing Your Runs in LangSmith

1. Log in to [smith.langchain.com](https://smith.langchain.com/), select your
   workspace, and create an API key under Settings → API Keys. This is a
   **LangSmith key**, separate from model-provider credentials; Ollama needs no key.
2. Put the four `LANGCHAIN_*` settings shown above in `.env`, replacing the
   placeholder key. US accounts use `https://api.smith.langchain.com`; EU
   accounts use `https://eu.api.smith.langchain.com`. Keep the key private.
   If you use `LANGSMITH_*` aliases, they take priority over `LANGCHAIN_*` names.
3. Fully stop and restart Streamlit after changing settings. `config.py` loads
   the project `.env` before LangChain/LangGraph imports for both the UI and
   direct graph use. The server log reports the tracing configuration and destination project without keys.
   The UI has no tracing controls; use the LangSmith console directly.
4. Run `python tracing.py` from the project folder to verify tracing. The check uploads a small synthetic verification run,
   reads it back, and displays a run ID on success. Configured status alone
   does not confirm credentials or remote delivery. If verification fails,
   check the key's workspace permissions, endpoint/region, and network access.
   HTTP 401/403 means access was rejected: generate a LangSmith key in the
   target workspace with trace write/create and run read permissions. Use the
   LangSmith key rather than a model-provider key. After editing `.env`, stop
   the Streamlit server process and start it again; refreshing the browser
   does not reload cached server credentials. Never paste your key into chat.
5. In project-based workspaces, open **Tracing / Projects** and find
   **nl-ops-db-mvp** (or your custom `LANGCHAIN_PROJECT`). It is normally created
   when the first trace is ingested. Search for **LangSmith tracing verification**
   to find the smoke test. If your account instead uses the newer agent-based
   workspace UI, follow the [agent addressing guide](https://docs.langchain.com/langsmith/log-traces-to-agent):
   create an agent/environment and use its `tracer_session_name` compatibility
   value as `LANGCHAIN_PROJECT`.
6. Submit a helpdesk question, wait for completion, and open the
   **NL-Ops Helpdesk Request** run. Its expandable visual trace tree shows
   input, safety, SQL generation, SQLite execution, and response nodes. Expand
   **Intent & Safety Agent** and **SQL Generation Agent** to inspect prompts,
   model responses, execution latency, errors, and input/output token counts
   when reported by Ollama. Blocked requests stop before SQL generation.
   The verification run itself contains no model calls or token counts.

LangSmith's trace details and project dashboards show token usage and cost
breakdowns when usage and model pricing are available. Local Ollama inference
does **not** draw from OpenAI credits. Costs for this local model may be blank
or zero unless custom pricing is configured; LangSmith observability billing
is separate and depends on your plan. See the official
[cost tracking guide](https://docs.langchain.com/langsmith/cost-tracking).
Tracing sends workflow inputs, SQL, results, and metadata to LangSmith.
Use `LANGCHAIN_TRACING_V2=false` when you want to keep those traces local.

The app also initializes the database automatically. It creates
`mock_ enterprise.db` beside `database.py` (the space in the filename is
intentional). Seeding is idempotent and preserves records. Customer dates are
relative to the initial seed date; delete the mock database to regenerate dates.
“Last week” means the previous Monday through Sunday. Europe maps to the UK,
Germany, and France represented in this dataset.

## Try it

- Show me all users who signed up in Europe last week.
- Show average salary by department.
- List open IT tickets with employee names.
- Delete all customers. (Should be blocked.)

Run from Python after initializing the database:

```python
from graph import start_request, resume_request

thread_id, state = start_request("Show employee 1's name", row_limit=50)
if state["status"] == "awaiting_approval":
    print(state["sql"])  # Present this to the reviewer.
    # Call only after an explicit human decision:
    state = resume_request(thread_id, "approve", state["approval_token"])
print(state["status"], state["message"])
```

## Local caching

At startup, after loading `.env` and before importing agents, `cache.py` calls
`set_llm_cache(SQLiteCache(database_path=...))`. The supported current import
is `from langchain_community.cache import SQLiteCache`; the older
`langchain.cache` import is unavailable in LangChain 1.x. LLM generations persist
in `.langchain_cache.db` beside the app. Keys include the exact rendered prompt
and model settings. Non-streaming model calls enable caching; graph node status
updates still stream to the UI. Both agents use this cache, including direct
`graph.invoke()` calls.

The approval-enabled UI deliberately does not call the workflow-result cache:
every new request is validated and reviewed before SQL execution. The exact
result-cache utility remains available to other callers, but must not wrap a
paused graph because a pending approval is not a completed request.
LLM-response caching remains enabled; it does not skip validation or approval.

This is **exact caching, not embedding-based semantic caching**. Paraphrases may
miss; differences in names, numbers, negation, case, or quoted values are never
silently merged. A semantic cache would require a separate embedding index and
accuracy validation before safely reusing SQL results. Time-relative answers
can be up to five minutes old; clear caches when immediate freshness matters.

The result panel reports **result cache**, **live agents**, or **agents / LLM
cache**, with actual LLM cache-hit counts and elapsed time. Result-cache hits
bypass the graph, so they do not generate a new agent trace in LangSmith.
To reset both local caches from the project folder, run:

```powershell
python cache.py
```

After editing synthetic tables, schemas, or prompts, clear both caches to force
a fully live test. Reset uses DELETE operations, preserving usable SQLite files
and open connections. Cached prompts and results remain local and are ignored
by Git. Cache reuse is designed for this local single-user synthetic-data MVP.

## XML execution plans

Plan inspection exports `execution_plan` as an XML document rooted at
`SQLiteQueryPlan`, with SQLite version, SQL statement, and nested `Operator`
elements. Each operator retains its node ID, parent ID, auxiliary value, and
original detail, plus an access classification. Parent-child relationships are
preserved; SQL and operator text are XML escaped. No costs or actual row counts
are invented. This is the application's own **estimated** plan format, not SQL
Server ShowPlan XML or a measured actual execution plan.

The UI shows a single **Download SQL Plan** hyperlink without inline XML/JSON.
Only XML plans are generated; companion SQL scripts are no longer created or served.
Plans persist under
`plan_artifacts/`, named by content hash. A loopback-only download server starts
with the UI on port 8502, serves only these files, and sends attachment headers
so browsers download rather than render them. New audit entries store `plan_links`
instead of raw XML. The audit table displays short clickable labels. DBA emails
contain matching HTML hyperlinks and plain-text URLs, without inline plan XML
or attachments. Old audit records and captured messages remain unchanged on disk;
their raw plan markup is hidden in the UI. Legacy XML attachments can be exported
as linked files automatically.

The downloaded XML can be inspected by an XML-capable editor. It is a SQLite
estimated plan, not an SSMS `.sqlplan`. No query is executed by downloading it.

Defaults are `PLAN_SERVER_PORT=8502` and
`PLAN_BASE_URL=http://127.0.0.1:8502`. Keep the UI running while following links.
These local links work only on this computer, including links in captured test
emails. Real remote DBA recipients would need an authenticated, reachable
artifact host and an appropriate `PLAN_BASE_URL`; this MVP's unauthenticated
loopback server must not be exposed as a production file-sharing service.
Cache resets preserve saved plan files. No artifact retention policy is applied.

## DBA email alerts for all failed requests

Every blocked or errored workflow request passes through `notify_failure`,
which calls `send_dba_alert(sql_query, execution_plan, reason)` before the audit
record is written. This includes invalid input, database/schema availability,
intent/safety refusals or errors, SQL generation errors, AST rejections/errors,
plan rejections/errors, and database execution failures. Audit-write failures
are checked after audit persistence and also trigger an alert if none has yet
been attempted. Unexpected UI/workflow exceptions use the same failure helper.
Successful executions and successful cache hits do not trigger alerts.

Emails include UTC timestamp, request ID, raw user prompt, generated SQL when
available, and the failure reason. When a plan exists, a **Download SQL Plan**
link points to the saved XML. Earlier-stage failures explicitly say SQL/plan is
unavailable rather than attaching a fake plan. Email wording does not assume
that the request failed before execution, since audit errors can occur afterward.

Configure `.env`, then fully restart Streamlit:

```dotenv
SMTP_SERVER=127.0.0.1
SMTP_PORT=1025
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=nl-ops@enterprise-mock.com
DBA_EMAIL_RECIPIENT=dba-alerts@enterprise-mock.com
SMTP_STARTTLS=false
```

The bundled local SMTP capture service requires no account or password. Start
it in a separate terminal and leave it running:

```powershell
cd "D:\Gen Academy\Projects\AI projects\nl-ops-mvp"
.\.venv\Scripts\python.exe local_smtp.py
```

Then start Streamlit, submit **Show all employees**, and expand **DBA Email
Inbox (Local Test)**. Click **Refresh local inbox** to see the captured alert
with its SQL, plan download links, reason, and prompt. The capture server listens only on
`127.0.0.1:1025`, writes durable `.eml` files under `local_mailbox/`, and never
forwards messages externally. An SMTP `Sent` result in this mode means captured
locally, not delivered to a real DBA inbox. Restart this service after rebooting
your machine. You can open the `.eml` files in an email client too.

Localhost permits blank credentials and defaults to no TLS. Remote SMTP
requires both credentials and defaults to STARTTLS; configure a real recipient
before production delivery, replace SMTP_SERVER/SMTP_PORT, and set
`SMTP_STARTTLS=true` for a STARTTLS provider. Keep credentials private in `.env`. No email is
sent just by starting the application or opening its UI.

Sending runs inline with a five-second timeout per SMTP socket operation.
Missing remote credentials return `Skipped`; connection, TLS, authentication,
or configuration errors return `Failed` and log a safe diagnostic. The query
remains blocked and the UI continues normally. `Sent` means the SMTP server
accepted the message, not that inbox delivery was confirmed. The audit record
and UI report delivery status; the audit record retains links to saved plan files.
There is no retry queue. Each failed request attempts one alert, including
when notification delivery itself fails; the final audit-failure check does
not retry an already attempted notification. A new submission is a new request.
If an audit file cannot be written, the email attempt is still made, but the
same file cannot persist its outcome until the storage issue is resolved.

## Enterprise audit logging

Every graph run ends at the `audit` node, including safety, syntax, plan, and
execution rejections. `audit_logger.py` appends a JSON object per line to
`enterprise_audit.log` beside the application, flushing and fsyncing each write.
Records contain UTC ISO request timestamps, unique request IDs, the original
untrimmed natural-language input, safety status, generated SQL, separate AST
and EXPLAIN statuses, execution outcome, and returned row count.

Successful runs are `Success`; result-cache submissions append a separate
`Cached` record with a fresh request ID/timestamp and inherited validation
statuses. Cache hits do not rerun those checks. Blocked SQL/plans are
`Blocked-by-Guardrail`; safety refusals are `Blocked-by-Safety`. Invalid input
and operational failures are labeled `Invalid-Request` and `Error` rather than
misrepresented as guardrail refusals. Unreached checks are `NotRun`, and failed
checks are `Error`. Row counts represent returned/cached rows, capped by the
request limit, not the total possible match count.

Expand **Enterprise Audit Logs** in Streamlit to view the latest 100 records
with separate AST/EXPLAIN columns; use **Refresh audit logs** to reload. The
viewer skips malformed JSON lines without discarding valid records. Clearing
caches preserves audit history. Audit writes that fail are reported in the UI;
a query may already have executed before an end-of-run audit failure.

The file is ignored by Git and persists across app restarts. It contains raw
requests and SQL, but no result rows or API credentials. This MVP uses a local
file and an in-process lock, without authentication, cross-process locking,
retention rotation, or tamper evidence. Production compliance would require
appropriate access controls and a managed append-only audit destination.

## Native LLM tool calling

Both agents bind the decorated functions in `tools.py` using
`ChatOllama.bind_tools(...)`. This preserves the installed local model and
requires no OpenAI API key. The configured model must support native function
calling (`qwen3:4b-instruct` does).

- `get_table_metadata(table_name)` returns real SQLite column names/types,
  primary-key/nullability flags, and a row count for an allowlisted table.
  It returns no employee/customer data rows. Schema inspection uses a trusted
  read-only connection; the COUNT uses the existing authorized reader and time
  budget. The identifier is checked before interpolation.
- `validate_sql_syntax(sql_query)` parses SQLite SQL with SQLGlot and returns
  syntax validity, errors, and a separate read-only/table policy decision. It
  never executes SQL. Valid syntax alone does not imply permitted SQL.

Inside `graph.py`, each agent invokes a compiled tool subgraph:
`agent → tools (ToolNode) → agent`. `tools_condition` routes native tool calls;
ToolNode dispatches the selected Python function and feeds matching ToolMessages
back into the conversation. Final JSON is validated with the existing Pydantic
schemas. The safety agent can inspect metadata when needed. The SQL agent must
successfully inspect metadata and lint a candidate before finalizing; if it
tries to skip those calls, the graph requests the missing tools rather than
silently accepting an unverified answer.

The loop allows at most six model turns, eight tool calls total, and four calls
in one model response. Tool/argument errors return sanitized error messages to
the model. Budget exhaustion or malformed final JSON fails closed through the
normal failure/audit/DBA-alert path. Tool results do not replace the independent
AST validator, EXPLAIN inspection, or read-only execution boundary. No execution
tool is exposed to the model.

Tool subgraphs and tool runs appear under the named agent traces in LangSmith.
The UI result cache still skips the full graph on a hit; model-cache misses or
direct graph runs may inspect current metadata before generating SQL. Changes
to `tools.py` invalidate saved workflow results. Live evaluation redirects tool
metadata reads to its disposable database, while offline evaluation substitutes
reference agent outputs. `tests/test_tool_calling.py` exercises actual tool
dispatch with scripted native model messages, including full guarded execution,
unauthorized metadata requests, unknown tools, and loop-limit behavior.

## Structured schema context

`schema_context.py` contains `SCHEMA_CATALOG`, with business descriptions,
column types and meanings, and explicit foreign-key mappings. Its data types
match SQLite (for example, salary is REAL and signup_date is ISO-formatted TEXT).
Update this catalog whenever the database schema or business definitions change;
tests compare its columns and foreign keys with a newly initialized database.

`get_relevant_schema(user_query)` matches whole entity keywords and returns a
scoped JSON block for the SQL Generation Agent. Employee/salary/department
requests include only `employees`; customer/user/signup requests include only
`customers`. Ticket requests include `tickets` and its referenced `employees`
table so joins have both sides and the foreign-key mapping. Multiple entity
matches are combined; unmatched vocabulary falls back to the full small catalog.
This deterministic heuristic is not semantic retrieval or an access-control rule.

The SQL generation node injects this structured context instead of raw DDL;
the original database schema remains available to the Intent & Safety Agent.
`schema_context` is retained in workflow state and LangSmith traces for review.
Changing the catalog invalidates result-cache entries; changed context also
changes SQL-agent LLM cache keys. No new dependency is required.

## SQL validation guardrail

`guardrails.validate_sql(sql_query, allowed_tables)` returns `(allowed, message)`.
It parses SQLite SQL with SQLGlot, requires a single SELECT root, and checks all
physical table references, including joins and nested subqueries, against the
server-owned allowlist (`employees`, `customers`, `tickets`). Schema-qualified
access is restricted to `main`. Stacked statements, writes, SELECT INTO, CTEs,
set operations, table-valued functions, oversized SQL, and overly large ASTs
are rejected. Comments and string literals are parsed as SQL rather than checked
with keyword matching.

The `validate_sql_node` runs after `sql_generation` and appears
in LangSmith traces. Rejections go directly to the response node, which displays
the validator's reason in the UI. Unexpected validator errors fail closed.
SQLite dry run and authorization still enforce executable syntax, function
restrictions, and read-only access: SQLGlot is not a full SQLite engine and does
not validate column existence or provide a complete production security boundary.
Changing guardrail source also invalidates workflow result-cache entries.

Run all tests, including guardrail and graph-routing coverage:

```powershell
python -m pytest tests -q
```

Pytest disables trace uploads for tests. The validator suite includes valid
SELECTs, DROP rejection, unauthorized tables, malformed SQL, nested table
access, statement stacking, and other bypass attempts.

## Boundaries and errors

Before execution, `inspect_query_plan_node` opens a managed read-only connection
with the same table/function authorizer and timeout as query execution. It calls
`inspect_query_plan(connection, sql)` to compile `EXPLAIN QUERY PLAN`, fetch the
plan rows, and close the cursor without running the submitted SELECT. Failed
inspection routes directly to the response and displays an optimization warning.
The node and its decision also appear in LangSmith. Execution no longer repeats
the old dry run because this inspection already compiles and authorizes the SQL.

The MVP rejects unindexed SCAN operations on core tables/aliases without a
WHERE condition referencing that table's columns, and rejects temporary b-tree
sorting/grouping. It accepts indexed SEARCH operations, index-backed scans,
constant queries, and filtered scans without temporary sorts. Filters are
checked within each SELECT scope; a comment or unrelated nested filter does
not exempt an outer scan. A LIMIT alone does not exempt an unfiltered scan.

This intentionally strict heuristic applies even to the tiny synthetic tables.
Broad questions such as "list all employees", aggregate/grouping requests, or
unindexed sorting may now be blocked. Try "Show employee with id 1" for an
indexed query. No index is automatically created and rejected SQL is not retried.
A filter's presence does not prove selectivity, and plan approval is not a
guarantee of optimal runtime. SQLite supplies plan descriptions rather than
reliable row-count/cost estimates; plan text can change across SQLite versions.
See [SQLite EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html).

The LLM safety decision is advisory. SQLite independently enforces a read-only
connection, query-only mode, a table/function authorizer, one SELECT statement,
a maximum of 200 returned rows, and an approximately two-second SQLite VM
execution budget. Dry run uses `EXPLAIN QUERY PLAN` with the same authorizer.
CTEs, PRAGMA, system tables, extensions, and write operations are unsupported.
Queries that fail validation, authorization, or execution limits return a warning;
Ollama failures stop the workflow with a service/model hint. Details go to server logs.
The time budget is cooperative, not a process-level memory or wall-clock limit.

This local synthetic-data MVP has no authentication, user permissions, ticket
persistence, or production database integration. Keep it on synthetic data.
Tracing, when enabled, can send requests, SQL, and graph state (including results)
to LangSmith. Disable it when such transmission is inappropriate.

API references: [LangGraph StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
and [ChatOllama structured output](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama/with_structured_output).

## Enterprise pipeline evaluation

`eval_dataset.json` defines ten cases: four safe requests with golden results,
DELETE/DROP/UPDATE attacks, an unauthorized-table request, and two performance
edge cases (unfiltered scan and temporary sort). Golden outcomes use a fresh
disposable database seeded by `database.py`, including the country index.
The application database is never modified by the evaluator, and its temporary
evaluation database is checked for unexpected changes after all runs.

Run **live model evaluation** with your configured local Ollama service:

```powershell
.\.venv\Scripts\python.exe eval_pipeline.py --report eval_reports/latest_live.json
```

Live mode invokes the actual compiled graph and both agents. It disables the LLM
cache during the evaluation and bypasses the UI result cache so requests exercise
the pipeline. LangSmith traces, when configured, are tagged `enterprise-eval`
with case IDs. The evaluator captures DBA notification attempts without sending
email by default; add `--send-alerts` only when real SMTP delivery is intended.
Alert-attempt coverage is separate from delivery success.

For a repeatable check without Ollama, run:

```powershell
.\.venv\Scripts\python.exe eval_pipeline.py --offline --report eval_reports/latest_offline.json
.\.venv\Scripts\python.exe -m pytest test_enterprise_eval.py -v
```

Offline mode supplies reference intent decisions and SQL from the dataset while
running the real graph, SQLGlot validator, EXPLAIN inspection, execution, alert
trigger path, and JSON audit writer. It is labeled OFFLINE in the report and
does **not** measure LLM accuracy. The pytest wrapper defaults to offline and
includes negative checks for wrong goldens, hallucinated tables, and missing
audit records. Enable the separate live pytest test explicitly:

```powershell
$env:RUN_LIVE_EVAL = "1"
.\.venv\Scripts\python.exe -m pytest test_enterprise_eval.py -v -s
```

The report includes security block success, destructive AST block rate, table
grounding precision, table hallucination rate, golden execution accuracy, audit
completeness, per-case checks, and latency. Destructive natural-language requests
may be refused by the Intent & Safety Agent before SQL exists; those are valid
end-to-end safety blocks. **Separate direct AST probes** verify DELETE, DROP,
UPDATE, and unauthorized-table rejection without bypassing the intent agent in
live runs. The report does not falsely label an early safety refusal as an AST
node interception. Generated table references must exist both in the injected
schema context and the real catalog. Table grounding does not claim that every
column or business interpretation is correct; golden row checks measure the
selected known outcomes, including values, projection, count, and ordering.

Each workflow run must have exactly one persisted audit entry matching its
request ID, original prompt, and generated SQL. By default evaluation records
are appended to the app's `enterprise_audit.log`. Use `--audit-log PATH` to
isolate them. Pytest uses temporary logs and never uploads test traces or sends
real email. Live evaluation should run as its own process, not from a Streamlit
thread, because database/notification bindings are temporarily substituted.

Overall PASS requires every case to pass, 100% security/AST/grounding/audit/golden
rates, and an unchanged evaluation database. CLI exit codes are 0 for PASS,
1 for metric failures, and 2 for invalid setup. These targets describe this small
test dataset; they are not guarantees for all possible prompts.

## Local checks (no API key required)

```powershell
python -m pytest tests test_enterprise_eval.py -q
```


## Checkpointed human approval (Week 3)

There is one canonical business workflow in `graph.py`. `enterprise_graph` is an
alias for `graph`, not a separate flow or state contract. Both use `WorkflowState`
and accept `request`/`row_limit`. `build_graph(checkpointer=MemorySaver())` is useful
for isolated tests; MemorySaver only survives within the running process.
The default graph uses `SqliteSaver` and `.workflow_checkpoints.db` so pending
requests survive server restarts. Keep that file private and persistent.

The flow is input → safety → targeted schema retrieval → generation → SQL policy
validation → EXPLAIN plan → prepare approval → interrupt → execution → response
→ failure notification when applicable → terminal audit. Correctable parser and
SQLite errors return to generation, with a maximum of three failed corrections.
Security and performance policy rejections stop without executing SQL.

Each submission has a fresh `thread_id` which is also its request ID. Streamlit
retains that ID in session state, reads the saved snapshot on reruns, and shows
Approve and execute, Reject request, or Validate revised SQL. Edited SQL returns
to policy and plan checks, then pauses for a new approval token. Three human edits
are permitted. Approval applies to the exact SQL in that review, and execution
rechecks policy and the plan after the pause. Rejection is a terminal audited
outcome and does not dispatch a database-failure alert.

For low-level LangGraph use, compile with a checkpointer, supply
`config={"configurable": {"thread_id": request_id}, "recursion_limit": 64}`,
and resume using `graph.invoke(Command(resume={"action": "approve",
"approval_token": token}), config)`. Use the same thread ID. Do not pass fresh
input on resume, `Command(goto=...)`, or `update_state` to bypass validation.
Prefer `start_request` and `resume_request`, which serialize decisions locally and
make duplicate terminal submissions no-ops. These helpers return checkpoint state
rather than a raw interrupt payload. A stale review token is rejected.

Audit appends are idempotent by request ID in this single-process application.
No external effects occur before `interrupt()` in the approval node. SQL execution
is read-only; a process crash between execution and its checkpoint can replay that
read. The implementation does not claim transactional exactly-once database IO.
For multiple server workers, use an appropriate shared checkpointer, authenticated
reviewer/request ownership, distributed coordination, and transactional audit
storage. The local Streamlit reviewer is not an enterprise authentication system.

A new browser session after server restart needs the original thread ID to recover
a pending request through the Python runner; browser session state itself is not
stored by the graph. Do not expose unrestricted thread-ID lookup in a public UI.
The evaluation harness uses a clearly labelled simulated reviewer against its
temporary database. It never disables the application approval gate.


## Recovery lifecycle and failure evidence

`run_state` describes request lifecycle independently of the terminal `status`:

| run_state | Meaning | Next action |
| --- | --- | --- |
| running | Work is in progress | Read the checkpoint |
| awaiting_approval | Validated SQL is waiting for review | Approve, reject, or edit |
| failed_recoverable | A temporary failure has saved progress | Retry saved request or cancel |
| finished | Completed, rejected, blocked, or exhausted | Display the terminal status |

`get_request_state(thread_id)` reads these states without executing anything.
Use `recover_request(thread_id, recovery_token=state["recovery_token"])` to retry
a saved transient failure. Use `action="cancel"` to stop it. This is separate
from `resume_request`, which is exclusively for SQL review. Neither operation
can authorize the other's pause. Recovery tokens change on each failure so old
browser decisions cannot replay a retry. Reruns never retry automatically.
There are at most three explicit recovery retries for the whole request, separate
from SQL correction, tool-call, and human-edit budgets. A new generation always
requires its own validation and approval, even after a prior execution failure.

The recovery interrupt retains the failed node, pre-failure status, SQL, review
history, and verified tool transcript. Retrying audit persistence only writes the
audit; it never executes SQL again. Unexpected escaped task failures are recorded
as failures of that task and routed into the same recovery protocol by the runner.
For an interrupted native checkpoint between nodes, `recover_request(thread_id)`
resumes with `invoke(None, config)` instead of passing fresh initial input.
Successful upstream nodes are not replayed. If the checkpoint store itself fails,
the runner raises instead of claiming progress was saved.

`workflow_errors.py` classifies timeout, model_format, tool, database, configuration,
SQL syntax, and infrastructure errors. Security/performance policy rejections stay
terminal. Missing-column/syntax failures trigger SQL correction; database locks
trigger checkpoint recovery instead of needless SQL regeneration. Audit records
include error categories, chronological failure history, and recovery counts.
Audit JSON serialization errors are infrastructure/data failures, not model errors;
original file IO failures propagate to the graph for classified recovery. A torn
trailing JSONL record is separated before writing a complete replayed record.

Structured-output repair receives the full preceding tool conversation, including
metadata, tool-call IDs, failed tool messages, and the exact SQL validation result.
Model timeouts carry serialized tool state through the outer SQLite checkpoint;
recovery preserves already verified metadata and the agent's tool/turn budgets.
Tool results remain untrusted data, and repaired SQL still passes the independent
SQL guardrails and human review before execution.
