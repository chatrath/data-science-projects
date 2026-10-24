# Local pipeline performance

Measured on this machine (6 physical CPU cores), llama3.1:8b generation and
nomic-embed-text embeddings, with 170 indexed chunks across all source kinds.
Question: What is context engineering? Answer briefly with citations.

| Measurement | Before | After |
|---|---:|---:|
| First retrieval | 3.27 s | 3.03 s |
| Generation | 61.91 s | 50.86 s |
| Context chunks | 5 | 3 |
| Context bytes | 2740 | 2334 |
| Model prompt tokens | 1056 | 973 |
| Generated tokens | 86 | 48 |

Repeated retrieval after query-embedding caching: 0.106 s median
(three runs, range 0.103-0.156 s).

cProfile attributed 3.14 of 3.27 baseline retrieval seconds to query embeddings;
database work plus BM25/RRF took about 0.12 s. Baseline model loading took
10.93 s and optimized loading took
10.02 s. These are individual local runs, not a statistically
controlled benchmark. Prompt-cache/model residency, answer length and CPU load
change timings. Both runs used the same question; the optimized answer passed
citation validation. No general latency or answer-quality guarantee is implied.

The regression suite passes 44 tests, including batching/deduplication, cache
expiry/size, native Ollama options, bounded contexts, citations, OCR/source
coverage and chat image delivery helpers. No re-embedding is required for these
performance changes.
