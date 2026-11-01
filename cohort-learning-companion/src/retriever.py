"""PostgreSQL/pgvector ingestion and BM25 + dense reciprocal-rank retrieval.

python -m src.retriever ingest
python -m src.retriever search "How does hybrid retrieval work?" --top-k 5

BM25 is computed in memory over the stored corpus, suitable for cohort-sized
collections. Dense search uses exact cosine distance in PostgreSQL. Only text
is embedded; visual layouts and citation metadata remain attached to results.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import logging
import time
from threading import Lock
import os
from pathlib import Path
import re
from typing import Iterable

from dotenv import load_dotenv
from openai import APIError
from ollama import ResponseError
import httpx
from .provider import ProviderSettings, create_embedding_client
from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
_INGEST_CONNECTION = ContextVar("ingest_connection", default=False)
# Fixed names keep SQL identifiers separate from user input.
SCHEMA = '''
CREATE TABLE IF NOT EXISTS companion_embedding_config (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    model text NOT NULL, dimensions integer NOT NULL
);
CREATE TABLE IF NOT EXISTS companion_chunks (
    id text PRIMARY KEY,
    dataset text NOT NULL,
    content text NOT NULL,
    content_hash text NOT NULL,
    metadata jsonb NOT NULL,
    embedding vector({dimensions}) NOT NULL
);
CREATE INDEX IF NOT EXISTS companion_chunks_dataset_idx ON companion_chunks(dataset);
-- Legacy collections were OpenAI-only; preserve their identity during migration.
ALTER TABLE companion_embedding_config ADD COLUMN IF NOT EXISTS provider text NOT NULL DEFAULT 'openai';
ALTER TABLE companion_embedding_config ADD COLUMN IF NOT EXISTS base_url text NOT NULL DEFAULT 'https://api.openai.com/v1/';
'''


SLIDE_IMAGE_MIN_COSINE = 0.75
_SLIDE_RENDER_LOCK = Lock()
logger = logging.getLogger(__name__)


def high_confidence_visual(score: object) -> bool:
    """Use the chunk's raw cosine similarity, never its fused/boosted score.

    Missing, malformed and non-finite similarities fail closed.
    """
    return (isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and SLIDE_IMAGE_MIN_COSINE <= score <= 1.0)


def normalize_chunk(value) -> dict | None:

    """Accept hit dictionaries, document-shaped records or serialized JSON.



    Unusable payloads are skipped at retrieval boundaries. JSONL ingestion still

    reports malformed records explicitly rather than silently losing input.

    """

    if isinstance(value, str):

        try:

            value = json.loads(value)

        except (ValueError, TypeError):

            return None

    if not isinstance(value, dict):

        return None

    nested = value.get('metadata', {})

    if not isinstance(nested, dict):

        return None

    text = next((value[k] for k in ('page_content', 'text', 'content') if isinstance(value.get(k), str)), None)

    if text is None:

        return None

    # Decode a serialized source record, not arbitrary JSON/code in course text.

    if text.lstrip().startswith('{'):

        try:

            inner = json.loads(text)

        except ValueError:

            inner = None

        if isinstance(inner, dict) and (inner.get('source') or isinstance(inner.get('metadata'), dict) and inner['metadata'].get('source')):

            normalized = normalize_chunk(inner)

            if normalized is not None:

                text = normalized['text']

                nested = {**normalized['metadata'], **nested}

    metadata = {k: v for k, v in value.items() if k not in {

        'text', 'page_content', 'content', 'metadata', 'score', 'dense_score',

        'dense_rank', 'bm25_score', 'bm25_rank', 'rerank_score', 'rrf_score',

        'source_boost', 'selection_reason', 'baseline_relevance_met', 'keyword_coverage'}}

    metadata.update(nested)

    for key in ('page_number', 'page_start', 'page_end'):

        if isinstance(metadata.get(key), str) and metadata[key].isdigit():

            metadata[key] = int(metadata[key])

    if metadata.get('kind') in (None, '', 'material'):

        source_parts = str(metadata.get('source', '')).replace('\\', '/').split('/')

        if 'page_number' in metadata:

            metadata['kind'] = 'slide'

        elif 'page_start' in metadata and 'pdfs' in source_parts:

            metadata['kind'] = 'handout'

    if metadata.get('kind') == 'slide' and 'page_number' not in metadata and isinstance(metadata.get('page_start'), int):

        metadata['page_number'] = metadata['page_start']

    return {**value, 'text': text, 'page_content': text, 'metadata': metadata,

            'score': value.get('score', 0.0)}


def relevant_source_render_path(hit: dict) -> str | None:

    hit = normalize_chunk(hit)

    if hit is None:

        return None

    metadata = hit['metadata']

    if metadata.get('kind') not in {'slide', 'handout'} or not high_confidence_visual(hit.get('dense_score')):

        return None

    existing = resolve_render_path(metadata)

    source = metadata.get('source')

    page_number = metadata.get('page_number') if metadata['kind'] == 'slide' else metadata.get('page_start')

    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:

        return None

    if not isinstance(source, str):

        return existing

    try:

        source_root = (ROOT / 'corpus' / ('slides' if metadata['kind'] == 'slide' else 'pdfs')).resolve()

        source_path = Path(source.replace('\\', '/'))

        candidates = [source_path] if source_path.is_absolute() else [ROOT / source_path, source_root / source_path]

        parts = source.replace('\\', '/').split('/')

        for index in range(len(parts) - 1):

            if parts[index:index + 2] == ['corpus', source_root.name]:

                candidates.append(source_root.joinpath(*parts[index + 2:]))

        source_path = next((p.resolve() for p in candidates if p.resolve().is_relative_to(source_root) and p.suffix.lower() == '.pdf' and p.is_file()), None)

    except (OSError, ValueError):

        return None

    if source_path is None:

        return existing

    with _SLIDE_RENDER_LOCK:

        try:

            from .ingestion import render_slide_page, render_path_for, valid_render

            expected = render_path_for(source_path, page_number, ROOT / 'corpus' / 'rendered')

            if expected.is_file() and valid_render(expected):

                return str(expected.resolve())

            return render_slide_page(source_path, page_number, ROOT / 'corpus' / 'rendered')

        except Exception as exc:

            logger.warning('PDF preview unavailable (%s)', type(exc).__name__)

            return None


def relevant_slide_render_path(hit: dict) -> str | None:

    """Compatibility helper restricted to slide-deck records."""

    hit = normalize_chunk(hit)

    if hit is None or hit['metadata'].get('kind') != 'slide':

        return None

    return relevant_source_render_path(hit)


def resolve_render_path(metadata: dict) -> str | None:

    """Resolve an existing slide preview only within this project's renders.



    Old absolute paths can be rebased after moving the project. Canonical path

    checks prevent serving unrelated files through imported JSONL metadata.

    """

    raw = metadata.get('render_path')

    if not isinstance(raw, str) or not raw.strip():

        return None

    root = (ROOT / 'corpus' / 'rendered').resolve()

    normalized = raw.replace('\\', '/')

    candidates = [Path(normalized)] if Path(normalized).is_absolute() else [ROOT / normalized, root / normalized]

    parts = normalized.split('/')

    for index in range(len(parts) - 1):

        if parts[index:index + 2] == ['corpus', 'rendered']:

            candidates.append(root.joinpath(*parts[index + 2:]))

    for candidate in candidates:

        try:

            candidate = candidate.resolve()

            if candidate.is_relative_to(root) and candidate.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} and candidate.is_file():

                from .ingestion import valid_render

                if valid_render(candidate):

                    return str(candidate)

        except (OSError, ValueError):

            continue

    return None


def tokenize(text: str) -> list[str]:
    """Keep code identifiers intact; apply identical normalization to queries."""
    return re.findall(r'\w+', text.casefold())


@dataclass
class Chunk:
    id: str
    dataset: str
    content: str
    content_hash: str
    metadata: dict


def read_jsonl(paths: Iterable[Path]) -> tuple[list[Chunk], list[str], int]:

    chunks, datasets, skipped = [], [], 0

    for path in paths:

        path = Path(path).resolve()

        dataset = str(path)

        if dataset in datasets:

            raise ValueError(f'Duplicate input: {path}')

        datasets.append(dataset)

        seen = set()

        with path.open(encoding='utf-8-sig') as stream:

            for number, line in enumerate(stream, 1):

                if not line.strip():

                    continue

                try:

                    record = json.loads(line)

                except json.JSONDecodeError as exc:

                    raise ValueError(f'{path.name}:{number}: invalid JSON') from exc

                normalized = normalize_chunk(record)

                if normalized is None:

                    raise ValueError(f'{path.name}:{number}: expected an object with string text')

                record = {**normalized['metadata'], 'text': normalized['text']}

                if isinstance(normalized.get('id'), str):

                    record['id'] = normalized['id']

                if not isinstance(record.get('source'), str) or not record['source'].strip():

                    raise ValueError(f'{path.name}:{number}: missing source')

                text = record['text']

                if not text.strip():

                    skipped += 1

                    continue

                # Line identity is stable across reruns; removed lines are pruned

                # within this input dataset when its replacement is committed.

                identity = str(record.get('id') or f"{record['source']}:{record.get('page_number', number)}")

                chunk_id = hashlib.sha256(f'{dataset}\0{identity}'.encode()).hexdigest()

                if chunk_id in seen:

                    raise ValueError(f'{path.name}:{number}: duplicate record identity')

                seen.add(chunk_id)

                metadata = {k: v for k, v in record.items() if k != 'text'}

                metadata.setdefault('kind', 'slide' if 'page_number' in record else 'material')

                metadata['jsonl_source'] = dataset

                metadata['jsonl_line'] = number

                chunks.append(Chunk(chunk_id, dataset, text, hashlib.sha256(text.encode()).hexdigest(), metadata))

    return chunks, datasets, skipped


class BM25Index:
    """Okapi BM25 with positive Robertson IDF, k1=1.5 and b=0.75."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.counts = [Counter(tokenize(row['content'])) for row in rows]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(rows) if rows else 0
        self.df = Counter(token for count in self.counts for token in count)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        if not terms or not self.average:
            return []
        scored = []
        for row, counts, length in zip(self.rows, self.counts, self.lengths):
            score = 0.0
            for token in terms:
                frequency = counts[token]
                if not frequency:
                    continue
                df = self.df[token]
                idf = math.log(1 + (len(self.rows) - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / self.average))
            if score > 0:
                scored.append((row['id'], score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]


def fuse(keyword: list[tuple[str, float]], dense: list[tuple[str, float]], weight: float = 0.5, rrf_k: int = 60) -> list[dict]:
    """Weighted reciprocal rank fusion; raw scores are diagnostics only."""
    if not 0 <= weight <= 1 or rrf_k < 1:
        raise ValueError('BM25 weight must be in [0, 1] and rrf_k positive')
    scores = defaultdict(float)
    details = defaultdict(dict)
    for label, results, channel_weight in [('bm25', keyword, weight), ('dense', dense, 1 - weight)]:
        if channel_weight == 0:
            continue
        for rank, (chunk_id, raw_score) in enumerate(results, 1):
            scores[chunk_id] += channel_weight / (rrf_k + rank)
            details[chunk_id].update({f'{label}_rank': rank, f'{label}_score': raw_score})
    return [{'id': key, 'score': value, **details[key]} for key, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


SOURCE_KINDS = ('handout', 'slide', 'transcript')
RELEVANCE_STOPWORDS = frozenset('a an and are as at be by can could do does explain for from how i in is it me of on or our that the their this to was we what when where which who why with would you your tell about'.split())


def category_candidates(ranked: list[tuple[str, float]], lookup: dict, per_source: int) -> list[tuple[str, float]]:
    """Keep each category's allowance in the original comparable global order."""
    counts = Counter()
    selected = []
    for chunk_id, score in ranked:
        kind = lookup[chunk_id]['metadata'].get('kind', 'material')
        if counts[kind] < per_source:
            selected.append((chunk_id, score))
            counts[kind] += 1
    return selected


def relevance(hit: dict, content: str, query: str, min_cosine: float, min_bm25: float, min_keyword_coverage: float) -> tuple[bool, float]:
    terms = set(tokenize(query)) - RELEVANCE_STOPWORDS
    coverage = len(terms.intersection(tokenize(content))) / len(terms) if terms else 0.0
    return (hit.get('dense_score', -1.0) >= min_cosine or
            (bool(terms) and hit.get('bm25_score', 0.0) >= min_bm25 and coverage >= min_keyword_coverage)), coverage


def boost_transcripts(ranked: list[dict], lookup: dict, query: str, factor: float = 1.1, *, min_cosine: float = 0.6, min_bm25: float = 2.0, min_keyword_coverage: float = 0.25) -> list[dict]:
    """Modest RRF multiplier for relevant transcripts; raw relevance is unchanged."""
    if not 1 <= factor <= 1.5:
        raise ValueError('Transcript boost must be in [1, 1.5]')
    result = []
    for hit in ranked:
        row = lookup[hit['id']]
        eligible, _ = relevance(hit, row['content'], query, min_cosine, min_bm25, min_keyword_coverage)
        boost = factor if row['metadata'].get('kind') == 'transcript' and eligible else 1.0
        result.append({**hit, 'rrf_score': hit['score'], 'source_boost': boost, 'score': hit['score'] * boost})
    return sorted(result, key=lambda hit: (-hit['score'], hit['id']))


def stratify_results(ranked: list[dict], lookup: dict, query: str, top_k: int, *, min_cosine: float = 0.6, min_bm25: float = 2.0, min_keyword_coverage: float = 0.25) -> list[dict]:
    """Reserve one qualifying hit per known source; backfill by global RRF.

    If fewer slots than qualifying categories are requested, the strongest RRF
    representatives win. Missing/weak categories consume no reserved slots.
    Thresholds gate reservations, not the global backfill ranking.
    """
    representatives = {}
    annotated = []
    for hit in ranked:
        row = lookup[hit['id']]
        kind = row['metadata'].get('kind', 'material')
        eligible, coverage = relevance(hit, row['content'], query, min_cosine, min_bm25, min_keyword_coverage)
        candidate = {**hit, 'baseline_relevance_met': eligible, 'keyword_coverage': coverage}
        annotated.append(candidate)
        if kind in SOURCE_KINDS and candidate['baseline_relevance_met'] and kind not in representatives:
            representatives[kind] = candidate
    # ranked is globally ordered, so representative choice and ties are stable.
    representative_ids = {hit['id'] for hit in representatives.values()}
    reserved = [hit for hit in annotated if hit['id'] in representative_ids][:top_k]
    reserved_ids = {hit['id'] for hit in reserved}
    selected_ids = set(reserved_ids)
    for hit in annotated:
        if len(selected_ids) >= top_k:
            break
        selected_ids.add(hit['id'])
    return [{**hit, 'selection_reason': 'source_reservation' if hit['id'] in reserved_ids else 'global_rank'}
            for hit in annotated if hit['id'] in selected_ids]


RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
# Raw cross-encoder logit floor, not cosine similarity or probability.
MIN_RERANK_SCORE = 0.5
_RERANK_LOCK = Lock()
_RERANKER = None


def rerank_candidates(query: str, hits: list[dict], top_k: int, min_score: float = MIN_RERANK_SCORE) -> list[dict]:
    """Batch joint query/passage scoring; no source quotas after reranking.

    Scores are raw model logits, not cosine similarities or probabilities.
    The model is loaded lazily once per process and runs locally on CPU.
    Selected chunks put rank 1 first, rank 2 last, and ranks 3+ in between.
    """
    if not math.isfinite(min_score):
        raise ValueError('Minimum rerank score must be finite')
    min_score = max(MIN_RERANK_SCORE, min_score)
    if not hits:
        return []
    global _RERANKER
    with _RERANK_LOCK:
        try:
            if _RERANKER is None:
                from sentence_transformers import CrossEncoder
                import torch
                torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
                _RERANKER = CrossEncoder(RERANK_MODEL, device='cpu', max_length=512,
                                        cache_folder=str(ROOT / '.local-db' / 'reranker'))
            scores = _RERANKER.predict([(query, hit['text']) for hit in hits],
                                      batch_size=8, show_progress_bar=False,
                                      activation_fn=lambda logits: logits)
        except Exception as exc:
            raise RuntimeError('Local reranker failed. Install requirements.txt and ensure the '
                               'MS MARCO model is downloaded (internet is needed on first use). '
                               'Use --no-rerank only to explicitly bypass reranking.') from exc
    if len(scores) != len(hits):
        raise RuntimeError('Local reranker returned an invalid number of scores')
    ranked = []
    for hit, score in zip(hits, scores):
        score = float(score)
        if not math.isfinite(score):
            raise RuntimeError('Local reranker returned a non-finite score')
        if score >= min_score:
            ranked.append({**hit, 'rerank_score': score})
    # Select by relevance before rearranging: strongest evidence at each end.
    # Stable sorting retains hybrid order on equal joint relevance scores.
    selected = sorted(ranked, key=lambda hit: -hit['rerank_score'])[:top_k]
    return selected[:1] + selected[2:] + selected[1:2]


class HybridRetriever:
    def __init__(self, database_url: str | None = None, client=None, settings: ProviderSettings | None = None):
        load_dotenv(ROOT / '.env')
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError('Set DATABASE_URL in the project .env')
        self.ingestion_database_url = database_url or os.getenv('DATABASE_INGEST_URL') or self.database_url
        self.settings = settings or ProviderSettings.from_env()
        self.client = client if client is not None else create_embedding_client(self.settings)
        self.encoding = tiktoken.get_encoding('cl100k_base') if self.settings.provider == 'openai' else None
        self._query_cache = OrderedDict()
        self._query_cache_lock = Lock()

    def _connect(self):
        url = self.ingestion_database_url if _INGEST_CONNECTION.get() else self.database_url
        return psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=10)

    @contextmanager
    def _connect_ingest(self):
        # Context-local selection keeps concurrent searches read-only and retains
        # the existing _connect() interface used by integrations/test doubles.
        token = _INGEST_CONNECTION.set(True)
        try:
            with self._connect() as conn:
                yield conn
        finally:
            _INGEST_CONNECTION.reset(token)

    def _check_config(self, conn):
        config = conn.execute('SELECT provider, base_url, model, dimensions FROM companion_embedding_config WHERE singleton').fetchone()
        expected = (self.settings.provider, self.settings.base_url, self.settings.embedding_model, self.settings.embedding_dimensions)
        if not config or tuple(config[key] for key in ('provider', 'base_url', 'model', 'dimensions')) != expected:
            raise ValueError('Stored embedding provider, endpoint, model or dimensions differ. Use a separate database and re-ingest for the new configuration; existing vectors cannot be mixed.')

    def _validate_texts(self, texts):
        for text in texts:
            # cl100k is specific to the OpenAI default. For other providers, enforce a
            # conservative UTF-8 byte budget instead of using the wrong tokenizer.
            length = len(self.encoding.encode(text, disallowed_special=())) if self.encoding is not None else len(text.encode('utf-8'))
            unit = 'tokens' if self.encoding is not None else 'UTF-8 bytes (conservative local/compatible-provider input budget)'
            if not text.strip() or length > 8191:
                raise ValueError(f'Embedding input is empty or exceeds 8191 {unit}; split the source into smaller contextual chunks')

    def _embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError('Embedding input must not be empty')
        if self.settings.provider == 'ollama':
            if is_query and len(texts) != 1:
                raise ValueError('Query embedding requires exactly one input')
            prefix = ''
            if self.settings.embedding_model.split(':', 1)[0] == 'nomic-embed-text':
                prefix = 'search_query: ' if is_query else 'search_document: '
            texts = [prefix + text for text in texts]
        self._validate_texts(texts)
        original_texts = texts
        texts = list(dict.fromkeys(texts))
        result = []
        # Bound batches to 32 * 8191 tokens for compatible-provider limits.
        batch_size = self.settings.embedding_batch_size
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            if self.settings.provider == 'ollama':
                vectors = [self.client.embed_query(batch[0])] if is_query else self.client.embed_documents(batch)
                if len(vectors) != len(batch):
                    raise ValueError('Ollama returned an incorrect number of embeddings')
            else:
                response = self.client.embeddings.create(model=self.settings.embedding_model, input=batch, dimensions=self.settings.embedding_dimensions, encoding_format='float')
                data = sorted(response.data, key=lambda entry: entry.index)
                if [entry.index for entry in data] != list(range(len(batch))):
                    raise ValueError('Embedding API returned missing or duplicate inputs')
                vectors = [entry.embedding for entry in data]
            for vector in vectors:
                if len(vector) != self.settings.embedding_dimensions or not all(math.isfinite(v) for v in vector) or not any(vector):
                    raise ValueError('Embedding API returned an invalid vector')
                result.append(vector)
        by_text = dict(zip(texts, result))
        return [by_text[text] for text in original_texts]

    def _query_embedding(self, query: str) -> list[float]:
        # Cache only embeddings, never ranked hits: every search still reads the
        # current database, so new/updated sources are immediately searchable.
        now = time.monotonic()
        with self._query_cache_lock:
            cached = self._query_cache.get(query)
            if cached and cached[0] > now:
                self._query_cache.move_to_end(query)
                return cached[1]
            self._query_cache.pop(query, None)
        vector = self._embed([query], is_query=True)[0]
        with self._query_cache_lock:
            self._query_cache[query] = (time.monotonic() + 300, vector)
            self._query_cache.move_to_end(query)
            while len(self._query_cache) > 128:
                self._query_cache.popitem(last=False)
        return vector

    def ingest(self, paths: Iterable[Path]) -> dict:
        chunks, datasets, skipped = read_jsonl(paths)
        if not datasets:
            raise ValueError('No input files supplied')
        self._validate_texts([chunk.content for chunk in chunks])
        with self._connect_ingest() as conn:
            # Extension installation may require DBA privileges on a managed DB.
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
            with conn.transaction():
                # Serialize ingestion; failed embedding calls roll back the full update.
                conn.execute("SELECT pg_advisory_xact_lock(780194217)")
                conn.execute(SCHEMA.format(dimensions=self.settings.embedding_dimensions))
                conn.execute('INSERT INTO companion_embedding_config(singleton, provider, base_url, model, dimensions) VALUES (true, %s, %s, %s, %s) ON CONFLICT DO NOTHING', (self.settings.provider, self.settings.base_url, self.settings.embedding_model, self.settings.embedding_dimensions))
                self._check_config(conn)
                previous = {row['id']: row['content_hash'] for row in conn.execute('SELECT id, content_hash FROM companion_chunks WHERE dataset = ANY(%s)', (datasets,))}
                changed = [chunk for chunk in chunks if previous.get(chunk.id) != chunk.content_hash]
                vectors = iter(self._embed([chunk.content for chunk in changed]))
                for chunk in chunks:
                    if previous.get(chunk.id) == chunk.content_hash:
                        conn.execute('UPDATE companion_chunks SET metadata=%s WHERE id=%s', (Jsonb(chunk.metadata), chunk.id))
                        continue
                    conn.execute('''INSERT INTO companion_chunks(id, dataset, content, content_hash, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(id) DO UPDATE SET content=EXCLUDED.content, content_hash=EXCLUDED.content_hash,
                        metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding''',
                        (chunk.id, chunk.dataset, chunk.content, chunk.content_hash, Jsonb(chunk.metadata), Vector(next(vectors))))
                # Synchronize only explicitly supplied datasets, leaving others intact.
                deleted = conn.execute('DELETE FROM companion_chunks WHERE dataset = ANY(%s) AND NOT (id = ANY(%s))', (datasets, [c.id for c in chunks])).rowcount
        return {'stored': len(chunks), 'embedded': len(changed), 'reused': len(chunks) - len(changed), 'deleted_stale': deleted, 'skipped_empty': skipped,
                'by_kind': dict(Counter(chunk.metadata.get('kind', 'material') for chunk in chunks)),
                'unique_embedding_inputs': len({c.content for c in changed}),
                'embedding_batches': math.ceil(len({c.content for c in changed}) / self.settings.embedding_batch_size)}

    def search(self, query: str, top_k: int = 3, candidates: int = 25, bm25_weight: float = 0.5, *, rerank: bool = True, rerank_pool: int = 15, min_rerank_score: float = MIN_RERANK_SCORE, stratified: bool = True, transcript_boost: float = 1.1, min_cosine: float = 0.6, min_bm25: float = 2.0, min_keyword_coverage: float = 0.25) -> list[dict]:
        if rerank_pool < 1 or not math.isfinite(min_rerank_score):
            raise ValueError('Require positive rerank pool and finite minimum rerank score')
        if not query.strip() or not tokenize(query):
            raise ValueError('Query must contain searchable text')
        if top_k < 1 or candidates < top_k or not 0 <= bm25_weight <= 1:
            raise ValueError('Require 1 <= top_k <= candidates and BM25 weight in [0, 1]')
        if not -1 <= min_cosine <= 1 or not math.isfinite(min_bm25) or min_bm25 <= 0 or not 0 <= min_keyword_coverage <= 1:
            raise ValueError('Require cosine threshold in [-1, 1], positive finite BM25 threshold and keyword coverage in [0, 1]')
        if not 1 <= transcript_boost <= 1.5:
            raise ValueError('Transcript boost must be in [1, 1.5]')
        with self._connect() as conn:
            register_vector(conn)
            with conn.transaction():
                conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                self._check_config(conn)
                rows = conn.execute('SELECT id, content, metadata FROM companion_chunks ORDER BY id').fetchall()
                if not rows:
                    return []
                lookup = {row['id']: row for row in rows}
                keyword = BM25Index(rows).search(query, len(rows) if stratified else candidates) if bm25_weight > 0 else []
                if stratified:
                    keyword = category_candidates(keyword, lookup, candidates)
                dense = []
                if bm25_weight < 1:
                    vector = Vector(self._query_embedding(query))
                    if stratified:
                        # One vector and one SQL round trip, with independent candidate
                        # slots per category so a dominant source cannot hide others.
                        dense_rows = conn.execute('''SELECT id, distance FROM (
                            SELECT id, embedding <=> %s AS distance,
                                   row_number() OVER (
                                       PARTITION BY COALESCE(metadata->>'kind', 'material')
                                       ORDER BY embedding <=> %s, id
                                   ) AS source_rank
                            FROM companion_chunks
                        ) ranked WHERE source_rank <= %s ORDER BY distance, id''', (vector, vector, candidates)).fetchall()
                    else:
                        dense_rows = conn.execute('SELECT id, embedding <=> %s AS distance FROM companion_chunks ORDER BY embedding <=> %s, id LIMIT %s', (vector, vector, candidates)).fetchall()
                    dense = [(row['id'], 1 - float(row['distance'])) for row in dense_rows]
        ranked = fuse(keyword, dense, bm25_weight)
        if stratified:
            ranked = boost_transcripts(ranked, lookup, query, transcript_boost, min_cosine=min_cosine, min_bm25=min_bm25, min_keyword_coverage=min_keyword_coverage)
        pool_size = max(top_k, rerank_pool) if rerank else top_k
        selected = stratify_results(ranked, lookup, query, pool_size, min_cosine=min_cosine, min_bm25=min_bm25, min_keyword_coverage=min_keyword_coverage) if stratified else ranked[:pool_size]
        hits = [{**hit, 'text': lookup[hit['id']]['content'], 'metadata': lookup[hit['id']]['metadata'],
                 'render_path': resolve_render_path(lookup[hit['id']]['metadata'])}
                for hit in selected]
        hits = [chunk for hit in hits if (chunk := normalize_chunk(hit)) is not None and chunk['text'].strip()]
        hits = rerank_candidates(query, hits, top_k, min_rerank_score) if rerank else hits
        for hit in hits:
            hit['render_path'] = relevant_source_render_path(hit)
            if hit['render_path']:
                hit['metadata']['render_path'] = hit['render_path']
        return hits


def database_error_message(exc: psycopg.Error) -> str:
    """Explain common setup failures without disclosing connection secrets."""
    if isinstance(exc, psycopg.errors.InvalidPassword):
        return 'PostgreSQL authentication failed. Check the username and password in DATABASE_URL.'
    if isinstance(exc, psycopg.errors.InvalidCatalogName):
        return 'The configured database does not exist. Run .\\start_database.ps1 for the project-local database or create the database named in DATABASE_URL.'
    if isinstance(exc, psycopg.errors.InsufficientPrivilege):
        return 'PostgreSQL permission denied. The ingestion role needs CREATE on the database/schema and access to the companion tables; a DBA may need to enable vector first.'
    if isinstance(exc, psycopg.errors.UndefinedTable) or isinstance(exc, psycopg.errors.UndefinedColumn):
        return 'Retrieval tables are missing or use an older schema. Run python -m src.retriever ingest before search.'
    if isinstance(exc, (psycopg.errors.FeatureNotSupported, psycopg.errors.UndefinedObject)):
        return 'PostgreSQL vector support is unavailable. Install pgvector on this server and enable CREATE EXTENSION vector in the target database.'
    if isinstance(exc, psycopg.OperationalError):
        return 'Cannot connect to PostgreSQL. For the project-local server, run .\\start_database.ps1, then retry. Check DATABASE_URL host and port for other servers.'
    return f'PostgreSQL operation failed (SQLSTATE {exc.sqlstate or "unknown"}). Check the server log for details.'


def format_search_results(results: list[dict]) -> str:
    """Render complete chunk text without changing code indentation."""
    if not results:
        return 'No matching chunks found.'
    entries = [f'Search results: {len(results)}']
    for rank, hit in enumerate(results, 1):
        source = str(hit.get('metadata', {}).get('source') or 'Unknown source')
        filename = source.replace('\\', '/').rsplit('/', 1)[-1] or 'Unknown source'
        entries.append('\n'.join([
            '=' * 72,
            f'Result {rank}',
            f'Source: {filename}',
            f"Type: {hit.get('metadata', {}).get('kind', 'unknown')}",
            f'Score (hybrid RRF): {hit["score"]:.6f}',
            *([f"Score (cross-encoder logit): {hit['rerank_score']:.4f}"] if 'rerank_score' in hit else []),
            '',
            'Chunk text:',
            hit['text'],
        ]))
    return '\n\n'.join(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest', help='Parse all raw corpus sources and synchronize their embeddings')
    ingest.add_argument('files', type=Path, nargs='*', help='Optional preprocessed JSONL files; supplying these bypasses raw parsing')
    ingest.add_argument('--corpus', type=Path, default=ROOT / 'corpus')
    ingest.add_argument('--processed-only', action='store_true', help='Use existing JSONL snapshots without refreshing raw sources')
    search = sub.add_parser('search', help='Run hybrid retrieval')
    search.add_argument('query')
    search.add_argument('--top-k', type=int, default=3)
    search.add_argument('--no-rerank', action='store_true', help='Explicitly bypass local cross-encoder reranking')
    search.add_argument('--rerank-pool', type=int, default=15)
    search.add_argument('--min-rerank-score', type=float, default=MIN_RERANK_SCORE, help='Minimum raw cross-encoder logit (floor 0.5); not a probability')
    search.add_argument('--candidates', type=int, default=25)
    search.add_argument('--bm25-weight', type=float, default=0.5)
    search.add_argument('--transcript-boost', type=float, default=1.1, help='Relevant transcript RRF multiplier, 1 to 1.5; 1 disables')
    search.add_argument('--global-ranking', action='store_true', help='Disable source reservations and use the previous global ranking')
    search.add_argument('--min-cosine', type=float, default=0.6, help='Cosine similarity needed for a source reservation')
    search.add_argument('--min-bm25', type=float, default=2.0, help='BM25 score needed for a lexical source reservation')
    search.add_argument('--min-keyword-coverage', type=float, default=0.25, help='Fraction of meaningful query words required for lexical qualification')
    args = parser.parse_args()
    try:
        retriever = HybridRetriever()
        if args.command == 'ingest':
            raw_report = None
            if args.files:
                paths = args.files
            elif args.processed_only:
                paths = [args.corpus / 'slides.jsonl', args.corpus / 'materials.jsonl']
            else:
                from .corpus_ingestion import prepare_corpus
                print('Refreshing handouts, slides (local OCR), and transcripts...', flush=True)
                paths, raw_report = prepare_corpus(args.corpus)
                for kind, stats in raw_report.items():
                    print(f"{kind}: {stats['files']} files, {stats['chunks']} searchable chunks, {stats['empty']} empty, {stats['ocr_pages']} OCR pages", flush=True)
            result = retriever.ingest(paths)
            if raw_report is not None:
                result['raw_sources'] = raw_report
        else:
            result = retriever.search(args.query, args.top_k, args.candidates, args.bm25_weight, rerank=not args.no_rerank, rerank_pool=args.rerank_pool, min_rerank_score=args.min_rerank_score, stratified=not args.global_ranking, transcript_boost=args.transcript_boost, min_cosine=args.min_cosine, min_bm25=args.min_bm25, min_keyword_coverage=args.min_keyword_coverage)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    except (ResponseError, httpx.HTTPError, ConnectionError):
        parser.exit(1, 'Local/compatible provider request failed. For Ollama, start ollama serve, pull the configured models, and check OLLAMA_BASE_URL.\n')
    except APIError:
        parser.exit(1, 'Provider API request failed. Check the selected provider key, base URL, activated model, and account quota.\n')
    except psycopg.Error as exc:
        parser.exit(1, database_error_message(exc) + '\n')
    if args.command == 'search':
        print(format_search_results(result))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
