"""Reversible, local semantic response cache. Disable with ENABLE_SEMANTIC_CACHE=false."""
from __future__ import annotations

import asyncio
from array import array
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from threading import Lock
import time

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
_MODEL = None
_MODEL_LOCK = Lock()
_CACHE = None
_FACTORY_LOCK = Lock()


def local_embedding(query: str) -> list[float]:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from sentence_transformers import SentenceTransformer
            import torch
            torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
            _MODEL = SentenceTransformer(MODEL, device='cpu', cache_folder=str(ROOT / '.local-db/semantic-cache-model'))
        return _MODEL.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


def namespace() -> str:
    """Invalidate on source, ingestion output, code, config or provider changes."""
    digest = hashlib.sha256(MODEL.encode())
    for path in sorted([ROOT / 'app.py', ROOT / '.env', ROOT / 'requirements.txt', *ROOT.joinpath('src').glob('*.py')]):
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    for folder in ['pdfs', 'slides', 'transcripts']:
        for path in sorted((ROOT / 'corpus' / folder).rglob('*')):
            if path.is_file():
                stat = path.stat()
                digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    for name in ['slides.jsonl', 'materials.jsonl']:
        path = ROOT / 'corpus' / name
        if path.is_file():
            stat = path.stat()
            digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    # Values are hashed only; credentials are never stored in the cache key.
    for key, value in sorted(os.environ.items()):
        if key.startswith(('OLLAMA_', 'RAG_', 'NEBIUS_', 'OPENAI_')) or key in {'DATABASE_URL', 'LLM_PROVIDER', 'ENABLE_AGENT_GRAPH'}:
            digest.update(f'{key}={value}'.encode())
    return digest.hexdigest()


def signature(query: str) -> str:
    """Avoid obvious false equivalences involving numbers, negation or literals."""
    terms = re.findall(r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|not|no|never|without|except|don't|doesn't|cannot|can't)\b", query.casefold())
    literals = re.findall(r'`[^`]+`|"[^"]+"|https?://\S+', query)
    return json.dumps([sorted(terms), sorted(literals)])


def normalized(vector) -> list[float]:
    result = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in result))
    if not result or not math.isfinite(norm) or norm == 0:
        raise ValueError('Invalid semantic cache embedding')
    return [v / norm for v in result]


def successful(result) -> bool:
    return (isinstance(result, dict) and not result.get('refused', False)
            and isinstance(result.get('answer'), str) and bool(result['answer'].strip())
            and isinstance(result.get('sources'), list) and bool(result['sources']))


def images_available(result: dict) -> bool:
    from .retriever import resolve_render_path
    return all((isinstance(image, (str, dict)) and resolve_render_path(
                   {'render_path': image if isinstance(image, str) else image.get('path')}))
               for image in result.get('slide_images', []))


@dataclass
class CacheLookup:
    namespace: str
    result: dict | None = None
    embedding: list[float] | None = None
    similarity: float | None = None


class SemanticCache:
    def __init__(self, path: Path | None = None, *, threshold: float = .95,
                 ttl: float = 86400, max_entries: int = 256, embedder=local_embedding,
                 namespace_factory=namespace):
        if not .0 <= threshold <= 1 or not math.isfinite(threshold) or ttl <= 0 or not math.isfinite(ttl) or max_entries < 1:
            raise ValueError('Invalid semantic cache settings')
        self.path = Path(path) if path is not None else ROOT / '.local-db/cache/semantic.sqlite3'
        self.threshold, self.ttl, self.max_entries = threshold, ttl, max_entries
        self.embedder, self.namespace_factory = embedder, namespace_factory

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=1)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS responses (
                namespace TEXT NOT NULL, query TEXT NOT NULL, signature TEXT NOT NULL,
                vector BLOB NOT NULL, result TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(namespace, query))''')
        except Exception:
            db.close()
            raise
        return db

    async def lookup(self, query: str) -> CacheLookup:
        return await asyncio.to_thread(self._lookup, query.strip())

    def _lookup(self, query: str) -> CacheLookup:
        lookup = CacheLookup(self.namespace_factory())
        cutoff = time.time() - self.ttl
        def decoded(payload):
            try:
                result = json.loads(payload)
                return result if successful(result) and images_available(result) else None
            except (ValueError, TypeError, KeyError, OSError):
                return None
        with closing(self._open()) as db:
            row = db.execute('SELECT result FROM responses WHERE namespace=? AND query=? AND created>?',
                             (lookup.namespace, query, cutoff)).fetchone()
            if row is not None:
                lookup.result = decoded(row[0])
                if lookup.result is not None:
                    lookup.similarity = 1.0
                    return lookup
            rows = db.execute('SELECT vector, result FROM responses WHERE namespace=? AND signature=? AND created>? AND query<>?',
                              (lookup.namespace, signature(query), cutoff, query)).fetchall()
        if not rows or len(query.split()) > 128:
            return lookup
        lookup.embedding = normalized(self.embedder(query))
        candidates = []
        for vector, payload in rows:
            try:
                stored = normalized(array('d', vector))
                if len(stored) != len(lookup.embedding):
                    continue
                similarity = sum(a * b for a, b in zip(stored, lookup.embedding))
                if similarity + 1e-12 >= self.threshold:
                    candidates.append((similarity, payload))
            except (ValueError, TypeError):
                continue
        for similarity, payload in sorted(candidates, key=lambda pair: -pair[0]):
            result = decoded(payload)
            if result is not None:
                lookup.result, lookup.similarity = result, similarity
                break
        return lookup

    async def store(self, query: str, result: dict, lookup: CacheLookup) -> None:
        if successful(result):
            await asyncio.to_thread(self._store, query.strip(), result, lookup)

    def _store(self, query: str, result: dict, lookup: CacheLookup):
        if self.namespace_factory() != lookup.namespace:
            return  # Corpus/settings changed while the answer was being generated.
        vector = lookup.embedding or normalized(self.embedder(query))
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with closing(self._open()) as db, db:
            db.execute('DELETE FROM responses WHERE created<=?', (time.time() - self.ttl,))
            db.execute('INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?)',
                       (lookup.namespace, query, signature(query), array('d', vector).tobytes(), payload, time.time()))
            db.execute('DELETE FROM responses WHERE rowid IN (SELECT rowid FROM responses ORDER BY created DESC LIMIT -1 OFFSET ?)', (self.max_entries,))

    async def clear(self) -> None:
        await asyncio.to_thread(self._clear)

    def _clear(self):
        with closing(self._open()) as db, db:
            db.execute('DELETE FROM responses')


def get_semantic_cache() -> SemanticCache | None:
    """Read the toggle on every request; .env edits take effect without restart."""
    global _CACHE
    values = dotenv_values(ROOT / '.env')
    def value(key, default):
        return str(values.get(key) or os.getenv(key, default)).strip()
    if value('ENABLE_SEMANTIC_CACHE', 'true').lower() in {'0', 'false', 'no', 'off'}:
        return None
    settings = (float(value('SEMANTIC_CACHE_THRESHOLD', '.95')),
                float(value('SEMANTIC_CACHE_TTL_SECONDS', '86400')),
                int(value('SEMANTIC_CACHE_MAX_ENTRIES', '256')))
    with _FACTORY_LOCK:
        if _CACHE is None or (_CACHE.threshold, _CACHE.ttl, _CACHE.max_entries) != settings:
            _CACHE = SemanticCache(threshold=settings[0], ttl=settings[1], max_entries=settings[2])
    return _CACHE


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['clear'])
    parser.parse_args()
    asyncio.run(SemanticCache().clear())
    print('Local semantic response cache cleared; corpus and models untouched.')
