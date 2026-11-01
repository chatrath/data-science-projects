"""Incrementally scan local PDFs: python update_corpus.py [--corpus corpus].

Nomic uses F16 model weights. New vector values are rounded to F16 precision;
the existing pgvector vector(768) column remains float32 for compatibility.
BM25 is rebuilt from that same table on every search, including active chats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from dotenv import load_dotenv
import httpx
import psycopg
import tiktoken

from src.corpus_ingestion import atomic_json
from src.ingest_materials import load_handout
from src.ingestion import load_slides, render_slide_page, render_path_for, valid_render
from src.provider import ProviderSettings
from src.retriever import HybridRetriever, BM25Index, database_error_message, read_jsonl

ROOT = Path(__file__).resolve().parent
VERSION = 1


def fingerprint(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def discover_pdfs(corpus: Path) -> list[Path]:
    return sorted(path.resolve() for path in corpus.rglob('*')
                  if path.is_file() and path.suffix.lower() == '.pdf'
                  and path.resolve().is_relative_to(corpus)
                  and not any(part.startswith('.') or part.lower() == 'rendered'
                              for part in path.relative_to(corpus).parts[:-1]))


def token_windows(text: str, size: int, overlap: int):
    """Use a local reference tokenizer; char offsets preserve Unicode exactly.

    Counts use cl100k_base, not Nomic's tokenizer. The additional UTF-8 byte
    ceiling respects the retriever's conservative embedding input budget.
    """
    encoding = tiktoken.get_encoding('cl100k_base')
    tokens = encoding.encode(text, disallowed_special=())
    _, offsets = encoding.decode_with_offsets(tokens)
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        left = offsets[start]
        right = offsets[end] if end < len(tokens) else len(text)
        while end > start and len(text[left:right].encode('utf-8')) > 7800:
            end -= 1
            right = offsets[end]
        if end <= start or right <= left:
            raise ValueError('Unable to split PDF text within the embedding budget')
        yield text[left:right], left, right
        if end == len(tokens):
            break
        following = max(start + 1, end - overlap)
        while following < end and offsets[following] <= left:
            following += 1
        start = following


def parse_pdf(source: Path, corpus: Path, size: int, overlap: int) -> list[dict]:
    slide = source.relative_to(corpus).parts[0].lower() == 'slides'
    # Preserve section boundaries for handouts and page boundaries for slides.
    records = load_slides(source, corpus / 'rendered') if slide else load_handout(source, max_chars=10**9)
    result = []
    for record in records:
        if record.get('ocr_error') or record.get('render_error'):
            raise ValueError(f'PDF OCR/render failed: {source.name}, page {record.get("page_number")}')
        text = record.get('text', '')
        if not text.strip():
            continue  # Blank pages are harmless; an entirely empty PDF fails below.
        spans, offset = [], 0
        for block in record.get('blocks', []):
            spans.append((offset, offset + len(block['text']), block))
            offset += len(block['text']) + 2
        for part, (content, left, right) in enumerate(token_windows(text, size, overlap), 1):
            if not content.strip():
                continue
            item = {**record, 'text': content, 'window': part, 'chunk_tokens': size,
                    'overlap_tokens': overlap, 'embedding_precision': 'float16'}
            blocks = [{**block, 'text': text[max(left, a):min(right, b)]}
                      for a, b, block in spans if a < right and b > left]
            if blocks:
                item.update(blocks=blocks, page_start=min(b['page'] for b in blocks),
                            page_end=max(b['page'] for b in blocks))
            if not slide:
                page = item['page_start']
                expected = render_path_for(source, page, corpus / 'rendered')
                item['render_path'] = str(expected.resolve()) if expected.is_file() and valid_render(expected) else render_slide_page(source, page, corpus / 'rendered')
            identity = [str(source), record.get('section_index'), record.get('section'),
                        record.get('page_number'), left, right, content]
            item['id'] = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
            result.append(item)
    if not result:
        raise ValueError(f'PDF has no searchable text: {source.name}')
    return result


class LocalPDFRetriever(HybridRetriever):
    def _embed(self, texts, *, is_query=False):
        vectors = super()._embed(texts, is_query=is_query)
        try:
            rounded = [list(struct.unpack('<768e', struct.pack('<768e', *v))) for v in vectors]
        except (OverflowError, struct.error) as exc:
            raise ValueError('Embedding cannot be represented at F16 precision') from exc
        if any(not any(v) for v in rounded):
            raise ValueError('F16 rounding produced an empty embedding')
        return rounded


def indexed_sources(retriever) -> set[str]:
    with retriever._connect() as conn:
        if conn.execute("SELECT to_regclass('companion_chunks') AS name").fetchone()['name'] is None:
            return set()
        rows = conn.execute("SELECT DISTINCT metadata->>'source' AS source FROM companion_chunks").fetchall()
        if rows:
            retriever._check_config(conn)
        return {str(Path(row['source']).resolve()) for row in rows if row['source']}


def snapshot_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if any(not isinstance(record, dict) or not isinstance(record.get('source'), str) for record in records):
        raise ValueError(f'Invalid snapshot: {path.name}')
    return records


def atomic_jsonl(path: Path, records: list[dict]):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def verify_model(settings):
    response = httpx.post(settings.base_url.rstrip('/') + '/api/show',
                          json={'model': settings.embedding_model}, timeout=30)
    response.raise_for_status()
    if response.json().get('details', {}).get('quantization_level') != 'F16':
        raise ValueError('nomic-embed-text must have F16 weights; run ollama pull nomic-embed-text')


def update_corpus(corpus: Path, *, chunk_tokens=800, overlap_tokens=150,
                  reprocess=False, retriever=None) -> dict:
    corpus = Path(corpus).resolve()
    if not corpus.is_dir():
        raise ValueError(f'Corpus directory does not exist: {corpus}')
    if not 100 <= chunk_tokens <= 1000 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError('Require 100 <= chunk_tokens <= 1000 and 0 <= overlap_tokens < chunk_tokens')
    load_dotenv(ROOT / '.env')
    settings = ProviderSettings.from_env({**os.environ, 'LLM_PROVIDER': 'ollama'})
    if settings.embedding_model.split(':')[0] != 'nomic-embed-text' or settings.embedding_dimensions != 768:
        raise ValueError('This updater requires local nomic-embed-text with 768 dimensions')
    retriever = retriever if retriever is not None else LocalPDFRetriever(settings=settings)
    cache = corpus / '.ingestion-cache'
    cache.mkdir(exist_ok=True)
    manifest_path = cache / 'pdf-update-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': {}}
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        raise ValueError('Invalid local PDF update manifest; restore or remove the manifest and retry')
    known = indexed_sources(retriever)
    inputs = discover_pdfs(corpus)
    configuration = [VERSION, chunk_tokens, overlap_tokens]
    pending, saved = [], dict(manifest['files'])
    for source in inputs:
        digest = fingerprint(source)
        previous = saved.get(str(source))
        if not reprocess and str(source) in known and (previous is None or previous == {'sha256': digest, 'configuration': configuration}):
            saved[str(source)] = {'sha256': digest, 'configuration': configuration}
            continue
        pending.append((source, digest))
    summary = {'scanned_files': len(inputs), 'new_files': sum(str(p) not in known for p, _ in pending),
               'updated_files': sum(str(p) in known for p, _ in pending), 'added_chunks': 0,
               'processed_chunks': 0, 'bm25_documents': 0}
    if not pending:
        atomic_json(manifest_path, {'files': saved})
        return summary
    verify_model(settings)
    updates = {}
    for source, digest in pending:
        print(f'Processing {source.relative_to(corpus)}', flush=True)
        records = parse_pdf(source, corpus, chunk_tokens, overlap_tokens)
        if fingerprint(source) != digest:
            raise ValueError(f'PDF changed during processing; retry: {source.name}')
        name = 'slides.jsonl' if source.relative_to(corpus).parts[0].lower() == 'slides' else 'materials.jsonl'
        updates.setdefault(name, {})[str(source)] = records
        summary['processed_chunks'] += len(records)
        saved[str(source)] = {'sha256': digest, 'configuration': configuration}
    paths = []
    # Keep complete snapshots: ingest synchronizes each supplied dataset.
    # Passing only new records would delete its existing slides/transcripts.
    for name, sources in updates.items():
        target = corpus / name
        existing = snapshot_records(target)
        # A missing/incomplete snapshot must never prune older DB records when
        # the existing synchronizer replaces this dataset.
        snapshot_ids = {chunk.id for chunk in read_jsonl([target])[0]} if target.exists() else set()
        with retriever._connect() as conn:
            stored_ids = {row['id'] for row in conn.execute(
                'SELECT id FROM companion_chunks WHERE dataset = %s', (str(target.resolve()),)
            ).fetchall()} if known else set()
        if not stored_ids.issubset(snapshot_ids):
            raise ValueError(f'Processed snapshot {target.name} is missing indexed records; restore it before updating')
        records = [r for r in existing if str(Path(r['source']).resolve()) not in sources]
        records += [record for items in sources.values() for record in items]
        atomic_jsonl(target, records)
        paths.append(target)
    result = retriever.ingest(paths)  # Batched embeddings; atomic DB transaction.
    with retriever._connect() as conn:
        rows = conn.execute('SELECT id, content FROM companion_chunks ORDER BY id').fetchall()
    BM25Index(rows)  # Validate a fresh vocabulary; live search does this too.
    summary.update(added_chunks=result['embedded'], bm25_documents=len(rows))
    # A failed parse, embedding call or transaction never marks sources complete.
    atomic_json(manifest_path, {'files': saved})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus')
    parser.add_argument('--chunk-tokens', type=int, default=800)
    parser.add_argument('--overlap-tokens', type=int, default=150)
    parser.add_argument('--reprocess', action='store_true', help='Reparse all discovered PDFs; reuse unchanged vectors')
    args = parser.parse_args()
    try:
        result = update_corpus(args.corpus, chunk_tokens=args.chunk_tokens,
                               overlap_tokens=args.overlap_tokens, reprocess=args.reprocess)
    except psycopg.Error as exc:
        parser.exit(1, database_error_message(exc) + '\n')
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        parser.exit(1, f'Corpus update failed: {exc}\n')
    print(f"Processed {result['new_files']} new files, updated {result['updated_files']} files, "
          f"added/updated {result['added_chunks']} chunks. "
          + (f"Vector store and BM25 updated successfully ({result['bm25_documents']} searchable chunks)."
             if result['processed_chunks'] else 'No new or modified PDFs; indices unchanged.'))


if __name__ == '__main__':
    main()
