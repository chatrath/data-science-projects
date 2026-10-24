"""Refresh processed corpus snapshots from all raw source folders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

from .ingestion import load_slides
from .ingest_materials import load_handout, load_transcript

CACHE_VERSIONS = {'handout': 1, 'slide': 1, 'transcript': 3}


def atomic_json(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False)
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def slide_exports(path: Path):
    """Markdown slide separators / text form feeds define individual slides."""
    text = path.read_text(encoding='utf-8-sig')
    for number, text in enumerate(re.split(r'(?m)^\s*---\s*$|\f', text), 1):
        yield {'kind': 'slide', 'source': str(path.resolve()), 'page_number': number, 'text': text.strip()}


def prepare_corpus(corpus: Path) -> tuple[list[Path], dict]:
    """Parse handouts, slide decks and transcripts; reuse unchanged file caches.

    Every processed snapshot is rebuilt from the discovered file set, so stale
    records disappear on synchronization. Source hashes invalidate cached OCR
    and transcript parses; missing slide renders trigger regeneration.
    """
    corpus = Path(corpus).resolve()
    for folder in ('pdfs', 'slides', 'transcripts'):
        if not (corpus / folder).is_dir():
            raise ValueError(f'Missing raw corpus folder: {corpus / folder}')
    specifications = [
        ('handout', 'pdfs', {'.pdf'}),
        ('slide', 'slides', {'.pdf', '.md', '.txt'}),
        ('transcript', 'transcripts', {'.txt', '.vtt', '.srt'}),
    ]
    inputs = [(kind, path) for kind, folder, extensions in specifications
              for path in sorted((corpus / folder).rglob('*'))
              if path.is_file() and path.suffix.lower() in extensions]
    if not inputs:
        raise ValueError('No supported raw handouts, slide decks or transcripts were found')
    cache_dir = corpus / '.ingestion-cache'
    cache_dir.mkdir(exist_ok=True)
    snapshots = {'slides.jsonl': [], 'materials.jsonl': []}
    report = {kind: {'files': 0, 'chunks': 0, 'empty': 0, 'ocr_pages': 0, 'cached_files': 0}
              for kind, _, _ in specifications}
    for kind, path in inputs:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        cache_path = cache_dir / (hashlib.sha256(str(path).encode()).hexdigest() + '.json')
        cached = None
        if cache_path.exists():
            try:
                saved = json.loads(cache_path.read_text(encoding='utf-8'))
                records = saved['records']
                if (saved['version'] == CACHE_VERSIONS[kind] and saved['sha256'] == digest
                        and saved['source'] == str(path) and len(records) == saved['count']
                        and all(isinstance(r, dict) and isinstance(r.get('text'), str) for r in records)
                        and all(not r.get('render_path') or Path(r['render_path']).is_file() for r in records)):
                    cached = records
            except (ValueError, KeyError, TypeError, OSError):
                pass
        if cached is not None:
            records = cached
            report[kind]['cached_files'] += 1
        else:
            if kind == 'handout':
                records = list(load_handout(path))
            elif kind == 'transcript':
                records = list(load_transcript(path))
            elif path.suffix.lower() == '.pdf':
                records = list(load_slides(path, corpus / 'rendered'))
            else:
                records = list(slide_exports(path))
            for item in records:
                item['kind'] = kind
            atomic_json(cache_path, {'version': CACHE_VERSIONS[kind], 'sha256': digest, 'source': str(path),
                                     'count': len(records), 'records': records})
        report[kind]['files'] += 1
        report[kind]['chunks'] += sum(bool(r['text'].strip()) for r in records)
        report[kind]['empty'] += sum(not bool(r['text'].strip()) for r in records)
        if kind == 'slide':
            report[kind]['ocr_pages'] += sum(bool(r.get('ocr_applied')) for r in records)
        elif kind == 'handout':
            report[kind]['ocr_pages'] += len({b['page'] for r in records for b in r.get('blocks', []) if b.get('ocr_applied')})
        snapshots['slides.jsonl' if kind == 'slide' else 'materials.jsonl'].extend(records)
    temporary_files = []
    paths = []
    try:
        for name, records in snapshots.items():
            target = corpus / name
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=corpus, delete=False) as stream:
                temporary = Path(stream.name)
                temporary_files.append(temporary)
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            paths.append(target)
        for temporary, target in zip(temporary_files, paths):
            temporary.replace(target)
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)
    return paths, report
