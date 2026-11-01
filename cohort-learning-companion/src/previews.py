"""Exact cited-page previews, independent of approximate image discovery.

Never select another slide by broad topic overlap. A preview must be the exact
single PDF page of a supplied, cited, reranker-approved text chunk. The original
query cosine is retained unchanged; this is a provenance route, not a boost.
"""
from __future__ import annotations
import math
from pathlib import Path
import re

from .retriever import ROOT, MIN_RERANK_SCORE, resolve_render_path, _SLIDE_RENDER_LOCK
from .ingestion import render_path_for, render_slide_page


def _reference(source: dict):
    score = source.get('rerank_score')
    if not (isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and score >= MIN_RERANK_SCORE):
        return None
    if not re.fullmatch(r'S\d+', str(source.get('citation', ''))):
        return None
    metadata = source.get('metadata', {})
    if not isinstance(metadata, dict) or metadata.get('kind') not in {'slide', 'handout'}:
        return None
    page = metadata.get('page_number') if metadata['kind'] == 'slide' else metadata.get('page_start')
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return None
    # A multi-page section cannot identify one exact image for its claims.
    if metadata['kind'] == 'handout' and metadata.get('page_end', page) != page:
        return None
    raw = metadata.get('source')
    if not isinstance(raw, str) or not raw.strip():
        return None
    root = (ROOT / 'corpus' / ('slides' if metadata['kind'] == 'slide' else 'pdfs')).resolve()
    raw = raw.replace('\\', '/')
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [ROOT / path, root / path]
    parts = raw.split('/')
    for i in range(len(parts) - 1):
        if parts[i:i + 2] == ['corpus', root.name]:
            candidates.append(root.joinpath(*parts[i + 2:]))
    path = next((p.resolve() for p in candidates if p.resolve().is_relative_to(root)
                 and p.suffix.lower() == '.pdf' and p.is_file()), None)
    if path is None:
        return None
    import pymupdf
    with pymupdf.open(path) as document:
        if document.needs_pass or page > len(document):
            return None
    expected = render_path_for(path, page, ROOT / 'corpus/rendered').resolve()
    if not expected.is_relative_to((ROOT / 'corpus/rendered').resolve()):
        return None
    return path, page, expected


def cited_page_image(source: dict) -> dict | None:
    """Render only a directly cited, unambiguous PDF page; callers fail closed."""
    reference = _reference(source)
    if reference is None:
        return None
    pdf, page, expected = reference
    with _SLIDE_RENDER_LOCK:
        path = resolve_render_path({'render_path': str(expected)})
        if path is None:
            render_slide_page(pdf, page, ROOT / 'corpus/rendered')
            path = resolve_render_path({'render_path': str(expected)})
    if path is None:
        return None
    source['metadata'] = {**source['metadata'], 'render_path': path}
    return {'path': path, 'label': source.get('label', pdf.name),
            'citation': source['citation'], 'chunk_id': source.get('chunk_id'),
            'page_number': page, 'source_kind': source['metadata']['kind'],
            'dense_score': source.get('dense_score'), 'rerank_score': source['rerank_score'],
            'preview_type': 'cited_page'}


def verified_cited_page_image(image: dict, sources: list[dict]) -> bool:
    """Recheck authoritative final-source provenance at the UI boundary."""
    if image.get('preview_type') != 'cited_page':
        return False
    source = next((s for s in sources if s.get('citation') == image.get('citation')), None)
    if source is None:
        return False
    try:
        reference = _reference(source)
        if reference is None:
            return False
        _, page, expected = reference
        path = resolve_render_path({'render_path': image.get('path')})
        return (path == str(expected) and not isinstance(image.get('page_number'), bool)
                and image.get('page_number') == page
                and image.get('source_kind') == source['metadata']['kind']
                and image.get('chunk_id') == source.get('chunk_id'))
    except Exception:
        # Malformed PDF/image metadata cannot interrupt an otherwise valid answer.
        return False
