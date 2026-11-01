"""Ingest handouts and transcripts without crossing section or code boundaries.

Run: python -m src.ingest_materials --output corpus/materials.jsonl
Heading detection is heuristic. Image-only PDF pages use local OCR.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator

import pymupdf
from .ingestion import extract_page

ROOT = Path(__file__).resolve().parents[1]
TIMING = re.compile(r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})")
FILLER = re.compile(r"\[(?:inaudible|unintelligible|silence|pause|laughter|laughs|music|applause|crosstalk|um|uh)\]|<(?:/?v(?:\s+[^>]+)?|/?c(?:\.[^>]+)?|/?lang(?:\s+[^>]+)?|\d{2}:\d{2}:\d{2}\.\d{3})>", re.I)


def pack_blocks(blocks: list[dict], max_chars: int) -> Iterator[list[dict]]:
    """Pack whole blocks; a single large code/cue block may exceed the target."""
    batch, size = [], 0
    for block in blocks:
        length = len(block['text'])
        if batch and size + length + 2 > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(block)
        size += length + 2
    if batch:
        yield batch


def record(source: Path, kind: str, section: str, blocks: list[dict], part: int) -> dict:
    text = '\n\n'.join(block['text'] for block in blocks)
    identity = json.dumps([str(source.resolve()), kind, section, part, blocks], sort_keys=True)
    return {'id': hashlib.sha256(identity.encode()).hexdigest(), 'source': str(source.resolve()),
            'kind': kind, 'section': section, 'part': part, 'text': text, 'blocks': blocks}


def load_handout(path: str | Path, max_chars: int = 4000) -> Iterator[dict]:
    """Use bookmark page starts, larger-font headings, and PDF text blocks.

    Page references survive section splits. Full text blocks stay intact, so
    max_chars is a soft target. Image-only pages are OCRed before sectioning.
    """
    if max_chars < 1:
        raise ValueError('max_chars must be positive')
    source = Path(path).resolve()
    with pymupdf.open(source) as doc:
        if doc.needs_pass:
            raise ValueError(f'Password-protected PDF: {source}')
        pages = []
        for page in doc:
            _, _, textpage = extract_page(page)
            kwargs = {'textpage': textpage} if textpage is not None else {}
            info = page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT, sort=True, **kwargs)
            info['ocr_applied'] = textpage is not None
            pages.append(info)
        sizes = Counter()
        for page in pages:
            for block in page['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        sizes[round(span['size'], 1)] += len(span['text'].strip())
        body_size = sizes.most_common(1)[0][0] if sizes else 12
        bookmarks = {page: title for _, title, page in doc.get_toc()}
        section, blocks = source.stem, []
        section_index = 0

        def flush():
            for part, group in enumerate(pack_blocks(blocks, max_chars), 1):
                item = record(source, 'handout', section, group, part)
                item.update(section_index=section_index, page_start=group[0]['page'], page_end=group[-1]['page'], ocr_applied=any(b.get('ocr_applied') for b in group))
                yield item

        for number, page in enumerate(pages, 1):
            if number in bookmarks:
                yield from flush()
                blocks = []
                section = bookmarks[number]
                section_index += 1
            found_text = False
            for block in page['blocks']:
                lines = block.get('lines', [])
                text = '\n'.join(''.join(s['text'] for s in line['spans']) for line in lines).strip()
                if not text:
                    continue
                found_text = True
                spans = [s for line in lines for s in line['spans'] if s['text'].strip()]
                heading = len(text) <= 180 and spans and min(s['size'] for s in spans) >= body_size * 1.15
                if heading:
                    yield from flush()
                    blocks = []
                    section = text.replace('\n', ' ')
                    section_index += 1
                blocks.append({'text': text, 'page': number, 'bbox': list(block['bbox']), 'is_heading': bool(heading), **({'ocr_applied': True} if page['ocr_applied'] else {})})
            if not found_text:
                # Keep a visible record of pages requiring manual/OCR processing.
                yield {'source': str(source.resolve()), 'kind': 'handout', 'page_start': number,
                       'page_end': number, 'text': '', 'needs_ocr': True, 'blocks': []}
        yield from flush()


def clean_lines(lines: list[str]) -> str:
    """Remove only known annotation tags outside code; retain indentation."""
    result, fence = [], None
    for line in lines:
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token[0]
            elif token[0] == fence:
                fence = None
            result.append(line.rstrip())
        elif fence or line.startswith(('    ', '\t')):
            result.append(line.rstrip())
        else:
            result.append(FILLER.sub('', line).rstrip())
    return '\n'.join(result).strip('\n')


def transcript_blocks(text: str) -> list[dict]:
    """Parse VTT/SRT cues or plain paragraphs; never split fenced code."""
    lines = text.lstrip('\ufeff').splitlines()
    blocks, current = [], []
    start = end = None
    fence = None

    def emit():
        cleaned = clean_lines(current)
        if not cleaned.strip():
            return
        speaker = None
        match = re.match(r'^([A-Za-z][\w .\-\u2019]{0,70}):\s+', cleaned)
        if match and match.group(1).lower() not in {'http', 'https', 'note', 'warning'}:
            speaker = match.group(1)
        blocks.append({'text': cleaned, 'speaker': speaker, 'start': start, 'end': end,
                       'contains_code': bool(re.search(r'(?m)^\s*(```|~~~)|^(?:    |\t)\S', cleaned))})

    for index, line in enumerate(lines):
        timing = TIMING.search(line) if fence is None else None
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if timing:
            emit()
            current = []
            start, end = timing['start'], timing['end']
            continue
        if fence is None:
            if line.strip() == 'WEBVTT':
                continue
            # Numeric cue IDs are metadata only when followed by a timestamp.
            if index + 1 < len(lines) and TIMING.search(lines[index + 1]) and not line.strip().startswith(('```', '~~~')):
                continue
            if not line.strip():
                emit()
                current = []
                continue
        current.append(line)
        if marker:
            token = marker.group(1)[0]
            fence = token if fence is None else (None if token == fence else fence)
    emit()
    return blocks


# Extracted topic labels only: no model calls or invented summaries.
TOPIC_STOPWORDS = frozenset("a an and are as at be been but by can could did do does doing for from get going had has have here how i if in into is it its just know like me more my no not now of okay on one or our out really right so some than that the their them then there these they thing things think this those through to too um uh up us use using very want was we well were what when where which who why will with would yeah yes you your also actually basically let's all about cool guys easy way particular need per let look might trying over two essentially should hear able lot something anything everything because got much make sure see say said take other any even still only such kind sort little good great today thanks thank please really alright done okay yuva".split())


def transcript_topics(blocks: list[dict]) -> list[str]:
    speakers = {word.casefold() for b in blocks for word in re.findall(r"\w+", b.get('speaker') or '')}
    # Rank repeated content words; remove speaker labels and conversational glue.
    body = '\n'.join(re.sub(r'^' + re.escape(b['speaker']) + r':\s*', '', b['text']) if b.get('speaker') else b['text'] for b in blocks)
    counts = Counter(word.casefold().rstrip('.-') for word in re.findall(r'[A-Za-z][A-Za-z0-9_+#.-]{2,}', body))
    terms = [word for word, _ in counts.most_common() if word not in TOPIC_STOPWORDS and word not in speakers]
    return terms[:8]


def transcript_windows(blocks: list[dict], max_chars: int, overlap_chars: int) -> Iterator[tuple[int, list[dict]]]:
    """Sliding windows of intact cues/code blocks, with a soft size limit.

    Each window adds new blocks. Oversized individual blocks remain intact.
    Overlap is rounded up to whole trailing blocks, but always advances.
    """
    start = 0
    while start < len(blocks):
        end, size = start, 0
        while end < len(blocks):
            length = len(blocks[end]['text']) + (2 if end > start else 0)
            if end > start and size + length > max_chars:
                break
            size += length
            end += 1
        yield start, blocks[start:end]
        if end == len(blocks):
            break
        next_start, overlap = end, 0
        while next_start > start + 1 and overlap < overlap_chars:
            next_start -= 1
            overlap += len(blocks[next_start]['text']) + 2
        start = next_start


def load_transcript(path: str | Path, max_chars: int = 4500, overlap_chars: int | None = None) -> Iterator[dict]:
    if max_chars < 1:
        raise ValueError('max_chars must be positive')
    overlap_chars = max_chars // 5 if overlap_chars is None else overlap_chars
    if not 0 <= overlap_chars < max_chars:
        raise ValueError('Require 0 <= overlap_chars < max_chars')
    source = Path(path)
    blocks = transcript_blocks(source.read_text(encoding='utf-8-sig'))
    for part, (offset, group) in enumerate(transcript_windows(blocks, max_chars, overlap_chars), 1):
        item = record(source, 'transcript', source.stem, group, part)
        topics = transcript_topics(group)
        header = '[Transcript Segment - Topic keywords: ' + (', '.join(topics) or 'live session discussion') + ']'
        item.update(context_header=header, topic_keywords=topics, block_start=offset,
                    block_end=offset + len(group), chunk_max_chars=max_chars,
                    overlap_target_chars=overlap_chars)
        item['text'] = header + '\n\n' + item['text']
        item['start'] = next((b['start'] for b in group if b['start']), None)
        item['end'] = next((b['end'] for b in reversed(group) if b['end']), None)
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-chars', type=int, default=4000, help='Handout block target')
    parser.add_argument('--transcript-max-chars', type=int, default=4500)
    parser.add_argument('--transcript-overlap-chars', type=int, default=900)
    args = parser.parse_args()
    if args.max_chars < 1:
        parser.error('--max-chars must be positive')
    if args.transcript_max_chars < 1 or not 0 <= args.transcript_overlap_chars < args.transcript_max_chars:
        parser.error('Require 0 <= --transcript-overlap-chars < --transcript-max-chars')
    inputs = [(p, load_handout) for p in sorted((args.corpus / 'pdfs').rglob('*')) if p.is_file() and p.suffix.lower() == '.pdf']
    inputs += [(p, load_transcript) for p in sorted((args.corpus / 'transcripts').rglob('*')) if p.is_file() and p.suffix.lower() in {'.txt', '.vtt', '.srt'}]
    if not inputs:
        parser.error('No PDF handouts or TXT transcripts found')
    if args.output.resolve() in {p.resolve() for p, _ in inputs}:
        parser.error('Output must not overwrite an input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Stage output so a corrupt PDF cannot replace an existing successful run.
    import tempfile
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent, delete=False) as output:
            temporary = Path(output.name)
            for path, loader in inputs:
                for chunk in (loader(path, args.transcript_max_chars, args.transcript_overlap_chars) if loader is load_transcript else loader(path, args.max_chars)):
                    output.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    count += 1
        temporary.replace(args.output)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    print(f'Extracted {count} chunks from {len(inputs)} files to {args.output}')


if __name__ == '__main__':
    main()
