"""Extract PDF slide decks as individual, layout-aware JSON records.

Usage: python -m src.ingestion corpus/slides/deck.pdf --output corpus/slides.jsonl
Add --render-dir corpus/rendered to preserve full slide images for visual review.
Image-only pages use local Tesseract OCR; visual flags do not interpret diagrams.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]


def ocr_textpage(page):
    """OCR locally using MuPDF's built-in Tesseract and English language data."""
    folder = Path(os.getenv('TESSDATA_PREFIX') or ROOT / '.local-db' / 'tessdata')
    if not (folder / 'eng.traineddata').is_file():
        raise RuntimeError('English OCR data is missing. Set TESSDATA_PREFIX to a folder containing eng.traineddata, or install it in .local-db/tessdata. See RETRIEVAL.md.')
    return page.get_textpage_ocr(language='eng', dpi=200, full=True, tessdata=str(folder.resolve()))


def extract_page(page, *, ocr: bool = True):
    """Return text blocks and a reusable OCR TextPage when native text is absent."""
    native = page.get_text('text').strip()
    textpage = ocr_textpage(page) if ocr and not native else None
    kwargs = {'textpage': textpage} if textpage is not None else {}
    blocks = [
        {'bbox': list(block[:4]), 'text': block[4].strip(), 'number': block[5]}
        for block in page.get_text('blocks', sort=True, **kwargs)
        if block[6] == 0 and block[4].strip()
    ]
    # Avoid the huge synthetic whitespace in sorted OCR plain-text extraction.
    text = '\n\n'.join(block['text'] for block in blocks)
    return text, blocks, textpage


def load_slides(pdf_path: str | Path, render_dir: str | Path | None = None, *, ocr: bool = True) -> Iterator[dict]:
    """Yield one record per page, including empty and image-only slides.

    Bounding boxes use PyMuPDF's unrotated page coordinates in points.
    Sorted text is an approximate reading order; block positions and optional
    renders preserve context when a slide has columns or a complex layout.
    """
    source = Path(pdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF slide deck: {source}")
    with fitz.open(source) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDF: {source}")
        for page in document:
            text, blocks, textpage = extract_page(page, ocr=ocr)
            images = [
                {"bbox": list(info["bbox"]), "width": info["width"], "height": info["height"]}
                for info in page.get_image_info()
            ]
            vectors = [
                {"bbox": list(path["rect"]), "type": path["type"], "primitive_count": len(path["items"])}
                for path in page.get_drawings()
            ]
            record = {
                "source": str(source),
                "kind": "slide",
                "ocr_applied": textpage is not None,
                "page_number": page.number + 1,
                "page_count": len(document),
                "text": text,
                "layout": {
                    "bbox": list(page.cropbox),
                    "rotation": page.rotation,
                    "text_blocks": blocks,
                    "images": images,
                    "vector_paths": vectors,
                },
                "has_images": bool(images),
                "has_vector_graphics": bool(vectors),
                "possible_diagram": bool(images or vectors),
                "needs_ocr": not bool(text),
                "needs_visual_review": bool(images or vectors) or not text,
            }
            if render_dir is not None:
                # Separate same-named decks from different source directories.
                import hashlib

                source_id = hashlib.sha256(str(source).encode()).hexdigest()[:12]
                destination = Path(render_dir) / f"{source.stem}-{source_id}"
                destination.mkdir(parents=True, exist_ok=True)
                rendered = destination / f"page_{page.number + 1:04d}.png"
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(rendered)
                record["render_path"] = str(rendered.resolve())
            yield record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF slide decks")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--render-dir", type=Path, help="Optional full-slide PNG directory")
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in args.pdfs}:
        parser.error("Output must not overwrite an input PDF")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for pdf in args.pdfs:
            for slide in load_slides(pdf, args.render_dir):
                output.write(json.dumps(slide, ensure_ascii=False) + "\n")
                count += 1
    print(f"Extracted {count} slides to {args.output}")


if __name__ == "__main__":
    main()
