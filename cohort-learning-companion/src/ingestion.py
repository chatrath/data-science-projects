"""Extract PDF slide decks as individual, layout-aware JSON records.



Usage: python -m src.ingestion corpus/slides/deck.pdf --output corpus/slides.jsonl

Full slide images are rendered to corpus/rendered by default; --render-dir overrides it.

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





def valid_render(path: str | Path) -> bool:

    try:

        image = fitz.Pixmap(str(path))

        return image.width > 0 and image.height > 0

    except Exception:

        return False





def render_path_for(source: str | Path, page_number: int, render_dir: str | Path | None = None) -> Path:

    import hashlib

    source = Path(source).resolve()

    root = ROOT / 'corpus' / 'rendered' if render_dir is None else Path(render_dir)

    source_id = hashlib.sha256(str(source).encode()).hexdigest()[:12]

    return root / f'{source.stem}-{source_id}' / f'page_{page_number:04d}.png'





def render_page(page, source: str | Path, render_dir: str | Path | None = None) -> str:

    """Atomically render a PDF page, independent of text extraction or OCR."""

    import tempfile

    rendered = render_path_for(source, page.number + 1, render_dir)

    destination = rendered.parent

    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=destination, suffix='.png', delete=False) as temporary:

        temporary_path = Path(temporary.name)

    try:

        for scale in (1.5, 1.0):

            try:

                page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False).save(temporary_path)

                if not valid_render(temporary_path):

                    raise RuntimeError('Invalid rendered image')

                break

            except Exception:

                if scale == 1.0:

                    raise

        temporary_path.replace(rendered)

    finally:

        temporary_path.unlink(missing_ok=True)

    return str(rendered.resolve())





def render_slide_page(source: str | Path, page_number: int, render_dir: str | Path | None = None) -> str:

    """Repair a single missing preview without re-OCRing or re-embedding a deck."""

    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:

        raise ValueError('Slide page number must be a positive integer')

    with fitz.open(source) as document:

        if document.needs_pass or page_number > len(document):

            raise ValueError('Slide page unavailable')

        return render_page(document[page_number - 1], source, render_dir)





def load_slides(pdf_path: str | Path, render_dir: str | Path | None = None, *, ocr: bool = True) -> Iterator[dict]:

    """Yield one record per page, including empty and image-only slides.



    Bounding boxes use PyMuPDF's unrotated page coordinates in points.

    Sorted text is an approximate reading order; block positions and full-slide

    renders preserve context when a slide has columns or a complex layout.

    """

    render_dir = ROOT / 'corpus' / 'rendered' if render_dir is None else Path(render_dir)

    source = Path(pdf_path).resolve()

    if not source.is_file():

        raise FileNotFoundError(source)

    if source.suffix.lower() != ".pdf":

        raise ValueError(f"Expected a PDF slide deck: {source}")

    with fitz.open(source) as document:

        if document.needs_pass:

            raise ValueError(f"Password-protected PDF: {source}")

        for page in document:

            errors = {}

            try:

                render_path = render_page(page, source, render_dir)

            except Exception as exc:

                render_path = None

                errors['render_error'] = type(exc).__name__

            try:

                text, blocks, textpage = extract_page(page, ocr=ocr)

            except Exception as exc:

                errors['ocr_error'] = type(exc).__name__

                try:

                    text, blocks, textpage = extract_page(page, ocr=False)

                except Exception:

                    text, blocks, textpage = '', [], None

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

            record['render_path'] = render_path

            record.update(errors)

            yield record





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF slide decks")

    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")

    parser.add_argument("--render-dir", type=Path, default=ROOT / "corpus" / "rendered", help="Full-slide PNG directory (default: corpus/rendered)")

    parser.add_argument("--no-ocr", action="store_true", help="Render with native text only")

    args = parser.parse_args()

    if args.output.resolve() in {path.resolve() for path in args.pdfs}:

        parser.error("Output must not overwrite an input PDF")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    import tempfile

    count = 0

    temporary = None

    try:

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent, delete=False) as output:

            temporary = Path(output.name)

            for pdf in args.pdfs:

                for slide in load_slides(pdf, args.render_dir, ocr=not args.no_ocr):

                    output.write(json.dumps(slide, ensure_ascii=False) + '\n')

                    count += 1

        temporary.replace(args.output)

    finally:

        if temporary is not None:

            temporary.unlink(missing_ok=True)

    print(f"Extracted {count} slides to {args.output}")





if __name__ == "__main__":

    main()
