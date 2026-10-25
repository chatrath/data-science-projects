import json

from pathlib import Path

import tempfile

import unittest

from unittest.mock import Mock, patch



import pymupdf



from src.corpus_ingestion import prepare_corpus

from src.ingestion import extract_page

from src.retriever import read_jsonl, BM25Index, fuse





class CorpusIngestionTests(unittest.TestCase):

    def setUp(self):

        self.temp=tempfile.TemporaryDirectory()

        self.addCleanup(self.temp.cleanup)

        self.corpus=Path(self.temp.name)/'corpus'

        for folder in ['pdfs','slides','transcripts']:

            (self.corpus/folder).mkdir(parents=True)

        for folder,name,text in [('pdfs','handout.pdf','Handout grounding evidence'),('slides','deck.pdf','Slide search architecture')]:

            with pymupdf.open() as doc:

                page=doc.new_page()

                page.insert_text((72,72),text)

                doc.save(self.corpus/folder/name)

        (self.corpus/'transcripts/demo.txt').write_text('WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nTutor: Demo implementation [laughter]\n```python\n    print("retrieval")\n```\n',encoding='utf-8')



    def test_all_three_source_types_enter_the_same_searchable_snapshot(self):

        paths,report=prepare_corpus(self.corpus)

        chunks,_,skipped=read_jsonl(paths)

        self.assertEqual({c.metadata['kind'] for c in chunks},{'handout','slide','transcript'})

        self.assertEqual(skipped,0)

        transcript=next(c for c in chunks if c.metadata['kind']=='transcript')

        self.assertNotIn('[laughter]',transcript.content)

        self.assertIn('    print("retrieval")',transcript.content)

        self.assertEqual(transcript.metadata['start'],'00:00:01.000')

        slide=next(c for c in chunks if c.metadata['kind']=='slide')

        self.assertTrue(Path(slide.metadata['render_path']).is_file())

        rows=[{'id':c.id,'content':c.content} for c in chunks]

        keyword=BM25Index(rows).search('grounding architecture implementation',25)

        fused=fuse(keyword,[(c.id,.8) for c in chunks])

        self.assertEqual({hit['id'] for hit in fused},{c.id for c in chunks})



    def test_unchanged_files_are_cached_and_new_transcript_is_discovered(self):

        first,_=prepare_corpus(self.corpus)

        _,report=prepare_corpus(self.corpus)

        self.assertEqual(sum(s['cached_files'] for s in report.values()),3)

        extra=self.corpus/'transcripts/added.vtt'

        extra.write_text('WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTutor: Newly added demonstration.\n',encoding='utf-8')

        paths,report=prepare_corpus(self.corpus)

        chunks,_,_=read_jsonl(paths)

        self.assertEqual(report['transcript']['files'],2)

        self.assertTrue(any('Newly added' in c.content for c in chunks))

        extra.unlink()

        paths,_=prepare_corpus(self.corpus)

        chunks,_,_=read_jsonl(paths)

        self.assertFalse(any('Newly added' in c.content for c in chunks))



    def test_source_edit_invalidates_cache_and_missing_render_is_rebuilt(self):

        paths,_=prepare_corpus(self.corpus)

        slide=json.loads((self.corpus/'slides.jsonl').read_text().splitlines()[0])

        Path(slide['render_path']).unlink()

        (self.corpus/'transcripts/demo.txt').write_text('Updated transcript paragraph.',encoding='utf-8')

        paths,report=prepare_corpus(self.corpus)

        self.assertEqual(report['handout']['cached_files'],1)

        self.assertEqual(report['slide']['cached_files'],0)

        self.assertEqual(report['transcript']['cached_files'],0)

        self.assertTrue(Path(slide['render_path']).is_file())



    def test_pdf_cache_without_render_references_is_rebuilt(self):

        prepare_corpus(self.corpus)

        import hashlib

        source=(self.corpus/'slides/deck.pdf').resolve()

        cache=self.corpus/'.ingestion-cache'/(hashlib.sha256(str(source).encode()).hexdigest()+'.json')

        saved=json.loads(cache.read_text(encoding='utf-8'))

        for record in saved['records']:record.pop('render_path',None)

        cache.write_text(json.dumps(saved),encoding='utf-8')

        _,report=prepare_corpus(self.corpus)

        self.assertEqual(report['slide']['cached_files'],0)

        records=[json.loads(line) for line in (self.corpus/'slides.jsonl').read_text(encoding='utf-8').splitlines()]

        self.assertTrue(all(Path(record['render_path']).read_bytes().startswith(b'\x89PNG') for record in records))



    def test_corrupt_render_and_incomplete_page_metadata_invalidate_cache(self):

        from src.ingestion import valid_render

        prepare_corpus(self.corpus)

        slide=json.loads((self.corpus/'slides.jsonl').read_text().splitlines()[0])

        Path(slide['render_path']).write_bytes(b'corrupt image')

        _,report=prepare_corpus(self.corpus)

        self.assertEqual(report['slide']['cached_files'],0)

        self.assertTrue(valid_render(slide['render_path']))

        import hashlib

        source=(self.corpus/'slides/deck.pdf').resolve()

        cache=self.corpus/'.ingestion-cache'/(hashlib.sha256(str(source).encode()).hexdigest()+'.json')

        saved=json.loads(cache.read_text())

        saved['records'][0].pop('page_number')

        cache.write_text(json.dumps(saved))

        _,report=prepare_corpus(self.corpus)

        self.assertEqual(report['slide']['cached_files'],0)

        self.assertEqual(json.loads((self.corpus/'slides.jsonl').read_text().splitlines()[0])['page_number'],1)



    def test_image_only_pdf_uses_ocr_textpage_for_block_extraction(self):

        page=Mock()

        page.get_text.side_effect=['',[(0,0,100,20,'Recovered diagram title',0,0)]]

        textpage=object()

        with patch('src.ingestion.ocr_textpage',return_value=textpage) as ocr:

            text,blocks,returned=extract_page(page)

        ocr.assert_called_once_with(page)

        self.assertEqual(text,'Recovered diagram title')

        self.assertIs(returned,textpage)

        self.assertIn('textpage',page.get_text.call_args.kwargs)





if __name__=='__main__':

    unittest.main()
