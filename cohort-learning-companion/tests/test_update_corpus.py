from contextlib import nullcontext
from pathlib import Path
import struct,tempfile,unittest
from unittest.mock import Mock,patch
import pymupdf
import update_corpus as updater
from src.retriever import BM25Index,read_jsonl

class MemoryRetriever:
    def __init__(self):self.rows=[];self.calls=0;self.fail=False
    def _connect(self):
        conn=Mock()
        def execute(sql,params=None):
            rows=[r for r in self.rows if r['dataset']==params[0]] if 'WHERE dataset' in sql else self.rows
            return Mock(fetchall=Mock(return_value=rows))
        conn.execute.side_effect=execute
        return nullcontext(conn)
    def ingest(self,paths):
        self.calls+=1
        if self.fail:raise RuntimeError('Embedding unavailable')
        chunks,datasets,_=read_jsonl(paths);old={r['id']:r for r in self.rows}
        embedded=sum(c.id not in old or old[c.id]['content']!=c.content for c in chunks)
        self.rows=[r for r in self.rows if r['dataset'] not in datasets]+[
            {'id':c.id,'dataset':c.dataset,'content':c.content,'metadata':c.metadata} for c in chunks]
        return {'embedded':embedded}

class UpdateCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.corpus=Path(self.temp.name)/'corpus';self.corpus.mkdir();self.retriever=MemoryRetriever()
        self.patches=[patch('update_corpus.verify_model'),patch('update_corpus.load_dotenv'),
            patch.dict('os.environ',{'OLLAMA_EMBEDDING_MODEL':'nomic-embed-text','OLLAMA_EMBEDDING_DIMENSIONS':'768'}),
            patch('update_corpus.indexed_sources',side_effect=lambda _: {r['metadata']['source'] for r in self.retriever.rows})]
        for item in self.patches:item.start();self.addCleanup(item.stop)
    def pdf(self,name,text='New vocabulary quasarindex searchable evidence.',blank=False):
        path=self.corpus/name;path.parent.mkdir(parents=True,exist_ok=True)
        with pymupdf.open() as doc:
            page=doc.new_page();page.insert_text((72,72),text)
            if blank:doc.new_page()
            doc.save(path)
        return path
    def run_update(self,**kwargs):return updater.update_corpus(self.corpus,retriever=self.retriever,**kwargs)
    def test_new_pdf_indexed_once_with_metadata_and_bm25_vocabulary(self):
        source=self.pdf('pdfs/new.pdf',blank=True);result=self.run_update()
        self.assertEqual((result['new_files'],result['added_chunks']),(1,1))
        self.assertTrue(BM25Index(self.retriever.rows).search('quasarindex',3))
        meta=self.retriever.rows[0]['metadata'];self.assertEqual(meta['source'],str(source.resolve()))
        self.assertEqual((meta['page_start'],meta['page_end']),(1,1));self.assertTrue(meta['section'])
        self.assertTrue(Path(meta['render_path']).is_file())
        self.assertEqual(self.run_update()['added_chunks'],0);self.assertEqual(self.retriever.calls,1)
    def test_new_nested_pdf_preserves_existing_handouts_and_transcripts(self):
        self.pdf('pdfs/first.pdf');self.run_update();target=self.corpus/'materials.jsonl'
        records=updater.snapshot_records(target)
        records.append({'id':'transcript','source':str(self.corpus/'transcripts/demo.txt'),'kind':'transcript','text':'Existing code demo.'})
        updater.atomic_jsonl(target,records);self.retriever.ingest([target]);before={r['id'] for r in self.retriever.rows}
        self.pdf('pdfs/nested/second.PDF','Additional vocabulary galaxyindex.')
        self.assertEqual(self.run_update()['new_files'],1)
        self.assertTrue(before.issubset({r['id'] for r in self.retriever.rows}))
        self.assertTrue(BM25Index(self.retriever.rows).search('galaxyindex',3))
    def test_slides_keep_page_level_rendered_images(self):
        self.pdf('slides/deck.pdf');self.run_update();meta=self.retriever.rows[0]['metadata']
        self.assertEqual((meta['kind'],meta['page_number']),('slide',1));self.assertTrue(Path(meta['render_path']).is_file())
    def test_root_pdf_supported_and_internal_artifacts_ignored(self):
        source=self.pdf('new.pdf');self.pdf('rendered/ignore.pdf');self.pdf('.ingestion-cache/ignore.pdf')
        self.assertEqual(updater.discover_pdfs(self.corpus),[source.resolve()]);self.assertEqual(self.run_update()['new_files'],1)
    def test_changed_pdf_replaces_chunks_without_duplicates(self):
        source=self.pdf('pdfs/doc.pdf');self.run_update();source.unlink();self.pdf('pdfs/doc.pdf','Changed replacementindex evidence.')
        result=self.run_update();self.assertEqual((result['new_files'],result['updated_files']),(0,1))
        self.assertEqual(len(self.retriever.rows),1);self.assertIn('replacementindex',self.retriever.rows[0]['content'])
    def test_failed_ingestion_not_marked_complete_and_retry_succeeds(self):
        self.pdf('pdfs/doc.pdf');self.retriever.fail=True
        with self.assertRaises(RuntimeError):self.run_update()
        self.assertFalse((self.corpus/'.ingestion-cache/pdf-update-manifest.json').exists())
        self.retriever.fail=False;self.assertEqual(self.run_update()['new_files'],1)
    def test_missing_snapshot_does_not_delete_indexed_data(self):
        self.pdf('pdfs/first.pdf');self.run_update();before=list(self.retriever.rows)
        (self.corpus/'materials.jsonl').unlink();self.pdf('pdfs/second.pdf')
        with self.assertRaisesRegex(ValueError,'missing indexed records'):self.run_update()
        self.assertEqual(self.retriever.rows,before)
    def test_empty_scan_noop_and_bad_parameters_rejected(self):
        self.assertEqual(self.run_update()['scanned_files'],0);self.assertEqual(self.retriever.calls,0)
        for kwargs in [{'chunk_tokens':0},{'overlap_tokens':800},{'overlap_tokens':-1}]:
            with self.assertRaises(ValueError):self.run_update(**kwargs)
    def test_token_windows_overlap_and_preserve_unicode(self):
        text=('Semantic caf'+chr(233)+' '+chr(25968)+chr(25454)+' code value = 123. ')*400
        windows=list(updater.token_windows(text,100,20));self.assertGreater(len(windows),2)
        self.assertEqual(windows[0][1],0);self.assertEqual(windows[-1][2],len(text))
        for (chunk,left,right),(_,start,_) in zip(windows,windows[1:]):
            self.assertEqual(chunk,text[left:right]);self.assertNotIn(chr(65533),chunk)
            self.assertLess(start,right);self.assertLessEqual(len(chunk.encode('utf-8')),7800)
    def test_vectors_have_768_dimensions_and_f16_precision(self):
        retriever=updater.LocalPDFRetriever.__new__(updater.LocalPDFRetriever);value=.123456789
        with patch('src.retriever.HybridRetriever._embed',return_value=[[value]*768]) as embed:result=retriever._embed(['evidence'])
        embed.assert_called_once_with(['evidence'],is_query=False);self.assertEqual(len(result[0]),768)
        self.assertEqual(result[0][0],struct.unpack('<e',struct.pack('<e',value))[0])
    def test_non_f16_model_is_rejected(self):
        self.patches[0].stop();settings=Mock(base_url='http://localhost:11434/',embedding_model='nomic-embed-text')
        response=Mock();response.json.return_value={'details':{'quantization_level':'Q4_K_M'}}
        with patch('update_corpus.httpx.post',return_value=response):
            with self.assertRaisesRegex(ValueError,'F16 weights'):updater.verify_model(settings)

    def test_full_ingestion_also_discovers_root_pdf(self):
        from src.corpus_ingestion import prepare_corpus
        for folder in ['pdfs','slides','transcripts']:(self.corpus/folder).mkdir()
        source=self.pdf('new.PDF')
        paths,report=prepare_corpus(self.corpus)
        chunks,_,_=read_jsonl(paths)
        self.assertTrue(any(c.metadata['source']==str(source.resolve()) for c in chunks))
        self.assertEqual(report['handout']['files'],1)
