import asyncio
import math
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from src.cache import SemanticCache,CacheLookup,get_semantic_cache,namespace

RESULT={'answer':'Grounded answer [S1].','sources':[{'citation':'S1','label':'guide.pdf'}],'refused':False,'slide_images':[]}

class CacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name);self.version='v1'
        self.embedder=Mock(return_value=[1.,0.])
        self.cache=SemanticCache(self.root/'cache.sqlite3',embedder=self.embedder,namespace_factory=lambda:self.version)

    async def seed(self,query='What is context engineering?',result=None):
        lookup=await self.cache.lookup(query)
        await self.cache.store(query,RESULT if result is None else result,lookup)

    async def test_exact_repeat_is_persistent_and_does_not_embed_again(self):
        await self.seed()
        fresh=SemanticCache(self.cache.path,embedder=self.embedder,namespace_factory=lambda:self.version)
        self.embedder.reset_mock()
        result=await fresh.lookup('What is context engineering?')
        self.assertEqual(result.result,RESULT)
        self.assertEqual(result.similarity,1)
        self.embedder.assert_not_called()

    async def test_semantic_threshold_accepts_close_paraphrase_and_rejects_weak_match(self):
        await self.seed()
        self.embedder.return_value=[.96,math.sqrt(1-.96**2)]
        result=await self.cache.lookup('Explain context engineering')
        self.assertEqual(result.result,RESULT)
        self.assertAlmostEqual(result.similarity,.96)
        self.embedder.return_value=[.94,math.sqrt(1-.94**2)]
        self.assertIsNone((await self.cache.lookup('Another question')).result)

    async def test_numbers_negation_and_code_literals_do_not_collide(self):
        for original,other in [('List four components','List five components'),('Use tools','Do not use tools'),('Explain `a()`','Explain `b()`')]:
            await self.seed(original)
            self.assertIsNone((await self.cache.lookup(other)).result)

    async def test_expiry_and_namespace_changes_invalidate_entries(self):
        with patch('src.cache.time.time',return_value=100):await self.seed()
        with patch('src.cache.time.time',return_value=100+self.cache.ttl+1):
            self.assertIsNone((await self.cache.lookup('What is context engineering?')).result)
        await self.seed();self.version='v2'
        self.assertIsNone((await self.cache.lookup('What is context engineering?')).result)

    async def test_change_during_generation_prevents_stale_write(self):
        lookup=await self.cache.lookup('Question');self.version='v2'
        await self.cache.store('Question',RESULT,lookup)
        self.assertIsNone((await self.cache.lookup('Question')).result)
        self.embedder.assert_not_called()

    async def test_refusals_and_empty_answers_are_not_cached(self):
        for result in [{**RESULT,'refused':True},{**RESULT,'answer':''},{**RESULT,'sources':[]}]:
            await self.cache.store('Question',result,CacheLookup('v1'))
        self.embedder.assert_not_called()
        self.assertIsNone((await self.cache.lookup('Question')).result)

    async def test_image_metadata_roundtrips_and_missing_image_forces_miss(self):
        import pymupdf
        image=self.root/'corpus/rendered/slide.png';image.parent.mkdir(parents=True)
        with pymupdf.open() as document:document.new_page().get_pixmap().save(image)
        result={**RESULT,'slide_images':[{'path':str(image),'label':'deck | page 1','dense_score':.8}]}
        with patch('src.retriever.ROOT',self.root):
            await self.seed(result=result)
            self.assertEqual((await self.cache.lookup('What is context engineering?')).result['slide_images'],result['slide_images'])
            image.unlink()
            self.assertIsNone((await self.cache.lookup('What is context engineering?')).result)

    async def test_cache_io_and_embedding_run_off_the_async_loop(self):
        threads=[];loop_thread=threading.get_ident()
        self.cache.embedder=lambda query:(threads.append(threading.get_ident()) or [1.,0.])
        await self.seed();await self.cache.lookup('Explain context engineering')
        self.assertTrue(threads)
        self.assertTrue(all(thread!=loop_thread for thread in threads))

    async def test_eviction_and_clear(self):
        self.cache.max_entries=2
        for i in range(3):
            with patch('src.cache.time.time',return_value=100+i):await self.seed(f'Query {i}')
        with closing(sqlite3.connect(self.cache.path)) as db, db:
            self.assertEqual(db.execute('SELECT count(*) FROM responses').fetchone()[0],2)
        await self.cache.clear()
        self.assertIsNone((await self.cache.lookup('Query 2')).result)

    async def test_corrupt_entry_is_ignored(self):
        await self.seed()
        with closing(sqlite3.connect(self.cache.path)) as db, db:db.execute("UPDATE responses SET result='not JSON'")
        self.assertIsNone((await self.cache.lookup('What is context engineering?')).result)

    def test_disabled_toggle_never_constructs_cache(self):
        with patch('src.cache.dotenv_values',return_value={'ENABLE_SEMANTIC_CACHE':'false'}),patch('src.cache.SemanticCache') as constructor:
            self.assertIsNone(get_semantic_cache());constructor.assert_not_called()

    def test_corpus_and_config_changes_change_namespace(self):
        with patch('src.cache.ROOT',self.root):
            first=namespace()
            (self.root/'.env').write_text('LLM_PROVIDER=ollama')
            second=namespace();self.assertNotEqual(first,second)
            source=self.root/'corpus/transcripts/demo.txt';source.parent.mkdir(parents=True);source.write_text('New evidence')
            self.assertNotEqual(second,namespace())
