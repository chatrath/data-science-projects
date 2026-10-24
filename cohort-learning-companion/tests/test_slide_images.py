from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from chainlit.context import context_var
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import app
from src.chain import RAGChain, REFUSAL, collect_slide_images
from src.provider import ProviderSettings
from src.retriever import resolve_render_path, relevant_slide_render_path, high_confidence_visual


class SlideImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.frame=self.root/'corpus/rendered/deck/page_0025.png'
        self.frame.parent.mkdir(parents=True)
        self.frame.write_bytes(b'test-image-placeholder')
        patcher=patch('src.retriever.ROOT',self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.metadata={'kind':'slide','source':'deck.pdf','page_number':25,'render_path':'corpus/rendered/deck/page_0025.png'}
        self.hit={'id':'slide25','text':'Slide text','score':.016,'dense_score':.75,'metadata':self.metadata}

    def test_render_resolution_supports_relative_and_relocated_paths(self):
        self.assertEqual(resolve_render_path(self.metadata),str(self.frame.resolve()))
        self.assertEqual(resolve_render_path({'render_path':'X:/old/project/corpus/rendered/deck/page_0025.png'}),str(self.frame.resolve()))

    def test_missing_outside_and_traversal_paths_are_ignored(self):
        outside=self.root/'private.png'
        outside.write_bytes(b'private')
        for path in [str(outside),'corpus/rendered/../../private.png','corpus/rendered/missing.png']:
            self.assertIsNone(resolve_render_path({'render_path':path}))

    def test_visual_threshold_uses_only_valid_raw_cosine(self):
        for score in [None, .749999, float('nan'), float('inf'), '0.9', True, 1.1]:
            with self.subTest(score=score):
                self.assertFalse(high_confidence_visual(score))
                hit = {**self.hit, 'dense_score':score, 'score':999, 'bm25_score':100}
                self.assertIsNone(relevant_slide_render_path(hit))
                self.assertEqual(collect_slide_images([hit]), [])
        self.assertTrue(high_confidence_visual(.75))
        self.assertTrue(high_confidence_visual(.9))
        self.assertIsNone(relevant_slide_render_path({**self.hit, 'metadata':{**self.metadata,'kind':'transcript'}}))

    def test_transcript_answer_does_not_force_a_weak_slide(self):
        weak = {**self.hit, 'dense_score':.74, 'selection_reason':'source_reservation'}
        retriever = Mock()
        retriever.search.return_value = [
            {'id':'transcript','text':'Transcript evidence','score':.02,'dense_score':.9,
             'metadata':{'kind':'transcript','source':'demo.txt'}}, weak]
        chain = RAGChain(settings=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}),
                         retriever=retriever,model=RunnableLambda(lambda prompt,**kwargs:AIMessage(content='Transcript answer [S1]')))
        result = chain.ask('Question')
        self.assertEqual(result['slide_images'], [])
        self.assertEqual(app.slide_image_elements(result), [])

    def test_ui_rechecks_score_even_for_precollected_frames(self):
        image = collect_slide_images([self.hit])[0]
        for score in [None,.74,float('nan')]:
            self.assertEqual(app.slide_image_elements({'slide_images':[{**image,'dense_score':score}]}), [])

    def test_frames_survive_refusal_and_are_deduplicated(self):
        self.assertEqual(len(collect_slide_images([self.hit,self.hit])),1)
        retriever=Mock()
        retriever.search.return_value=[self.hit]
        chain=RAGChain(settings=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}),retriever=retriever,model=RunnableLambda(lambda prompt,**kwargs:AIMessage(content=REFUSAL)))
        result=chain.ask('Question')
        self.assertTrue(result['refused'])
        self.assertEqual(result['sources'],[])
        self.assertEqual(result['slide_images'][0]['path'],str(self.frame.resolve()))

    def test_chainlit_image_is_inline_and_has_page_caption(self):
        token=context_var.set(SimpleNamespace(session=SimpleNamespace(thread_id='test-thread')))
        self.addCleanup(context_var.reset,token)
        result={'slide_images':collect_slide_images([self.hit,self.hit])}
        images=app.slide_image_elements(result)
        self.assertEqual(len(images),1)
        self.assertEqual(images[0].display,'inline')
        self.assertIn('deck.pdf | page 25',images[0].name)
        self.assertEqual(images[0].path,str(self.frame.resolve()))
        self.frame.unlink()
        self.assertEqual(app.slide_image_elements(result),[])


if __name__=='__main__':
    unittest.main()
