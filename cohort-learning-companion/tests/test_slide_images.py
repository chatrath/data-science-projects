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

        import pymupdf

        with pymupdf.open() as doc:

            doc.new_page().get_pixmap().save(self.frame)

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

        self.assertEqual(result['slide_images'],[])



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



    def test_missing_preview_is_repaired_from_local_pdf_without_ocr(self):

        import pymupdf

        from src.ingestion import render_slide_page

        pdf=self.root/'corpus/slides/deck.pdf'

        pdf.parent.mkdir(parents=True)

        with pymupdf.open() as document:

            page=document.new_page();page.insert_text((72,72),'Repair this slide')

            document.save(pdf)

        raw={'source':str(pdf),'page_number':1,'text':'Repair this slide','dense_score':.8}

        with patch('src.ingestion.ocr_textpage',side_effect=AssertionError('No OCR for preview repair')):

            path=relevant_slide_render_path(raw)

        self.assertTrue(Path(path).is_file())

        with pymupdf.open(path) as image:self.assertGreater(image[0].rect.width,0)

        Path(path).unlink()

        self.assertTrue(Path(collect_slide_images([raw])[0]['path']).is_file())

        Path(path).unlink()

        relative={**raw,'source':'deck.pdf'}

        self.assertTrue(Path(relevant_slide_render_path(relative)).is_file())

        import json

        self.assertEqual(len(collect_slide_images(['invalid JSON',None,json.dumps(raw)])),1)



    def test_weak_or_external_slide_does_not_trigger_rendering(self):

        with patch('src.ingestion.render_slide_page') as render:

            self.assertIsNone(relevant_slide_render_path({**self.hit,'dense_score':.7,'metadata':{**self.metadata,'render_path':None}}))

            self.assertIsNone(relevant_slide_render_path({**self.hit,'metadata':{**self.metadata,'source':str(self.root/'private.pdf'),'render_path':None}}))

            render.assert_not_called()



    def test_default_ingestion_generates_actual_png_without_explicit_render_dir(self):

        import pymupdf

        from src.ingestion import load_slides

        pdf=self.root/'deck.pdf'

        with pymupdf.open() as document:

            page=document.new_page();page.insert_text((72,72),'Native slide content')

            document.save(pdf)

        with patch('src.ingestion.ROOT',self.root):records=list(load_slides(pdf))

        self.assertEqual(len(records),1)

        self.assertTrue(Path(records[0]['render_path']).is_relative_to(self.root/'corpus/rendered'))

        self.assertTrue(Path(records[0]['render_path']).read_bytes().startswith(b'\x89PNG'))





    def test_task_context_handout_gets_its_own_page_preview(self):

        import pymupdf

        from src.retriever import relevant_source_render_path

        from src.chain import collect_source_images

        pdf=self.root/'corpus/pdfs/context.pdf'

        pdf.parent.mkdir(parents=True)

        with pymupdf.open() as document:

            document.new_page().insert_text((72,72),'Cover')

            document.new_page().insert_text((72,72),'Four components: Instructions Evidence Tools State')

            document.save(pdf)

        hit={'id':'handout','text':'Four components of task context','dense_score':.813,'score':.016,

             'metadata':{'kind':'handout','source':str(pdf),'page_start':2,'page_end':2}}

        self.assertIsNone(relevant_slide_render_path(hit))

        path=relevant_source_render_path(hit)

        self.assertTrue(Path(path).read_bytes().startswith(b'\x89PNG'))

        images=collect_source_images([hit])

        self.assertEqual(images[0]['page_number'],2)

        self.assertEqual(images[0]['source_kind'],'handout')

        self.assertIn('context.pdf | page 2',images[0]['label'])

        with patch('src.ingestion.render_slide_page') as render:

            self.assertIsNone(relevant_source_render_path({**hit,'dense_score':.74}))

            self.assertIsNone(relevant_source_render_path({**hit,'metadata':{**hit['metadata'],'kind':'transcript'}}))

            render.assert_not_called()





    def test_ocr_failure_does_not_stop_page_rendering_or_later_pages(self):

        import pymupdf

        from src.ingestion import load_slides,valid_render

        pdf=self.root/'ocr-failure.pdf'

        with pymupdf.open() as doc:

            doc.new_page().draw_rect(pymupdf.Rect(10,10,100,100))

            doc.new_page().insert_text((72,72),'Later native-text page')

            doc.save(pdf)

        with patch('src.ingestion.ocr_textpage',side_effect=RuntimeError('OCR unavailable')):

            records=list(load_slides(pdf,self.root/'corpus/rendered'))

        self.assertEqual(len(records),2)

        self.assertEqual(records[0]['ocr_error'],'RuntimeError')

        self.assertTrue(all(valid_render(r['render_path']) for r in records))

        self.assertIn('Later native-text',records[1]['text'])



    def test_one_failed_render_does_not_discard_the_rest_of_a_deck(self):

        import pymupdf

        from src.ingestion import load_slides,render_page,valid_render

        pdf=self.root/'render-failure.pdf'

        with pymupdf.open() as doc:

            for text in ['First','Second']:doc.new_page().insert_text((72,72),text)

            doc.save(pdf)

        def sometimes(page,*args):

            if page.number==0:raise RuntimeError('Cannot render this page')

            return render_page(page,*args)

        with patch('src.ingestion.render_page',side_effect=sometimes):

            records=list(load_slides(pdf,self.root/'corpus/rendered'))

        self.assertEqual(records[0]['render_error'],'RuntimeError')

        self.assertIsNone(records[0]['render_path'])

        self.assertTrue(valid_render(records[1]['render_path']))



    def test_render_retries_at_lower_resolution_and_preserves_good_file_on_failure(self):

        import pymupdf

        from src.ingestion import render_page,valid_render

        with pymupdf.open() as doc:

            page=doc.new_page();pixmap=page.get_pixmap()

            fake=Mock(number=0)

            fake.get_pixmap.side_effect=[RuntimeError('High resolution failed'),pixmap]

            path=render_page(fake,self.root/'retry.pdf',self.root/'corpus/rendered')

            self.assertTrue(valid_render(path))

            self.assertEqual([c.kwargs['matrix'].a for c in fake.get_pixmap.call_args_list],[1.5,1.0])

            before=Path(path).read_bytes()

            fake.get_pixmap.side_effect=RuntimeError('Rendering failed')

            with self.assertRaises(RuntimeError):render_page(fake,self.root/'retry.pdf',self.root/'corpus/rendered')

            self.assertEqual(Path(path).read_bytes(),before)



    def test_wrong_page_corrupt_preview_and_string_page_number_are_repaired(self):

        import pymupdf

        from src.ingestion import render_slide_page,valid_render

        from src.retriever import relevant_source_render_path

        pdf=self.root/'corpus/slides/pages.pdf';pdf.parent.mkdir(parents=True)

        with pymupdf.open() as doc:

            for text in ['First page','Second page']:doc.new_page().insert_text((72,72),text)

            doc.save(pdf)

        wrong=render_slide_page(pdf,1,self.root/'corpus/rendered')

        raw={'text':'Second page','dense_score':.8,'metadata':{'source':str(pdf),'page_number':'2','render_path':wrong}}

        correct=relevant_source_render_path(raw)

        self.assertNotEqual(wrong,correct)

        self.assertTrue(correct.endswith('page_0002.png'))

        Path(correct).write_bytes(b'broken PNG')

        self.assertEqual(relevant_source_render_path(raw),correct)

        self.assertTrue(valid_render(correct))

        alternate={**raw,'metadata':{'kind':'slide','source':str(pdf),'page_start':'2'}}

        self.assertEqual(relevant_source_render_path(alternate),correct)



    def test_standalone_failure_does_not_replace_previous_jsonl_snapshot(self):

        import pymupdf

        from src.ingestion import main

        pdf=self.root/'valid.pdf';locked=self.root/'locked.pdf';output=self.root/'slides.jsonl'

        with pymupdf.open() as doc:

            doc.new_page().insert_text((72,72),'Valid page');doc.save(pdf)

            doc.save(locked,encryption=pymupdf.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='secret')

        output.write_text('previous successful snapshot',encoding='utf-8')

        argv=['ingestion',str(pdf),str(locked),'--output',str(output),'--render-dir',str(self.root/'corpus/rendered')]

        with patch('sys.argv',argv),self.assertRaises(ValueError):main()

        self.assertEqual(output.read_text(encoding='utf-8'),'previous successful snapshot')






    def test_reranker_preserves_exact_source_page_and_dense_confidence(self):
        import pymupdf
        from src.ingestion import load_slides
        from src.retriever import rerank_candidates
        pdf=self.root/'corpus/slides/ranked.pdf';pdf.parent.mkdir(parents=True)
        with pymupdf.open() as document:
            for text in ['First slide','Best slide','Irrelevant slide']:
                document.new_page().insert_text((72,72),text)
            document.save(pdf)
        records=list(load_slides(pdf,self.root/'corpus/rendered'))
        hits=[{'id':str(i),'text':r['text'],'score':.016,'dense_score':.8,
               'metadata':{k:v for k,v in r.items() if k!='text'}} for i,r in enumerate(records)]
        model=Mock();model.predict.return_value=[4.,9.,-5.]
        with patch('src.retriever._RERANKER',model):selected=rerank_candidates('Question',hits,3)
        images=collect_slide_images(selected)
        self.assertEqual([image['page_number'] for image in images],[2,1])
        self.assertTrue(images[0]['path'].endswith('page_0002.png'))
        self.assertTrue(images[1]['path'].endswith('page_0001.png'))
        self.assertTrue(all(image['dense_score']==.8 for image in images))



if __name__=='__main__':

    unittest.main()
