"""Exact cited-page identity is separate from approximate visual matching."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pymupdf

import app
from src.chain import RAGChain, REFUSAL, format_context
from src.previews import cited_page_image, verified_cited_page_image
from src.agent_graph import agent_result


class ExactCitedPageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pdf=self.root/'corpus/slides/deck.pdf';self.pdf.parent.mkdir(parents=True)
        with pymupdf.open() as doc:
            for text in ['Embedding definition','Popular embedding models: bge-large, nomic-embed','Contrastive training']:
                doc.new_page().insert_text((40,60),text)
            doc.save(self.pdf)
        self.patches=[patch('src.previews.ROOT',self.root),patch('src.retriever.ROOT',self.root)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def source(self,page=2):
        return {'citation':'S2','label':'deck.pdf | page '+str(page),'chunk_id':'slide'+str(page),
                'rerank_score':1.7,'dense_score':.63,
                'metadata':{'source':str(self.pdf),'kind':'slide','page_number':page}}

    def test_cited_low_cosine_page_is_rendered_without_fabricating_score(self):
        source=self.source()
        image=cited_page_image(source)
        self.assertIsNotNone(image)
        self.assertEqual(image['dense_score'],.63)
        self.assertEqual(image['preview_type'],'cited_page')
        self.assertTrue(Path(image['path']).read_bytes().startswith(b'\x89PNG'))
        self.assertTrue(verified_cited_page_image(image,[source]))
        self.assertIn('page_0002',image['path'])

    def test_missing_or_corrupt_exact_preview_is_repaired(self):
        source=self.source();image=cited_page_image(source)
        Path(image['path']).write_bytes(b'broken')
        repaired=cited_page_image(source)
        self.assertTrue(Path(repaired['path']).read_bytes().startswith(b'\x89PNG'))

    def test_unknown_weak_or_nonfinite_rerank_scores_do_not_qualify(self):
        for score in [None,.49,float('nan'),float('inf'),True,'8.0']:
            source=self.source();source['rerank_score']=score
            self.assertIsNone(cited_page_image(source))

    def test_outside_pdf_invalid_page_and_ambiguous_section_fail_closed(self):
        sources=[]
        for page in [0,True,4,'2']:
            sources.append(self.source(page))
        outside=self.root/'private.pdf'
        outside.write_bytes(self.pdf.read_bytes())
        s=self.source();s['metadata']['source']=str(outside);sources.append(s)
        s=self.source();s['metadata']={'kind':'handout','source':str(self.pdf),'page_start':1,'page_end':3};sources.append(s)
        for source in sources:self.assertIsNone(cited_page_image(source))

    def test_wrong_page_path_missing_citation_and_forged_chunk_id_are_rejected(self):
        source=self.source();image=cited_page_image(source)
        other=cited_page_image(self.source(1))
        for forged in [{**image,'path':other['path']},{**image,'citation':'S99'},
                       {**image,'page_number':1},{**image,'chunk_id':'other'},
                       {**image,'source_kind':'transcript'}]:
            self.assertFalse(verified_cited_page_image(forged,[source]))
        self.assertFalse(verified_cited_page_image(image,[]))

    def test_only_active_cited_pages_added_and_refusals_remain_image_free(self):
        sources=[self.source(1),self.source(2),self.source(3)]
        for i,source in enumerate(sources,1):source['citation']='S'+str(i)
        prepared={'sources':sources,'slide_images':[],'timings':{}}
        result=RAGChain._finish('Definition [S1]. Model names [S2].',prepared)
        self.assertEqual([i['page_number'] for i in result['slide_images']],[1,2])
        self.assertEqual(RAGChain._finish(REFUSAL,prepared)['slide_images'],[])

    def test_duplicate_cited_page_is_displayed_once(self):
        source=self.source();duplicate={**self.source(),'citation':'S3'}
        result=RAGChain._finish('Supported [S2][S3].',{'sources':[source,duplicate],'slide_images':[],'timings':{}})
        self.assertEqual(len(result['slide_images']),1)

    def test_graph_final_result_repairs_omission_and_ui_accepts_verified_page(self):
        source=self.source()
        result=agent_result({'generation':'Model names [S2].','sources':[source],'slide_images':[]})
        self.assertEqual(len(result['slide_images']),1)
        with patch.object(app.cl,'Image',return_value='native image') as image:
            self.assertEqual(app.slide_image_elements(result),['native image'])
            self.assertEqual(image.call_args.kwargs['display'],'inline')
        with patch.object(app.cl,'Image') as image:
            self.assertEqual(app.slide_image_elements({**result,'sources':[]}),[])
            image.assert_not_called()

    def test_context_retains_real_scores_for_post_generation_verification(self):
        hit={'id':'slide2','text':'Model names','score':.02,'dense_score':.63,'rerank_score':1.7,
             'metadata':self.source()['metadata']}
        _,sources=format_context([hit])
        self.assertEqual(sources[0]['chunk_id'],'slide2')
        self.assertEqual(sources[0]['dense_score'],.63)
        self.assertEqual(sources[0]['rerank_score'],1.7)

    def test_failed_render_keeps_answer_and_citations(self):
        source=self.source()
        with patch('src.previews.render_slide_page',side_effect=RuntimeError('render unavailable')):
            result=RAGChain._finish('Supported [S2].',{'sources':[source],'slide_images':[],'timings':{}})
        self.assertFalse(result['refused'])
        self.assertEqual(result['answer'],'Supported [S2].')
        self.assertEqual(result['slide_images'],[])

    def test_old_windows_source_path_rebases_to_exact_local_pdf(self):
        source=self.source()
        source['metadata']['source']=r'C:\old-project\corpus\slides\deck.pdf'
        image=cited_page_image(source)
        self.assertIsNotNone(image)
        self.assertTrue(verified_cited_page_image(image,[source]))

    def test_corrupt_pdf_fails_ui_validation_without_discarding_answer(self):
        source=self.source();image=cited_page_image(source)
        self.pdf.write_bytes(b'corrupt PDF')
        self.assertFalse(verified_cited_page_image(image,[source]))

    def test_wrong_existing_frame_is_replaced_by_exact_cited_page(self):
        source=self.source(2)
        wrong=cited_page_image(self.source(1))
        source['metadata']['render_path']=wrong['path']
        result=RAGChain._finish('Model names [S2].',{'sources':[source],'slide_images':[wrong['path']],'timings':{}})
        self.assertEqual(len(result['slide_images']),1)
        self.assertEqual(result['slide_images'][0]['page_number'],2)
        self.assertTrue(verified_cited_page_image(result['slide_images'][0],result['sources']))
