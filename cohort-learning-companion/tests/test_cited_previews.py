import unittest
from unittest.mock import patch
from src.chain import RAGChain,filter_cited_images,REFUSAL
from src.agent_graph import agent_result


def resolve(metadata):
    raw=metadata.get('render_path')
    if not raw or raw=='missing':return None
    return raw.replace('old/','current/')


class CitedPreviewTests(unittest.TestCase):
    def sources(self):
        return [{'citation':f'S{i}','label':f'source{i}.pdf | page {i}',
                 'metadata':{'source':f'source{i}.pdf','page_number':i,'render_path':f'old/page{i}.png'}} for i in [1,2,3]]

    def test_standard_rag_drops_uncited_third_image_and_keeps_confidence(self):
        sources=self.sources()
        images=[{'path':f'current/page{i}.png','dense_score':.8,'label':f'page{i}'} for i in [1,2,3]]
        with patch('src.chain.resolve_render_path',side_effect=resolve):
            result=RAGChain._finish('Supported answer [S1][S2].',{'sources':sources,'slide_images':images,'timings':{}})
        self.assertEqual([x['citation'] for x in result['sources']],['S1','S2'])
        self.assertEqual([x['path'] for x in result['slide_images']],['current/page1.png','current/page2.png'])
        self.assertTrue(all(x['dense_score']==.8 for x in result['slide_images']))
        self.assertEqual(len(images),3)

    def test_agent_drops_uncited_frame_and_rebases_paths(self):
        with patch('src.chain.resolve_render_path',side_effect=resolve):
            result=agent_result({'generation':'Answer [S1][S2].','sources':self.sources()[:2],
                                 'slide_images':['current/page1.png','old/page2.png','current/page3.png']})
        self.assertEqual(result['slide_images'],['current/page1.png','current/page2.png'])

    def test_missing_provenance_and_missing_files_do_not_attach_frames(self):
        with patch('src.chain.resolve_render_path',side_effect=resolve):
            self.assertEqual(filter_cited_images(['page1.png'],[{'metadata':{}}]),[])
            self.assertEqual(filter_cited_images(['missing'],[{'metadata':{'render_path':'missing'}}]),[])
            result=agent_result({'generation':REFUSAL,'sources':self.sources(),'slide_images':['current/page1.png']})
        self.assertEqual(result['slide_images'],[])

    def test_same_page_is_displayed_once_for_multiple_citations(self):
        with patch('src.chain.resolve_render_path',side_effect=resolve):
            result=filter_cited_images(['old/page1.png','current/page1.png'],[
                {'metadata':{'render_path':'old/page1.png'}},
                {'metadata':{'render_path':'current/page1.png'}}])
        self.assertEqual(result,['current/page1.png'])
