import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
from src.agent_nodes import decompose_query, retrieve_documents, _generation_context, generate_answer
from src.provider import ProviderSettings
from src.query_planning import explicit_question_parts, topic_terms
from src.chain import REFUSAL

NEW='What is an embedding? Also, what are popular embedding models? Can you also explain how does an embedding model learns?'
OLD='What is an embedding and what are popular embedding models? How does an embedding model learns?'


def hit(identity,text,score=2,parts=None):
    return {'id':identity,'text':text,'score':.02,'rerank_score':score,'dense_score':.8,
            'metadata':{'source':identity+'.pdf',**({'retrieval_parts':parts} if parts else {})}}


class PlanningTests(unittest.TestCase):
    def test_reported_variants_preserve_all_three_topics_without_model(self):
        with patch('src.agent_nodes._get_decomposer',side_effect=AssertionError('No model needed')):
            plans=[decompose_query({'question':q})['sub_queries'] for q in [NEW,OLD]]
        self.assertEqual(plans[0],plans[1])
        self.assertEqual(len(plans[0]),3)
        self.assertEqual(topic_terms(plans[0][1]),['popular','embedding','models'])
        self.assertIn('learns',plans[0][2])

    def test_heading_focus_ignores_question_and_politeness_prefixes(self):
        for query in ['List popular embedding models','What are popular embedding models?',
                      'Can you also explain what are popular embedding models?']:
            self.assertEqual(topic_terms(query),['popular','embedding','models'])

    def test_pronouns_quoted_syntax_and_single_questions_use_planner(self):
        for query in ['What is an embedding? How does it learn?',
                      'What is "X? Y"? How does X work?',
                      'What is the difference between dense and sparse retrieval?']:
            self.assertIsNone(explicit_question_parts(query))

    def test_subject_conjunction_is_not_split(self):
        parts=explicit_question_parts('What are dense and sparse retrieval? Why use hybrid retrieval?')
        self.assertEqual(len(parts),2)
        self.assertIn('dense and sparse',parts[0])


class AdaptiveRetrievalTests(unittest.TestCase):
    def settings(self,**kwargs):
        return replace(ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}),**kwargs)

    def retrieve(self,queries,pools,settings=None):
        retriever=Mock();retriever.search.side_effect=pools
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=settings or self.settings()),patch('src.agent_nodes.HybridRetriever',return_value=retriever),patch('src.agent_nodes.collect_source_images',return_value=[]):
            result=retrieve_documents({'question':'Multi-part question','sub_queries':queries})
        return result,retriever

    def test_two_precise_chunks_per_part_up_to_six_with_all_anchors_first(self):
        pools=[[hit(f'{p}-{i}',f'Part {p} evidence {i}',score=4-i) for i in range(3)] for p in range(3)]
        result,retriever=self.retrieve(['First topic','Second topic','Third topic'],pools)
        self.assertEqual(len(result['documents']),6)
        self.assertEqual([h['metadata']['retrieval_parts'] for h in result['retrieved_chunks'][:3]],[[1],[2],[3]])
        self.assertTrue(all(call.kwargs['min_rerank_score']==.5 for call in retriever.search.call_args_list))
        self.assertNotIn('Part 0 evidence 2',result['documents'])

    def test_question_word_heading_match_preserves_model_list(self):
        result,_=self.retrieve(['What is an embedding?','What are popular embedding models?'],[
            [hit('def','An embedding is a vector.')],
            [hit('generic','Generic definition',8),hit('catalog','Popular embedding models\nCohere embed-v3, bge-large.',1.5)]])
        self.assertEqual(result['retrieved_chunks'][1]['id'],'catalog')

    def test_shared_evidence_is_deduplicated_and_provenance_is_merged(self):
        shared=hit('shared','One useful shared explanation.')
        result,_=self.retrieve(['First topic','Second topic'],[[shared],[{**shared,'dense_score':.9}]])
        self.assertEqual(len(result['documents']),1)
        self.assertEqual(result['retrieved_chunks'][0]['metadata']['retrieval_parts'],[1,2])
        self.assertEqual(result['retrieved_chunks'][0]['dense_score'],.9)
        self.assertNotIn('retrieval_parts',shared['metadata'])

    def test_single_question_still_uses_original_three_chunk_budget(self):
        result,_=self.retrieve(['Single topic'],[[hit(str(i),str(i)) for i in range(4)]])
        self.assertEqual(len(result['documents']),3)

    def test_configured_cap_still_reserves_each_of_three_parts(self):
        result,_=self.retrieve(['First topic','Second topic','Third topic'],[
            [hit(f'{p}-{i}',f'{p} useful evidence {i}') for i in range(2)] for p in range(3)],
            self.settings(rag_multi_max_chunks=3))
        self.assertEqual(len(result['documents']),3)
        self.assertEqual([h['metadata']['retrieval_parts'] for h in result['retrieved_chunks']],[[1],[2],[3]])

    def test_failed_middle_part_does_not_relabel_third_part(self):
        result,_=self.retrieve(['First topic','Second topic','Third topic'],[
            [hit('first','First evidence')],ConnectionError(),[hit('third','Third evidence')]])
        self.assertEqual([h['metadata']['retrieval_parts'] for h in result['retrieved_chunks']],[[1],[3]])
        state={'question':'Multi','sub_queries':['First topic','Second topic','Third topic'],**result}
        self.assertFalse(_generation_context(state,self.settings())['coverage_complete'])


class ContextAllocationTests(unittest.TestCase):
    def state(self,hits):
        return {'question':'First topic? Second topic? Third topic?',
                'sub_queries':['First topic','Second topic','Third topic'],
                'documents':[h['text'] for h in hits],'retrieved_chunks':hits,'slide_images':[]}

    def test_adaptive_context_retains_all_six_whole_chunks_when_they_fit(self):
        hits=[hit(str(i),f'Complete excerpt {i}\n```python\n    print({i})\n```',parts=[i%3+1]) for i in range(6)]
        prepared=_generation_context(self.state(hits),ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}))
        self.assertTrue(prepared['coverage_complete'])
        self.assertEqual(len(prepared['sources']),6)
        self.assertEqual(prepared['timings']['omitted_chunks'],0)
        for h in hits:self.assertIn(h['text'],prepared['context'])

    def test_context_reserves_parts_before_large_extra_chunks(self):
        anchors=[hit(str(i),'Anchor evidence '+str(i),parts=[i+1]) for i in range(3)]
        extra=hit('extra','x'*1500,parts=[1])
        settings=replace(ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}),rag_multi_context_bytes=500)
        prepared=_generation_context(self.state([anchors[0],extra,anchors[1],anchors[2]]),settings)
        self.assertTrue(prepared['coverage_complete'])
        self.assertEqual(prepared['timings']['omitted_chunks'],1)
        self.assertEqual(len(prepared['sources']),3)
        for h in anchors:self.assertIn(h['text'],prepared['context'])
        self.assertNotIn(extra['text'],prepared['context'])

    def test_overflow_that_loses_an_entire_part_refuses_instead_of_partial_answer(self):
        hits=[hit('one','First evidence',parts=[1]),hit('two','x'*1000,parts=[2]),hit('three','Third evidence',parts=[3])]
        settings=replace(ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}),rag_multi_context_bytes=400)
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=settings),patch('src.agent_nodes.create_provider_client') as model:
            result=generate_answer(self.state(hits))
        self.assertEqual(result['generation'],REFUSAL)
        model.assert_not_called()

    def test_new_settings_are_validated(self):
        for name,value in [('RAG_MULTI_MAX_CHUNKS','0'),('RAG_MULTI_CHUNKS_PER_QUERY','4'),('RAG_MULTI_CONTEXT_BYTES','1'),('RAG_MULTI_NUM_PREDICT','2000')]:
            with self.assertRaises(ValueError):ProviderSettings.from_env({'LLM_PROVIDER':'ollama',name:value})
