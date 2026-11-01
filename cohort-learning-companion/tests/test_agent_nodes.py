import unittest
from unittest.mock import Mock,patch
from langgraph.graph import StateGraph,START,END
from src.agent_nodes import decompose_query,_get_decomposer,DECOMPOSITION_SCHEMA,retrieve_documents
from src.agent_state import AgentState


class DecompositionTests(unittest.TestCase):
    def invoke(self,response,question='Compare sparse and dense retrieval.',step=0):
        model=Mock();model.invoke.return_value=response
        state={'question':question,'loop_step':step}
        before=state.copy()
        with patch('src.agent_nodes._get_decomposer',return_value=model):result=decompose_query(state)
        self.assertEqual(state,before)
        return result,model

    def test_simple_question_preserves_original_verbatim(self):
        question='  What is RAG?  '
        result,model=self.invoke({'is_complex':False,'sub_queries':['Model paraphrase']},question,4)
        self.assertEqual(result,{'sub_queries':[question],'loop_step':5})
        self.assertEqual(model.invoke.call_args.args[0][1]['content'],question)

    def test_complex_question_splits_into_two_or_three_queries(self):
        for queries in [['Explain BM25.','Explain dense retrieval.'],['Explain BM25.','Explain dense retrieval.','How are they combined?']]:
            result,_=self.invoke({'is_complex':True,'sub_queries':queries})
            self.assertEqual(result,{'sub_queries':queries,'loop_step':1})

    def test_model_and_configuration_errors_fall_back(self):
        for failure in [ConnectionError('offline'),RuntimeError('invalid configuration'),TimeoutError()]:
            with patch('src.agent_nodes._get_decomposer',side_effect=failure):
                self.assertEqual(decompose_query({'question':'Question'}),{'sub_queries':['Question'],'loop_step':1})
        model=Mock();model.invoke.side_effect=ValueError('Malformed model output')
        with patch('src.agent_nodes._get_decomposer',return_value=model):
            self.assertEqual(decompose_query({'question':'Question','loop_step':2})['sub_queries'],['Question'])

    def test_invalid_or_duplicate_plans_fall_back(self):
        bad=[None,'not structured',{}, {'is_complex':'true','sub_queries':['A','B']}]
        bad += [{'is_complex':True,'sub_queries':queries} for queries in [[],['A'],['A','B','C','D'],['A',' a '],['A',''],['A',None],['A','x'*1025]]]
        for response in bad:
            with self.subTest(response=response):
                result,_=self.invoke(response,question='Original')
                self.assertEqual(result,{'sub_queries':['Original'],'loop_step':1})

    def test_empty_question_does_not_call_model(self):
        with patch('src.agent_nodes._get_decomposer') as model:
            self.assertEqual(decompose_query({'question':''}),{'sub_queries':[''],'loop_step':1})
            model.assert_not_called()

    def test_factory_uses_configured_local_model_and_json_schema(self):
        with patch('src.agent_nodes.load_dotenv'),patch.dict('os.environ',{'LLM_PROVIDER':'openai','OLLAMA_BASE_URL':'http://localhost:11434','OLLAMA_CHAT_MODEL':'other-model'}),patch('src.agent_nodes.create_provider_client') as factory:
            _get_decomposer()
        settings=factory.call_args.args[0]
        self.assertEqual((settings.provider,settings.chat_model),('ollama','other-model'))
        factory.return_value.with_structured_output.assert_called_once_with(DECOMPOSITION_SCHEMA,method='json_schema')

    def test_node_integrates_as_a_langgraph_partial_state_update(self):
        graph=StateGraph(AgentState);graph.add_node('decompose',decompose_query)
        graph.add_edge(START,'decompose');graph.add_edge('decompose',END)
        model=Mock();model.invoke.return_value={'is_complex':True,'sub_queries':['First part?','Second part?']}
        with patch('src.agent_nodes._get_decomposer',return_value=model):
            result=graph.compile().invoke({'question':'Two parts?','documents':['Retained evidence'],'loop_step':3})
        self.assertEqual(result['sub_queries'],['First part?','Second part?'])
        self.assertEqual(result['documents'],['Retained evidence'])
        self.assertEqual(result['loop_step'],4)


class RetrievalNodeTests(unittest.TestCase):
    def run_node(self, state, responses, images=None):
        settings=Mock(rag_top_k=3)
        retriever=Mock();retriever.search.side_effect=responses
        with patch('src.agent_nodes.load_dotenv'), patch('src.agent_nodes.ProviderSettings.from_env',return_value=settings), patch('src.agent_nodes.HybridRetriever',return_value=retriever), patch('src.agent_nodes.collect_source_images',side_effect=images or (lambda hits: [])):
            result=retrieve_documents(state)
        return result,retriever

    def test_deduplicates_and_shares_top_k_across_queries(self):
        a={'id':'a','text':'First','metadata':{}}
        b={'id':'b','text':'Second','metadata':{}}
        c={'id':'c','text':'Third','metadata':{}}
        state={'question':'Compare','sub_queries':['One?','Two?'],'loop_step':4}
        before=state.copy()
        result,retriever=self.run_node(state,[[a,b,c],[a,{'id':'different','text':'First'},c]])
        self.assertEqual(result['documents'],['First','Second','Third'])
        self.assertEqual(result['slide_images'],[])
        self.assertEqual([hit['id'] for hit in result['retrieved_chunks']],['a','b','c'])
        self.assertEqual(state,before)
        self.assertEqual(retriever.search.call_count,2)
        self.assertTrue(all(call.kwargs['rerank'] for call in retriever.search.call_args_list))
        self.assertTrue(all(call.kwargs['min_rerank_score']==0.5 for call in retriever.search.call_args_list))

    def test_multi_question_keeps_exact_topic_heading_in_shared_budget(self):
        definition={'id':'definition','text':'An embedding is a vector.'}
        generic={'id':'generic','text':'An embedding model converts text to vectors.'}
        catalog={'id':'catalog','text':'Popular embedding models\nOpenAI text-embedding-3-small, bge-large, nomic-embed.'}
        training={'id':'training','text':'Contrastive learning pulls similar pairs together.'}
        result,retriever=self.run_node({'sub_queries':[
            'What is an embedding?', 'Popular embedding models', 'How does embedding model learn?']},
            [[definition], [generic,catalog], [training]])
        self.assertEqual([h['id'] for h in result['retrieved_chunks']],['definition','catalog','training'])
        self.assertEqual(len(result['documents']),3)
        self.assertEqual(retriever.search.call_count,3)

    def test_heading_priority_requires_full_line_not_incidental_phrase(self):
        first={'id':'first','text':'Primary relevant evidence'}
        incidental={'id':'other','text':'This paragraph mentions popular embedding models only in passing.'}
        result,_=self.run_node({'sub_queries':['Popular embedding models','Other question']},[[first,incidental],[]])
        self.assertEqual(result['retrieved_chunks'][0]['id'],'first')

    def test_single_query_keeps_reranker_order_even_with_heading_match(self):
        first={'id':'first','text':'Strongest evidence'}
        heading={'id':'heading','text':'Popular embedding models\nModel names'}
        result,_=self.run_node({'question':'Popular embedding models'},[[first,heading]])
        self.assertEqual([h['id'] for h in result['retrieved_chunks']],['first','heading'])

    def test_normalizes_json_and_keeps_images_only_for_selected_evidence(self):
        import json
        hits=[json.dumps({'page_content':'Slide text','metadata':{'source':'slides.pdf'}}),None,{'text':'Other','metadata':{}}]
        image=Mock(return_value=[{'path':'valid.png'}])
        result,_=self.run_node({'sub_queries':['Topic']},[hits],image)
        self.assertEqual(result['documents'],['Slide text','Other'])
        self.assertEqual(result['slide_images'],['valid.png'])
        self.assertEqual(result['retrieved_chunks'][0]['metadata']['source'],'slides.pdf')
        self.assertEqual(image.call_count,2)
        self.assertEqual(image.call_args_list[0].args[0][0]['metadata']['source'],'slides.pdf')

    def test_query_failure_retains_other_evidence(self):
        result,_=self.run_node({'sub_queries':['Broken','Working']},[ConnectionError(),[{'text':'Evidence'}]])
        self.assertEqual(result['documents'],['Evidence'])

    def test_image_failure_retains_text(self):
        result,_=self.run_node({'question':'Fallback'},[[{'text':'Evidence'}]],Mock(side_effect=OSError()))
        self.assertEqual(result['documents'],['Evidence'])
        self.assertEqual(result['slide_images'],[])

    def test_empty_input_and_initialization_failure(self):
        with patch('src.agent_nodes.HybridRetriever') as factory:
            self.assertEqual(retrieve_documents({'sub_queries':[]}),{'documents':[],'slide_images':[],'retrieved_chunks':[]})
            factory.assert_not_called()
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',side_effect=ValueError()):
            self.assertEqual(retrieve_documents({'question':'Question'}),{'documents':[],'slide_images':[],'retrieved_chunks':[]})

    def test_duplicate_queries_only_searched_once_and_empty_hits_skipped(self):
        result,retriever=self.run_node({'sub_queries':[' Topic ','Topic','',None]},[[{'text':' '},'invalid']])
        self.assertEqual(result,{'documents':[],'slide_images':[],'retrieved_chunks':[]})
        retriever.search.assert_called_once()

    def test_compiled_graph_partial_update(self):
        graph=StateGraph(AgentState);graph.add_node('retrieve',retrieve_documents)
        graph.add_edge(START,'retrieve');graph.add_edge('retrieve',END)
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=Mock(rag_top_k=3)),patch('src.agent_nodes.HybridRetriever') as factory,patch('src.agent_nodes.collect_source_images',return_value=[]):
            factory.return_value.search.return_value=[{'text':'Evidence'}]
            result=graph.compile().invoke({'question':'Question','sub_queries':['Question'],'loop_step':2})
        self.assertEqual(result['documents'],['Evidence'])
        self.assertEqual(result['loop_step'],2)


class DocumentGradingTests(unittest.TestCase):
    def test_nonempty_reranked_evidence_passes_without_model_or_configuration(self):
        from src.agent_nodes import grade_documents
        import copy
        state={'question':'Q','documents':['Evidence'], 'retrieved_chunks':[
            {'text':'Evidence','rerank_score':0.5,'dense_score':0.61}],
            'slide_images':['slide.png'],'generation_grade':'no','loop_step':2}
        before=copy.deepcopy(state)
        with patch('src.agent_nodes.create_provider_client',side_effect=AssertionError('No grading model')),patch('src.agent_nodes.load_dotenv',side_effect=AssertionError('No grading config')):
            self.assertEqual(grade_documents(state),{'generation_grade':'yes'})
        self.assertEqual(state,before)

    def test_missing_blank_and_invalid_evidence_is_no_without_model(self):
        from src.agent_nodes import grade_documents
        with patch('src.agent_nodes.create_provider_client') as factory:
            for docs in [None,[],['','  '],[None,{},123]]:
                self.assertEqual(grade_documents({'question':'Q','documents':docs}),{'generation_grade':'no'})
            self.assertEqual(grade_documents({'question':'Q'}),{'generation_grade':'no'})
            factory.assert_not_called()

    def test_does_not_apply_visual_cosine_threshold_to_text(self):
        from src.agent_nodes import grade_documents
        state={'question':'Q','documents':['Valid keyword evidence'],
               'retrieved_chunks':[{'text':'Valid keyword evidence','dense_score':0.2,'rerank_score':4.0}]}
        self.assertEqual(grade_documents(state),{'generation_grade':'yes'})

    def test_langgraph_partial_update_retains_evidence_and_images(self):
        from src.agent_nodes import grade_documents
        graph=StateGraph(AgentState);graph.add_node('grade',grade_documents)
        graph.add_edge(START,'grade');graph.add_edge('grade',END)
        result=graph.compile().invoke({'question':'Q','documents':['Evidence'],
                                      'slide_images':['preview.png'],'loop_step':2})
        self.assertEqual(result['generation_grade'],'yes')
        self.assertEqual(result['documents'],['Evidence'])
        self.assertEqual(result['slide_images'],['preview.png'])
        self.assertEqual(result['loop_step'],2)


class ComparisonPlanningTests(unittest.TestCase):
    def test_exact_reported_question_preserves_both_subjects_without_model(self):
        question='Explain difference between dense retrieval and sparse retrieval?'
        with patch('src.agent_nodes._get_decomposer') as factory:
            result=decompose_query({'question':question,'loop_step':2})
        self.assertEqual(result,{'sub_queries':['Explain dense retrieval.','Explain sparse retrieval.',question],'loop_step':3})
        factory.assert_not_called()

    def test_comparison_variants(self):
        from src.agent_nodes import _comparison_queries
        for question in ['What is the difference between dense retrieval and sparse retrieval?',
                         'Explain the differences between dense retrieval and sparse retrieval.']:
            result=_comparison_queries(question)
            self.assertEqual(result[:2],['Explain dense retrieval.','Explain sparse retrieval.'])
            self.assertEqual(result[-1],question)

    def test_constraints_and_ambiguous_subjects_use_existing_planner(self):
        from src.agent_nodes import _comparison_queries
        for question in ['Explain difference between dense and sparse retrieval?',
                         'Explain difference between dense retrieval and sparse retrieval for code?',
                         'Explain difference between dense retrieval and sparse retrieval and show code?',
                         'What is dense retrieval?']:
            self.assertIsNone(_comparison_queries(question))


class CompoundExplanationPlanningTests(unittest.TestCase):
    def test_reported_semantic_chunking_bm25_question_retains_named_topics(self):
        question='Explain semantic chunking and how BM25 is better than vectors?'
        with patch('src.agent_nodes._get_decomposer') as factory:
            result=decompose_query({'question':question,'loop_step':0})
        self.assertEqual(result,{'sub_queries':['Explain semantic chunking.','Explain BM25.',
                                              'When does BM25 beat vectors?'],'loop_step':1})
        factory.assert_not_called()

    def test_pattern_is_general_and_preserves_additional_constraints(self):
        from src.agent_nodes import _compound_explanation_queries
        self.assertEqual(_compound_explanation_queries('Explain network caching and how Redis is better than Memcached?'),
                         ['Explain network caching.','Explain Redis.','When does Redis beat Memcached?'])
        self.assertIsNone(_compound_explanation_queries('Explain semantic chunking and how BM25 is better than vectors for multilingual data?'))
