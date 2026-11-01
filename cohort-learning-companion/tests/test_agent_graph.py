import unittest
from unittest.mock import Mock, patch

from src.agent_nodes import generate_answer
from src.chain import REFUSAL
from src.provider import ProviderSettings
from src.agent_graph import build_agent_graph, route_after_grading, agent_result, AgentRAGChain


class GenerationTests(unittest.TestCase):
    def test_grounded_answer_and_image_paths_remain_in_state(self):
        state={'question':'What is RAG?', 'documents':['RAG retrieves evidence.'],
               'slide_images':['valid.png'], 'generation_grade':'yes'}
        before=state.copy()
        model=Mock();model.invoke.return_value.content='RAG retrieves evidence [S1].'
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'})),patch('src.agent_nodes.create_provider_client',return_value=model):
            result=generate_answer(state)
            self.assertEqual(result['generation'],'RAG retrieves evidence [S1].')
            self.assertEqual(result['sources'][0]['citation'],'S1')
        self.assertEqual(state,before)
        prompt=model.invoke.call_args.args[0].to_string()
        self.assertIn('RAG retrieves evidence.',prompt)
        self.assertIn('What is RAG?',prompt)
        self.assertNotIn('valid.png',prompt)

    def test_missing_or_rejected_evidence_refuses_without_model(self):
        for state in [{'documents':[]}, {'documents':['Evidence'],'generation_grade':'no'}]:
            with patch('src.agent_nodes.create_provider_client') as factory:
                self.assertEqual(generate_answer(state),{'generation':REFUSAL,'sources':[]})
                factory.assert_not_called()

    def test_uncited_or_invalid_citations_refuse(self):
        for text in ['Uncited answer','Wrong citation [S99].','']:
            model=Mock();model.invoke.return_value.content=text
            with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'})),patch('src.agent_nodes.create_provider_client',return_value=model):
                self.assertEqual(generate_answer({'question':'Q','documents':['Evidence']}),{'generation':REFUSAL,'sources':[]})

    def test_provider_errors_propagate(self):
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',side_effect=ConnectionError()):
            with self.assertRaises(ConnectionError):
                generate_answer({'question':'Q','documents':['Evidence']})


class GraphTests(unittest.TestCase):
    def run_graph(self, grades):
        def decompose(state):
            return {'sub_queries':[state['question']],'loop_step':state.get('loop_step',0)+1}
        with patch('src.agent_graph.decompose_query',side_effect=decompose),patch('src.agent_graph.retrieve_documents',return_value={'documents':['Evidence'],'slide_images':['slide.png']}) as retrieve,patch('src.agent_graph.grade_documents',side_effect=[{'generation_grade':g} for g in grades]),patch('src.agent_graph.generate_answer',return_value={'generation':'Answer [S1]'}) as generate:
            result=build_agent_graph().invoke({'question':'Q','loop_step':0})
        return result,retrieve,generate

    def test_yes_generates_without_retry(self):
        result,retrieve,_=self.run_graph(['yes'])
        self.assertEqual(retrieve.call_count,1)
        self.assertEqual(result['generation'],'Answer [S1]')
        self.assertEqual(result['slide_images'],['slide.png'])

    def test_negative_grade_retries_once(self):
        result,retrieve,_=self.run_graph(['no','yes'])
        self.assertEqual(retrieve.call_count,2)
        self.assertEqual(result['loop_step'],2)

    def test_repeated_negative_grade_terminates(self):
        result,retrieve,generate=self.run_graph(['no','no'])
        self.assertEqual(retrieve.call_count,2)
        self.assertEqual(generate.call_args.args[0]['generation_grade'],'no')
        self.assertEqual(route_after_grading({'generation_grade':'no','loop_step':100}),'generate')

    def test_result_revalidates_and_deduplicates_images(self):
        with patch('src.chain.resolve_render_path',side_effect=lambda meta: None if meta['render_path']=='missing' else 'valid.png'):
            result=agent_result({'generation':'Answer','sources':[{'citation':'S1','metadata':{'render_path':'valid.png'}}],'slide_images':['valid.png','valid.png','missing']})
        self.assertEqual(result['slide_images'],['valid.png'])
        self.assertEqual(agent_result({'generation':REFUSAL,'slide_images':['valid.png']})['slide_images'],[])


class AsyncGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_streams_generation_only_and_returns_final(self):
        async def events(state,**kwargs):
            yield 'messages',(Mock(content='hidden plan'),{'langgraph_node':'decompose'})
            yield 'updates',{'retrieve':{'documents':['Evidence'],'slide_images':[]}}
            yield 'messages',(Mock(content='Answer'),{'langgraph_node':'generate'})
            yield 'updates',{'generate':{'generation':'Answer'}}
        graph=Mock();graph.astream.side_effect=events
        result=[event async for event in AgentRAGChain(graph).astream('Question')]
        self.assertEqual(result[0],{'type':'token','text':'Answer'})
        self.assertEqual(result[1]['type'],'final')
        self.assertEqual(result[1]['result']['answer'],'Answer')
        self.assertEqual(len(result),2)

    async def test_real_graph_async_execution(self):
        with patch('src.agent_graph.decompose_query',return_value={'sub_queries':['Q'],'loop_step':1}),patch('src.agent_graph.retrieve_documents',return_value={'documents':[],'slide_images':[]}),patch('src.agent_graph.grade_documents',return_value={'generation_grade':'yes'}):
            result=await build_agent_graph().ainvoke({'question':'Q'})
        self.assertEqual(result['generation'],REFUSAL)


class AgentCacheIsolationTests(unittest.TestCase):
    def test_environment_toggle_changes_cache_namespace(self):
        import tempfile
        from pathlib import Path
        from src.cache import namespace
        with tempfile.TemporaryDirectory() as folder,patch('src.cache.ROOT',Path(folder)):
            with patch.dict('os.environ',{'ENABLE_AGENT_GRAPH':'false'}):
                classic=namespace()
            with patch.dict('os.environ',{'ENABLE_AGENT_GRAPH':'true'}):
                corrective=namespace()
        self.assertNotEqual(classic,corrective)


class CitationPreservationTests(unittest.TestCase):
    def test_metadata_citations_and_images_survive_compiled_graph(self):
        chunk={'id':'pdf-2','text':'Instructions, Evidence, Tools, State.',
               'metadata':{'source':'corpus/pdfs/context.pdf','kind':'handout','page_number':2,'render_path':'valid.png'},
               'score':0.1,'dense_score':0.82}
        model=Mock();model.invoke.return_value.content='Instructions, Evidence, Tools, State [S1].'
        with patch('src.agent_graph.decompose_query',return_value={'sub_queries':['Q'],'loop_step':1}),patch('src.agent_graph.retrieve_documents',return_value={'documents':[chunk['text']],'retrieved_chunks':[chunk],'slide_images':['valid.png']}),patch('src.agent_graph.grade_documents',return_value={'generation_grade':'yes'}),patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'})),patch('src.agent_nodes.create_provider_client',return_value=model):
            state=build_agent_graph().invoke({'question':'What are task context components?'})
        self.assertEqual(state['slide_images'],['valid.png'])
        self.assertEqual(state['retrieved_chunks'][0]['metadata'],chunk['metadata'])
        prompt=model.invoke.call_args.args[0].to_string()
        self.assertIn('--- SOURCE CHUNK [S1] [File: context.pdf | page 2] ---',prompt)
        self.assertIn('Cite each factual claim',prompt)
        self.assertTrue(chunk['metadata'].items() <= state['sources'][0]['metadata'].items())
        self.assertEqual(state['sources'][0]['metadata']['id'],'pdf-2')
        with patch('src.chain.resolve_render_path',return_value='valid.png'):
            result=agent_result(state)
        self.assertEqual(result['sources'][0]['label'],'context.pdf | page 2')
        self.assertEqual(result['slide_images'],['valid.png'])

    def test_retry_replaces_old_metadata_and_images(self):
        chunk={'text':'New evidence','metadata':{'source':'new.pdf','page_number':3}}
        model=Mock();model.invoke.return_value.content='New evidence [S1].'
        def decompose(state):
            return {'sub_queries':['Q'],'loop_step':state.get('loop_step',0)+1}
        with patch('src.agent_graph.decompose_query',side_effect=decompose),patch('src.agent_graph.retrieve_documents',side_effect=[{'documents':['Old evidence'],'retrieved_chunks':[{'text':'Old evidence','metadata':{'source':'old.pdf'}}],'slide_images':['old.png']},{'documents':['New evidence'],'retrieved_chunks':[chunk],'slide_images':['new.png']}]),patch('src.agent_graph.grade_documents',side_effect=[{'generation_grade':'no'},{'generation_grade':'yes'}]),patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'})),patch('src.agent_nodes.create_provider_client',return_value=model):
            state=build_agent_graph().invoke({'question':'Q'})
        self.assertEqual(state['slide_images'],['new.png'])
        self.assertEqual(state['sources'][0]['label'],'new.pdf | page 3')
        self.assertNotIn('Old evidence',model.invoke.call_args.args[0].to_string())


class AgentImageCacheTests(unittest.TestCase):
    def test_cached_agent_image_paths_are_revalidated(self):
        from src.cache import images_available
        with patch('src.retriever.resolve_render_path',side_effect=lambda meta: 'valid.png' if meta['render_path']=='valid.png' else None):
            self.assertTrue(images_available({'slide_images':['valid.png']}))
            self.assertFalse(images_available({'slide_images':['missing.png']}))


class FastGradingGraphTests(unittest.TestCase):
    def test_reranked_evidence_uses_only_generation_model(self):
        chunk={'text':'Evidence','metadata':{'source':'guide.pdf','page_number':2},'rerank_score':1.0}
        model=Mock();model.invoke.return_value.content='Answer [S1].'
        with patch('src.agent_graph.decompose_query',return_value={'sub_queries':['Q'],'loop_step':1}),patch('src.agent_graph.retrieve_documents',return_value={'documents':['Evidence'],'retrieved_chunks':[chunk],'slide_images':['valid.png']}) as retrieve,patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'})),patch('src.agent_nodes.create_provider_client',return_value=model) as factory:
            state=build_agent_graph().invoke({'question':'Q','generation_grade':'no'})
        factory.assert_called_once()
        model.invoke.assert_called_once()
        model.with_structured_output.assert_not_called()
        retrieve.assert_called_once()
        self.assertEqual(state['generation_grade'],'yes')
        self.assertEqual(state['generation'],'Answer [S1].')
        self.assertEqual(state['sources'][0]['label'],'guide.pdf | page 2')
        self.assertEqual(state['slide_images'],['valid.png'])

    def test_empty_retrieval_retries_and_refuses_without_any_model(self):
        def decompose(state):
            return {'sub_queries':['Q'],'loop_step':state.get('loop_step',0)+1}
        with patch('src.agent_graph.decompose_query',side_effect=decompose),patch('src.agent_graph.retrieve_documents',return_value={'documents':[],'retrieved_chunks':[],'slide_images':[]}) as retrieve,patch('src.agent_nodes.create_provider_client') as factory:
            state=build_agent_graph().invoke({'question':'Q'})
        self.assertEqual(retrieve.call_count,2)
        factory.assert_not_called()
        self.assertEqual(state['generation'],REFUSAL)


class RetrievalComparisonPreviewTests(unittest.TestCase):
    def test_reported_question_keeps_dense_and_sparse_slide_previews(self):
        from src.agent_nodes import retrieve_documents,decompose_query
        question='Explain difference between dense retrieval and sparse retrieval?'
        dense={'id':'dense','text':'Dense retrieval uses semantic vector similarity.',
               'dense_score':.7935,'rerank_score':3.1376,'score':.02,
               'metadata':{'source':'deck.pdf','kind':'slide','page_number':28}}
        sparse={'id':'sparse','text':'Sparse retrieval uses keyword BM25 scores.',
                'dense_score':.7958,'rerank_score':3.5569,'score':.02,
                'metadata':{'source':'deck.pdf','kind':'slide','page_number':29}}
        retriever=Mock()
        retriever.search.side_effect=lambda query,**kwargs: [dense] if query=='Explain dense retrieval.' else [sparse]
        with patch('src.agent_nodes._get_decomposer') as planner:
            state={'question':question};state.update(decompose_query(state))
        planner.assert_not_called()
        def images(hits):
            return [{'path':f"page_{hits[0]['metadata']['page_number']}.png"}]
        with patch('src.agent_nodes.load_dotenv'),patch('src.agent_nodes.ProviderSettings.from_env',return_value=Mock(rag_top_k=3)),patch('src.agent_nodes.HybridRetriever',return_value=retriever),patch('src.agent_nodes.collect_source_images',side_effect=images):
            state.update(retrieve_documents(state))
        self.assertEqual(state['documents'],[dense['text'],sparse['text']])
        self.assertEqual(state['slide_images'],['page_28.png','page_29.png'])
        self.assertEqual([h['metadata']['page_number'] for h in state['retrieved_chunks']],[28,29])


class ConfiguredGenerationModelTests(unittest.TestCase):
    def test_generation_honors_selected_model_without_changing_context_or_citations(self):
        for selected in ['qwen3:4b-instruct','llama3.1:8b']:
            with self.subTest(model=selected):
                model=Mock();model.invoke.return_value.content='Supported answer [S1].'
                state={'question':'Question','documents':['Evidence'], 'slide_images':['slide.png'],
                       'retrieved_chunks':[{'text':'Evidence','metadata':{'source':'guide.pdf','page_number':2}}]}
                with patch('src.agent_nodes.load_dotenv'),patch.dict('os.environ',{'LLM_PROVIDER':'openai','OLLAMA_CHAT_MODEL':selected}),patch('src.agent_nodes.create_provider_client',return_value=model) as factory:
                    result=generate_answer(state)
                config=factory.call_args.args[0]
                self.assertEqual(config.provider,'ollama')
                self.assertEqual(config.chat_model,selected)
                self.assertEqual(config.embedding_model,'nomic-embed-text')
                self.assertEqual(result['generation'],'Supported answer [S1].')
                self.assertEqual(result['sources'][0]['label'],'guide.pdf | page 2')
                self.assertEqual(state['slide_images'],['slide.png'])
                self.assertIn('--- SOURCE CHUNK [S1] [File: guide.pdf | page 2] ---',model.invoke.call_args.args[0].to_string())
