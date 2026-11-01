from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.chain import RAGChain
from src.provider import ProviderSettings
from src.retriever import HybridRetriever


class PerformanceTests(unittest.TestCase):
    def retriever(self, **env):
        config=ProviderSettings.from_env({'LLM_PROVIDER':'ollama',**env})
        embedder=Mock()
        embedder.embed_documents.side_effect=lambda texts:[[float(len(text))]*768 for text in texts]
        embedder.embed_query.return_value=[1.]*768
        with patch('src.retriever.load_dotenv'):
            return HybridRetriever('postgresql://unused',settings=config,client=embedder)

    def test_chunk_embeddings_use_batches_and_deduplicate_without_losing_order(self):
        retriever=self.retriever(EMBEDDING_BATCH_SIZE='2')
        vectors=retriever._embed(['first','second','first','third','fourth'])
        calls=retriever.client.embed_documents.call_args_list
        self.assertEqual([len(call.args[0]) for call in calls],[2,2])
        self.assertEqual(len(vectors),5)
        self.assertEqual(vectors[0],vectors[2])
        self.assertEqual(calls[0].args[0],['search_document: first','search_document: second'])

    def test_query_cache_reuses_embeddings_but_expires(self):
        retriever=self.retriever()
        with patch('src.retriever.time.monotonic',return_value=10):
            retriever._query_embedding('question')
            retriever._query_embedding('question')
        self.assertEqual(retriever.client.embed_query.call_count,1)
        with patch('src.retriever.time.monotonic',return_value=311):
            retriever._query_embedding('question')
        self.assertEqual(retriever.client.embed_query.call_count,2)

    def test_query_cache_is_bounded(self):
        retriever=self.retriever()
        for index in range(130):
            retriever._query_embedding(str(index))
        self.assertEqual(len(retriever._query_cache),128)
        self.assertNotIn('0',retriever._query_cache)

    def test_rag_uses_small_context_and_performance_options(self):
        settings=ProviderSettings.from_env({'LLM_PROVIDER':'ollama','OLLAMA_NUM_THREAD':'6'})
        retriever=Mock()
        retriever.search.return_value=[{'id':'one','text':'Supported text.','score':.016,'metadata':{'source':'guide.pdf'}}]
        captured=[]
        def invoke(prompt,**kwargs):
            captured.append(kwargs)
            return AIMessage(content='Supported answer [S1].')
        chain=RAGChain(settings=settings,retriever=retriever,model=RunnableLambda(invoke))
        result=chain.ask('Question')
        retriever.search.assert_called_once_with('Question',top_k=3,candidates=25,bm25_weight=.5)
        self.assertEqual(captured[0]['options'],{'num_ctx':8192,'num_predict':256,'temperature':0,'num_thread':6})
        self.assertEqual(captured[0]['keep_alive'],600)
        self.assertFalse(result['refused'])
        self.assertLessEqual(result['timings']['context_bytes'],5000)
        self.assertIn('generation_seconds',result['timings'])

    def test_invalid_performance_configuration_is_rejected(self):
        for key,value in [('EMBEDDING_BATCH_SIZE','0'),('EMBEDDING_BATCH_SIZE','33'),('OLLAMA_NUM_THREAD','-1'),('RAG_TOP_K','0')]:
            self.assertRaises(ValueError,ProviderSettings.from_env,{'LLM_PROVIDER':'ollama',key:value})


if __name__=='__main__':
    unittest.main()
