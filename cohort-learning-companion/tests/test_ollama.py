"""Local Ollama routing; no server, model download, keys, or DB required."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.provider import ProviderSettings, create_embedding_client, create_provider_client, generate_text
from src.retriever import HybridRetriever


def settings(**overrides):
    return ProviderSettings.from_env({'LLM_PROVIDER': 'ollama', **overrides})


class OllamaTests(unittest.TestCase):
    def test_defaults_and_native_langchain_factories_need_no_keys(self):
        config = settings()
        self.assertEqual((config.chat_model, config.embedding_model, config.embedding_dimensions), ('llama3.1:8b', 'nomic-embed-text', 768))
        self.assertEqual(config.api_key, '')
        with patch('src.provider.OpenAI') as hosted, patch('src.provider.ChatOllama') as chat, patch('src.provider.OllamaEmbeddings') as embed:
            create_provider_client(config)
            create_embedding_client(config)
            hosted.assert_not_called()
            self.assertEqual(chat.call_args.kwargs['model'], 'llama3.1:8b')
            self.assertEqual(embed.call_args.kwargs['model'], 'nomic-embed-text')
            self.assertEqual(embed.call_args.kwargs['num_ctx'], 8192)
            self.assertNotIn('api_key', embed.call_args.kwargs)

    def test_configurable_model_and_url(self):
        config = settings(OLLAMA_CHAT_MODEL='qwen2.5:7b', OLLAMA_BASE_URL='http://127.0.0.1:11435')
        self.assertEqual(config.chat_model, 'qwen2.5:7b')
        self.assertEqual(config.base_url, 'http://127.0.0.1:11435/')

    def test_local_generation_invokes_langchain_and_translates_token_budget(self):
        client = Mock()
        client.invoke.return_value = SimpleNamespace(content='local answer')
        messages = [{'role': 'user', 'content': 'Explain retrieval'}]
        self.assertEqual(generate_text(messages, settings=settings(), client=client, max_tokens=100), 'local answer')
        client.invoke.assert_called_once_with(messages, options={'num_predict':100})

    def test_document_and_query_prefixes_dimensions_and_empty_input(self):
        embedder = Mock()
        embedder.embed_documents.return_value = [[1.] * 768]
        embedder.embed_query.return_value = [2.] * 768
        with patch('src.retriever.load_dotenv'):
            retriever = HybridRetriever('postgresql://unused', client=embedder, settings=settings())
        self.assertIsNone(retriever.encoding)
        self.assertEqual(len(retriever._embed(['demo'])[0]), 768)
        embedder.embed_documents.assert_called_once_with(['search_document: demo'])
        self.assertEqual(retriever._embed(['question'], is_query=True)[0][0], 2.)
        embedder.embed_query.assert_called_once_with('search_query: question')
        self.assertEqual(retriever._embed([]), [])
        self.assertRaisesRegex(ValueError, 'must not be empty', retriever._embed, ['   '])
        embedder.embed_documents.return_value = [[1.] * 1536]
        self.assertRaisesRegex(ValueError, 'invalid vector', retriever._embed, ['demo'])

    def test_database_ingest_and_hybrid_search_route_only_to_local_embedder(self):
        from src.retriever import Chunk
        embedder = Mock()
        embedder.embed_documents.return_value = [[1.] * 768]
        embedder.embed_query.return_value = [1.] * 768
        with patch('src.retriever.load_dotenv'):
            retriever = HybridRetriever('postgresql://unused', client=embedder, settings=settings())
        conn = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=False)
        conn.transaction.side_effect = lambda: nullcontext()
        def execute(sql, params=None):
            cursor = Mock(rowcount=0)
            if sql.startswith('SELECT provider'):
                cursor.fetchone.return_value = {'provider': 'ollama', 'base_url': retriever.settings.base_url, 'model': 'nomic-embed-text', 'dimensions': 768}
            elif sql.startswith('SELECT id, content_hash'):
                cursor.__iter__ = Mock(return_value=iter([]))
            elif sql.startswith('SELECT id, content, metadata'):
                cursor.fetchall.return_value = [{'id': 'one', 'content': 'local retrieval', 'metadata': {'source': 'demo.pdf'}}]
            elif sql.startswith(('SELECT id, embedding', 'SELECT id, distance')):
                cursor.fetchall.return_value = [{'id': 'one', 'distance': 0.1}]
            return cursor
        conn.execute.side_effect = execute
        retriever._connect = Mock(return_value=conn)
        with patch('src.retriever.register_vector'), patch('src.retriever.rerank_candidates', side_effect=lambda query,hits,top_k,min_score: hits[:top_k]), patch('src.retriever.read_jsonl', return_value=([Chunk('one', 'dataset', 'local retrieval', 'hash', {'source': 'demo.pdf'})], ['dataset'], 0)):
            result = retriever.ingest(['unused.jsonl'])
            self.assertEqual(result['embedded'], 1)
            hits = retriever.search('retrieval')
        self.assertEqual(hits[0]['id'], 'one')
        self.assertIn('bm25_rank', hits[0])
        self.assertIn('dense_rank', hits[0])
        embedder.embed_documents.assert_called_once_with(['search_document: local retrieval'])
        embedder.embed_query.assert_called_once_with('search_query: retrieval')


if __name__ == '__main__':
    unittest.main()
