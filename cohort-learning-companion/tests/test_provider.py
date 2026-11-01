"""Offline provider routing and embedding-space compatibility checks."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from src.provider import NEBIUS_BASE_URL, ProviderSettings, create_provider_client, generate_text
from src.retriever import HybridRetriever, SCHEMA


class ProviderTests(unittest.TestCase):
    def test_nebius_defaults_and_blank_base_url(self):
        settings = ProviderSettings.from_env({'NEBIUS_API_KEY': 'fake-test-secret', 'NEBIUS_BASE_URL': ''})
        self.assertEqual(settings.provider, 'nebius')
        self.assertEqual(settings.base_url, NEBIUS_BASE_URL)
        self.assertEqual(settings.embedding_model, 'Qwen/Qwen3-Embedding-8B')
        self.assertEqual(settings.embedding_dimensions, 4096)
        self.assertNotIn('fake-test-secret', repr(settings))

    def test_explicit_provider_wins_without_key_fallback(self):
        settings = ProviderSettings.from_env({'LLM_PROVIDER': 'openai', 'NEBIUS_API_KEY': 'nebius-key'})
        self.assertEqual(settings.provider, 'openai')
        self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY', create_provider_client, settings)
        settings = ProviderSettings.from_env({'LLM_PROVIDER': 'nebius', 'OPENAI_API_KEY': 'openai-key'})
        self.assertRaisesRegex(ValueError, 'NEBIUS_API_KEY', create_provider_client, settings)

    def test_custom_endpoint_and_validation(self):
        settings = ProviderSettings.from_env({'LLM_PROVIDER': 'nebius', 'NEBIUS_BASE_URL': 'https://custom.example/v1', 'NEBIUS_EMBEDDING_DIMENSIONS': '1024'})
        self.assertEqual(settings.base_url, 'https://custom.example/v1/')
        self.assertEqual(settings.embedding_dimensions, 1024)
        for env in [{'LLM_PROVIDER': 'invalid'}, {'NEBIUS_API_KEY': 'key', 'NEBIUS_EMBEDDING_DIMENSIONS': 'no'}, {'NEBIUS_API_KEY': 'key', 'NEBIUS_BASE_URL': '[https://example.com](https://example.com)'}]:
            self.assertRaises(ValueError, ProviderSettings.from_env, env)

    def test_client_factory_uses_nebius_key_and_base_url(self):
        settings = ProviderSettings.from_env({'NEBIUS_API_KEY': 'test-key'})
        with patch('src.provider.OpenAI') as factory:
            create_provider_client(settings)
            factory.assert_called_once_with(api_key='test-key', base_url=NEBIUS_BASE_URL, timeout=60.0, max_retries=2)

    def test_nebius_embedding_and_generation_wire_requests(self):
        settings = ProviderSettings.from_env({'NEBIUS_API_KEY': 'test-key', 'NEBIUS_EMBEDDING_DIMENSIONS': '3', 'NEBIUS_CHAT_MODEL': 'activated/chat-model'})
        requests = []
        def handler(request):
            requests.append(request)
            payload = json.loads(request.content)
            self.assertEqual(request.headers['authorization'], 'Bearer test-key')
            if request.url.path.endswith('/embeddings'):
                self.assertEqual(payload['model'], settings.embedding_model)
                self.assertEqual(payload['dimensions'], 3)
                self.assertEqual(payload['encoding_format'], 'float')
                return httpx.Response(200, json={'object': 'list', 'data': [{'index': 0, 'object': 'embedding', 'embedding': [1., 2., 3.]}], 'model': settings.embedding_model, 'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
            self.assertEqual(payload['model'], 'activated/chat-model')
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': settings.chat_model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'answer'}, 'finish_reason': 'stop'}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            with OpenAI(api_key=settings.api_key, base_url=settings.base_url, http_client=http) as client:
                with patch('src.retriever.load_dotenv'):
                    retriever = HybridRetriever(database_url='postgresql://unused', client=client, settings=settings)
                self.assertEqual(retriever._embed(['demo']), [[1., 2., 3.]])
                self.assertEqual(generate_text([{'role': 'user', 'content': 'question'}], settings=settings, client=client), 'answer')
        self.assertEqual([str(r.url) for r in requests], [NEBIUS_BASE_URL + 'embeddings', NEBIUS_BASE_URL + 'chat/completions'])

    def test_embedding_space_mismatch_rejected(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.settings = ProviderSettings.from_env({'NEBIUS_API_KEY': 'test-key'})
        conn = Mock()
        config = {'provider': 'nebius', 'base_url': NEBIUS_BASE_URL, 'model': retriever.settings.embedding_model, 'dimensions': 4096}
        conn.execute.return_value.fetchone.return_value = config
        retriever._check_config(conn)
        for key, value in [('provider', 'openai'), ('base_url', 'https://other.example/v1/'), ('model', 'other-model'), ('dimensions', 1536)]:
            conn.execute.return_value.fetchone.return_value = {**config, key: value}
            self.assertRaisesRegex(ValueError, 'cannot be mixed', retriever._check_config, conn)
        self.assertIn('vector(4096)', SCHEMA.format(dimensions=4096))

    def test_generation_requires_activated_model_name(self):
        client = Mock()
        self.assertRaisesRegex(ValueError, 'NEBIUS_CHAT_MODEL', generate_text, [], settings=ProviderSettings.from_env({'NEBIUS_API_KEY': 'key'}), client=client)
        client.chat.completions.create.assert_not_called()

    def test_nebius_does_not_use_openai_tokenizer(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.encoding = None
        retriever._validate_texts(['code demo'])
        self.assertRaisesRegex(ValueError, 'UTF-8 bytes', retriever._validate_texts, ['x' * 8192])


if __name__ == '__main__':
    unittest.main()
