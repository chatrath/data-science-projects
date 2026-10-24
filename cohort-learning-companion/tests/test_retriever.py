import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.provider import ProviderSettings

DIMENSIONS = 1536

from src.retriever import BM25Index, HybridRetriever, fuse, read_jsonl


class RetrievalTests(unittest.TestCase):
    def test_database_errors_are_actionable_and_do_not_expose_secrets(self):
        import psycopg
        from src.retriever import database_error_message
        cases = [
            (psycopg.OperationalError('secret-password'), 'start_database.ps1'),
            (psycopg.errors.InvalidPassword('secret-password'), 'authentication'),
            (psycopg.errors.InsufficientPrivilege('secret-password'), 'permission'),
            (psycopg.errors.UndefinedTable('secret-password'), 'ingest'),
            (psycopg.errors.FeatureNotSupported('secret-password'), 'pgvector'),
        ]
        for error, expected in cases:
            message = database_error_message(error)
            self.assertIn(expected, message)
            self.assertNotIn('secret-password', message)

    def test_bm25_matches_rare_code_identifiers_and_omits_nonmatches(self):
        index = BM25Index([
            {'id': 'a', 'content': 'pgvector cosine_distance demo'},
            {'id': 'b', 'content': 'chunking paragraphs and headings'},
            {'id': 'c', 'content': ''},
        ])
        self.assertEqual(index.search('COSINE_DISTANCE', 5)[0][0], 'a')
        self.assertEqual(index.search('unseen_token', 5), [])
        self.assertEqual(BM25Index([]).search('query', 5), [])

    def test_fusion_promotes_shared_results_and_respects_weight(self):
        sparse = [('a', 8), ('shared', 3)]
        dense = [('b', .9), ('shared', .8)]
        self.assertEqual(fuse(sparse, dense)[0]['id'], 'shared')
        self.assertEqual([h['id'] for h in fuse(sparse, dense, 1)], ['a', 'shared'])
        self.assertEqual([h['id'] for h in fuse(sparse, dense, 0)], ['b', 'shared'])
        self.assertRaises(ValueError, fuse, sparse, dense, 2)

    def test_jsonl_identity_metadata_blank_skips_and_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'slides.jsonl'
            records = [dict(source='deck.pdf', text='First slide', page_number=1, layout={'images': []}),
                       dict(source='deck.pdf', text='', page_number=2)]
            path.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            chunks, datasets, skipped = read_jsonl([path])
            self.assertEqual(skipped, 1)
            self.assertEqual(chunks[0].metadata['page_number'], 1)
            self.assertIn('layout', chunks[0].metadata)
            records[0]['text'] = 'Updated slide'
            path.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            updated, _, _ = read_jsonl([path])
            self.assertEqual(chunks[0].id, updated[0].id)
            self.assertNotEqual(chunks[0].content_hash, updated[0].content_hash)
            self.assertRaises(ValueError, read_jsonl, [path, path])

    def test_duplicate_identity_and_malformed_record_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.jsonl'
            line = json.dumps(dict(source='handout.pdf', id='same', text='content'))
            path.write_text(line + '\n' + line, encoding='utf-8')
            self.assertRaisesRegex(ValueError, 'duplicate record', read_jsonl, [path])
            path.write_text('{broken json', encoding='utf-8')
            self.assertRaisesRegex(ValueError, 'invalid JSON', read_jsonl, [path])

    def test_embedding_order_shape_and_empty_batches(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.settings = ProviderSettings()
        retriever.encoding = Mock()
        retriever.encoding.encode.return_value = [1]
        retriever.client = Mock()
        vector_a, vector_b = [1.] * DIMENSIONS, [2.] * DIMENSIONS
        retriever.client.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=vector_b), SimpleNamespace(index=0, embedding=vector_a)])
        self.assertEqual(retriever._embed(['a', 'b']), [vector_a, vector_b])
        retriever.client.reset_mock()
        self.assertEqual(retriever._embed([]), [])
        retriever.client.embeddings.create.assert_not_called()
        retriever.client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.])])
        self.assertRaisesRegex(ValueError, 'invalid vector', retriever._embed, ['a'])

    def test_overlong_context_rejected_without_silent_truncation(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.settings = ProviderSettings()
        retriever.encoding = Mock()
        retriever.encoding.encode.return_value = [1] * 8192
        retriever.client = Mock()
        self.assertRaisesRegex(ValueError, '8191', retriever._embed, ['long context'])
        retriever.client.embeddings.create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
