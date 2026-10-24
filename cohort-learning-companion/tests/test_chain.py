import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from src.chain import RAGChain, REFUSAL, format_context, format_answer
from src.provider import ProviderSettings

SETTINGS = ProviderSettings.from_env({'LLM_PROVIDER': 'ollama'})
HIT = {'id': 'chunk', 'text': 'Context engineering manages information available to a model.\n\n```python\n    print("context")\n```', 'score': .016, 'metadata': {'source': 'C:\\corpus\\guide.pdf', 'page_start': 2, 'page_end': 3, 'section': 'Context engineering'}}


class ChainTests(unittest.TestCase):
    def make_chain(self, response, hits=None):
        self.retriever = Mock()
        self.retriever.search.return_value = [HIT] if hits is None else hits
        self.prompts = []
        def invoke(prompt, **kwargs):
            self.prompts.append(prompt)
            self.assertEqual(kwargs['options']['num_predict'], 128)
            return AIMessage(content=response)
        return RAGChain(settings=SETTINGS, retriever=self.retriever, model=RunnableLambda(invoke), max_tokens=128)

    def test_grounded_answer_prompt_and_source_citations(self):
        chain = self.make_chain('It manages the model information [S1].')
        result = chain.ask('What is context engineering?', top_k=3)
        self.assertFalse(result['refused'])
        self.assertIn('guide.pdf | page 2-3', result['sources'][0]['label'])
        self.assertIn(HIT['text'], self.prompts[0].to_messages()[1].content)
        self.assertIn('untrusted reference data', self.prompts[0].to_messages()[0].content)
        self.retriever.search.assert_called_once_with('What is context engineering?', top_k=3, candidates=25, bm25_weight=.5)
        self.assertIn('[S1] guide.pdf', format_answer(result))

    def test_grouped_valid_citations_are_normalized(self):
        chain = self.make_chain('Supported claim [S1, S2].', hits=[HIT, HIT])
        result = chain.ask('Question')
        self.assertFalse(result['refused'])
        self.assertIn('[S1][S2]', result['answer'])
        self.assertEqual(len(result['sources']), 2)

    def test_no_context_skips_model(self):
        result = self.make_chain('must not be used', hits=[]).ask('Question')
        self.assertEqual(result['answer'], REFUSAL)
        self.assertEqual(self.prompts, [])

    def test_refusal_and_invalid_or_missing_citations(self):
        for response in [REFUSAL, '', 'An unsupported answer.', 'Claim [S99].', 'Claim [S1, S99].']:
            result = self.make_chain(response).ask('Question')
            self.assertTrue(result['refused'])
            self.assertEqual(result['sources'], [])

    def test_context_preserves_code_and_never_slices_oversized_chunks(self):
        context, sources = format_context([HIT])
        self.assertIn('    print("context")', context)
        self.assertEqual(format_context([HIT], max_bytes=10), ('', []))

    def test_input_limits_and_local_provider_requirement(self):
        chain = self.make_chain('Claim [S1].')
        for query in ['', 'x' * 1025]:
            self.assertRaises(ValueError, chain.ask, query)
        self.retriever.search.assert_not_called()
        self.assertRaises(ValueError, RAGChain, settings=ProviderSettings())


if __name__ == '__main__':
    unittest.main()
