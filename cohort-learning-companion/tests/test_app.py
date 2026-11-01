from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from src.cache import CacheLookup

import app
from chainlit.data import get_data_layer


class ChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session={}
        self.sent=[]
        self.tokens=[]
        owner=self
        class Message:
            def __init__(self, content, **kwargs):
                self.content=content
                self.elements=kwargs.get("elements",[])
                self.updated=False
            async def send(self):
                owner.sent.append(self)
                # Deliberately return None: handlers must retain their own object.
            async def stream_token(self,token):
                self.content += token
                owner.tokens.append(token)
            async def update(self):
                self.updated=True
        self.chain=Mock()
        self.result={'answer':'Grounded answer [S1].','sources':[{'citation':'S1','label':'guide.pdf | page 2'}]}
        async def events(query):
            yield {'type':'token','text':'Grounded '}
            yield {'type':'token','text':'answer [S1].'}
            yield {'type':'final','result':self.result}
        self.chain.astream=Mock(side_effect=events)
        self.patches=[
            patch.object(app.cl,'Message',Message),
            patch.object(app.cl,'user_session',SimpleNamespace(get=self.session.get,set=lambda k,v:self.session.__setitem__(k,v))),
            patch.object(app,'RAGChain',return_value=self.chain),
            patch.object(app,'get_semantic_cache',return_value=None),
            patch.object(app,'dotenv_values',return_value={'ENABLE_AGENT_GRAPH':'false'}),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_welcome_and_message_deliver_cited_answer(self):
        await app.on_chat_start()
        self.assertIn('Welcome',self.sent[0].content)
        await app.on_message(SimpleNamespace(content='Question'))
        self.chain.astream.assert_called_once_with('Question')
        self.assertEqual(self.tokens,['Grounded ','answer [S1].'])
        self.assertIn('Grounded answer [S1]',self.sent[-1].content)
        self.assertIn('**Sources**',self.sent[-1].content)

    async def test_cache_hit_bypasses_rag_and_preserves_image_delivery(self):
        lookup=CacheLookup('v1',result=self.result,similarity=.98)
        cache=SimpleNamespace(lookup=AsyncMock(return_value=lookup),store=AsyncMock())
        image=SimpleNamespace(name='slide image')
        with patch.object(app,'get_semantic_cache',return_value=cache),patch.object(app,'get_chain',new_callable=AsyncMock) as get_chain,patch.object(app,'slide_image_elements',return_value=[image]):
            await app.on_message(SimpleNamespace(content='Question'))
        get_chain.assert_not_called();self.chain.astream.assert_not_called()
        cache.store.assert_not_called()
        self.assertEqual(self.tokens,[])
        self.assertEqual(self.sent[-1].elements,[image])
        self.assertIn('Grounded answer',self.sent[-1].content)

    async def test_cache_miss_streams_and_stores_only_after_final_ui_update(self):
        lookup=CacheLookup('v1')
        async def store(*args):self.assertTrue(self.sent[-1].updated)
        cache=SimpleNamespace(lookup=AsyncMock(return_value=lookup),store=AsyncMock(side_effect=store))
        with patch.object(app,'get_semantic_cache',return_value=cache):
            await app.on_message(SimpleNamespace(content='Question'))
        self.assertEqual(self.tokens,['Grounded ','answer [S1].'])
        cache.store.assert_awaited_once_with('Question',self.result,lookup)

    async def test_cache_failure_falls_back_to_streaming(self):
        cache=SimpleNamespace(lookup=AsyncMock(side_effect=RuntimeError('private cache error')),store=AsyncMock())
        with patch.object(app,'get_semantic_cache',return_value=cache):await app.on_message(SimpleNamespace(content='Question'))
        self.assertTrue(self.tokens)
        self.assertNotIn('private cache error',self.sent[-1].content)
        cache.store.assert_not_called()

    async def test_cache_write_failure_does_not_replace_successful_answer(self):
        cache=SimpleNamespace(lookup=AsyncMock(return_value=CacheLookup('v1')),store=AsyncMock(side_effect=RuntimeError('private cache error')))
        with patch.object(app,'get_semantic_cache',return_value=cache):await app.on_message(SimpleNamespace(content='Question'))
        self.assertIn('Grounded answer',self.sent[-1].content)
        self.assertNotIn('private cache error',self.sent[-1].content)

    async def test_refused_response_is_not_saved(self):
        from src.chain import REFUSAL
        self.result={'answer':REFUSAL,'sources':[],'refused':True}
        cache=SimpleNamespace(lookup=AsyncMock(return_value=CacheLookup('v1')),store=AsyncMock())
        with patch.object(app,'get_semantic_cache',return_value=cache):await app.on_message(SimpleNamespace(content='Question'))
        cache.store.assert_not_called()

    async def test_backend_failure_updates_progress_with_safe_error(self):
        self.session['rag_chain']=self.chain
        async def broken(query):
            yield {'type':'token','text':'Partial answer'}
            raise RuntimeError('sensitive backend details')
        self.chain.astream.side_effect=broken
        await app.on_message(SimpleNamespace(content='Question'))
        self.assertIn('Something went wrong',self.sent[-1].content)
        self.assertNotIn('sensitive',self.sent[-1].content)

    async def test_final_refusal_replaces_provisional_tokens(self):
        from src.chain import REFUSAL
        self.result={'answer':REFUSAL,'sources':[]}
        await app.on_message(SimpleNamespace(content='Question'))
        self.assertTrue(self.tokens)
        self.assertEqual(self.sent[-1].content,REFUSAL)
        self.assertNotIn('**Sources**',self.sent[-1].content)

    async def test_empty_content_is_handled_without_retrieval(self):
        await app.on_message(SimpleNamespace(content=None))
        self.chain.astream.assert_not_called()
        self.assertIn('Please enter a question',self.sent[-1].content)

    async def test_disconnected_message_update_does_not_escape(self):
        reply=SimpleNamespace(content='',update=AsyncMock(side_effect=ConnectionError()))
        await app.deliver('Answer',reply)

    def test_rag_database_does_not_enable_chat_history_persistence(self):
        self.assertIsNone(get_data_layer())

    async def test_agent_flag_selects_streaming_adapter(self):
        with patch.object(app,'dotenv_values',return_value={'ENABLE_AGENT_GRAPH':'true'}),patch.object(app,'AgentRAGChain',return_value=self.chain) as factory:
            await app.on_message(SimpleNamespace(content='Question'))
        factory.assert_called_once()
        self.assertEqual(self.tokens,['Grounded ','answer [S1].'])
        self.assertTrue(self.session['agent_graph_enabled'])

    def test_agent_image_paths_are_validated_without_text_elements(self):
        image=SimpleNamespace(name='Preview')
        with patch.object(app,'resolve_render_path',side_effect=lambda meta: 'valid.png' if meta['render_path']=='valid.png' else None),patch.object(app.cl,'Image',return_value=image) as factory:
            result=app.slide_image_elements({'slide_images':['valid.png','valid.png','missing']})
        self.assertEqual(result,[image])
        factory.assert_called_once()

    async def test_agent_final_response_displays_pdf_citations_and_image(self):
        self.result={'answer':'Evidence [S1].','sources':[{'citation':'S1','label':'context.pdf | page 2'}],
                     'slide_images':['valid.png']}
        image=SimpleNamespace(name='Preview')
        with patch.object(app,'dotenv_values',return_value={'ENABLE_AGENT_GRAPH':'true'}),patch.object(app,'AgentRAGChain',return_value=self.chain),patch.object(app,'resolve_render_path',return_value='valid.png'),patch.object(app.cl,'Image',return_value=image):
            await app.on_message(SimpleNamespace(content='Question'))
        self.assertIn('context',self.sent[-1].content)
        self.assertIn('[S1]',self.sent[-1].content)
        self.assertIn('page 2',self.sent[-1].content)
        self.assertEqual(self.sent[-1].elements,[image])

    async def test_multi_question_markdown_streams_verbatim_and_final_sections_are_spaced(self):
        parts=['### 1. First','\n**Answer** [S1].\n###',' 2. Second\n```python\n# comment\n    print(1)\n```\nAnswer [S1].']
        raw=''.join(parts)
        self.result={'answer':raw,'sources':[{'citation':'S1','label':'guide.pdf | page 2'}]}
        async def events(query):
            for text in parts:
                yield {'type':'token','text':text}
            yield {'type':'final','result':self.result}
        self.chain.astream.side_effect=events
        await app.on_message(SimpleNamespace(content='First question? Second question?'))
        self.assertEqual(self.tokens,parts)
        self.assertIn('### 1. First\n\n**Answer** [S1].\n\n### 2. Second\n\n',self.sent[-1].content)
        self.assertIn('```python\n# comment\n    print(1)\n```',self.sent[-1].content)
        self.assertIn('**Sources**',self.sent[-1].content)


if __name__=='__main__':
    unittest.main()
