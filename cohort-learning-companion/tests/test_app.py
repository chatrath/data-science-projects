from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

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
            async def send(self):
                owner.sent.append(self)
                # Deliberately return None: handlers must retain their own object.
            async def stream_token(self,token):
                self.content += token
                owner.tokens.append(token)
            async def update(self):
                pass
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


if __name__=='__main__':
    unittest.main()
