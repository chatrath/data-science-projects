import asyncio
import threading
import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableLambda
from src.chain import RAGChain, REFUSAL
from src.provider import ProviderSettings

SETTINGS=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'})
HIT={'text':'Grounded evidence.','score':.016,'metadata':{'source':'guide.pdf'}}


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_token_arrives_before_completion_and_retrieval_is_off_loop(self):
        release=asyncio.Event()
        loop_thread=threading.get_ident()
        retrieval_threads=[]
        def search(*args,**kwargs):
            retrieval_threads.append(threading.get_ident())
            return [HIT]
        retriever=Mock()
        retriever.search.side_effect=search
        async def generate(prompt,**kwargs):
            yield AIMessageChunk(content='Grounded ')
            await release.wait()
            yield AIMessageChunk(content='answer [S1].')
        chain=RAGChain(settings=SETTINGS,retriever=retriever,model=RunnableLambda(generate))
        stream=chain.astream('Question')
        first=await asyncio.wait_for(anext(stream),2)
        self.assertEqual(first,{'type':'token','text':'Grounded '})
        self.assertFalse(release.is_set())
        self.assertNotEqual(retrieval_threads[0],loop_thread)
        release.set()
        events=[event async for event in stream]
        self.assertEqual(events[-1]['type'],'final')
        result=events[-1]['result']
        self.assertEqual(result['answer'],'Grounded answer [S1].')
        self.assertEqual(result['sources'][0]['citation'],'S1')
        self.assertGreaterEqual(result['timings']['first_token_seconds'],0)

    async def test_bad_citations_are_refused_after_streaming(self):
        async def generate(prompt,**kwargs):
            yield AIMessageChunk(content='Claim [S99].')
        retriever=Mock()
        retriever.search.return_value=[HIT]
        chain=RAGChain(settings=SETTINGS,retriever=retriever,model=RunnableLambda(generate))
        events=[event async for event in chain.astream('Question')]
        self.assertEqual(events[0]['type'],'token')
        self.assertEqual(events[-1]['result']['answer'],REFUSAL)
        self.assertEqual(events[-1]['result']['sources'],[])

    async def test_empty_context_refuses_without_starting_generation(self):
        started=[]
        async def generate(prompt,**kwargs):
            started.append(True)
            yield AIMessageChunk(content='unused')
        retriever=Mock()
        retriever.search.return_value=[]
        chain=RAGChain(settings=SETTINGS,retriever=retriever,model=RunnableLambda(generate))
        events=[event async for event in chain.astream('Question')]
        self.assertEqual(len(events),1)
        self.assertTrue(events[0]['result']['refused'])
        self.assertEqual(started,[])

    async def test_stream_errors_propagate_for_ui_recovery(self):
        async def generate(prompt,**kwargs):
            yield AIMessageChunk(content='Partial ')
            raise ConnectionError('model stopped')
        retriever=Mock()
        retriever.search.return_value=[HIT]
        chain=RAGChain(settings=SETTINGS,retriever=retriever,model=RunnableLambda(generate))
        stream=chain.astream('Question')
        self.assertEqual((await anext(stream))['type'],'token')
        with self.assertRaises(ConnectionError):
            await anext(stream)


if __name__=='__main__':
    unittest.main()
