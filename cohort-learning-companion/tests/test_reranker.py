import unittest
from contextlib import nullcontext
from src.retriever import HybridRetriever
from src.provider import ProviderSettings
from unittest.mock import Mock, patch
from src.retriever import rerank_candidates, stratify_results

class RerankerTests(unittest.TestCase):
    def hits(self):
        return [{'id':str(i),'text':f'Passage {i}','score':.02-i*.001,'dense_score':.8,
                 'metadata':{'kind':'transcript' if i%2 else 'handout'}} for i in range(15)]

    def test_joint_batch_reorders_broad_pool_and_keeps_raw_scores(self):
        hits=self.hits()
        model=Mock()
        model.predict.return_value=[-3.]*12+[4.,8.,2.]
        with patch('src.retriever._RERANKER',model):
            result=rerank_candidates('Paraphrased question',hits,3)
        self.assertEqual([h['id'] for h in result],['13','14','12'])
        self.assertEqual(result[0]['rerank_score'],8.)
        self.assertEqual(result[0]['dense_score'],.8)
        self.assertEqual(result[0]['score'],hits[13]['score'])
        self.assertNotIn('rerank_score',hits[13])
        self.assertEqual(model.predict.call_args.args[0], [('Paraphrased question',h['text']) for h in hits])
        self.assertEqual(model.predict.call_args.kwargs['batch_size'],8)

    def test_weak_candidates_are_not_used_to_fill_slots(self):
        model=Mock();model.predict.return_value=[-4.,.5,-1.]
        with patch('src.retriever._RERANKER',model):
            result=rerank_candidates('Question',self.hits()[:3],3)
            self.assertEqual([h['id'] for h in result],['1'])
            self.assertEqual(rerank_candidates('Question',self.hits()[:3],3,min_score=1),[])

    def test_empty_pool_does_not_load_or_call_model(self):
        model=Mock()
        with patch('src.retriever._RERANKER',model):self.assertEqual(rerank_candidates('Question',[],3),[])
        model.predict.assert_not_called()

    def test_invalid_scores_and_inference_errors_fail_explicitly(self):
        model=Mock()
        with patch('src.retriever._RERANKER',model):
            for scores in [[float('nan')], [float('inf')], []]:
                model.predict.return_value=scores
                with self.assertRaises(RuntimeError):rerank_candidates('Question',self.hits()[:1],3)
            model.predict.side_effect=OSError('Model unavailable')
            with self.assertRaisesRegex(RuntimeError,'Local reranker failed'):rerank_candidates('Question',self.hits()[:1],3)

    def test_source_reservations_do_not_override_final_joint_relevance(self):
        hits=self.hits()
        lookup={h['id']:{'content':h['text'],'metadata':h['metadata']} for h in hits}
        pool=stratify_results(hits,lookup,'Question',15)
        model=Mock();model.predict.return_value=[8. if h['metadata']['kind']=='transcript' else -2. for h in pool]
        with patch('src.retriever._RERANKER',model):result=rerank_candidates('Question',pool,3)
        self.assertEqual(len(result),3)
        self.assertTrue(all(h['metadata']['kind']=='transcript' for h in result))

    def test_search_reranks_fifteen_candidates_before_selecting_three(self):
        rows = [{'id':str(i),'content':f'retrieval passage {i}',
                 'metadata':{'kind':'transcript' if i==17 else 'slide' if i==16 else 'handout'}} for i in range(18)]
        conn=Mock()
        conn.__enter__=Mock(return_value=conn)
        conn.__exit__=Mock(return_value=False)
        conn.transaction.side_effect=lambda:nullcontext()
        def execute(sql,params=None):
            cursor=Mock()
            if sql.startswith('SELECT id, content, metadata'):
                cursor.fetchall.return_value=rows
            elif sql.startswith('SELECT id, distance'):
                cursor.fetchall.return_value=[{'id':r['id'],'distance':.1+i*.001} for i,r in enumerate(rows)]
            return cursor
        conn.execute.side_effect=execute
        retriever=HybridRetriever('postgresql://unused',client=Mock(),settings=ProviderSettings.from_env({'LLM_PROVIDER':'ollama'}))
        model=Mock()
        model.predict.side_effect=lambda pairs,**kwargs:[8. if text.endswith('17') else -2. for _,text in pairs]
        with patch.object(retriever,'_connect',return_value=conn), patch.object(retriever,'_check_config'), patch.object(retriever,'_query_embedding',return_value=[1.]*768), patch('src.retriever.register_vector'), patch('src.retriever._RERANKER',model):
            result=retriever.search('retrieval')
        self.assertEqual(len(model.predict.call_args.args[0]),15)
        self.assertEqual([h['id'] for h in result],['17'])
        self.assertEqual(result[0]['metadata']['kind'],'transcript')

    def test_context_order_handles_small_pools_and_larger_top_k(self):
        from src.chain import format_context
        model=Mock()
        for count in [0,1,2,3,5]:
            with self.subTest(count=count):
                hits=self.hits()[:count]
                model.predict.return_value=list(range(count,0,-1))
                with patch('src.retriever._RERANKER',model):
                    result=rerank_candidates('Question',hits,count or 1)
                expected=list(range(count))
                if count>2:expected=[0]+list(range(2,count))+[1]
                self.assertEqual([int(h['id']) for h in result],expected)
                self.assertEqual(sorted(h['id'] for h in result),sorted(h['id'] for h in hits))
                if result:
                    context,_=format_context(result,max_bytes=10000)
                    positions=[context.index(f'Passage {i}') for i in expected]
                    self.assertEqual(positions,sorted(positions))

    def test_top_k_selection_precedes_context_reordering(self):
        model=Mock();model.predict.return_value=[5.,4.,3.,2.,1.]
        with patch('src.retriever._RERANKER',model):
            result=rerank_candidates('Question',self.hits()[:5],3)
        self.assertEqual([h['id'] for h in result],['0','2','1'])
        self.assertEqual([h['rerank_score'] for h in result],[5.,3.,4.])
