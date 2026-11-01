import unittest
from src.retriever import category_candidates, stratify_results


def row(kind,text='context engineering'):
    return {'metadata':{'kind':kind},'content':text}


class StratifiedRetrievalTests(unittest.TestCase):
    def setup_pool(self):
        lookup={f'p{i}':row('handout') for i in range(6)}
        lookup.update(slide=row('slide'),transcript=row('transcript'))
        hits=[{'id':f'p{i}','score':.02-i*.0001,'dense_score':.8,'bm25_score':5.} for i in range(6)]
        hits.extend([{'id':'slide','score':.015,'dense_score':.75}, {'id':'transcript','score':.014,'dense_score':.7}])
        return lookup,hits

    def test_dominant_handouts_do_not_crowd_out_qualified_sources(self):
        lookup,hits=self.setup_pool()
        selected=stratify_results(hits,lookup,'context engineering',3)
        self.assertEqual([h['id'] for h in selected],['p0','slide','transcript'])
        self.assertTrue(all(h['selection_reason']=='source_reservation' for h in selected))
        self.assertTrue(all(h['baseline_relevance_met'] for h in selected))

    def test_weak_categories_do_not_receive_forced_slots(self):
        lookup,hits=self.setup_pool()
        for hit in hits[-2:]:
            hit.update(dense_score=.2,bm25_score=0)
        selected=stratify_results(hits,lookup,'context engineering',3)
        self.assertEqual([h['id'] for h in selected],['p0','p1','p2'])

    def test_strong_keyword_match_can_qualify_without_dense_similarity(self):
        lookup,hits=self.setup_pool()
        hits[-2].update(dense_score=.1,bm25_score=8.)
        selected=stratify_results(hits,lookup,'context engineering',3)
        slide=next(h for h in selected if h['id']=='slide')
        self.assertEqual(slide['keyword_coverage'],1.)
        self.assertTrue(slide['baseline_relevance_met'])

    def test_stopword_only_overlap_does_not_qualify_for_keyword_reservation(self):
        lookup,hits=self.setup_pool()
        lookup['slide']['content']='the is what and unrelated'
        lookup['transcript']['content']='the is what and unrelated'
        for hit in hits[-2:]:
            hit.update(dense_score=.2,bm25_score=12.)
        selected=stratify_results(hits,lookup,'what is context engineering',3)
        self.assertEqual([h['id'] for h in selected],['p0','p1','p2'])

    def test_small_top_k_uses_strongest_representatives_without_fixed_type_priority(self):
        lookup={'t':row('transcript'),'s':row('slide'),'p':row('handout')}
        hits=[{'id':key,'score':score,'dense_score':.8} for key,score in [('t',.02),('s',.019),('p',.018)]]
        selected=stratify_results(hits,lookup,'context',2)
        self.assertEqual([h['id'] for h in selected],['t','s'])

    def test_extra_slots_backfill_globally_and_missing_categories_are_safe(self):
        lookup,hits=self.setup_pool()
        selected=stratify_results(hits,lookup,'context',5)
        self.assertEqual([h['id'] for h in selected],['p0','p1','p2','slide','transcript'])
        self.assertEqual(len({h['id'] for h in selected}),5)
        self.assertEqual(len(stratify_results(hits[:2],lookup,'context',5)),2)
        self.assertEqual(stratify_results([],{},'context',3),[])

    def test_candidate_allowance_preserves_minority_source_candidates(self):
        lookup,hits=self.setup_pool()
        ranked=[(h['id'],h['score']) for h in hits]
        selected=category_candidates(ranked,lookup,2)
        self.assertEqual([key for key,_ in selected],['p0','p1','slide','transcript'])

    def test_stricter_thresholds_can_disable_all_reservations(self):
        lookup,hits=self.setup_pool()
        selected=stratify_results(hits,lookup,'context engineering',3,min_cosine=.99,min_bm25=100.)
        self.assertEqual([h['id'] for h in selected],['p0','p1','p2'])
        self.assertTrue(all(h['selection_reason']=='global_rank' for h in selected))


if __name__=='__main__':
    unittest.main()
