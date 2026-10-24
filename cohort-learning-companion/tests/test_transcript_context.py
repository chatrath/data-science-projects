import tempfile
import unittest
from pathlib import Path
from src.ingest_materials import load_transcript, transcript_windows
from src.retriever import boost_transcripts

class TranscriptContextTests(unittest.TestCase):
    def test_windows_overlap_cover_all_and_preserve_code(self):
        blocks = [{'text': str(i) * 30} for i in range(10)]
        blocks[4]['text'] = '```python\n' + '    print(1)\n' * 20 + '```'
        windows = list(transcript_windows(blocks, 120, 40))
        self.assertEqual({id(b) for _, g in windows for b in g}, {id(b) for b in blocks})
        self.assertTrue(any(set(map(id,a)).intersection(map(id,b)) for (_,a),(_,b) in zip(windows,windows[1:])))
        self.assertTrue(all(windows[i+1][0] > windows[i][0] for i in range(len(windows)-1)))
        self.assertEqual(sum(b is blocks[4] for _,g in windows for b in g), 1)

    def test_prefix_uses_content_keywords_and_keeps_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'unrelated_filename.txt'
            path.write_text('WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nAlice: [um] embeddings embeddings retrieval vectors\n', encoding='utf-8')
            item = list(load_transcript(path))[0]
            self.assertTrue(item['text'].startswith('[Transcript Segment - Topic keywords:'))
            self.assertIn('embeddings',item['topic_keywords'])
            self.assertNotIn('alice',item['topic_keywords'])
            self.assertNotIn('unrelated',item['context_header'])
            self.assertNotIn('[um]',item['text'])
            self.assertEqual(item['start'],'00:00:01.000')
            with self.assertRaises(ValueError): list(load_transcript(path,100,100))

    def test_boost_only_relevant_transcripts(self):
        lookup = {k:{'metadata':{'kind':kind},'content':'vector retrieval'} for k,kind in [('p','handout'),('t','transcript'),('w','transcript')]}
        hits = [{'id':'p','score':.016,'dense_score':.8},{'id':'t','score':.015,'dense_score':.7},{'id':'w','score':.0155,'dense_score':.2}]
        ranked = boost_transcripts(hits,lookup,'semantic search')
        self.assertEqual([h['id'] for h in ranked],['t','p','w'])
        self.assertEqual(ranked[-1]['source_boost'],1)
        self.assertEqual(ranked[0]['rrf_score'],.015)
        self.assertEqual(hits[1]['score'],.015)
        self.assertEqual([h['id'] for h in boost_transcripts(hits,lookup,'semantic search',1)],['p','w','t'])
