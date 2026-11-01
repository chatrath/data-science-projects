import unittest
from app import format_answer_markdown,format_chat_answer
from src.chain import PROMPT,REFUSAL


class ResponseMarkdownTests(unittest.TestCase):
    def test_adjacent_question_sections_get_blank_lines_and_keep_citations(self):
        raw='### 1. Dense retrieval\n**Dense** uses meaning [S1].\n### 2. Sparse retrieval\n**Sparse** uses keywords [S2].'
        expected='### 1. Dense retrieval\n\n**Dense** uses meaning [S1].\n\n### 2. Sparse retrieval\n\n**Sparse** uses keywords [S2].'
        self.assertEqual(format_answer_markdown(raw),expected)

    def test_top_level_numbered_answers_are_separated_without_changing_nested_lists(self):
        raw='1. First answer [S1].\n   - Nested point\n2. Second answer [S2].'
        self.assertEqual(format_answer_markdown(raw),'1. First answer [S1].\n   - Nested point\n\n2. Second answer [S2].')

    def test_backtick_and_tilde_fences_preserve_code_verbatim(self):
        for fence in ['```','~~~~']:
            code=f'{fence}python\r\n# comment\r\n1. literal code\r\n    print("[S1]")\r\n{fence}\r\n'
            raw='### Demo\n'+code+'### Explanation\nText [S1].'
            result=format_answer_markdown(raw)
            self.assertIn(code,result)
            self.assertIn('### Explanation\n\nText [S1].',result)

    def test_plain_paragraphs_bullets_tables_and_inline_code_remain_unchanged(self):
        for raw in ['Paragraph [S1].','- **Key**: value [S1].\n- Other [S2].',
                    '| A | B |\n|---|---|\n| 1 | 2 |','Use `# heading` and `1. literal` [S1].']:
            self.assertEqual(format_answer_markdown(raw),raw)

    def test_spacing_is_idempotent(self):
        raw='**1. First question**\nAnswer [S1].\n**2. Second question**\nAnswer [S2].'
        once=format_answer_markdown(raw)
        self.assertEqual(format_answer_markdown(once),once)
        self.assertIn('Answer [S1].\n\n**2.',once)

    def test_refusal_and_source_citation_footer_remain_intact(self):
        self.assertEqual(format_chat_answer({'answer':REFUSAL,'sources':[]}),REFUSAL)
        result=format_chat_answer({'answer':'### Answer\nEvidence [S1].',
                                   'sources':[{'citation':'S1','label':'guide.pdf | page 2'}]})
        self.assertIn('### Answer\n\nEvidence [S1].',result)
        self.assertIn('**Sources**',result)
        self.assertIn('- **[S1]** guide',result)
        self.assertIn('page 2',result)

    def test_generation_prompt_requests_markdown_and_separate_answer_sections(self):
        messages=PROMPT.invoke({'context':'[S1] Evidence','question':'Question one? Question two?'}).to_messages()
        self.assertIn('**bold key terms**',messages[0].content)
        self.assertIn('multi-question or multi-part',messages[0].content)
        self.assertIn('numbered ### heading',messages[-1].content)
        self.assertIn('blank lines between answers',messages[-1].content)
        self.assertIn('1-2 short bullets per part',messages[-1].content)
        self.assertIn('at most 45 words per part',messages[-1].content)
        self.assertIn('Do not guess.',messages[0].content)
