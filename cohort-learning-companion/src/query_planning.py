"""Conservative, subject-independent helpers for explicit multi-part questions."""
import re


def clean_question(part: str) -> str:
    part = part.strip().strip('?.! ')
    part = re.sub(r'^(?:(?:also|and|then|finally|next)[\s,:]+)+', '', part, flags=re.I)
    part = re.sub(r'^(?:can|could|would)\s+you\s+(?:(?:please|also)\s+)*(?:explain|describe|tell\s+me)\s+', '', part, flags=re.I)
    return part.strip()


def explicit_question_parts(question: str) -> list[str] | None:
    """Preserve explicit questions; leave pronouns/quoted syntax to the model.

    No technical terms or sample-question answers are hardcoded. Coordinated
    interrogatives split, but 'X and Y' comparison subjects stay together.
    """
    if any(mark in question for mark in ['`', '"', 'http://', 'https://']):
        return None
    parts = []
    for sentence in question.split('?'):
        for clause in re.split(r'\s+and\s+(?=(?:what|how|why|when|where|which|who)\b)', sentence, flags=re.I):
            part = clean_question(clause)
            if part:
                parts.append(part)
    if not 2 <= len(parts) <= 5:
        return None
    if any(not re.match(r'^(?:what|how|why|when|where|which|who|define|list|explain|describe|compare)\b', p, re.I)
           or re.search(r'\b(?:it|its|they|their|them|this|that|these|those)\b', p, re.I)
           or not 3 <= len(p.split()) <= 30 for p in parts):
        return None
    queries = list(dict.fromkeys(p[:1].upper() + p[1:] + '?' for p in parts))
    return queries if len(queries) > 1 else None


def topic_terms(text: str) -> list[str]:
    """Compare a short focused question with a literal topic heading."""
    text = clean_question(text)
    text = re.sub(r'^(?:what|who|which)\s+(?:is|are|was|were)\s+', '', text, flags=re.I)
    text = re.sub(r'^how\s+(?:(?:does|do|can|is|are)\s+)?', '', text, flags=re.I)
    text = re.sub(r'^(?:define|list|explain|describe)\s+', '', text, flags=re.I)
    text = re.sub(r'^(?:a|an|the)\s+', '', text, flags=re.I)
    return re.findall(r'\w+', text.casefold())


def retrieval_wording(query: str) -> str:
    """Remove enumeration scaffolding without changing the requested topic."""
    text = clean_question(query)
    focused = re.sub(r'^(?:(?:what|which)\s+are\s+|list\s+)(?:the\s+)?', '', text, flags=re.I)
    return focused if focused != text and focused.strip() else query
