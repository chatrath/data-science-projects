"""Reranked RAG graph with a local evidence gate and streaming Chainlit adapter.

Use ``await graph.ainvoke({'question': question, 'loop_step': 0})`` or the
AgentRAGChain adapter, which exposes the same token/final events as RAGChain.
Each invocation starts fresh; no conversations share graph state.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agent_state import AgentState
from .agent_nodes import decompose_query, retrieve_documents, grade_documents, generate_answer
from .chain import REFUSAL, filter_cited_images

MAX_RETRIEVAL_ATTEMPTS = 2


def route_after_grading(state: AgentState) -> str:
    if state.get('generation_grade') == 'yes':
        return 'generate'
    return 'decompose' if state.get('loop_step', 0) < MAX_RETRIEVAL_ATTEMPTS else 'generate'


def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node('decompose', decompose_query)
    builder.add_node('retrieve', retrieve_documents)
    # This is a local nonempty-evidence gate, not an additional LLM invocation.
    # The hybrid retriever's cross-encoder already provides relevance filtering.
    builder.add_node('grade', grade_documents)
    builder.add_node('generate', generate_answer)
    builder.add_edge(START, 'decompose')
    builder.add_edge('decompose', 'retrieve')
    builder.add_edge('retrieve', 'grade')
    builder.add_conditional_edges('grade', route_after_grading,
                                  {'decompose': 'decompose', 'generate': 'generate'})
    builder.add_edge('generate', END)
    return builder.compile()


graph = build_agent_graph()


def agent_result(state: AgentState) -> dict:
    """Translate graph state without inventing source metadata or cosine scores.

    retrieve_documents has already applied the cosine guardrail; paths are
    revalidated here and again at display time. No paths appear in answer text.
    """
    answer = state.get('generation') or REFUSAL
    refused = answer == REFUSAL
    sources = [] if refused else state.get('sources', [])
    paths = filter_cited_images(state.get('slide_images', []), sources)
    return {'answer': answer, 'sources': sources,
            'refused': refused, 'slide_images': paths, 'timings': {}}


class AgentRAGChain:
    """Drop-in app adapter: planning/retrieval run off the event loop."""

    def __init__(self, compiled_graph=None):
        self.graph = compiled_graph if compiled_graph is not None else graph

    async def astream(self, query: str):
        query = query.strip()
        if not query or len(query.encode('utf-8')) > 1024:
            raise ValueError('Question must fit within 1024 UTF-8 bytes and not be empty')
        state = {'question': query, 'loop_step': 0, 'documents': [], 'slide_images': []}
        async for mode, payload in self.graph.astream(
                state, stream_mode=['messages', 'updates'], config={'recursion_limit': 16}):
            if mode == 'messages':
                message, metadata = payload
                if metadata.get('langgraph_node') == 'generate' and isinstance(message.content, str) and message.content:
                    yield {'type': 'token', 'text': message.content}
            elif mode == 'updates':
                for update in payload.values():
                    if isinstance(update, dict):
                        state.update(update)
        yield {'type': 'final', 'result': agent_result(state)}
