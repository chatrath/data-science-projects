"""Shared state schema for a self-correcting LangGraph RAG agent.

Pass this TypedDict to ``StateGraph(AgentState)``. Nodes can return partial
state updates; unannotated fields replace their previous values. The graph
must enforce its loop limit when routing, using ``loop_step``.
"""
from typing import List
from typing_extensions import TypedDict, NotRequired


class AgentState(TypedDict):
    """Retrieval, generation and grading state shared between graph nodes."""

    question: str
    sub_queries: List[str]
    documents: List[str]
    slide_images: List[str]
    generation_grade: str  # Grader output: "yes" or "no".
    generation: str
    loop_step: int

    # Preserve retrieval provenance while keeping documents as text for grading.
    retrieved_chunks: NotRequired[List[dict]]
    sources: NotRequired[List[dict]]

    timings: NotRequired[dict]
