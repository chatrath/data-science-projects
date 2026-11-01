"""Local cohort chat UI. Run: .\\start_app.ps1 (Python 3.13 UI environment)"""
from __future__ import annotations

import html
from contextlib import aclosing
import logging
import os
import re

import chainlit as cl
import httpx
from ollama import ResponseError
import psycopg
from dotenv import dotenv_values
from pathlib import Path

from src.cache import get_semantic_cache, successful as cacheable_result
from src.chain import RAGChain
from src.previews import verified_cited_page_image
from src.agent_graph import AgentRAGChain
from src.retriever import database_error_message, resolve_render_path, high_confidence_visual

from src.security import install_origin_security, RequestLimiter, configured_request_limit

install_origin_security()
_REQUEST_LIMITER = RequestLimiter(configured_request_limit())
logger = logging.getLogger(__name__)


@cl.data_layer
def chat_data_layer():
    # DATABASE_URL belongs to the RAG corpus. Do not let Chainlit automatically
    # use it for chat-history persistence (which requires a separate schema).
    return None


def format_answer_markdown(content: str) -> str:
    """Space model-produced answer sections without rewriting prose or code.

    Streaming stays verbatim so split Markdown tokens/fences are never buffered
    or rewritten. The final update repairs block spacing outside fenced code.
    """
    output = []
    fence = None
    after_heading = False
    for line in content.splitlines(keepends=True):
        body = line.rstrip('\r\n')
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', body)
        if fence is not None:
            output.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
            continue
        if after_heading and body.strip():
            output.append('\n')
        after_heading = False
        if marker:
            fence = (marker[1][0], len(marker[1]))
            output.append(line)
            continue
        heading = bool(re.match(r'^ {0,3}#{1,6}\s+\S', body) or
                       re.fullmatch(r'\*\*[^*]+\*\*:?[ \t]*', body))
        numbered = bool(re.match(r'^\d+[.)]\s+\S', body))
        if (heading or numbered) and output and output[-1].strip():
            output.append('\n')
        output.append(line)
        after_heading = heading
    return ''.join(output)


def format_chat_answer(result: dict) -> str:
    """Keep answer Markdown/code intact and add cited source labels below it."""
    content = format_answer_markdown(result['answer'])
    if result.get('sources'):
        lines = []
        for source in result['sources']:
            # Source labels are document metadata, not trusted Markdown or HTML.
            label = re.sub(r'([\\`*_{}\[\]()#+.!|>~-])', r'\\\1', html.escape(source['label']))
            lines.append(f"- **[{source['citation']}]** {label}")
        content += '\n\n---\n\n**Sources**\n\n' + '\n'.join(lines)
    return content


def slide_image_elements(result: dict) -> list[cl.Image]:
    """Display only high-confidence slide frames, once per path."""
    elements, seen = [], set()
    for image in result.get('slide_images', []):
        # AgentState carries paths selected by relevance-filtered retrieval.
        # Conventional RAG results carry image metadata and a raw cosine score.
        if isinstance(image, str):
            path = resolve_render_path({'render_path': image})
            if path is not None and path not in seen:
                seen.add(path)
                elements.append(cl.Image(path=path, name=Path(path).name, display='inline', size='large'))
            continue
        if not isinstance(image, dict):
            continue
        if image.get('preview_type') == 'cited_page':
            if not verified_cited_page_image(image, result.get('sources', [])):
                continue
        elif not high_confidence_visual(image.get('dense_score')):
            continue
        # Recheck paths at display time: a file may have moved since retrieval.
        path = resolve_render_path({'render_path': image.get('path')})
        if path is None or path in seen:
            continue
        seen.add(path)
        elements.append(cl.Image(path=path, name=image['label'], display='inline', size='large'))
    # Agent paths originate only from the graph's relevance-filtered retrieval.
    for raw in result.get('agent_image_paths', []):
        path = resolve_render_path({'render_path': raw})
        if path is not None and path not in seen:
            seen.add(path)
            elements.append(cl.Image(path=path, name=Path(path).name, display='inline', size='large'))
    return elements


async def get_chain() -> RAGChain:
    enabled = str(dotenv_values(Path(__file__).parent / '.env').get(
        'ENABLE_AGENT_GRAPH', os.getenv('ENABLE_AGENT_GRAPH', 'false'))).lower() in {'true', '1', 'yes'}
    chain = cl.user_session.get('rag_chain')
    if chain is None or cl.user_session.get('agent_graph_enabled') != enabled:
        chain = await cl.make_async(AgentRAGChain if enabled else RAGChain)()
        cl.user_session.set('agent_graph_enabled', enabled)
        cl.user_session.set('rag_chain', chain)
    return chain


def error_message(exc: Exception) -> str:
    if isinstance(exc, psycopg.Error):
        return database_error_message(exc)
    if isinstance(exc, (ResponseError, httpx.HTTPError, ConnectionError)):
        return 'I could not reach the local model. Please check that Ollama is running and the configured models are downloaded, then try again.'
    if isinstance(exc, ValueError):
        return 'The learning companion could not process this request. Please check the local provider configuration and try again.'
    return 'Something went wrong while preparing your answer. Please try again.'


async def deliver(content: str, reply: cl.Message | None = None, elements: list | None = None) -> None:
    """Send/update a response; a disconnected browser must not crash a handler."""
    try:
        if reply is None:
            await cl.Message(content=content, author='Learning Companion', elements=elements or []).send()
        else:
            reply.content = content
            reply.elements = elements or []
            await reply.update()
    except Exception as exc:
        logger.warning('Chat delivery failed (%s)', type(exc).__name__)


@cl.on_chat_start
async def on_chat_start():
    try:
        cl.user_session.set('rag_chain', None)
        await cl.Message(
            content=(
                '**Welcome to your Cohort Learning Companion!**\n\n'
                'Ask a question about the course handouts, slides, or live demos. '
                'I’ll answer from the available course materials and cite my sources.\n\n'
                'For example: *What is context engineering?*'
            ),
            author='Learning Companion',
        ).send()
        await get_chain()
    except Exception as exc:
        logger.warning('Chat initialization failed (%s)', type(exc).__name__)
        await deliver(error_message(exc))


@cl.on_message
async def on_message(msg: cl.Message):
    # Acquire without awaiting: neither simultaneous session messages nor other
    # sessions can create an unbounded queue of CPU/model work.
    if cl.user_session.get('request_active'):
        await deliver('Your previous question is still running. Please wait for it to finish.')
        return
    if not _REQUEST_LIMITER.acquire():
        await deliver('The learning companion is busy. Please try again shortly.')
        return
    try:
        cl.user_session.set('request_active', True)
        await _process_message(msg)
    finally:
        try:
            cl.user_session.set('request_active', False)
        finally:
            _REQUEST_LIMITER.release()


async def _process_message(msg: cl.Message):
    reply = None
    elements = []
    cache = lookup = result = None
    cache_hit = False
    response_succeeded = False
    try:
        query = (msg.content or '').strip()
        if not query:
            await deliver('Please enter a question about the course materials.')
            return
        if len(query.encode('utf-8')) > 1024:
            await deliver('Please shorten your question to fit within 1,024 UTF-8 bytes.')
            return
        reply = cl.Message(content='', author='Learning Companion')
        await reply.send()
        try:
            cache = await cl.make_async(get_semantic_cache)()
            if cache is not None:
                lookup = await cache.lookup(query)
                result = lookup.result
        except Exception as exc:
            logger.warning('Semantic cache lookup bypassed (%s)', type(exc).__name__)
            cache = lookup = None
        cache_hit = result is not None
        if cache_hit:
            logger.info('Semantic cache hit (cosine %.3f)', lookup.similarity or 1.0)
        if not cache_hit:
            chain = await get_chain()
            async with aclosing(chain.astream(query)) as stream:
                async for event in stream:
                    if event['type'] == 'token':
                        # Chainlit renders Markdown as tokens arrive. Preserve split
                        # markup/code fences; normalize spacing only in the final update.
                        await reply.stream_token(event['text'])
                    elif event['type'] == 'final':
                        result = event['result']
        if result is None:
            raise RuntimeError('Generation ended without a final result')
        # Normalize grouped citations and replace provisional text if the final
        # citation check refuses the answer. Attach sources/images only now.
        content = format_chat_answer(result)
        elements = await cl.make_async(slide_image_elements)(result)
        response_succeeded = True
        if elements:
            content += '\n\n**Cited PDF previews**\n\nThese pages accompany the cited text. The answer above uses text evidence.'
    except Exception as exc:
        logger.warning('Chat request failed (%s)', type(exc).__name__)
        content = error_message(exc)
    await deliver(content, reply, elements)
    # Write after final UI delivery, so cache embedding/disk I/O cannot delay tokens.
    if cache is not None and lookup is not None and response_succeeded and cacheable_result(result) and not cache_hit:
        try:
            await cache.store(query, result, lookup)
        except Exception as exc:
            logger.warning('Semantic cache write bypassed (%s)', type(exc).__name__)
