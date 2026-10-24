"""Local cohort chat UI. Run: .\\start_app.ps1 (Python 3.13 UI environment)"""
from __future__ import annotations

import html
from contextlib import aclosing
import logging
import re

import chainlit as cl
import httpx
from ollama import ResponseError
import psycopg

from src.chain import RAGChain
from src.retriever import database_error_message, resolve_render_path, high_confidence_visual

logger = logging.getLogger(__name__)


@cl.data_layer
def chat_data_layer():
    # DATABASE_URL belongs to the RAG corpus. Do not let Chainlit automatically
    # use it for chat-history persistence (which requires a separate schema).
    return None


def format_chat_answer(result: dict) -> str:
    """Keep answer Markdown/code intact and add cited source labels below it."""
    content = result['answer']
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
        if not high_confidence_visual(image.get('dense_score')):
            continue
        # Recheck paths at display time: a file may have moved since retrieval.
        path = resolve_render_path({'render_path': image.get('path')})
        if path is None or path in seen:
            continue
        seen.add(path)
        elements.append(cl.Image(path=path, name=image['label'], display='inline', size='large'))
    return elements


async def get_chain() -> RAGChain:
    chain = cl.user_session.get('rag_chain')
    if chain is None:
        chain = await cl.make_async(RAGChain)()
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
    reply = None
    elements = []
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
        chain = await get_chain()
        result = None
        async with aclosing(chain.astream(query)) as stream:
            async for event in stream:
                if event['type'] == 'token':
                    await reply.stream_token(event['text'])
                elif event['type'] == 'final':
                    result = event['result']
        if result is None:
            raise RuntimeError('Generation ended without a final result')
        # Normalize grouped citations and replace provisional text if the final
        # citation check refuses the answer. Attach sources/images only now.
        content = format_chat_answer(result)
        elements = slide_image_elements(result)
        if elements:
            content += '\n\n**Retrieved slide previews**\n\nThese slides accompany the retrieved text. The answer above uses text evidence.'
    except Exception as exc:
        logger.warning('Chat request failed (%s)', type(exc).__name__)
        content = error_message(exc)
    await deliver(content, reply, elements)
