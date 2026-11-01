"""Shared hosted-provider and local LangChain/Ollama configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_BASE_URL = 'http://localhost:11434/'
NEBIUS_BASE_URL = 'https://api.tokenfactory.nebius.com/v1/'


@dataclass(frozen=True)
class ProviderSettings:
    provider: str = 'openai'
    api_key: str = field(default='', repr=False)
    base_url: str = 'https://api.openai.com/v1/'
    embedding_model: str = 'text-embedding-3-small'
    embedding_dimensions: int = 1536
    chat_model: str = ''
    embedding_batch_size: int = 32
    rag_top_k: int = 3
    rag_context_bytes: int = 5000
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 256
    ollama_num_thread: int | None = None
    ollama_keep_alive: int = 600

    # Multi-part requests can retain two precise chunks per part, bounded overall.
    rag_multi_max_chunks: int = 6
    rag_multi_chunks_per_query: int = 2
    rag_multi_context_bytes: int = 10000
    rag_multi_num_predict: int = 384

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> 'ProviderSettings':
        if env is None:
            load_dotenv(ROOT / '.env')
            env = os.environ
        value = lambda key, default='': env.get(key, '').strip() or default
        provider = value('LLM_PROVIDER', 'nebius' if value('NEBIUS_API_KEY') else 'openai').lower()
        if provider not in {'nebius', 'openai', 'ollama'}:
            raise ValueError('LLM_PROVIDER must be ollama, nebius or openai')
        nebius = provider == 'nebius'
        local = provider == 'ollama'
        prefix = provider.upper()
        base_url = value(f'{prefix}_BASE_URL', OLLAMA_BASE_URL if local else NEBIUS_BASE_URL if nebius else 'https://api.openai.com/v1/').rstrip('/') + '/'
        parsed = urlsplit(base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError(f'{prefix}_BASE_URL must be a plain HTTP(S) API base URL, without credentials or query parameters')
        try:
            dimensions = int(value(f'{prefix}_EMBEDDING_DIMENSIONS', '768' if local else '4096' if nebius else '1536'))
        except ValueError as exc:
            raise ValueError(f'{prefix}_EMBEDDING_DIMENSIONS must be an integer') from exc
        if not 1 <= dimensions <= 16000:
            raise ValueError(f'{prefix}_EMBEDDING_DIMENSIONS must be between 1 and 16000')
        def integer(key, default, minimum, maximum):
            try:
                number = int(value(key, str(default)))
            except ValueError as exc:
                raise ValueError(f'{key} must be an integer') from exc
            if not minimum <= number <= maximum:
                raise ValueError(f'{key} must be between {minimum} and {maximum}')
            return number
        return cls(provider, '' if local else value(f'{prefix}_API_KEY'), base_url,
                   value(f'{prefix}_EMBEDDING_MODEL', 'nomic-embed-text' if local else 'Qwen/Qwen3-Embedding-8B' if nebius else 'text-embedding-3-small'),
                   dimensions, value(f'{prefix}_CHAT_MODEL', 'qwen3:4b-instruct' if local else ''),
                   embedding_batch_size=integer('EMBEDDING_BATCH_SIZE',32,1,32),
                   rag_top_k=integer('RAG_TOP_K',3,1,25),
                   rag_context_bytes=integer('RAG_CONTEXT_BYTES',5000,256,24000),
                   ollama_num_ctx=integer('OLLAMA_NUM_CTX',8192,2048,32768),
                   ollama_num_predict=integer('OLLAMA_NUM_PREDICT',256,1,1024),
                   ollama_num_thread=integer('OLLAMA_NUM_THREAD',1,1,256) if value('OLLAMA_NUM_THREAD') else None,
                   ollama_keep_alive=integer('OLLAMA_KEEP_ALIVE',600,0,86400),
                   rag_multi_max_chunks=integer('RAG_MULTI_MAX_CHUNKS',6,1,12),
                   rag_multi_chunks_per_query=integer('RAG_MULTI_CHUNKS_PER_QUERY',2,1,3),
                   rag_multi_context_bytes=integer('RAG_MULTI_CONTEXT_BYTES',10000,256,24000),
                   rag_multi_num_predict=integer('RAG_MULTI_NUM_PREDICT',384,1,1024))


def create_provider_client(settings: ProviderSettings | None = None) -> OpenAI | ChatOllama:
    settings = settings or ProviderSettings.from_env()
    if settings.provider == 'ollama':
        return ChatOllama(model=settings.chat_model, base_url=settings.base_url, temperature=0,
                          num_ctx=settings.ollama_num_ctx, num_predict=settings.ollama_num_predict,
                          num_thread=settings.ollama_num_thread, keep_alive=settings.ollama_keep_alive,
                          client_kwargs={'timeout': 600.0})
    if not settings.api_key:
        raise ValueError(f'Set {settings.provider.upper()}_API_KEY in the project .env')
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=60.0, max_retries=2)


def create_embedding_client(settings: ProviderSettings | None = None):
    """Return LangChain's native local embedder or a hosted OpenAI client."""
    settings = settings or ProviderSettings.from_env()
    if settings.provider == 'ollama':
        # nomic-embed-text defaults to 2K in Ollama; request its 8K context.
        # Leave dimensions unset: nomic returns its native 768-dimension vectors.
        return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.base_url,
                                num_ctx=8192, num_thread=settings.ollama_num_thread, keep_alive=settings.ollama_keep_alive,
                                client_kwargs={'timeout': 120.0})
    return create_provider_client(settings)


def generate_text(messages: list[dict], *, settings: ProviderSettings | None = None, client=None, **kwargs) -> str:
    """Chat-completions helper for the future answering chain (no retrieval)."""
    settings = settings or ProviderSettings.from_env()
    if not settings.chat_model:
        raise ValueError(f'Set {settings.provider.upper()}_CHAT_MODEL to an activated generation model ID')
    if settings.provider == 'ollama':
        model = client if client is not None else create_provider_client(settings)
        if 'max_tokens' in kwargs:
            if 'num_predict' in kwargs:
                raise ValueError('Use either max_tokens or num_predict, not both')
            kwargs['num_predict'] = kwargs.pop('max_tokens')
        options = dict(kwargs.pop('options', {}))
        for key in ('num_predict', 'num_ctx', 'num_thread', 'temperature', 'top_k', 'top_p'):
            if key in kwargs:
                options[key] = kwargs.pop(key)
        response = model.invoke(messages, **({'options': options} if options else {}), **kwargs)
        if not isinstance(response.content, str):
            raise ValueError('Local generation returned non-text content')
        return response.content
    with_client = client is None
    client = create_provider_client(settings) if with_client else client
    try:
        response = client.chat.completions.create(model=settings.chat_model, messages=messages, **kwargs)
        text = response.choices[0].message.content
        if text is None:
            raise ValueError('Generation returned no text; check model output and token budget')
        return text
    finally:
        if with_client:
            client.close()
