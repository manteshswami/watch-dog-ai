"""
WatchAI — Embeddings Factory
Singleton Ollama embedding function for ChromaDB.
Uses nomic-embed-text served via the local Ollama instance.
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = "nomic-embed-text"):
    """
    Singleton factory — reuse the same Ollama embedding model across the app.
    Requires Ollama running locally with nomic-embed-text pulled.
    """
    import config
    from chromadb.utils import embedding_functions

    base_url = config.OLLAMA_BASE_URL
    logger.info(f"Loading Ollama embedding model: {model_name} @ {base_url}")

    return embedding_functions.OllamaEmbeddingFunction(
        url=f"{base_url}/api/embeddings",
        model_name=model_name,
    )
