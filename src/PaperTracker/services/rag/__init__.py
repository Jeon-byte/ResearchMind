"""ResearchMind RAG engine.

The RAG engine owns rebuildable retrieval assets for Knowledge Bases while
SQLite remains the product source of truth.
"""

from PaperTracker.services.rag.service import CollectionRAGService

__all__ = ["CollectionRAGService"]
