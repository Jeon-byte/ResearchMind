# ResearchMind RAG Migration Plan

## Goal

ResearchMind keeps the existing paper tracking, Brief, Knowledge Base, and chat product model. MultiRAG-Doc contributes the RAG engine ideas: section-aware chunks, vector retrieval, reranking, grounded answers, and citations.

The first implementation target is Standard RAG:

```text
Knowledge Base PDF -> parse -> chunk -> embed -> index
Question -> retrieve -> rerank(optional) -> answer with citations
```

Decompose, agentic retrieval, and multimodal figure/table retrieval can be added later after the Standard path is stable.

## Storage Split

SQLite remains the product source of truth:

- Briefs and Knowledge Bases.
- Paper metadata and collection membership.
- PDF download, parse, and index status.
- Conversations, messages, and answer citations.

RAG workspaces store rebuildable retrieval assets:

```text
<db_dir>/rag/
  collections/
    kb_<collection_id>/
      chunks/
        all_chunks.json
      index/
        text_index.faiss     # when faiss is installed
        text_index.npz       # deterministic fallback
      manifest.json
```

Only Knowledge Base contents get RAG workspaces. Brief search results remain temporary research material until the user adds papers to a Knowledge Base.

## Migration Strategy From MultiRAG-Doc

Move concepts, not the whole project:

- Adopt section-aware chunking as the default chunk strategy.
- Adopt `all_chunks.json` metadata snapshots for RAG assets.
- Adopt FAISS `IndexIDMap + IndexFlatIP` when available.
- Adopt BGE-M3 embedding when `FlagEmbedding` is available.
- Keep deterministic fallback embedding/vector search so the web app and tests still run without GPU dependencies.
- Add BGE reranker later as an optional quality layer.
- Keep ResearchMind's SQLite schema and web API as the integration boundary.

## MVP Implementation

1. Add `PaperTracker.services.rag`.
2. Build one workspace per Knowledge Base.
3. During paper ingestion, persist chunks to SQLite and rebuild the KB RAG index.
4. During QA, retrieve from the KB RAG workspace first.
5. Fall back to existing SQLite FTS if a workspace does not exist or has no hits.
6. Store messages and citations in the existing conversation tables.

## Near-Term Quality Roadmap

Standard RAG v1:

- Dense vector retrieval over full-text chunks.
- Citation-backed answer generation.
- Deterministic evidence digest when no LLM is configured.
- Configured BGE-M3 embedding and BGE reranker model names.
- Explicit model download command: `paper-tracker rag-download-models --config config/example.yml`.
- Safe fallback to hashing retrieval when local model files are incomplete or unavailable.

Standard RAG v2:

- Hybrid retrieval: dense + SQLite FTS/BM25.
- Optional BGE reranker.
- Retrieval debug panel in the UI.
- Minimal decompose mode for multi-part questions.
- Agent-lite retrieval entry behind `rag.agent_enabled`.

Research RAG v3:

- Full query decomposition with planner prompts.
- Section-aware routing for method/result/experiment questions.
- Citation and recall evaluation datasets per Knowledge Base.

Multimodal v4:

- Table chunks.
- Figure caption chunks.
- Image retrieval and multimodal fusion.

## Current Implementation Status

Implemented now:

- KB-specific RAG workspaces.
- Section-aware text chunking.
- BGE-M3 configuration and local-model-first loading.
- FAISS vector index when `faiss` is installed, with numpy fallback.
- Hybrid retrieval using vector hits plus SQLite FTS/BM25 hits.
- Optional BGE reranker when local model files and `sentence-transformers` are available.
- Standard mode.
- Minimal decompose mode.
- Agent-lite mode behind `rag.agent_enabled`.
- Web retrieval debug panel.

Not complete yet:

- Full MultiRAG-Doc agent loop.
- Docling-based multimodal parsing.
- Figure/table/equation indexing.
- Multimodal fusion.
- Retrieval evaluation UI.
