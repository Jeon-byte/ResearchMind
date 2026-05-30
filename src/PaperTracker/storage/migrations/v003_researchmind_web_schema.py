"""Migration v003: ResearchMind web MVP schema."""

from __future__ import annotations

from PaperTracker.storage.migration import Migration

MIGRATION = Migration(
    version=3,
    description="ResearchMind web MVP collections, assets, chunks, and chat schema",
    sql="""
        CREATE TABLE IF NOT EXISTS collections (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE,
          description TEXT NOT NULL DEFAULT '',
          created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          updated_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
        );

        CREATE TABLE IF NOT EXISTS collection_papers (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          collection_id INTEGER NOT NULL,
          source TEXT NOT NULL,
          source_id TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'queued',
          added_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
          UNIQUE(collection_id, source, source_id)
        );

        CREATE INDEX IF NOT EXISTS idx_collection_papers_collection
          ON collection_papers(collection_id, added_at DESC);

        CREATE TABLE IF NOT EXISTS paper_assets (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source TEXT NOT NULL,
          source_id TEXT NOT NULL,
          pdf_url TEXT,
          local_path TEXT,
          download_status TEXT NOT NULL DEFAULT 'queued',
          download_error TEXT,
          parse_status TEXT NOT NULL DEFAULT 'queued',
          parse_error TEXT,
          index_status TEXT NOT NULL DEFAULT 'queued',
          index_error TEXT,
          downloaded_at INTEGER,
          parsed_at INTEGER,
          indexed_at INTEGER,
          updated_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          UNIQUE(source, source_id)
        );

        CREATE TABLE IF NOT EXISTS paper_chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source TEXT NOT NULL,
          source_id TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          section_title TEXT,
          page_start INTEGER,
          page_end INTEGER,
          content TEXT NOT NULL,
          token_count INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          UNIQUE(source, source_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_paper_chunks_source
          ON paper_chunks(source, source_id, chunk_index);

        CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunk_fts USING fts5(
          chunk_id UNINDEXED,
          source UNINDEXED,
          source_id UNINDEXED,
          content
        );

        CREATE TABLE IF NOT EXISTS conversations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          collection_id INTEGER NOT NULL,
          title TEXT NOT NULL DEFAULT '',
          created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          updated_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_collection
          ON conversations(collection_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          conversation_id INTEGER NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
          ON messages(conversation_id, created_at ASC);

        CREATE TABLE IF NOT EXISTS answer_citations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          message_id INTEGER NOT NULL,
          paper_chunk_id INTEGER NOT NULL,
          rank_order INTEGER NOT NULL DEFAULT 0,
          score REAL NOT NULL DEFAULT 0.0,
          quote_text TEXT NOT NULL DEFAULT '',
          FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
          FOREIGN KEY (paper_chunk_id) REFERENCES paper_chunks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_answer_citations_message
          ON answer_citations(message_id, rank_order ASC);
    """,
)
