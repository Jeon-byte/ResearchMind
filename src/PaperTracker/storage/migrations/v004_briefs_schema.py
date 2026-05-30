"""Migration v004: Brief schema for ResearchMind."""

from __future__ import annotations

from PaperTracker.storage.migration import Migration

MIGRATION = Migration(
    version=4,
    description="Add briefs and brief_papers tables for search workspace flow",
    sql="""
        CREATE TABLE IF NOT EXISTS briefs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          query_text TEXT NOT NULL,
          sources_json TEXT NOT NULL DEFAULT '[]',
          max_results INTEGER NOT NULL DEFAULT 10,
          created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          updated_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          last_run_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
        );

        CREATE TABLE IF NOT EXISTS brief_papers (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          brief_id INTEGER NOT NULL,
          source TEXT NOT NULL,
          source_id TEXT NOT NULL,
          added_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          FOREIGN KEY (brief_id) REFERENCES briefs(id) ON DELETE CASCADE,
          UNIQUE(brief_id, source, source_id)
        );

        CREATE INDEX IF NOT EXISTS idx_brief_papers_brief
          ON brief_papers(brief_id, added_at DESC);
    """,
)
