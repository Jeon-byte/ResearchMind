"""Migration v005: add multimodal metadata to paper chunks."""

from __future__ import annotations

from PaperTracker.storage.migration import Migration

MIGRATION = Migration(
    version=5,
    description="Add modality and image path metadata to paper chunks",
    sql="""
        ALTER TABLE paper_chunks
          ADD COLUMN modality TEXT NOT NULL DEFAULT 'text';

        ALTER TABLE paper_chunks
          ADD COLUMN image_path TEXT;

        CREATE INDEX IF NOT EXISTS idx_paper_chunks_modality
          ON paper_chunks(source, source_id, modality);
    """,
)
