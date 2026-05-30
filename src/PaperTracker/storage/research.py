"""ResearchMind storage repositories.

Provides collection, asset, chunk, and conversation persistence for the web MVP.
"""

from __future__ import annotations

import sqlite3
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PaperTracker.core.models import Paper
from PaperTracker.utils.log import log


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    """Parsed full-text chunk ready for persistence."""

    chunk_index: int
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    token_count: int = 0
    modality: str = "text"
    image_path: str | None = None


class ResearchStore:
    """SQLite-backed repository for ResearchMind web data."""

    def __init__(self, conn: sqlite3.Connection):
        """Initialize repository with an active SQLite connection.

        Args:
            conn: Active SQLite connection.
        """
        self.conn = conn

    def create_collection(self, name: str, description: str = "") -> dict[str, Any]:
        """Create a new collection record.

        Args:
            name: User-facing collection name.
            description: Optional collection description.

        Returns:
            Created collection payload.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO collections (name, description)
            VALUES (?, ?)
            """,
            (name.strip(), description.strip()),
        )
        self.conn.commit()
        return self.get_collection(int(cursor.lastrowid))

    def update_collection(self, collection_id: int, *, name: str, description: str) -> dict[str, Any]:
        """Update a collection's editable fields.

        Args:
            collection_id: Collection primary key.
            name: Updated collection name.
            description: Updated collection description.

        Returns:
            Updated collection payload.
        """
        self.conn.execute(
            """
            UPDATE collections
            SET name = ?, description = ?, updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (name.strip(), description.strip(), collection_id),
        )
        self.conn.commit()
        return self.get_collection(collection_id)

    def delete_collection(self, collection_id: int) -> None:
        """Delete one collection and its dependent rows.

        Args:
            collection_id: Collection primary key.
        """
        self.conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        self.conn.commit()

    def create_brief(
        self,
        *,
        title: str,
        query_text: str,
        sources: Sequence[str],
        max_results: int,
    ) -> dict[str, Any]:
        """Create a new brief workspace record.

        Args:
            title: Brief title shown in navigation.
            query_text: Original query text.
            sources: Search sources used for the brief.
            max_results: Search result limit used to create the brief.

        Returns:
            Created brief payload.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO briefs (title, query_text, sources_json, max_results)
            VALUES (?, ?, ?, ?)
            """,
            (
                title.strip(),
                query_text.strip(),
                json.dumps(list(sources), ensure_ascii=False),
                max_results,
            ),
        )
        self.conn.commit()
        return self.get_brief(int(cursor.lastrowid))

    def update_brief(self, brief_id: int, *, title: str) -> dict[str, Any]:
        """Rename one brief.

        Args:
            brief_id: Brief primary key.
            title: Updated title.

        Returns:
            Updated brief payload.
        """
        self.conn.execute(
            """
            UPDATE briefs
            SET title = ?, updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (title.strip(), brief_id),
        )
        self.conn.commit()
        return self.get_brief(brief_id)

    def update_brief_max_results(self, brief_id: int, max_results: int) -> dict[str, Any]:
        """Update rerun result limit for one brief.

        Args:
            brief_id: Brief primary key.
            max_results: Search result limit for subsequent reruns.

        Returns:
            Updated brief payload.
        """
        self.conn.execute(
            """
            UPDATE briefs
            SET max_results = ?, updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (max_results, brief_id),
        )
        self.conn.commit()
        return self.get_brief(brief_id)

    def delete_brief(self, brief_id: int) -> None:
        """Delete one brief and all its paper links.

        Args:
            brief_id: Brief primary key.
        """
        self.conn.execute("DELETE FROM briefs WHERE id = ?", (brief_id,))
        self.conn.commit()

    def list_briefs(self) -> list[dict[str, Any]]:
        """Return all briefs ordered by last update."""
        rows = self.conn.execute(
            """
            SELECT
                b.id,
                b.title,
                b.query_text,
                b.sources_json,
                b.max_results,
                b.created_at,
                b.updated_at,
                b.last_run_at,
                COUNT(bp.id) AS paper_count
            FROM briefs AS b
            LEFT JOIN brief_papers AS bp
              ON bp.brief_id = b.id
            GROUP BY b.id
            ORDER BY b.updated_at DESC, b.id DESC
            """
        ).fetchall()
        return [_brief_payload_from_row(row) for row in rows]

    def get_brief(self, brief_id: int) -> dict[str, Any]:
        """Load one brief by id.

        Args:
            brief_id: Brief primary key.

        Returns:
            Brief payload with metadata.

        Raises:
            LookupError: If the brief does not exist.
        """
        row = self.conn.execute(
            """
            SELECT
                b.id,
                b.title,
                b.query_text,
                b.sources_json,
                b.max_results,
                b.created_at,
                b.updated_at,
                b.last_run_at,
                COUNT(bp.id) AS paper_count
            FROM briefs AS b
            LEFT JOIN brief_papers AS bp
              ON bp.brief_id = b.id
            WHERE b.id = ?
            GROUP BY b.id
            """,
            (brief_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"Brief not found: {brief_id}")
        return _brief_payload_from_row(row)

    def find_brief_by_query(self, query_text: str) -> dict[str, Any] | None:
        """Find the most recent brief with the same query text.

        Args:
            query_text: Original search query text.

        Returns:
            Matching brief payload, or None when no brief exists.
        """
        row = self.conn.execute(
            """
            SELECT
                b.id,
                b.title,
                b.query_text,
                b.sources_json,
                b.max_results,
                b.created_at,
                b.updated_at,
                b.last_run_at,
                COUNT(bp.id) AS paper_count
            FROM briefs AS b
            LEFT JOIN brief_papers AS bp
              ON bp.brief_id = b.id
            WHERE lower(trim(b.query_text)) = lower(trim(?))
            GROUP BY b.id
            ORDER BY b.updated_at DESC, b.id DESC
            LIMIT 1
            """,
            (query_text,),
        ).fetchone()
        return _brief_payload_from_row(row) if row is not None else None

    def touch_brief(self, brief_id: int, *, rerun: bool = False) -> None:
        """Update brief timestamps.

        Args:
            brief_id: Brief primary key.
            rerun: Whether to also refresh last_run_at.
        """
        if rerun:
            self.conn.execute(
                """
                UPDATE briefs
                SET updated_at = CAST(strftime('%s','now') AS INTEGER),
                    last_run_at = CAST(strftime('%s','now') AS INTEGER)
                WHERE id = ?
                """,
                (brief_id,),
            )
        else:
            self.conn.execute(
                """
                UPDATE briefs
                SET updated_at = CAST(strftime('%s','now') AS INTEGER)
                WHERE id = ?
                """,
                (brief_id,),
            )
        self.conn.commit()

    def add_papers_to_brief(self, brief_id: int, papers: Sequence[Paper]) -> int:
        """Append papers to a brief, ignoring existing rows.

        Args:
            brief_id: Brief primary key.
            papers: Papers to add.

        Returns:
            Number of newly inserted brief rows.
        """
        inserted = 0
        for paper in papers:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO brief_papers (brief_id, source, source_id)
                VALUES (?, ?, ?)
                """,
                (brief_id, paper.source, paper.id),
            )
            inserted += cursor.rowcount
        self.conn.execute(
            """
            UPDATE briefs
            SET updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (brief_id,),
        )
        self.conn.commit()
        return inserted

    def list_brief_papers(self, brief_id: int) -> list[dict[str, Any]]:
        """Return all papers currently stored in a brief.

        Args:
            brief_id: Brief primary key.

        Returns:
            Brief paper rows with latest stored metadata.
        """
        rows = self.conn.execute(
            """
            SELECT
                bp.brief_id,
                bp.source,
                bp.source_id,
                bp.added_at,
                pc.title,
                pc.authors,
                pc.abstract,
                pc.abstract_url,
                pc.pdf_url,
                pc.doi,
                pa.download_status,
                pa.parse_status,
                pa.index_status
            FROM brief_papers AS bp
            LEFT JOIN paper_assets AS pa
              ON pa.source = bp.source AND pa.source_id = bp.source_id
            LEFT JOIN paper_content AS pc
              ON pc.id = (
                SELECT id
                FROM paper_content
                WHERE source = bp.source AND source_id = bp.source_id
                ORDER BY fetched_at DESC, id DESC
                LIMIT 1
              )
            WHERE bp.brief_id = ?
            ORDER BY bp.added_at DESC, bp.id DESC
            """,
            (brief_id,),
        ).fetchall()
        return [
            {
                "brief_id": row[0],
                "source": row[1],
                "source_id": row[2],
                "added_at": row[3],
                "title": row[4] or row[2],
                "authors": row[5] or "[]",
                "abstract": row[6] or "",
                "abstract_url": row[7],
                "pdf_url": row[8],
                "doi": row[9],
                "download_status": row[10] or "queued",
                "parse_status": row[11] or "queued",
                "index_status": row[12] or "queued",
            }
            for row in rows
        ]

    def remove_brief_paper(self, brief_id: int, source: str, source_id: str) -> None:
        """Delete one paper from a brief.

        Args:
            brief_id: Brief primary key.
            source: Source identifier.
            source_id: Source-level paper identifier.
        """
        self.conn.execute(
            """
            DELETE FROM brief_papers
            WHERE brief_id = ? AND source = ? AND source_id = ?
            """,
            (brief_id, source, source_id),
        )
        self.conn.execute(
            """
            UPDATE briefs
            SET updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (brief_id,),
        )
        self.conn.commit()

    def get_collection(self, collection_id: int) -> dict[str, Any]:
        """Load a collection by id.

        Args:
            collection_id: Collection primary key.

        Returns:
            Collection payload.

        Raises:
            LookupError: If the collection does not exist.
        """
        row = self.conn.execute(
            """
            SELECT id, name, description, created_at, updated_at
            FROM collections
            WHERE id = ?
            """,
            (collection_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"Collection not found: {collection_id}")
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def list_collections(self) -> list[dict[str, Any]]:
        """Return all collections ordered by last update time."""
        rows = self.conn.execute(
            """
            SELECT id, name, description, created_at, updated_at
            FROM collections
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }
            for row in rows
        ]

    def touch_collection(self, collection_id: int) -> None:
        """Update collection modified timestamp.

        Args:
            collection_id: Collection primary key.
        """
        self.conn.execute(
            """
            UPDATE collections
            SET updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (collection_id,),
        )
        self.conn.commit()

    def add_papers_to_collection(self, collection_id: int, papers: Sequence[Paper]) -> None:
        """Add papers to a collection and initialize asset rows.

        Args:
            collection_id: Collection primary key.
            papers: Papers to associate with the collection.
        """
        for paper in papers:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO collection_papers (
                    collection_id, source, source_id, status
                ) VALUES (?, ?, ?, 'queued')
                """,
                (collection_id, paper.source, paper.id),
            )
            self.conn.execute(
                """
                INSERT INTO paper_assets (
                    source, source_id, pdf_url, download_status, parse_status, index_status
                ) VALUES (?, ?, ?, 'queued', 'queued', 'queued')
                ON CONFLICT(source, source_id) DO UPDATE SET
                    pdf_url = COALESCE(excluded.pdf_url, paper_assets.pdf_url),
                    updated_at = CAST(strftime('%s','now') AS INTEGER)
                """,
                (paper.source, paper.id, paper.links.pdf),
            )
        self.conn.execute(
            """
            UPDATE collections
            SET updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (collection_id,),
        )
        self.conn.commit()

    def list_collection_papers(self, collection_id: int) -> list[dict[str, Any]]:
        """Return papers and ingestion status for a collection.

        Args:
            collection_id: Collection primary key.

        Returns:
            Collection paper rows with latest known metadata and asset state.
        """
        rows = self.conn.execute(
            """
            SELECT
                cp.collection_id,
                cp.source,
                cp.source_id,
                cp.status,
                pc.title,
                pc.authors,
                pc.abstract,
                pc.abstract_url,
                pc.pdf_url,
                pc.doi,
                pa.local_path,
                pa.download_status,
                pa.download_error,
                pa.parse_status,
                pa.parse_error,
                pa.index_status,
                pa.index_error
            FROM collection_papers AS cp
            LEFT JOIN paper_assets AS pa
              ON pa.source = cp.source AND pa.source_id = cp.source_id
            LEFT JOIN paper_content AS pc
              ON pc.id = (
                SELECT id
                FROM paper_content
                WHERE source = cp.source AND source_id = cp.source_id
                ORDER BY fetched_at DESC, id DESC
                LIMIT 1
              )
            WHERE cp.collection_id = ?
            ORDER BY cp.added_at DESC, cp.id DESC
            """,
            (collection_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "collection_id": row[0],
                    "source": row[1],
                    "source_id": row[2],
                    "status": row[3],
                    "title": row[4] or row[2],
                    "authors": row[5] or "[]",
                    "abstract": row[6] or "",
                    "abstract_url": row[7],
                    "pdf_url": row[8],
                    "doi": row[9],
                    "local_path": row[10],
                    "download_status": row[11] or "queued",
                    "download_error": row[12],
                    "parse_status": row[13] or "queued",
                    "parse_error": row[14],
                    "index_status": row[15] or "queued",
                    "index_error": row[16],
                }
            )
        return results

    def get_latest_paper_record(self, source: str, source_id: str) -> dict[str, Any] | None:
        """Return latest stored paper metadata for one paper.

        Args:
            source: Source identifier.
            source_id: Source-level paper identifier.

        Returns:
            Latest paper metadata row, or None if not found.
        """
        row = self.conn.execute(
            """
            SELECT title, authors, abstract, abstract_url, pdf_url, doi
            FROM paper_content
            WHERE source = ? AND source_id = ?
            ORDER BY fetched_at DESC, id DESC
            LIMIT 1
            """,
            (source, source_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "title": row[0],
            "authors": row[1],
            "abstract": row[2],
            "abstract_url": row[3],
            "pdf_url": row[4],
            "doi": row[5],
        }

    def update_collection_paper_status(
        self,
        collection_id: int,
        source: str,
        source_id: str,
        status: str,
    ) -> None:
        """Update aggregate ingestion status for one collection paper.

        Args:
            collection_id: Collection primary key.
            source: Source identifier.
            source_id: Source-level paper identifier.
            status: Aggregate status value.
        """
        self.conn.execute(
            """
            UPDATE collection_papers
            SET status = ?
            WHERE collection_id = ? AND source = ? AND source_id = ?
            """,
            (status, collection_id, source, source_id),
        )
        self.conn.execute(
            """
            UPDATE collections
            SET updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (collection_id,),
        )
        self.conn.commit()

    def upsert_asset_state(
        self,
        source: str,
        source_id: str,
        *,
        pdf_url: str | None = None,
        local_path: str | None = None,
        download_status: str | None = None,
        download_error: str | None = None,
        parse_status: str | None = None,
        parse_error: str | None = None,
        index_status: str | None = None,
        index_error: str | None = None,
    ) -> None:
        """Create or update asset state for one paper.

        Args:
            source: Source identifier.
            source_id: Source-level paper identifier.
            pdf_url: Optional PDF URL.
            local_path: Optional local PDF path.
            download_status: Optional download status.
            download_error: Optional download error.
            parse_status: Optional parse status.
            parse_error: Optional parse error.
            index_status: Optional index status.
            index_error: Optional index error.
        """
        existing = self.conn.execute(
            """
            SELECT id FROM paper_assets WHERE source = ? AND source_id = ?
            """,
            (source, source_id),
        ).fetchone()
        if existing is None:
            self.conn.execute(
                """
                INSERT INTO paper_assets (
                    source, source_id, pdf_url, local_path,
                    download_status, download_error,
                    parse_status, parse_error,
                    index_status, index_error,
                    downloaded_at, parsed_at, indexed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
                """,
                (
                    source,
                    source_id,
                    pdf_url,
                    local_path,
                    download_status or "queued",
                    download_error,
                    parse_status or "queued",
                    parse_error,
                    index_status or "queued",
                    index_error,
                    _status_timestamp(download_status),
                    _status_timestamp(parse_status),
                    _status_timestamp(index_status),
                ),
            )
        else:
            self.conn.execute(
                """
                UPDATE paper_assets
                SET
                    pdf_url = COALESCE(?, pdf_url),
                    local_path = COALESCE(?, local_path),
                    download_status = COALESCE(?, download_status),
                    download_error = ?,
                    parse_status = COALESCE(?, parse_status),
                    parse_error = ?,
                    index_status = COALESCE(?, index_status),
                    index_error = ?,
                    downloaded_at = COALESCE(?, downloaded_at),
                    parsed_at = COALESCE(?, parsed_at),
                    indexed_at = COALESCE(?, indexed_at),
                    updated_at = CAST(strftime('%s','now') AS INTEGER)
                WHERE source = ? AND source_id = ?
                """,
                (
                    pdf_url,
                    local_path,
                    download_status,
                    download_error,
                    parse_status,
                    parse_error,
                    index_status,
                    index_error,
                    _status_timestamp(download_status),
                    _status_timestamp(parse_status),
                    _status_timestamp(index_status),
                    source,
                    source_id,
                ),
            )
        self.conn.commit()

    def replace_paper_chunks(
        self,
        source: str,
        source_id: str,
        chunks: Sequence[ParsedChunk],
    ) -> None:
        """Replace all stored full-text chunks for one paper.

        Args:
            source: Source identifier.
            source_id: Source-level paper identifier.
            chunks: Parsed chunks to persist.
        """
        rows = self.conn.execute(
            """
            SELECT id FROM paper_chunks WHERE source = ? AND source_id = ?
            """,
            (source, source_id),
        ).fetchall()
        chunk_ids = [row[0] for row in rows]
        if chunk_ids:
            self.conn.executemany(
                "DELETE FROM paper_chunk_fts WHERE chunk_id = ?",
                [(str(chunk_id),) for chunk_id in chunk_ids],
            )
        self.conn.execute(
            """
            DELETE FROM paper_chunks WHERE source = ? AND source_id = ?
            """,
            (source, source_id),
        )
        for chunk in chunks:
            cursor = self.conn.execute(
                """
                INSERT INTO paper_chunks (
                    source, source_id, chunk_index, section_title,
                    page_start, page_end, content, token_count, modality, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    source_id,
                    chunk.chunk_index,
                    chunk.section_title,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.content,
                    chunk.token_count,
                    chunk.modality,
                    chunk.image_path,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO paper_chunk_fts (chunk_id, source, source_id, content)
                VALUES (?, ?, ?, ?)
                """,
                (str(cursor.lastrowid), source, source_id, chunk.content),
            )
        self.conn.commit()
        log.info("Replaced full-text chunks: source=%s id=%s count=%d", source, source_id, len(chunks))

    def list_paper_chunks_for_rag(self, source: str, source_id: str) -> list[dict[str, Any]]:
        """Return persisted chunks with DB ids and latest paper title for RAG indexing."""
        rows = self.conn.execute(
            """
            SELECT
                pc.id,
                pc.source,
                pc.source_id,
                pc.chunk_index,
                pc.section_title,
                pc.page_start,
                pc.page_end,
                pc.content,
                pc.token_count,
                pc.modality,
                pc.image_path,
                content.title
            FROM paper_chunks AS pc
            LEFT JOIN paper_content AS content
              ON content.id = (
                SELECT id FROM paper_content
                WHERE source = pc.source AND source_id = pc.source_id
                ORDER BY fetched_at DESC, id DESC
                LIMIT 1
              )
            WHERE pc.source = ? AND pc.source_id = ?
            ORDER BY pc.chunk_index ASC
            """,
            (source, source_id),
        ).fetchall()
        return [
            {
                "id": row[0],
                "source": row[1],
                "source_id": row[2],
                "chunk_index": row[3],
                "section_title": row[4],
                "page_start": row[5],
                "page_end": row[6],
                "content": row[7],
                "token_count": row[8],
                "modality": row[9] or "text",
                "image_path": row[10],
                "paper_title": row[11] or row[2],
            }
            for row in rows
        ]

    def search_collection_chunks(
        self,
        collection_id: int,
        query: str,
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Search stored chunks within one collection using FTS and fallback scan.

        Args:
            collection_id: Collection primary key.
            query: User question or search text.
            limit: Maximum number of chunks to return.

        Returns:
            Ranked chunk payloads.
        """
        normalized = _normalize_match_query(query)
        if normalized:
            rows = self.conn.execute(
                """
                SELECT
                    pc.id,
                    pc.source,
                    pc.source_id,
                    pc.chunk_index,
                    pc.section_title,
                    pc.page_start,
                    pc.page_end,
                    pc.content,
                    pc.modality,
                    pc.image_path,
                    bm25(paper_chunk_fts) AS score,
                    content.title
                FROM paper_chunk_fts AS pf
                JOIN paper_chunks AS pc
                  ON pc.id = CAST(pf.chunk_id AS INTEGER)
                JOIN collection_papers AS cp
                  ON cp.source = pc.source AND cp.source_id = pc.source_id
                LEFT JOIN paper_content AS content
                  ON content.id = (
                    SELECT id FROM paper_content
                    WHERE source = pc.source AND source_id = pc.source_id
                    ORDER BY fetched_at DESC, id DESC
                    LIMIT 1
                  )
                WHERE cp.collection_id = ?
                  AND paper_chunk_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (collection_id, normalized, limit),
            ).fetchall()
            if rows:
                return [_chunk_payload_from_row(row, use_bm25=True) for row in rows]

        rows = self.conn.execute(
            """
            SELECT
                pc.id,
                pc.source,
                pc.source_id,
                pc.chunk_index,
                pc.section_title,
                pc.page_start,
                pc.page_end,
                pc.content,
                pc.modality,
                pc.image_path,
                0.0 AS score,
                content.title
            FROM paper_chunks AS pc
            JOIN collection_papers AS cp
              ON cp.source = pc.source AND cp.source_id = pc.source_id
            LEFT JOIN paper_content AS content
              ON content.id = (
                SELECT id FROM paper_content
                WHERE source = pc.source AND source_id = pc.source_id
                ORDER BY fetched_at DESC, id DESC
                LIMIT 1
              )
            WHERE cp.collection_id = ?
            ORDER BY pc.id DESC
            LIMIT ?
            """,
            (collection_id, limit),
        ).fetchall()
        return [_chunk_payload_from_row(row, use_bm25=False) for row in rows]

    def create_conversation(self, collection_id: int, title: str) -> int:
        """Create a conversation for one collection.

        Args:
            collection_id: Collection primary key.
            title: Conversation title.

        Returns:
            Conversation id.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO conversations (collection_id, title)
            VALUES (?, ?)
            """,
            (collection_id, title.strip() or "Research conversation"),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        """Append a message to a conversation.

        Args:
            conversation_id: Conversation primary key.
            role: Message role such as user or assistant.
            content: Message body.

        Returns:
            Message id.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        self.conn.execute(
            """
            UPDATE conversations
            SET updated_at = CAST(strftime('%s','now') AS INTEGER)
            WHERE id = ?
            """,
            (conversation_id,),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def add_citations(self, message_id: int, citations: Sequence[dict[str, Any]]) -> None:
        """Persist assistant citations for one answer.

        Args:
            message_id: Assistant message id.
            citations: Citation payloads with chunk ids and scores.
        """
        for idx, citation in enumerate(citations, start=1):
            self.conn.execute(
                """
                INSERT INTO answer_citations (
                    message_id, paper_chunk_id, rank_order, score, quote_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    citation["chunk_id"],
                    idx,
                    citation.get("score", 0.0),
                    citation.get("quote_text", ""),
                ),
            )
        self.conn.commit()

    def get_collection_conversations(self, collection_id: int) -> list[dict[str, Any]]:
        """List conversations for one collection."""
        rows = self.conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE collection_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (collection_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

    def get_conversation_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        """Return messages for one conversation."""
        conversation_row = self.conn.execute(
            "SELECT collection_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        collection_id = int(conversation_row[0]) if conversation_row else None
        rows = self.conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id,),
        ).fetchall()
        messages = [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3], "citations": []}
            for row in rows
        ]
        message_ids = [message["id"] for message in messages if message["role"] == "assistant"]
        if not message_ids:
            return messages

        placeholders = ",".join("?" for _ in message_ids)
        citation_rows = self.conn.execute(
            f"""
            SELECT
                ac.message_id,
                ac.rank_order,
                ac.score,
                ac.quote_text,
                pc.page_start,
                pc.page_end,
                pc.section_title,
                pc.modality,
                pc.image_path,
                content.title,
                pc.source,
                pc.source_id
            FROM answer_citations AS ac
            LEFT JOIN paper_chunks AS pc
              ON pc.id = ac.paper_chunk_id
            LEFT JOIN paper_content AS content
              ON content.id = (
                SELECT id
                FROM paper_content
                WHERE source = pc.source AND source_id = pc.source_id
                ORDER BY fetched_at DESC, id DESC
                LIMIT 1
              )
            WHERE ac.message_id IN ({placeholders})
            ORDER BY ac.message_id ASC, ac.rank_order ASC
            """,
            message_ids,
        ).fetchall()
        citations_by_message: dict[int, list[dict[str, Any]]] = {}
        for row in citation_rows:
            citations_by_message.setdefault(row[0], []).append(
                {
                    "rank": row[1],
                    "score": row[2],
                    "quote_text": row[3] or "",
                    "page_start": row[4],
                    "page_end": row[5],
                    "section_title": row[6],
                    "modality": row[7] or "text",
                    "image_path": row[8],
                    "image_url": _figure_image_url(collection_id, row[8]) if row[8] else None,
                    "paper_title": row[9] or row[11] or "Untitled",
                    "source": row[10],
                    "source_id": row[11],
                }
            )
        for message in messages:
            message["citations"] = citations_by_message.get(message["id"], [])
        return messages


def ensure_paper_directory(base_dir: Path, source: str) -> Path:
    """Ensure local PDF directory exists for one source.

    Args:
        base_dir: Root directory for stored PDFs.
        source: Source identifier.

    Returns:
        Source-specific directory path.
    """
    path = base_dir / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_timestamp(status: str | None) -> int | None:
    """Return SQL current timestamp marker when a status indicates completion."""
    if status != "completed":
        return None
    return int(time.time())


def _normalize_match_query(query: str) -> str:
    """Normalize free text into a conservative FTS query string."""
    terms = []
    for raw in query.replace('"', " ").split():
        token = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"_", "-"})
        if len(token) < 2:
            continue
        if token:
            terms.append(f'"{token}"')
    return " OR ".join(terms[:8])


def _chunk_payload_from_row(row: Sequence[Any], *, use_bm25: bool) -> dict[str, Any]:
    """Convert one chunk query row into API payload."""
    score = float(row[10])
    return {
        "chunk_id": row[0],
        "source": row[1],
        "source_id": row[2],
        "chunk_index": row[3],
        "section_title": row[4],
        "page_start": row[5],
        "page_end": row[6],
        "content": row[7],
        "modality": row[8] or "text",
        "image_path": row[9],
        "score": -score if use_bm25 else score,
        "paper_title": row[11] or row[2],
    }


def _figure_image_url(collection_id: int | None, image_path: str | None) -> str | None:
    if collection_id is None or not image_path:
        return None
    return f"/api/collections/{collection_id}/figures/{Path(image_path).name}"


def _brief_payload_from_row(row: Sequence[Any]) -> dict[str, Any]:
    """Convert one brief row into API payload."""
    sources = []
    raw_sources = row[3] or "[]"
    try:
        sources = json.loads(raw_sources)
    except json.JSONDecodeError:
        sources = []
    return {
        "id": row[0],
        "title": row[1],
        "query_text": row[2],
        "sources": sources,
        "max_results": row[4],
        "created_at": row[5],
        "updated_at": row[6],
        "last_run_at": row[7],
        "paper_count": row[8],
    }
