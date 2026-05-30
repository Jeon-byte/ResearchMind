"""Tests for ResearchMind web storage and QA helpers."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PaperTracker.services.qa import CollectionQAService
from PaperTracker.services.rag import CollectionRAGService
from PaperTracker.core.models import Paper, PaperLinks
from PaperTracker.storage.migration import run_migrations
from PaperTracker.storage.research import ParsedChunk, ResearchStore


class TestResearchMindStore(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        run_migrations(self.conn)
        self.store = ResearchStore(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_create_collection_and_chunks_search(self) -> None:
        collection = self.store.create_collection("agents", "agent planning papers")
        self.conn.execute(
            """
            INSERT INTO seen_papers (id, source, source_id, title)
            VALUES (?, ?, ?, ?)
            """,
            (1, "arxiv", "1234.5678", "Planner Paper"),
        )
        self.conn.execute(
            """
            INSERT INTO paper_content (
                seen_paper_id, source, source_id, title, authors, abstract
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "arxiv", "1234.5678", "Planner Paper", "[]", "abstract"),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO collection_papers (collection_id, source, source_id, status)
            VALUES (?, ?, ?, 'queued')
            """,
            (collection["id"], "arxiv", "1234.5678"),
        )
        self.store.replace_paper_chunks(
            "arxiv",
            "1234.5678",
            [
                ParsedChunk(
                    chunk_index=0,
                    content="We introduce a planning agent with tool use and verification.",
                    page_start=1,
                    page_end=1,
                    token_count=10,
                )
            ],
        )

        rows = self.store.search_collection_chunks(collection["id"], "planning verification", limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["paper_title"], "Planner Paper")

    def test_brief_create_append_and_remove(self) -> None:
        brief = self.store.create_brief(
            title="agent planning",
            query_text="agent planning",
            sources=("arxiv", "openalex"),
            max_results=8,
        )
        paper = Paper(
            source="arxiv",
            id="2401.0001",
            title="Agent Planning",
            authors=("Alice",),
            abstract="abstract",
            published=None,
            updated=None,
            links=PaperLinks(abstract="https://x", pdf="https://pdf"),
        )
        self.conn.execute(
            "INSERT INTO seen_papers (id, source, source_id, title) VALUES (?, ?, ?, ?)",
            (1, "arxiv", "2401.0001", "Agent Planning"),
        )
        self.conn.execute(
            """
            INSERT INTO paper_content (
                seen_paper_id, source, source_id, title, authors, abstract, abstract_url, pdf_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "arxiv", "2401.0001", "Agent Planning", '["Alice"]', "abstract", "https://x", "https://pdf"),
        )

        inserted_first = self.store.add_papers_to_brief(brief["id"], (paper,))
        inserted_second = self.store.add_papers_to_brief(brief["id"], (paper,))
        rows_before = self.store.list_brief_papers(brief["id"])

        self.assertEqual(inserted_first, 1)
        self.assertEqual(inserted_second, 0)
        self.assertEqual(len(rows_before), 1)

        self.store.remove_brief_paper(brief["id"], "arxiv", "2401.0001")
        rows_after = self.store.list_brief_papers(brief["id"])

        self.assertEqual(rows_after, [])

    def test_find_brief_by_query(self) -> None:
        created = self.store.create_brief(
            title="VLM",
            query_text="vlm",
            sources=("arxiv",),
            max_results=5,
        )

        found = self.store.find_brief_by_query(" VLM ")

        self.assertIsNotNone(found)
        self.assertEqual(found["id"], created["id"])

    def test_update_brief_max_results(self) -> None:
        brief = self.store.create_brief(
            title="agent planning",
            query_text="agent planning",
            sources=("arxiv",),
            max_results=5,
        )

        updated = self.store.update_brief_max_results(brief["id"], 12)

        self.assertEqual(updated["max_results"], 12)

    def test_hyphenated_terms_do_not_break_chunk_search(self) -> None:
        collection = self.store.create_collection("reasoning", "")
        self.conn.execute(
            """
            INSERT OR IGNORE INTO collection_papers (collection_id, source, source_id, status)
            VALUES (?, ?, ?, 'indexed')
            """,
            (collection["id"], "arxiv", "2401.0002"),
        )
        self.store.replace_paper_chunks(
            "arxiv",
            "2401.0002",
            [
                ParsedChunk(
                    chunk_index=0,
                    content="The paper studies test-time scaling and self-correction.",
                    page_start=1,
                    page_end=1,
                    token_count=8,
                )
            ],
        )

        rows = self.store.search_collection_chunks(collection["id"], "test-time scaling", limit=5)

        self.assertEqual(len(rows), 1)

    def test_qa_service_returns_citations_without_llm(self) -> None:
        collection = self.store.create_collection("vision", "")
        self.conn.execute(
            """
            INSERT INTO seen_papers (id, source, source_id, title)
            VALUES (?, ?, ?, ?)
            """,
            (1, "arxiv", "9999.0001", "Vision Paper"),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO collection_papers (collection_id, source, source_id, status)
            VALUES (?, ?, ?, 'indexed')
            """,
            (collection["id"], "arxiv", "9999.0001"),
        )
        self.conn.execute(
            """
            INSERT INTO paper_content (
                seen_paper_id, source, source_id, title, authors, abstract
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "arxiv", "9999.0001", "Vision Paper", "[]", "abstract"),
        )
        self.store.replace_paper_chunks(
            "arxiv",
            "9999.0001",
            [
                ParsedChunk(
                    chunk_index=0,
                    content="The method uses a two-stage encoder and improves segmentation accuracy.",
                    page_start=2,
                    page_end=2,
                    token_count=11,
                )
            ],
        )

        service = CollectionQAService(store=self.store)
        result = service.ask(collection["id"], "What test-time method does the paper use?")

        self.assertTrue(result.answer)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0]["paper_title"], "Vision Paper")

    def test_rag_workspace_retrieves_collection_chunks(self) -> None:
        collection = self.store.create_collection("rag", "")
        self.conn.execute(
            """
            INSERT INTO seen_papers (id, source, source_id, title)
            VALUES (?, ?, ?, ?)
            """,
            (1, "arxiv", "2401.0003", "RAG Paper"),
        )
        self.conn.execute(
            """
            INSERT INTO paper_content (
                seen_paper_id, source, source_id, title, authors, abstract
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "arxiv", "2401.0003", "RAG Paper", "[]", "abstract"),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO collection_papers (collection_id, source, source_id, status)
            VALUES (?, ?, ?, 'indexed')
            """,
            (collection["id"], "arxiv", "2401.0003"),
        )
        self.store.replace_paper_chunks(
            "arxiv",
            "2401.0003",
            [
                ParsedChunk(
                    chunk_index=0,
                    content="Retrieval augmented generation uses retrieved evidence to ground answers.",
                    page_start=1,
                    page_end=1,
                    token_count=9,
                )
            ],
        )
        rows = self.store.list_paper_chunks_for_rag("arxiv", "2401.0003")
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = CollectionRAGService(root=Path(tmpdir), store=self.store)
            rag.index_paper(
                collection_id=collection["id"],
                source="arxiv",
                source_id="2401.0003",
                chunk_rows=rows,
            )
            hits = rag.retrieve(collection["id"], "retrieved evidence answers", top_k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.paper_title, "RAG Paper")

    def test_qa_service_prefers_rag_workspace(self) -> None:
        collection = self.store.create_collection("rag qa", "")
        self.conn.execute(
            "INSERT INTO seen_papers (id, source, source_id, title) VALUES (?, ?, ?, ?)",
            (1, "arxiv", "2401.0004", "Grounded QA"),
        )
        self.conn.execute(
            """
            INSERT INTO paper_content (
                seen_paper_id, source, source_id, title, authors, abstract
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "arxiv", "2401.0004", "Grounded QA", "[]", "abstract"),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO collection_papers (collection_id, source, source_id, status)
            VALUES (?, ?, ?, 'indexed')
            """,
            (collection["id"], "arxiv", "2401.0004"),
        )
        self.store.replace_paper_chunks(
            "arxiv",
            "2401.0004",
            [
                ParsedChunk(
                    chunk_index=0,
                    content="The method grounds generation with citations from retrieved chunks.",
                    page_start=3,
                    page_end=3,
                    token_count=9,
                )
            ],
        )
        rows = self.store.list_paper_chunks_for_rag("arxiv", "2401.0004")
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = CollectionRAGService(root=Path(tmpdir), store=self.store)
            rag.index_paper(
                collection_id=collection["id"],
                source="arxiv",
                source_id="2401.0004",
                chunk_rows=rows,
            )
            service = CollectionQAService(store=self.store, rag_service=rag)
            result = service.ask(collection["id"], "How does the method ground generation?")

        self.assertIn("retrieval-backed evidence digest", result.answer)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0]["paper_title"], "Grounded QA")
        self.assertEqual(result.retrieval_debug["source"], "rag")
        self.assertTrue(result.retrieval_debug["hits"])

    def test_rag_decompose_mode_returns_hits(self) -> None:
        collection = self.store.create_collection("rag decompose", "")
        self.conn.execute(
            "INSERT INTO seen_papers (id, source, source_id, title) VALUES (?, ?, ?, ?)",
            (1, "arxiv", "2401.0005", "Decompose QA"),
        )
        self.conn.execute(
            """
            INSERT INTO paper_content (
                seen_paper_id, source, source_id, title, authors, abstract
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "arxiv", "2401.0005", "Decompose QA", "[]", "abstract"),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO collection_papers (collection_id, source, source_id, status)
            VALUES (?, ?, ?, 'indexed')
            """,
            (collection["id"], "arxiv", "2401.0005"),
        )
        self.store.replace_paper_chunks(
            "arxiv",
            "2401.0005",
            [
                ParsedChunk(
                    chunk_index=0,
                    content="Dataset construction is described with careful filtering.",
                    page_start=2,
                    page_end=2,
                    token_count=7,
                ),
                ParsedChunk(
                    chunk_index=1,
                    content="Evaluation uses accuracy and recall metrics.",
                    page_start=5,
                    page_end=5,
                    token_count=6,
                ),
            ],
        )
        rows = self.store.list_paper_chunks_for_rag("arxiv", "2401.0005")
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = CollectionRAGService(root=Path(tmpdir), store=self.store)
            rag.index_paper(
                collection_id=collection["id"],
                source="arxiv",
                source_id="2401.0005",
                chunk_rows=rows,
            )
            hits = rag.retrieve(
                collection["id"],
                "How is the dataset constructed and how is evaluation done?",
                top_k=2,
                mode="decompose",
            )

        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("decompose", hits[0].retrieval_path)


if __name__ == "__main__":
    unittest.main()
