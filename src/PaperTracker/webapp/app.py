"""ResearchMind FastAPI application.

Exposes web APIs and a minimal browser UI for search, collection ingest, and QA.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from PaperTracker.config import AppConfig
from PaperTracker.core.models import Paper
from PaperTracker.core.query import FieldQuery, SearchQuery
from PaperTracker.llm.client import LLMApiClient
from PaperTracker.services import create_search_service
from PaperTracker.services.ingestion import IngestionService
from PaperTracker.services.qa import CollectionQAService
from PaperTracker.services.rag import CollectionRAGService
from PaperTracker.storage import create_storage
from PaperTracker.storage.research import ResearchStore


@dataclass(slots=True)
class WebAppState:
    """Long-lived shared state for the FastAPI application."""

    config: AppConfig
    search_service: Any
    research_store: ResearchStore
    ingestion_service: IngestionService
    qa_service: CollectionQAService
    dedup_store: Any
    content_store: Any


class BriefSearchRequest(BaseModel):
    """Brief search creation request model."""

    query: str = Field(min_length=1)
    max_results: int | None = Field(default=None, ge=1, le=50)


class CreateCollectionRequest(BaseModel):
    """Collection creation request model."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)


class RenameBriefRequest(BaseModel):
    """Brief rename request model."""

    title: str = Field(min_length=1, max_length=160)


class RerunBriefRequest(BaseModel):
    """Brief rerun request model."""

    max_results: int | None = Field(default=None, ge=1, le=50)


class UpdateCollectionRequest(BaseModel):
    """Collection update request model."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)


class PaperPayload(BaseModel):
    """Serialized paper payload used by collection endpoints."""

    source: str
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    abstract_url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    published_ts: int | None = None
    updated_ts: int | None = None


class AddPapersRequest(BaseModel):
    """Batch add papers request model."""

    papers: list[PaperPayload] = Field(default_factory=list)
    ingest_immediately: bool = True


class AddBriefPapersToCollectionRequest(BaseModel):
    """Add selected brief papers into one collection."""

    collection_id: int
    selected_papers: list[dict[str, str]] = Field(default_factory=list)
    ingest_immediately: bool = True


class AskRequest(BaseModel):
    """Question answering request model."""

    collection_id: int
    question: str = Field(min_length=1)
    conversation_id: int | None = None
    mode: str | None = None


def create_app(config: AppConfig) -> FastAPI:
    """Create the ResearchMind FastAPI application.

    Args:
        config: Loaded PaperTracker application configuration.

    Returns:
        Configured FastAPI app.
    """
    db_manager, dedup_store, content_store = create_storage(config)
    if db_manager is None or dedup_store is None or content_store is None:
        raise RuntimeError("Web app requires storage.enabled=true and content_storage_enabled=true")

    search_service = create_search_service(config, dedup_store=dedup_store)
    research_store = ResearchStore(db_manager.get_connection())
    storage_dir = Path(config.storage.db_path).resolve().parent
    papers_dir = storage_dir / "papers"
    rag_service = CollectionRAGService(
        root=storage_dir / "rag",
        config=config.rag,
        store=research_store,
    )

    llm_client = None
    llm_model = None
    if config.llm.enabled:
        llm_client = LLMApiClient(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            retry_base_delay=config.llm.retry_base_delay,
            retry_max_delay=config.llm.retry_max_delay,
            timeout_multiplier=config.llm.retry_timeout_multiplier,
        )
        llm_model = config.llm.model

    ingestion_service = IngestionService(
        store=research_store,
        papers_dir=papers_dir,
        rag_service=rag_service,
        llm_client=llm_client,
        llm_model=llm_model,
    )

    qa_service = CollectionQAService(
        store=research_store,
        llm_client=llm_client,
        llm_model=llm_model,
        rag_service=rag_service,
        max_chunks=config.rag.top_k,
    )

    state = WebAppState(
        config=config,
        search_service=search_service,
        research_store=research_store,
        ingestion_service=ingestion_service,
        qa_service=qa_service,
        dedup_store=dedup_store,
        content_store=content_store,
    )

    app = FastAPI(title="ResearchMind", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.researchmind = state

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        close_func = getattr(state.search_service, "close", None)
        if callable(close_func):
            close_func()
        if db_manager:
            db_manager.close()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        return {
            "briefs": state.research_store.list_briefs(),
            "collections": state.research_store.list_collections(),
            "search_defaults": {
                "max_results": state.config.search.max_results,
                "sources": list(state.config.search.sources),
            },
            "llm_enabled": bool(llm_client and llm_model),
            "rag": {
                "default_mode": state.config.rag.default_mode,
                "debug_enabled": state.config.rag.debug_enabled,
                "decompose_enabled": state.config.rag.decompose_enabled,
                "agent_enabled": state.config.rag.agent_enabled,
                "multimodal_enabled": state.config.rag.multimodal_enabled,
                "vl_caption_enabled": state.config.rag.vl_caption_enabled,
                "image_vector_enabled": state.config.rag.image_vector_enabled,
            },
        }

    @app.get("/api/briefs")
    def list_briefs() -> dict[str, Any]:
        return {"briefs": state.research_store.list_briefs()}

    @app.patch("/api/briefs/{brief_id}")
    def rename_brief(brief_id: int, request: RenameBriefRequest) -> dict[str, Any]:
        try:
            brief = state.research_store.update_brief(brief_id, title=request.title)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"brief": brief}

    @app.delete("/api/briefs/{brief_id}")
    def delete_brief(brief_id: int) -> dict[str, Any]:
        try:
            state.research_store.get_brief(brief_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        state.research_store.delete_brief(brief_id)
        return {"ok": True}

    @app.post("/api/briefs/search")
    def create_brief_from_search(request: BriefSearchRequest) -> dict[str, Any]:
        max_results = request.max_results or state.config.search.max_results
        existing = state.research_store.find_brief_by_query(request.query)
        if existing:
            brief = existing
            existing_brief = True
        else:
            brief = state.research_store.create_brief(
                title=request.query.strip(),
                query_text=request.query.strip(),
                sources=state.config.search.sources,
                max_results=max_results,
            )
            existing_brief = False
        warning = None
        try:
            brief, papers, inserted = _run_brief_search(
                state,
                query_text=request.query,
                max_results=max_results,
                brief_id=brief["id"],
            )
        except Exception as error:  # noqa: BLE001 - brief should still exist even if sources fail
            warning = _humanize_search_error(error)
            papers = []
            inserted = 0
            brief = state.research_store.get_brief(brief["id"])
        return {
            "brief": brief,
            "warning": warning,
            "existing_brief": existing_brief,
            "inserted_count": inserted,
            "papers": [_normalize_brief_paper(row) for row in state.research_store.list_brief_papers(brief["id"])],
            "search_results": [_serialize_paper(paper) for paper in papers],
        }

    @app.get("/api/briefs/{brief_id}")
    def get_brief(brief_id: int) -> dict[str, Any]:
        try:
            brief = state.research_store.get_brief(brief_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "brief": brief,
            "papers": [_normalize_brief_paper(row) for row in state.research_store.list_brief_papers(brief_id)],
        }

    @app.post("/api/briefs/{brief_id}/rerun")
    def rerun_brief(
        brief_id: int,
        request: RerunBriefRequest = Body(default_factory=RerunBriefRequest),
    ) -> dict[str, Any]:
        try:
            brief = state.research_store.get_brief(brief_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        max_results = request.max_results or brief["max_results"]
        if max_results != brief["max_results"]:
            brief = state.research_store.update_brief_max_results(brief_id, max_results)
        updated_brief, papers, inserted = _run_brief_search(
            state,
            query_text=brief["query_text"],
            max_results=max_results,
            brief_id=brief_id,
        )
        return {
            "brief": updated_brief,
            "inserted_count": inserted,
            "papers": [_normalize_brief_paper(row) for row in state.research_store.list_brief_papers(brief_id)],
            "search_results": [_serialize_paper(paper) for paper in papers],
        }

    @app.delete("/api/briefs/{brief_id}/papers/{source}/{source_id}")
    def delete_brief_paper(brief_id: int, source: str, source_id: str) -> dict[str, Any]:
        try:
            state.research_store.get_brief(brief_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        state.research_store.remove_brief_paper(brief_id, source, source_id)
        return {
            "ok": True,
            "brief": state.research_store.get_brief(brief_id),
            "papers": [_normalize_brief_paper(row) for row in state.research_store.list_brief_papers(brief_id)],
        }

    @app.post("/api/briefs/{brief_id}/add-to-collection")
    def add_brief_papers_to_collection(
        brief_id: int,
        request: AddBriefPapersToCollectionRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        try:
            state.research_store.get_brief(brief_id)
            state.research_store.get_collection(request.collection_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        papers = _load_selected_brief_papers(state, brief_id, request.selected_papers)
        if not papers:
            raise HTTPException(status_code=400, detail="No brief papers selected")

        _save_papers_for_workspace(state, papers)
        state.research_store.add_papers_to_collection(request.collection_id, papers)

        if request.ingest_immediately:
            for paper in papers:
                background_tasks.add_task(
                    state.ingestion_service.ingest_paper,
                    request.collection_id,
                    paper.source,
                    paper.id,
                )

        return {
            "ok": True,
            "count": len(papers),
            "collection": state.research_store.get_collection(request.collection_id),
            "papers": [
                _normalize_collection_paper(row)
                for row in state.research_store.list_collection_papers(request.collection_id)
            ],
        }

    @app.get("/api/collections")
    def list_collections() -> dict[str, Any]:
        return {"collections": state.research_store.list_collections()}

    @app.post("/api/collections")
    def create_collection(request: CreateCollectionRequest) -> dict[str, Any]:
        try:
            collection = state.research_store.create_collection(request.name, request.description)
        except Exception as error:  # noqa: BLE001 - constraint failures become HTTP errors
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"collection": collection}

    @app.patch("/api/collections/{collection_id}")
    def update_collection(collection_id: int, request: UpdateCollectionRequest) -> dict[str, Any]:
        try:
            collection = state.research_store.update_collection(
                collection_id,
                name=request.name,
                description=request.description,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"collection": collection}

    @app.delete("/api/collections/{collection_id}")
    def delete_collection(collection_id: int) -> dict[str, Any]:
        try:
            state.research_store.get_collection(collection_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        state.research_store.delete_collection(collection_id)
        return {"ok": True}

    @app.get("/api/collections/{collection_id}")
    def get_collection(collection_id: int) -> dict[str, Any]:
        try:
            collection = state.research_store.get_collection(collection_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        papers = state.research_store.list_collection_papers(collection_id)
        conversations = state.research_store.get_collection_conversations(collection_id)
        return {
            "collection": collection,
            "papers": [_normalize_collection_paper(row) for row in papers],
            "conversations": conversations,
        }

    @app.get("/api/collections/{collection_id}/figures/{file_name}")
    def get_collection_figure(collection_id: int, file_name: str) -> FileResponse:
        safe_name = Path(file_name).name
        figures_dir = state.qa_service.rag_service.workspace(collection_id).collection_dir / "figures"
        image_path = (figures_dir / safe_name).resolve()
        if figures_dir.resolve() not in image_path.parents or not image_path.exists():
            raise HTTPException(status_code=404, detail="Figure image not found")
        return FileResponse(image_path)

    @app.post("/api/collections/{collection_id}/papers")
    def add_papers(
        collection_id: int,
        request: AddPapersRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        try:
            state.research_store.get_collection(collection_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        papers = [_paper_from_payload(payload) for payload in request.papers]
        if not papers:
            raise HTTPException(status_code=400, detail="No papers provided")

        state.dedup_store.mark_seen(papers)
        state.content_store.save_papers(papers)
        state.research_store.add_papers_to_collection(collection_id, papers)

        if request.ingest_immediately:
            for paper in papers:
                background_tasks.add_task(
                    state.ingestion_service.ingest_paper,
                    collection_id,
                    paper.source,
                    paper.id,
                )

        return {
            "ok": True,
            "count": len(papers),
            "papers": state.research_store.list_collection_papers(collection_id),
        }

    @app.post("/api/collections/{collection_id}/ingest")
    def ingest_collection(collection_id: int, background_tasks: BackgroundTasks) -> dict[str, Any]:
        papers = state.research_store.list_collection_papers(collection_id)
        if not papers:
            raise HTTPException(status_code=404, detail="Collection has no papers")
        for paper in papers:
            background_tasks.add_task(
                state.ingestion_service.ingest_paper,
                collection_id,
                paper["source"],
                paper["source_id"],
            )
        return {"ok": True, "scheduled": len(papers)}

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        mode = (request.mode or state.config.rag.default_mode).strip().lower()
        if mode == "decompose" and not state.config.rag.decompose_enabled:
            raise HTTPException(status_code=400, detail="Decompose mode is disabled")
        if mode == "agent" and not state.config.rag.agent_enabled:
            raise HTTPException(status_code=400, detail="Agent mode is disabled")
        try:
            result = state.qa_service.ask(
                request.collection_id,
                request.question,
                conversation_id=request.conversation_id,
                mode=mode,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - keep API failures readable for the web UI
            raise HTTPException(status_code=500, detail=str(error)) from error
        retrieval_debug = result.retrieval_debug if state.config.rag.debug_enabled else None
        if retrieval_debug is not None:
            retrieval_debug = {
                **retrieval_debug,
                "ask_elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "llm_enabled": bool(state.qa_service.llm_client and state.qa_service.llm_model),
            }
        return {
            "conversation_id": result.conversation_id,
            "message_id": result.answer_message_id,
            "answer": result.answer,
            "citations": result.citations,
            "retrieval_debug": retrieval_debug,
        }

    @app.post("/api/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        started_at = time.perf_counter()
        mode = (request.mode or state.config.rag.default_mode).strip().lower()
        if mode == "decompose" and not state.config.rag.decompose_enabled:
            raise HTTPException(status_code=400, detail="Decompose mode is disabled")
        if mode == "agent" and not state.config.rag.agent_enabled:
            raise HTTPException(status_code=400, detail="Agent mode is disabled")

        def event_stream():
            try:
                for event in state.qa_service.ask_stream(
                    request.collection_id,
                    request.question,
                    conversation_id=request.conversation_id,
                    mode=mode,
                ):
                    if event.get("type") == "done":
                        retrieval_debug = event.get("retrieval_debug") if state.config.rag.debug_enabled else None
                        if retrieval_debug is not None:
                            retrieval_debug = {
                                **retrieval_debug,
                                "ask_elapsed_seconds": round(time.perf_counter() - started_at, 3),
                                "llm_enabled": bool(state.qa_service.llm_client and state.qa_service.llm_model),
                            }
                        event = {**event, "retrieval_debug": retrieval_debug}
                    yield _sse(event)
            except LookupError as error:
                yield _sse({"type": "error", "detail": str(error)})
            except Exception as error:  # noqa: BLE001 - stream errors must be delivered as events
                yield _sse({"type": "error", "detail": str(error)})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/rag/status")
    def rag_status() -> dict[str, Any]:
        return state.qa_service.rag_service.debug_status() if state.qa_service.rag_service else {}

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: int) -> dict[str, Any]:
        return {
            "messages": state.research_store.get_conversation_messages(conversation_id)
        }

    return app


def _serialize_paper(paper: Paper) -> dict[str, Any]:
    """Convert internal paper model to API payload."""
    return {
        "source": paper.source,
        "source_id": paper.id,
        "title": paper.title,
        "authors": list(paper.authors),
        "abstract": paper.abstract,
        "abstract_url": paper.links.abstract,
        "pdf_url": paper.links.pdf,
        "doi": paper.doi,
        "published_ts": int(paper.published.timestamp()) if paper.published else None,
        "updated_ts": int(paper.updated.timestamp()) if paper.updated else None,
        "extra": json.loads(json.dumps(dict(paper.extra), ensure_ascii=False)),
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _paper_from_payload(payload: PaperPayload) -> Paper:
    """Rebuild a Paper object from API payload."""
    from datetime import datetime, timezone

    from PaperTracker.core.models import PaperLinks

    published = (
        datetime.fromtimestamp(payload.published_ts, tz=timezone.utc)
        if payload.published_ts is not None
        else None
    )
    updated = (
        datetime.fromtimestamp(payload.updated_ts, tz=timezone.utc)
        if payload.updated_ts is not None
        else None
    )
    return Paper(
        source=payload.source,
        id=payload.source_id,
        title=payload.title,
        authors=tuple(payload.authors),
        abstract=payload.abstract,
        published=published,
        updated=updated,
        links=PaperLinks(abstract=payload.abstract_url, pdf=payload.pdf_url),
        doi=payload.doi,
        extra={},
    )


def _normalize_collection_paper(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize collection paper row for API output."""
    authors: Sequence[str] | str = row.get("authors") or "[]"
    if isinstance(authors, str):
        try:
            author_list = json.loads(authors)
        except json.JSONDecodeError:
            author_list = [authors]
    else:
        author_list = list(authors)
    return {
        **row,
        "authors": author_list,
    }


def _normalize_brief_paper(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize brief paper row for API output."""
    return _normalize_collection_paper(row)


def _run_brief_search(
    state: WebAppState,
    *,
    query_text: str,
    max_results: int,
    brief_id: int | None,
) -> tuple[dict[str, Any], list[Paper], int]:
    """Execute search and persist results into a new or existing brief."""
    query = SearchQuery(
        name=query_text,
        fields={"TEXT": FieldQuery(OR=(query_text.strip(),))},
    )
    papers = list(
        state.search_service.search(
            query,
            max_results=max_results,
        )
    )
    _save_papers_for_workspace(state, papers)
    inserted = state.research_store.add_papers_to_brief(brief_id, papers)
    state.research_store.touch_brief(brief_id, rerun=True)
    return state.research_store.get_brief(brief_id), papers, inserted


def _save_papers_for_workspace(state: WebAppState, papers: Sequence[Paper]) -> None:
    """Persist searched papers so briefs and collections can reuse metadata."""
    if not papers:
        return
    state.dedup_store.mark_seen(papers)
    state.content_store.save_papers(papers)


def _load_selected_brief_papers(
    state: WebAppState,
    brief_id: int,
    selected: Sequence[dict[str, str]],
) -> list[Paper]:
    """Load selected brief papers as canonical Paper objects."""
    selected_keys = {
        (item.get("source", "").strip(), item.get("source_id", "").strip())
        for item in selected
        if item.get("source") and item.get("source_id")
    }
    if not selected_keys:
        return []

    brief_rows = state.research_store.list_brief_papers(brief_id)
    selected_rows = [
        row for row in brief_rows
        if (row["source"], row["source_id"]) in selected_keys
    ]
    payloads = [
        PaperPayload(
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            authors=_normalize_authors(row.get("authors")),
            abstract=row.get("abstract", ""),
            abstract_url=row.get("abstract_url"),
            pdf_url=row.get("pdf_url"),
            doi=row.get("doi"),
        )
        for row in selected_rows
    ]
    return [_paper_from_payload(payload) for payload in payloads]


def _normalize_authors(authors: Any) -> list[str]:
    """Normalize serialized authors into a list."""
    if isinstance(authors, list):
        return [str(item) for item in authors]
    if isinstance(authors, str):
        try:
            data = json.loads(authors)
            if isinstance(data, list):
                return [str(item) for item in data]
        except json.JSONDecodeError:
            return [authors]
    return []


def _humanize_search_error(error: Exception) -> str:
    """Convert raw source errors into user-facing search warnings."""
    message = str(error)
    if "429" in message or "Rate exceeded" in message:
        return (
            "arXiv temporarily rate-limited this search (HTTP 429 / Rate exceeded). "
            "Wait a moment and try rerun search again."
        )
    return message
