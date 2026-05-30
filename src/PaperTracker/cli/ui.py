"""Click CLI interface definitions.

Defines the command-line interface structure and routes commands
to their respective runners.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
import uvicorn

from PaperTracker.cli.runner import CommandRunner
from PaperTracker.config import load_config_with_defaults
from PaperTracker.services.rag.downloader import download_rag_models
from PaperTracker.webapp import create_app


@click.group(help="PaperTracker: search papers and print in terminal.")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CLI entry group.

    Loads environment variables from .env file.

    Args:
        ctx: Click context.
    """
    # Load environment variables from the project directory used to start the CLI.
    load_dotenv(Path.cwd() / ".env")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@cli.command("search")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("config/example.yml"),
    show_default=True,
    help="Path to YAML config file (overrides defaults).",
)
@click.pass_context
def search_cmd(ctx: click.Context, config_path: Path) -> None:
    """Search papers and print to console via logging.

    All parameters are read from the YAML config.

    Args:
        ctx: Click context.
        config_path: Path to YAML config file.

    Raises:
        click.Abort: When the search fails.
    """
    cfg = load_config_with_defaults(config_path)
    runner = CommandRunner(cfg)
    runner.run_search(action=ctx.command.name)


@cli.command("serve")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("config/example.yml"),
    show_default=True,
    help="Path to YAML config file (overrides defaults).",
)
@click.pass_context
def serve_cmd(ctx: click.Context, config_path: Path) -> None:
    """Run the ResearchMind web server.

    Args:
        ctx: Click context.
        config_path: Path to YAML config file.
    """
    del ctx
    cfg = load_config_with_defaults(config_path)
    app = create_app(cfg)
    uvicorn.run(app, host="127.0.0.1", port=8000)


@cli.command("rag-download-models")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("config/example.yml"),
    show_default=True,
    help="Path to YAML config file (overrides defaults).",
)
def rag_download_models_cmd(config_path: Path) -> None:
    """Download configured RAG embedding and reranker models."""
    cfg = load_config_with_defaults(config_path)
    paths = download_rag_models(
        models_dir=cfg.rag.models_dir,
        embedding_model=cfg.rag.embedding_model,
        reranker_model=cfg.rag.reranker_model,
    )
    for path in paths:
        click.echo(f"Downloaded: {path}")
