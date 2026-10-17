"""Run from repository root: python -m backend check | serve."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import uvicorn

from ai_service.forte_ai.pipeline import load_config

from .app import ROOT, create_application
from .catalogue import CATALOGUE_VERSION
from .errors import BackendError, CatalogueError
from .repository import DictionaryRegulationRepository


def check_configuration(config: dict) -> dict:
    """Check trusted citations against the team's configured IDs without inference."""
    repository = DictionaryRegulationRepository()
    entries = json.loads(Path(config["rules_path"]).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise CatalogueError("AI rule catalogue must be a list")
    ids = set()
    needs_review = []
    active_count = 0
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("enabled")) is not bool:
            raise CatalogueError("Each AI rule needs a boolean enabled flag")
        configured_source = entry.get("source")
        if not isinstance(configured_source, dict) or type(configured_source.get("verified")) is not bool:
            raise CatalogueError("Each AI rule needs explicit source verification")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise CatalogueError("Invalid or duplicate AI rule identifier")
        ids.add(identifier)
        if not entry["enabled"] or not configured_source["verified"]:
            continue
        active_count += 1
        rule = repository.find(identifier)
        source = rule.source() if rule else None
        if source is None:
            needs_review.append(identifier)
            continue
        if any(configured_source.get(field) != source[field] for field in ("url", "section", "verified")):
            raise CatalogueError("AI rule and backend source differ; review both catalogues")
    if active_count == 0:
        raise CatalogueError("The analyzer requires at least one enabled verified rule")
    return {
        "status": "ok", "catalogue_version": CATALOGUE_VERSION,
        "rules": len(repository.all()),
        "verified_sources": sum(rule.source_verified for rule in repository.all()),
        "ai_rules_requiring_review": needs_review,
        "model_inference_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ForteLens Python application backend")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "check"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=str(ROOT / "ai_service/config/local.json"))
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        check = check_configuration(config)
        if args.command == "check":
            print(json.dumps(check, ensure_ascii=False))
            return 0
        key = os.environ.get("FORTE_API_KEY", "")
        if len(key) < 16:
            print("Set FORTE_API_KEY to a random secret of at least 16 characters.", file=sys.stderr)
            return 2
        if not 1 <= args.port <= 65535:
            raise ValueError("Invalid listening port")
        app = create_application(config, host=args.host, port=args.port, api_key=key)
        # One process: in-memory job ownership and the single-GPU queue are shared.
        uvicorn.run(app, host=args.host, port=args.port, workers=1,
                    access_log=False, proxy_headers=False, server_header=False)
        return 0
    except (BackendError, OSError, ValueError, KeyError, TypeError, AttributeError):
        # Configuration can contain secrets and private paths; don't print its values.
        print(json.dumps({"status": "failed", "error": "backend_configuration_error",
                          "message": "Check local config, regulation mappings and free port."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
