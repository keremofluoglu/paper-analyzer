"""Incremental update: re-run every saved topic for papers submitted since its last run, then ingest."""
import sys
from datetime import datetime, timezone

from .fetch import fetch, load_topics
from .store import ingest


def update(n_per_topic: int = 200) -> dict:
    report = {}
    for query, info in load_topics().items():
        since = datetime.fromisoformat(info["last_run"]) if info.get("last_run") else None
        if since and since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        print(f"Updating topic {query!r} since {since}", file=sys.stderr)
        report[query] = fetch(query, n=n_per_topic, expand_citations=info.get("expand", True), since=since)
    report["_ingest"] = ingest()
    return report
