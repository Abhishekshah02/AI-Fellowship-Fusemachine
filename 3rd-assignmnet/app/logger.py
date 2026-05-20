import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

logger = logging.getLogger("text2sql")
if not logger.handlers:
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(stream)

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(file_handler)

# Structured event log (one JSON object per line) for benchmark analysis.
EVENTS_PATH = LOG_DIR / "events.jsonl"


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "event": event,
        **fields,
    }
    line = json.dumps(payload, default=str)
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.info(f"{event} | {fields}")
