""" Structured progress events for CLI and GUI consumers. """

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

EGOMODELKIT_PROGRESS_PREFIX: Final[str] = "EGOMODELKIT_PROGRESS "

@dataclass(frozen = True, slots = True)
class ProgressEvent:
    """ One model-run progress update. """
    stage: str
    message: str
    current: int | None = None
    total: int | None = None
    unit: str | None = None
    
    @property
    def display_text(self) -> str:
        """ Return the user-facing progress text. """
        if self.current is None or self.total is None or self.total <= 0:
            return self.message

        suffix = f" {self.unit}" if self.unit else ""

        return f"{self.message}: {self.current:,} / {self.total:,}{suffix}"
    
@dataclass(frozen = True, slots = True)
class ExternalProgressUpdate:
    """ Progress update emitted from an external model container. """
    kind: str
    payload: dict[str, object]

def parse_external_progress_line(line: str) -> ExternalProgressUpdate | None:
    """ Parse one structured progress line emitted inside a model container. """
    prefix_index = line.find(EGOMODELKIT_PROGRESS_PREFIX)

    if prefix_index < 0:
        return None

    raw_payload = line[prefix_index + len(EGOMODELKIT_PROGRESS_PREFIX):].strip()

    try:
        data, _ = json.JSONDecoder().raw_decode(raw_payload)
    except ValueError:
        return None

    if not isinstance(data, dict):
        return None

    kind = data.get("kind")

    if not isinstance(kind, str):
        return None

    payload = {
        key: value
        for key, value in data.items()
        if key != "kind"
    }

    return ExternalProgressUpdate(kind=kind, payload=payload)
    
def external_progress_line(kind: str, **payload: object) -> str:
    """ Return one structured progress line for external model containers. """
    return EGOMODELKIT_PROGRESS_PREFIX + json.dumps(
        {
            "kind": kind,
            **payload,
        },
        sort_keys = True,
    )

def write_runtime_log_line(path: Path, line: str) -> None:
    """ Append one raw runtime line to runtime.log. """
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("a", encoding = "utf-8") as stream:
        stream.write(line)
        stream.write("\n")    

def write_progress_event(path: Path, event: ProgressEvent) -> None:
    """ Append one JSONL progress event. """
    path.parent.mkdir(parents = True, exist_ok = True)
    
    with path.open("a", encoding = "utf-8") as stream:
        stream.write(json.dumps(asdict(event), sort_keys = True) + "\n")

def read_progress_events(path: Path) -> list[ProgressEvent]:
    """ Read progress events, skipping malformed lines. """
    if not path.exists():
        return []

    events: list[ProgressEvent] = []
    
    for line in path.read_text(encoding = "utf-8").splitlines():
        try:
            data = json.loads(line)
            events.append(ProgressEvent(**data))
        except (TypeError, ValueError):
            continue
        
    return events
