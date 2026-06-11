""" Structured progress events for CLI and GUI consumers. """

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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
        """ Return a concise user-facing progress line. """
        if self.current is None or self.total is None:
            return self.message
    
        suffix = f" {self.unit}" if self.unit else ""
    
        return f"{self.message}: {self.current} / {self.total}{suffix}"
    
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
