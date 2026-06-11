from __future__ import annotations

from pathlib import Path

from egomodelkit.progress import ProgressEvent, read_progress_events, write_progress_event


def test_progress_event_display_text_with_counts() -> None:
    event = ProgressEvent(
        stage = "runtime",
        message = "Extracting frames",
        current = 240,
        total = 1200,
        unit = "frames",
    )

    assert event.display_text == "Extracting frames: 240 / 1200 frames"

def test_progress_event_display_text_without_counts_or_unit() -> None:
    assert (
        ProgressEvent(stage = "setup", message = "Preparing input").display_text ==
        "Preparing input"
    )
    
    assert (
        ProgressEvent(
            stage = "runtime",
            message = "Frames",
            current = 1,
            total = 2).display_text ==
        "Frames: 1 / 2"
    )

def test_progress_log_round_trip_skips_malformed_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "progress.jsonl"
    
    assert read_progress_events(log_path) == []
    
    write_progress_event(
        log_path,
        ProgressEvent(
            stage = "runtime", 
            message = "Extracting", 
            current = 1, 
            total = 3, 
            unit = "frames"
        )
    )
    
    with log_path.open("a", encoding = "utf-8") as stream:
        stream.write("not-json\n")
        stream.write('{"stage": "bad"}\n')
    
    events = read_progress_events(log_path)
    
    assert (
        events == 
        [
            ProgressEvent(
                stage = "runtime", 
                message = "Extracting", 
                current = 1, 
                total = 3, 
                unit = "frames"
            ),
        ]
    )
