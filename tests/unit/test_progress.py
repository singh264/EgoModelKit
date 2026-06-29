from __future__ import annotations

from pathlib import Path

from egomodelkit.progress import (
    ExternalProgressUpdate,
    ProgressEvent,
    parse_external_progress_line,
    read_progress_events,
    write_progress_event,
)


def test_progress_event_display_text_with_counts() -> None:
    event = ProgressEvent(
        stage = "runtime",
        message = "Extracting frames",
        current = 240,
        total = 1200,
        unit = "frames",
    )

    assert event.display_text == "Extracting frames: 240 / 1,200 frames"

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

def test_parse_external_progress_line_reads_model_payload() -> None:
    update = parse_external_progress_line(
        'EGOMODELKIT_PROGRESS {"kind": "detic_frame_processed", "current": 3, "total": 10}'
    )

    assert update == ExternalProgressUpdate(
        kind = "detic_frame_processed",
        payload = {"current": 3, "total": 10},
    )

def test_parse_external_progress_line_ignores_plain_or_malformed_lines() -> None:
    assert parse_external_progress_line("plain runtime log") is None
    assert parse_external_progress_line("EGOMODELKIT_PROGRESS not-json") is None
    assert parse_external_progress_line('EGOMODELKIT_PROGRESS {"current": 1}') is None

def test_progress_event_display_text_omits_unknown_zero_total() -> None:
    event = ProgressEvent(
        stage = "extract_frames",
        message = "Extracting frames: waiting",
        current = 0,
        total = 0,
        unit = "frames",
    )

    assert event.display_text == "Extracting frames: waiting"

def test_parse_external_progress_line_reads_payload_after_tqdm_prefix() -> None:
    update = parse_external_progress_line(
        '95%|█████████▌| 19/20 [00:23<00:01]'
        'EGOMODELKIT_PROGRESS {"kind": "detic_frame_processed", "current": 20, "total": 20}'
    )

    assert update == ExternalProgressUpdate(
        kind = "detic_frame_processed",
        payload = {"current": 20, "total": 20},
    )

def test_parse_external_progress_line_ignores_json_arrays() -> None:
    assert parse_external_progress_line(
        'EGOMODELKIT_PROGRESS ["not", "an", "object"]'
    ) is None
