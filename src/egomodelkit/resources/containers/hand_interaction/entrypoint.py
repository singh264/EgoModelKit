"""Extract Bandini-configured frames for the standalone hand-interaction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

SUPPORTED_VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4"})
DEFAULT_SESSION_ID: Final[str] = "session001"
INPUT_MANIFEST_FILENAME: Final[str] = "hand_interaction_input_manifest.csv"
SUBCLIP_MANIFEST_FILENAME: Final[str] = "hand_interaction_subclip_manifest.csv"
METRICS_CONFIG_FILENAME: Final[str] = "metrics_config.json"
PROGRESS_PREFIX: Final[str] = "EGOMODELKIT_PROGRESS "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir-name", required=True)
    parser.add_argument("--subclip-length", type=int, required=True)
    parser.add_argument("--processing-fps", type=int, required=True)
    parser.add_argument("--resize-width", type=int, required=True)
    parser.add_argument("--resize-height", type=int, required=True)
    parser.add_argument("--pooling-window-seconds", type=float, required=True)
    parser.add_argument("--interaction-contact-state-threshold", type=int, required=True)
    parser.add_argument("--dominant-hand", choices=["left", "right"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    work_dir = output_dir / args.work_dir_name
    extracted_frames_dir = work_dir / "extracted_frames"
    temporary_frames_dir = work_dir / "temporary_frames"

    if work_dir.exists():
        shutil.rmtree(work_dir)

    extracted_frames_dir.mkdir(parents=True, exist_ok=True)
    temporary_frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _supported_videos(input_path)
    if not videos:
        raise RuntimeError("No supported MP4 video files were found for hand interaction.")

    input_rows: list[dict[str, object]] = []
    subclip_rows: list[dict[str, object]] = []
    total_extracted_frames = 0

    for video_index, video_path in enumerate(videos, start=1):
        staged_video_name = f"video{video_index:03d}.MP4"
        staged_video_stem = Path(staged_video_name).stem
        metadata = _probe_source_video_metadata(video_path)

        _emit_progress(
            "hand_interaction_video_checked",
            current=video_index,
            total=len(videos),
        )

        video_frames_dir = temporary_frames_dir / staged_video_stem
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        _extract_frames(
            video_path=video_path,
            output_pattern=video_frames_dir / "frame_%06d.jpg",
            processing_fps=args.processing_fps,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
        )

        frame_paths = sorted(video_frames_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"Frame extraction produced no frames for: {video_path.name}")

        source_duration = float(metadata["source_duration_seconds"])
        if source_duration <= 0:
            source_duration = len(frame_paths) / args.processing_fps
            metadata["source_duration_seconds"] = source_duration

        input_rows.append(
            {
                "session_id": DEFAULT_SESSION_ID,
                "session_sort_index": video_index,
                "input_name": video_path.name,
                "staged_video_name": staged_video_name,
                "staged_video_stem": staged_video_stem,
                "input_modified_time": _input_modified_time(video_path),
                **metadata,
            }
        )

        frames_per_subclip = args.subclip_length * args.processing_fps

        for subclip_index, start in enumerate(
            range(0, len(frame_paths), frames_per_subclip),
            start=1,
        ):
            subclip_name = f"{staged_video_stem}--{subclip_index}"
            subclip_dir = extracted_frames_dir / subclip_name
            subclip_dir.mkdir(parents=True, exist_ok=True)
            subclip_frames = frame_paths[start : start + frames_per_subclip]

            for frame_path in subclip_frames:
                shutil.move(str(frame_path), subclip_dir / frame_path.name)

            source_start = (subclip_index - 1) * args.subclip_length
            processed_duration = len(subclip_frames) / args.processing_fps
            valid_duration = min(
                processed_duration,
                max(0.0, source_duration - source_start),
            )

            subclip_rows.append(
                {
                    "session_id": DEFAULT_SESSION_ID,
                    "input_name": video_path.name,
                    "staged_video_stem": staged_video_stem,
                    "subclip_name": subclip_name,
                    "subclip_index": subclip_index,
                    "source_start_seconds": source_start,
                    "source_end_seconds": source_start + valid_duration,
                    "valid_duration_seconds": valid_duration,
                    "processing_fps": args.processing_fps,
                    "processing_subclip_duration_seconds": processed_duration,
                }
            )

        total_extracted_frames += len(frame_paths)
        _emit_progress(
            "hand_interaction_frame_extracted",
            current=total_extracted_frames,
            total=total_extracted_frames,
        )

    _write_csv(output_dir / INPUT_MANIFEST_FILENAME, input_rows)
    _write_csv(output_dir / SUBCLIP_MANIFEST_FILENAME, subclip_rows)
    _write_metrics_config(
        output_dir / METRICS_CONFIG_FILENAME,
        subclip_length_seconds=args.subclip_length,
        processing_fps=args.processing_fps,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        pooling_window_seconds=args.pooling_window_seconds,
        interaction_contact_state_threshold=args.interaction_contact_state_threshold,
        dominant_hand=args.dominant_hand,
    )
    shutil.rmtree(temporary_frames_dir, ignore_errors=True)
    print("EgoModelKit runtime: hand-interaction frames are ready.", flush=True)


def _supported_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(input_path.iterdir(), key=lambda path: _natural_sort_key(path.name))

    return [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    ]


def _probe_source_video_metadata(video_path: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    source_fps = _parse_rate(stream.get("avg_frame_rate"))
    source_total_frames = _parse_int(stream.get("nb_frames"))
    source_duration = _parse_float(stream.get("duration"))

    if source_duration <= 0:
        source_duration = _parse_float((payload.get("format") or {}).get("duration"))

    if source_total_frames <= 0 and source_duration > 0 and source_fps > 0:
        source_total_frames = round(source_duration * source_fps)

    if source_duration <= 0 and source_total_frames > 0 and source_fps > 0:
        source_duration = source_total_frames / source_fps

    return {
        "source_duration_seconds": source_duration,
        "source_fps": source_fps,
        "source_total_frames": source_total_frames,
    }


def _extract_frames(
    *,
    video_path: Path,
    output_pattern: Path,
    processing_fps: int,
    resize_width: int,
    resize_height: int,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={processing_fps},scale={resize_width}:{resize_height}:flags=lanczos",
            "-q:v",
            "2",
            str(output_pattern),
        ],
        check=True,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write an empty manifest: {path.name}")

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_config(
    path: Path,
    *,
    subclip_length_seconds: int,
    processing_fps: int,
    resize_width: int,
    resize_height: int,
    pooling_window_seconds: float,
    interaction_contact_state_threshold: int,
    dominant_hand: str,
) -> None:
    payload = {
        "subclip_length_seconds": subclip_length_seconds,
        "subclip_fps": processing_fps,
        "frame_fps": processing_fps,
        "processing_fps": processing_fps,
        "resize_width": resize_width,
        "resize_height": resize_height,
        "pooling_window_seconds": pooling_window_seconds,
        "pooling_window_frames": max(1, round(processing_fps * pooling_window_seconds)),
        "interaction_contact_state_threshold": interaction_contact_state_threshold,
        "dominant_hand": dominant_hand,
        "non_dominant_hand": "left" if dominant_hand == "right" else "right",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_modified_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _parse_rate(value: object) -> float:
    text = str(value or "0")
    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)
        numerator = _parse_float(numerator_text)
        denominator = _parse_float(denominator_text)
        return numerator / denominator if denominator else 0.0
    return _parse_float(text)


def _parse_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _parse_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _natural_sort_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _emit_progress(kind: str, **payload: object) -> None:
    print(
        PROGRESS_PREFIX + json.dumps({"kind": kind, **payload}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
