"""Run EgoVizML-compatible subclip extraction with per-frame progress."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Final

import cv2
from moviepy.editor import VideoFileClip

PROGRESS_PREFIX: Final[str] = "EGOMODELKIT_PROGRESS "
STAGED_VIDEO_SUFFIX: Final[str] = ".MP4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split staged ADL videos into EgoVizML subclips and frames."
    )
    parser.add_argument("directory")
    parser.add_argument("--subclip-length", type=int, required=True)
    parser.add_argument("--subclip-fps", type=int, required=True)
    parser.add_argument("--frame-fps", type=int, required=True)
    parser.add_argument("--progress-total", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_videos(
        directory=Path(args.directory),
        subclip_length=args.subclip_length,
        subclip_fps=args.subclip_fps,
        frame_fps=args.frame_fps,
        progress_total=args.progress_total,
    )


def extract_videos(
    *,
    directory: Path,
    subclip_length: int,
    subclip_fps: int,
    frame_fps: int,
    progress_total: int,
) -> int:
    current = 0
    staged_videos = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == STAGED_VIDEO_SUFFIX
    )

    subclips_dir = directory / "subclips"
    subclips_dir.mkdir(parents=True, exist_ok=True)

    for video_path in staged_videos:
        clip = VideoFileClip(str(video_path))
        try:
            duration_seconds = int(math.floor(clip.duration))
            for segment_index, start_seconds in enumerate(
                range(0, duration_seconds, subclip_length),
                start=1,
            ):
                end_seconds = start_seconds + subclip_length
                subclip = clip.subclip(start_seconds, end_seconds)
                subclip_path = (
                    subclips_dir / f"{video_path.stem}--{segment_index}.MP4"
                )
                try:
                    subclip.write_videofile(
                        str(subclip_path),
                        fps=subclip_fps,
                        audio=False,
                    )
                finally:
                    subclip.close()

                frame_output_dir = subclip_path.with_suffix("")
                frame_output_dir.mkdir(parents=True, exist_ok=True)
                current = _extract_subclip_frames(
                    video_path=subclip_path,
                    output_dir=frame_output_dir,
                    frame_fps=frame_fps,
                    current=current,
                    progress_total=progress_total,
                )
        finally:
            clip.close()

    if current <= 0:
        raise RuntimeError("EgoVizML frame extraction produced no inference frames.")

    if current != progress_total:
        _emit_progress(
            "adl_frame_extracted",
            current=current,
            total=current,
        )

    return current


def _extract_subclip_frames(
    *,
    video_path: Path,
    output_dir: Path,
    frame_fps: int,
    current: int,
    progress_total: int,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    source_fps = int(capture.get(cv2.CAP_PROP_FPS))
    downsample = source_fps // frame_fps
    if downsample <= 0:
        capture.release()
        raise RuntimeError(
            f"Configured frame FPS {frame_fps} is incompatible with {video_path}."
        )

    source_frame_index = 0
    try:
        while capture.isOpened():
            available, frame = capture.read()
            if not available:
                break

            if source_frame_index % downsample == 0:
                frame_path = output_dir / f"frame_{source_frame_index}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write extracted frame: {frame_path}")
                current += 1
                _emit_progress(
                    "adl_frame_extracted",
                    current=current,
                    total=max(progress_total, current),
                )

            source_frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return current


def _emit_progress(kind: str, **payload: object) -> None:
    print(
        PROGRESS_PREFIX + json.dumps({"kind": kind, **payload}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
