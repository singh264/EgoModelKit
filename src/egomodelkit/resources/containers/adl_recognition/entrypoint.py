""" Container entry point for EgoModelKit ADL recognition core stages.

This file keeps EgoModelKit-specific logic small.

It does not reimplement EgoVizML's ADL processing, instead it:

1. stages user videos into the folder shape EgoVizML expects
2. calls EgoVizML's video_to_subclips_and_frames.py
3. adapts nested Detic/Shan outputs into the flat folders expected by process_all_preds.py
4. calls EgoVizML's process_all_preds.py
5. calls a prediction wrapper that uses egoviz.models.processing and egoviz.models.inference
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

SUPPORTED_VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4"})
EGOVIZML_STAGED_VIDEO_SUFFIX: Final[str] = ".MP4"

DEFAULT_SESSION_ID: Final[str] = "session001"
ADL_INPUT_MANIFEST_FILENAME: Final[str] = "adl_input_manifest.csv"
ADL_SUBCLIP_MANIFEST_FILENAME: Final[str] = "adl_subclip_manifest.csv"
METRICS_CONFIG_FILENAME: Final[str] = "metrics_config.json"

EGOVIZML_HOME: Final[Path] = Path("/opt/EgoVizML")

VIDEO_TO_SUBCLIPS_SCRIPT: Final[Path] = (
    EGOVIZML_HOME / "scripts" / "video_to_subclips_and_frames.py"
)

PROCESS_ALL_PREDS_SCRIPT: Final[Path] = EGOVIZML_HOME / "scripts" / "process_all_preds.py"

PREDICT_ADL_SCRIPT: Final[Path] = Path("/opt/egomodelkit_predict_adl.py")

EGOVIZML_ADL_DIRS: Final[tuple[str, ...]] = (
    "communication-management",
    "functional-mobility",
    "grooming-health-management",
    "home-management",
    "leisure-other-activities",
    "meal-preparation-cleanup",
    "self-feeding",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices = ["extract", "predict", "finalize"], required = True)
    parser.add_argument("--input-path", required = True)
    parser.add_argument("--output-dir", required = True)
    parser.add_argument("--work-dir-name", required = True)
    parser.add_argument("--egoviz-data-dir-name", required = True)
    parser.add_argument("--adl-dir-name", required = True)
    parser.add_argument("--subclip-length", type = int, required = True)
    parser.add_argument("--fps", type = int, required = True)
    parser.add_argument("--frame-fps", type = int, required = True)
    parser.add_argument("--resize-width", type = int, required = True)
    parser.add_argument("--resize-height", type = int, required = True)
    parser.add_argument("--pooling-window-seconds", type = float, required = True)
    parser.add_argument("--interaction-contact-state-threshold", type = int, required = True)
    parser.add_argument("--dominant-hand", choices = ["left", "right"], required = True)   
    parser.add_argument("--active-iou", type = float, required = True)
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents = True, exist_ok = True)
    
    if args.stage == "extract":
        extract_frames(
            input_path = input_path,
            output_dir = output_dir,
            work_dir_name = args.work_dir_name,
            egoviz_data_dir_name = args.egoviz_data_dir_name,
            adl_dir_name = args.adl_dir_name,
            subclip_length = args.subclip_length,
            fps = args.fps,
            frame_fps = args.frame_fps,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            pooling_window_seconds=args.pooling_window_seconds,
            interaction_contact_state_threshold=args.interaction_contact_state_threshold,
            dominant_hand = args.dominant_hand,
        )
        
        return
    
    if args.stage == "predict":
        predict_from_all_preds(
            all_preds_path = input_path,
            output_dir = output_dir,
        )
        
        return

    finalize_predictions(
        output_dir = output_dir,
        work_dir_name = args.work_dir_name,
        egoviz_data_dir_name = args.egoviz_data_dir_name,
        adl_dir_name = args.adl_dir_name,
        active_iou = args.active_iou,
    )

def extract_frames(
    *,
    input_path: Path,
    output_dir: Path,
    work_dir_name: str,
    egoviz_data_dir_name: str,
    adl_dir_name: str,
    subclip_length: int,
    fps: int,
    frame_fps: int,
    resize_width: int,
    resize_height: int,
    pooling_window_seconds: float,
    interaction_contact_state_threshold: int,
    dominant_hand: str,
) -> None:
    print("EgoModelKit runtime: preparing EgoVizML video workspace.", flush = True)
    
    data_root = _egoviz_data_root( 
        output_dir = output_dir,
        work_dir_name = work_dir_name,
        egoviz_data_dir_name = egoviz_data_dir_name,
    )
    
    adl_dir = data_root / adl_dir_name
    
    if adl_dir.exists():
        shutil.rmtree(adl_dir)
        
    adl_dir.mkdir(parents = True, exist_ok = True)
    
    staged_rows = _stage_input_videos(
        input_path = input_path,
        adl_dir = adl_dir,
        manifest_path = output_dir / ADL_INPUT_MANIFEST_FILENAME,
    )

    if not staged_rows:
        raise RuntimeError("No supported video files were staged for ADL recognition.")
    
    _write_metrics_config(
        output_dir / METRICS_CONFIG_FILENAME,
        subclip_length_seconds = subclip_length,
        subclip_fps = fps,
        frame_fps = frame_fps,
        resize_width = resize_width,
        resize_height = resize_height,
        pooling_window_seconds = pooling_window_seconds,
        interaction_contact_state_threshold = interaction_contact_state_threshold,
        dominant_hand = dominant_hand,
    )
    
    print("EgoModelKit runtime: calling EgoVizML frame extraction.", flush = True)
    
    _run(
        [
            sys.executable,
            str(VIDEO_TO_SUBCLIPS_SCRIPT),
            str(adl_dir),
            "--subclip_length",
            str(subclip_length),
            "--fps",
            str(fps),
            "--frame_fps",
            str(frame_fps),
        ]
    )
    
    _write_subclip_manifest(
        manifest_path = output_dir / ADL_SUBCLIP_MANIFEST_FILENAME,
        staged_rows = staged_rows,
        subclips_dir = adl_dir / "subclips",
        subclip_length_seconds = subclip_length,
        processing_fps = frame_fps,
    )
    
    print("EgoModelKit runtime: resizing extracted ADL frames.", flush = True)

    _resize_extracted_frames(
        adl_dir / "subclips",
        resize_width = resize_width,
        resize_height = resize_height,
    )
    
    print("EgoModelKit runtime: EgoVizML frame extraction finished.", flush = True)

def finalize_predictions(
    *,
    output_dir: Path,
    work_dir_name: str,
    egoviz_data_dir_name: str,
    adl_dir_name: str,
    active_iou: float,
) -> None:
    print("EgoModelKit runtime: preparing EgoVizML prediction folders.", flush = True)
    
    data_root = _egoviz_data_root(
        output_dir = output_dir,
        work_dir_name = work_dir_name,
        egoviz_data_dir_name = egoviz_data_dir_name,
    )
    
    adl_dir = data_root / adl_dir_name
    
    _ensure_egovizml_adl_folders(data_root)
    _flatten_nested_model_outputs(adl_dir)
    
    print("EgoModelKit runtime: calling EgoVizML process_all_preds.py", flush = True)
    
    _run(
        [
            sys.executable,
            str(PROCESS_ALL_PREDS_SCRIPT),
            str(data_root),
            "--active_iou",
            str(active_iou),
        ]
    )
    
    generated_all_preds = data_root / "all_preds.pkl"
    final_all_preds = output_dir / "all_preds.pkl"
    
    if not generated_all_preds.exists():
        raise RuntimeError(f"EgoVizML did not write expected file: {generated_all_preds}")
    
    shutil.copy2(generated_all_preds, final_all_preds)
    
    predict_from_all_preds(
        all_preds_path = final_all_preds,
        output_dir = output_dir,
    )

def predict_from_all_preds(
    *,
    all_preds_path: Path,
    output_dir: Path,
) -> None:
    print("EgoModelKit runtime: calling EgoVizML classifier wrapper.", flush = True)
    
    _run(
        [
            sys.executable,
            str(PREDICT_ADL_SCRIPT),
            "--all-preds-input",
            str(all_preds_path),
            "--output",
            str(output_dir / "adl_predictions.csv"),
            "--summary-output",
            str(output_dir / "adl_predictions_summary.csv"),
        ]
    )
    
    print("EgoModelKit runtime: ADL prediction outputs are ready.", flush = True)

def _stage_input_videos(
    *,
    input_path: Path,
    adl_dir: Path,
    manifest_path: Path,
) -> list[dict[str, object]]:
    input_files = (
        [input_path]
        if input_path.is_file()
        else sorted(
            input_path.iterdir(),
            key = lambda path: _natural_sort_key(path.name),
        )
    )

    staged_count = 0
    staged_rows: list[dict[str, object]] = []

    for path in input_files:
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            continue

        source_metadata = _probe_source_video_metadata(path)
        staged_video_name = f"video{staged_count + 1:03d}{EGOVIZML_STAGED_VIDEO_SUFFIX}"
        staged_path = adl_dir / staged_video_name
        shutil.copy2(path, staged_path)
        staged_count += 1
        
        staged_rows.append(
            {
                "session_id": DEFAULT_SESSION_ID,
                "session_sort_index": staged_count,
                "input_name": path.name,
                "staged_video_name": staged_video_name,
                "staged_video_stem": staged_path.stem,
                "input_modified_time": _input_modified_time(path),
                "source_duration_seconds": source_metadata["source_duration_seconds"],
                "source_fps": source_metadata["source_fps"],
                "source_total_frames": source_metadata["source_total_frames"],
            }
        )

    _write_input_manifest(manifest_path, staged_rows)

    return staged_rows

def _ensure_egovizml_adl_folders(data_root: Path) -> None:
    for adl_name in EGOVIZML_ADL_DIRS:
        (data_root / adl_name / "detic").mkdir(parents = True, exist_ok = True)
        (data_root / adl_name / "shan").mkdir(parents = True, exist_ok = True)

def _flatten_nested_model_outputs(adl_dir: Path) -> None:
    detic_raw_dir = adl_dir / "detic_raw"
    shan_raw_dir = adl_dir / "subclips_shan"
    
    detic_flat_dir = adl_dir / "detic"
    shan_flat_dir = adl_dir / "shan"
    
    detic_flat_dir.mkdir(parents = True, exist_ok = True)
    shan_flat_dir.mkdir(parents = True, exist_ok = True)
    
    for old_file in detic_flat_dir.glob("*.pkl"):
        old_file.unlink()
    
    for old_file in shan_flat_dir.glob("*.pkl"):
        old_file.unlink()
    
    detic_by_key: dict[tuple[str, str], Path] = {}
    shan_by_key: dict[tuple[str, str], Path] = {}
    
    for detic_path in sorted(detic_raw_dir.rglob("*_detic.pkl")):
        clip_name = detic_path.parent.name
        frame_stem = detic_path.stem.removesuffix("_detic")
        detic_by_key[(clip_name, frame_stem)] = detic_path
    
    for shan_path in sorted(shan_raw_dir.rglob("*_shan.pkl")):
        clip_name = shan_path.parent.name
        frame_stem = shan_path.stem.removesuffix("_shan")
        shan_by_key[(clip_name, frame_stem)] = shan_path
    
    detic_keys = set(detic_by_key)
    shan_keys = set(shan_by_key)
    
    missing_shan = [
        str(detic_by_key[key])
        for key in sorted(detic_keys - shan_keys)
    ]
    
    missing_detic = [
        str(shan_by_key[key])
        for key in sorted(shan_keys - detic_keys)
    ]
    
    if missing_shan:
        raise RuntimeError(
            "Missing hand-object-contact outputs for Detic files: "
            + ", ".join(missing_shan[:5])
        )
    
    if missing_detic:
        raise RuntimeError(
            "Missing Detic outputs for hand-object-contact files: "
            + ", ".join(missing_detic[:5])
        )

    paired_count = 0
    
    for clip_name, frame_stem in sorted(detic_keys):
        detic_path = detic_by_key[(clip_name, frame_stem)]
        shan_path = shan_by_key[(clip_name, frame_stem)]
        
        frame_token = frame_stem.replace("frame_", "frame")
        output_base = f"{clip_name}_{frame_token}"
        
        shutil.copy2(detic_path, detic_flat_dir / f"{output_base}_detic.pkl")
        shutil.copy2(shan_path, shan_flat_dir / f"{output_base}_shan.pkl")
        
        paired_count += 1
    
    if paired_count == 0:
        raise RuntimeError("No paired Detic and hand-object-contact predictions were found.")
    
    print(f"EgoModelKit runtime: paired {paired_count} Detic/Shan frame outputs.", flush = True)

def _egoviz_data_root(
    *,
    output_dir: Path,
    work_dir_name: str,
    egoviz_data_dir_name: str,
) -> Path:
    return output_dir / work_dir_name / egoviz_data_dir_name

def _run(command: list[str]) -> None:
    print("EgoModelKit runtime: " + " ".join(command), flush = True)
    subprocess.run(command, check = True)

def _natural_sort_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]

def _input_modified_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(
        microsecond = 0
    ).isoformat()

def _write_input_manifest(
    manifest_path: Path,
    rows: list[dict[str, object]],
) -> None:
    manifest_path.parent.mkdir(parents = True, exist_ok = True)

    with manifest_path.open("w", encoding = "utf-8", newline = "") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames = [
                "session_id",
                "session_sort_index",
                "input_name",
                "staged_video_name",
                "staged_video_stem",
                "input_modified_time",
                "source_duration_seconds",
                "source_fps",
                "source_total_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

def _probe_source_video_metadata(path: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check = True,
        capture_output = True,
        text = True,
    )

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", []) if isinstance(payload, dict) else []

    if (
        not isinstance(streams, list)
        or not streams
        or not isinstance(streams[0], dict)
    ):
        raise RuntimeError(f"Could not read source video metadata: {path}")

    stream = streams[0]
    format_payload = payload.get("format", {})

    format_duration = (
        format_payload.get("duration")
        if isinstance(format_payload, dict)
        else None
    )

    source_duration_seconds = _first_positive_float(
        stream.get("duration"),
        format_duration,
    )

    if source_duration_seconds <= 0:
        raise RuntimeError(
            f"Could not determine source video duration: {path}"
        )

    source_fps = _parse_ffprobe_frame_rate(
        stream.get("avg_frame_rate")
    )

    if source_fps <= 0:
        source_fps = _parse_ffprobe_frame_rate(
            stream.get("r_frame_rate")
        )

    source_total_frames = _positive_int(
        stream.get("nb_frames")
    )

    if source_total_frames == 0 and source_fps > 0:
        source_total_frames = round(
            source_duration_seconds * source_fps
        )

    return {
        "source_duration_seconds": source_duration_seconds,
        "source_fps": source_fps,
        "source_total_frames": source_total_frames,
    }

def _first_positive_float(*values: object) -> float:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue

        if parsed > 0:
            return parsed

    return 0.0

def _positive_int(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0

    return max(0, parsed)

def _parse_ffprobe_frame_rate(value: object) -> float:
    if not isinstance(value, str):
        return 0.0

    if "/" not in value:
        return _first_positive_float(value)

    numerator_text, denominator_text = value.split("/", 1)

    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except ValueError:
        return 0.0

    if denominator == 0:
        return 0.0

    return max(0.0, numerator / denominator)

def _write_subclip_manifest(
    *,
    manifest_path: Path,
    staged_rows: list[dict[str, object]],
    subclips_dir: Path,
    subclip_length_seconds: int,
    processing_fps: int,
) -> None:
    rows: list[dict[str, object]] = []

    for staged_row in staged_rows:
        staged_video_stem = str(staged_row["staged_video_stem"])
        source_duration_seconds = float(staged_row["source_duration_seconds"])

        matching_subclips = (
            sorted(
                [
                    path
                    for path in subclips_dir.iterdir()
                    if path.is_dir()
                    and path.name.startswith(
                        f"{staged_video_stem}--"
                    )
                ],
                key = lambda path: _subclip_index_from_name(
                    path.name
                ),
            )
            if subclips_dir.is_dir()
            else []
        )

        for subclip_path in matching_subclips:
            subclip_index = _subclip_index_from_name(subclip_path.name)
            source_start_seconds = (subclip_index - 1) * subclip_length_seconds

            valid_duration_seconds = max(
                0.0,
                min(
                    float(subclip_length_seconds),
                    source_duration_seconds - source_start_seconds,
                ),
            )

            source_end_seconds = source_start_seconds + valid_duration_seconds

            rows.append(
                {
                    "session_id": staged_row["session_id"],
                    "input_name": staged_row["input_name"],
                    "staged_video_stem": staged_video_stem,
                    "subclip_name": subclip_path.name,
                    "subclip_index": subclip_index,
                    "source_start_seconds": source_start_seconds,
                    "source_end_seconds": source_end_seconds,
                    "valid_duration_seconds": valid_duration_seconds,
                    "processing_fps": processing_fps,
                    "processing_subclip_duration_seconds": subclip_length_seconds,
                }
            )

    manifest_path.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    with manifest_path.open(
        "w",
        encoding = "utf-8",
        newline = "",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames = [
                "session_id",
                "input_name",
                "staged_video_stem",
                "subclip_name",
                "subclip_index",
                "source_start_seconds",
                "source_end_seconds",
                "valid_duration_seconds",
                "processing_fps",
                "processing_subclip_duration_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

def _subclip_index_from_name(
    subclip_name: str,
) -> int:
    match = re.search(
        r"--(\d+)$",
        subclip_name,
    )

    if match is None:
        raise RuntimeError(
            "Unexpected EgoVizML subclip name: "
            f"{subclip_name}"
        )

    return int(match.group(1))

def _write_metrics_config(
    metrics_config_path: Path,
    *,
    subclip_length_seconds: int,
    subclip_fps: int,
    frame_fps: int,
    resize_width: int,
    resize_height: int,
    pooling_window_seconds: float,
    interaction_contact_state_threshold: int,
    dominant_hand: str,
) -> None:
    metrics_config_path.parent.mkdir(parents = True, exist_ok = True)
    pooling_window_frames = max(1, round(frame_fps * pooling_window_seconds))
    
    metrics_config_path.write_text(
        json.dumps(
            {
                "subclip_length_seconds": subclip_length_seconds,
                "subclip_fps": subclip_fps,
                "frame_fps": frame_fps,
                "resize_width": resize_width,
                "resize_height": resize_height,
                "pooling_window_seconds": pooling_window_seconds,
                "pooling_window_frames": pooling_window_frames,
                "interaction_contact_state_threshold": interaction_contact_state_threshold,
                "dominant_hand": dominant_hand,
            },
            indent = 2,
            sort_keys = True,
        ) + "\n",
        encoding = "utf-8",
    )

def _resize_extracted_frames(
    subclips_dir: Path,
    *,
    resize_width: int,
    resize_height: int,
) -> None:
    if not subclips_dir.is_dir():
        return

    import cv2  # noqa: PLC0415

    for image_path in sorted(subclips_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        resized = cv2.resize(image, (resize_width, resize_height))
        
        cv2.imwrite(str(image_path), resized)

if __name__ == "__main__":
    main()
