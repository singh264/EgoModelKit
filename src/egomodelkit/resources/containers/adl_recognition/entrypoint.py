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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

SUPPORTED_VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4"})

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
    
    staged_count = _stage_input_videos(
        input_path = input_path,
        adl_dir = adl_dir
    )
    
    if staged_count == 0:
        raise RuntimeError("No supported video files were staged for ADL recognition.")
    
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
) -> int:
    input_files = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.iterdir())
    )
    
    staged_count = 0
    
    for path in input_files:
        if not path.is_file():
            continue
                
        if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            continue
        
        staged_path = adl_dir / f"video{staged_count + 1:03d}.MP4"
        shutil.copy2(path, staged_path)
        staged_count += 1
    
    return staged_count

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

if __name__ == "__main__":
    main()
