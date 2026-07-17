""" Container entry point for Shan hand-object-contact image inference. """

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

PYTHON_EXECUTABLE: Final[str] = sys.executable
SHAN_DEMO_SCRIPT: Final[str] = "demo.py"

INPUT_PATH_FLAG: Final[str] = "--input-path"
OUTPUT_DIR_FLAG: Final[str] = "--output-dir"

SHAN_CUDA_FLAG: Final[str] = "--cuda"
SHAN_NETWORK_FLAG: Final[str] = "--net"
SHAN_DATASET_FLAG: Final[str] = "--dataset"
SHAN_CHECKSESSION_FLAG: Final[str] = "--checksession"
SHAN_CHECKEPOCH_FLAG: Final[str] = "--checkepoch"
SHAN_CHECKPOINT_FLAG: Final[str] = "--checkpoint"
SHAN_IMAGE_DIR_FLAG: Final[str] = "--image_dir"
SHAN_SAVE_DIR_FLAG: Final[str] = "--save_dir"
SHAN_LOAD_DIR_FLAG: Final[str] = "--load_dir"

LOAD_DIR_ENV: Final[str] = "EGOMODELKIT_SHAN_LOAD_DIR"
NETWORK_ENV: Final[str] = "EGOMODELKIT_SHAN_NETWORK_NAME"
DATASET_ENV: Final[str] = "EGOMODELKIT_SHAN_DATASET_NAME"
CHECKPOINT_SESSION_ENV: Final[str] = "EGOMODELKIT_SHAN_CHECKPOINT_SESSION"
CHECKPOINT_EPOCH_ENV: Final[str] = "EGOMODELKIT_SHAN_CHECKPOINT_EPOCH"
CHECKPOINT_STEP_ENV: Final[str] = "EGOMODELKIT_SHAN_CHECKPOINT_STEP"

STAGED_IMAGE_DIR: Final[Path] = Path(
    "/tmp/egomodelkit_hand_object_contact_input"
)

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser() 
    parser.add_argument(INPUT_PATH_FLAG, required = True)
    parser.add_argument(OUTPUT_DIR_FLAG, required = True)
    
    return parser.parse_args()

def _copy_supported_directory_images(
    input_dir: Path,
    staged_dir: Path,
) -> None:
    """ Copy supported image files from one mounted directory into staging. """
    for child in sorted(input_dir.iterdir()):
        if (
            child.is_file()
            and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ):
            shutil.copy2(child, staged_dir / child.name)

def prepare_image_dir(input_path: Path) -> Path:
    """ Stage mounted image input for directory-based demo API. """
    print("EgoModelKit runtime: staging input image(s) for hand-object-contact.", flush = True)

    if STAGED_IMAGE_DIR.exists():
        shutil.rmtree(STAGED_IMAGE_DIR)
    
    STAGED_IMAGE_DIR.mkdir(parents = True, exist_ok = True)
    
    if input_path.is_dir():
        _copy_supported_directory_images(
            input_dir = input_path,
            staged_dir = STAGED_IMAGE_DIR,
        )
    else:
        shutil.copy2(
            input_path,
            STAGED_IMAGE_DIR / input_path.name,
        )
    
    return STAGED_IMAGE_DIR

def main() -> None:
    args = parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    print("EgoModelKit runtime: preparing output directory.", flush = True)
    output_dir.mkdir(parents = True, exist_ok = True)

    image_dir = prepare_image_dir(input_path)

    command = [
        PYTHON_EXECUTABLE,
        SHAN_DEMO_SCRIPT,
        SHAN_CUDA_FLAG,
        SHAN_NETWORK_FLAG,
        os.environ[NETWORK_ENV],
        SHAN_DATASET_FLAG,
        os.environ[DATASET_ENV],
        SHAN_CHECKSESSION_FLAG,
        os.environ[CHECKPOINT_SESSION_ENV],
        SHAN_CHECKEPOCH_FLAG,
        os.environ[CHECKPOINT_EPOCH_ENV],
        SHAN_CHECKPOINT_FLAG,
        os.environ[CHECKPOINT_STEP_ENV],
        SHAN_IMAGE_DIR_FLAG,
        str(image_dir),
        SHAN_SAVE_DIR_FLAG,
        str(output_dir),
        SHAN_LOAD_DIR_FLAG,
        os.environ[LOAD_DIR_ENV],
    ]
    
    print("EgoModelKit runtime: launching hand-object-contact demo inference.", flush = True)
    subprocess.run(command, check = True)
    print("EgoModelKit runtime: hand-object-contact inference finished.", flush = True)

if __name__ == "__main__":
    main()