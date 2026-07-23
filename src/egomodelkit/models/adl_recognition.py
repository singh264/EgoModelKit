from dataclasses import dataclass
from pathlib import Path
from typing import Final

ADL_RECOGNITION_MODEL_ID: Final[str] = "adl-recognition"
COMBINED_PREDS_FILENAME: Final[str] = "all_preds.pkl"

ADL_SEGMENT_LENGTH_SECONDS: Final[int] = 60
ADL_SUBCLIP_ENCODING_FPS: Final[int] = 10
ADL_INFERENCE_FRAME_FPS: Final[int] = 1
ADL_ACTIVE_OBJECT_IOU_THRESHOLD: Final[float] = 0.8

ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4"})

ADL_RECOGNITION_DRY_RUN_VALIDATION_MESSAGE: Final[str] = (
    f"Dry run: {ADL_RECOGNITION_MODEL_ID} request is valid."
)


class AdlRecognitionInputError(ValueError):
    """Raised when an ADL recognition request is invalid."""


@dataclass(frozen=True, slots=True)
class AdlRecognitionRequest:
    """One video-based ADL recognition request."""

    input_path: Path
    output_dir: Path


def validate_adl_recognition_request(request: AdlRecognitionRequest) -> None:
    """Validate an ADL recognition request."""
    input_path = request.input_path

    if not input_path.exists():
        raise AdlRecognitionInputError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if _is_combined_predictions_file(input_path):
            pass
        elif input_path.suffix.lower() in ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES:
            pass
        else:
            raise AdlRecognitionInputError(
                "Input must be an EgoVizML all_preds.pkl file, an MP4 video file, "
                "or a directory containing MP4 video files."
            )
    elif input_path.is_dir():
        if not _directory_contains_supported_videos(input_path):
            raise AdlRecognitionInputError(
                "Input directory does not contain any MP4 video files."
            )
    else:
        raise AdlRecognitionInputError("Input path must be a file or directory.")

    if request.output_dir.exists() and not request.output_dir.is_dir():
        raise AdlRecognitionInputError(
            f"Output path exists but is not a directory: {request.output_dir}"
        )


def _is_combined_predictions_file(input_path: Path) -> bool:
    return input_path.is_file() and input_path.name == COMBINED_PREDS_FILENAME


def _directory_contains_supported_videos(input_dir: Path) -> bool:
    return any(
        child.is_file()
        and child.suffix.lower() in ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES
        for child in input_dir.iterdir()
    )
