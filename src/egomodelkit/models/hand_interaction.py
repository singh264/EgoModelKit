"""Validation helpers for standalone hand-interaction runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from egomodelkit.bandini_metrics import (
    DEFAULT_DOMINANT_HAND,
    LEFT_HAND_LABEL,
    RIGHT_HAND_LABEL,
    HandLabel,
)

HAND_INTERACTION_MODEL_ID: Final[str] = "hand-interaction"
HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4"})
HAND_INTERACTION_DRY_RUN_VALIDATION_MESSAGE: Final[str] = (
    f"Dry run: {HAND_INTERACTION_MODEL_ID} request is valid."
)


class HandInteractionInputError(ValueError):
    """Raised when a hand-interaction request is invalid."""


@dataclass(frozen=True, slots=True)
class HandInteractionRequest:
    """One video-based standalone hand-interaction request."""

    input_path: Path
    output_dir: Path
    dominant_hand: HandLabel = DEFAULT_DOMINANT_HAND


def validate_hand_interaction_request(request: HandInteractionRequest) -> None:
    """Validate one standalone hand-interaction request."""
    input_path = request.input_path

    if request.dominant_hand not in {LEFT_HAND_LABEL, RIGHT_HAND_LABEL}:
        raise HandInteractionInputError("Dominant hand must be 'left' or 'right'.")

    if not input_path.exists():
        raise HandInteractionInputError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES:
            raise HandInteractionInputError(
                "Input must be an MP4 video file or a directory containing MP4 video files."
            )
    elif input_path.is_dir():
        if not _directory_contains_supported_videos(input_path):
            raise HandInteractionInputError(
                "Input directory does not contain any MP4 video files."
            )
    else:
        raise HandInteractionInputError(
            "Input path must be an MP4 video file or a directory containing MP4 video files."
        )

    if request.output_dir.exists() and not request.output_dir.is_dir():
        raise HandInteractionInputError(
            f"Output path exists but is not a directory: {request.output_dir}"
        )


def _directory_contains_supported_videos(input_dir: Path) -> bool:
    """Return whether a directory directly contains at least one MP4 video."""
    return any(
        child.is_file()
        and child.suffix.lower() in HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES
        for child in input_dir.iterdir()
    )
