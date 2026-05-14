""" Validation helpers for Shan hand-object-contact runs. """

from dataclasses import dataclass
from pathlib import Path
from typing import Final

HAND_OBJECT_CONTACT_MODEL_ID: Final[str] = "hand-object-contact"

SUPPORTED_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

DRY_RUN_VALIDATION_MESSAGE: Final[str] = (
    f"Dry run: {HAND_OBJECT_CONTACT_MODEL_ID} request is valid."
)


class HandObjectContactInputError(ValueError):
    """ Raised when the hand-object-contact request is invalid. """


@dataclass(frozen=True, slots=True)
class HandObjectContactRequest:
    """ One image-based Shan hand-object-contact request. """
    input_path: Path
    output_dir: Path


def validate_request(request: HandObjectContactRequest) -> None:
    """ Validate one image-based hand-object-contact request. """
    if not request.input_path.is_file():
        raise HandObjectContactInputError(
            f"Input image does not exist: {request.input_path}"
        )
    
    input_suffix = request.input_path.suffix.lower()
    
    if input_suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise HandObjectContactInputError(
            "Unsupported image file type:",
            f"{request.input_path.suffix or '<none>'}"
        )
    
    if request.output_dir.exists() and not request.output_dir.is_dir():
        raise HandObjectContactInputError(
            f"Output path exists but is not a directory: {request.output_dir}"
        )
