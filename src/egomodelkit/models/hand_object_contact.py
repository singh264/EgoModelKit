""" Validation helpers for hand-object-contact runs. """

from dataclasses import dataclass
from pathlib import Path
from typing import Final

HAND_OBJECT_CONTACT_MODEL_ID: Final[str] = "hand-object-contact"

SUPPORTED_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

HAND_OBJECT_CONTACT_DRY_RUN_VALIDATION_MESSAGE: Final[str] = (
    f"Dry run: {HAND_OBJECT_CONTACT_MODEL_ID} request is valid."
)


class HandObjectContactInputError(ValueError):
    """ Raised when the hand-object-contact request is invalid. """


@dataclass(frozen=True, slots=True)
class HandObjectContactRequest:
    """ One image-based hand-object-contact request. """
    input_path: Path
    output_dir: Path


def validate_hand_object_contact_request(request: HandObjectContactRequest) -> None:
    """ Validate one image-based hand-object-contact request. """
    input_path = request.input_path
    
    if not input_path.exists():
        raise HandObjectContactInputError(
            f"Input path does not exist: {input_path}"
        )
    
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            supported_suffixes = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))

            raise HandObjectContactInputError(
                "Unsupported input image suffix "
                f"'{input_path.suffix}.' Supported suffixes: "
                f"{supported_suffixes}"
            )
    elif input_path.is_dir():
        if not _directory_contains_supported_images(input_path):
            raise HandObjectContactInputError(
                "Input directory does not contain any supported image files."
            )
    else:
        raise HandObjectContactInputError(
            "Input path must be a supported image file or a directory "
            "containing supported image files."
        )
    
    if request.output_dir.exists() and not request.output_dir.is_dir():
        raise HandObjectContactInputError(
            f"Output path exists but is not a directory: {request.output_dir}"
        )

def _directory_contains_supported_images(input_dir: Path) -> bool:
    """ Return whether a directory contains at least one supported image file. """
    return any(
        child.is_file()
        and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        for child in input_dir.iterdir()
    )
