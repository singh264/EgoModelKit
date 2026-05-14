from pathlib import Path

import pytest

from egomodelkit.models.hand_object_contact import (
    HandObjectContactInputError,
    HandObjectContactRequest,
    validate_request,
)


def test_validate_request_accepts_supported_image(tmp_path: Path) -> None:
    input_path = tmp_path / "frame.JPG"
    input_path.write_bytes(b"fake-image")
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    validate_request(request)

def test_validate_request_rejects_missing_image(tmp_path: Path) -> None:
    request = HandObjectContactRequest(
        input_path = tmp_path / "missing.jpg",
        output_dir = tmp_path / "results",
    )
    
    with pytest.raises(
        HandObjectContactInputError,
        match="Input image does not exist",
    ):
        validate_request(request)

def test_validate_request_rejects_unsupported_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not an image", encoding="utf-8")
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    with pytest.raises (
        HandObjectContactInputError,
        match="Unsupported image file type",
    ):
        validate_request(request)

def test_validate_request_allows_missing_output_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "frame.jpg"
    input_path.write_bytes(b"fake-image")
    
    output_dir = tmp_path / "outputs"
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = output_dir,
    )
    
    validate_request(request)

def test_validate_request_allows_existing_output_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "frame.jpg"
    input_path.write_bytes(b"fake-image")
    
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = output_dir,
    )
    
    validate_request(request)
    
def test_validate_request_rejects_output_path_that_is_a_file(tmp_path: Path) -> None:
    input_path = tmp_path / "frame.jpg"
    input_path.write_bytes(b"fake-image")

    output_path = tmp_path / "outputs"
    output_path.write_text("this is file, not a directory")
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = output_path,
    )
    
    with pytest.raises(
        HandObjectContactInputError,
        match="Output path exists but is not a directory",
    ):
        validate_request(request)
