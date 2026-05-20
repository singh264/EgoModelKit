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
        match="Input path does not exist",
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
        match="Unsupported input image suffix",
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

def test_validate_request_accepts_directory_with_supported_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    
    (input_dir / "frame_001.jpg").write_bytes(b"fake-image")
    (input_dir / "frame_002.png").write_bytes(b"fake-image")
    
    request = HandObjectContactRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )
    
    validate_request(request)

def test_validate_request_rejects_directory_without_supported_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    
    (input_dir / "notes.txt").write_text("not an image")
    
    request = HandObjectContactRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )
    
    with pytest.raises(
        HandObjectContactInputError,
        match = "directory does not contain any supported image files",
    ):
        validate_request(request)

def test_validate_request_accepts_directory_with_images_and_non_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    
    (input_dir / "frame_001.jpg").write_bytes(b"fake-image")
    (input_dir / "README.txt").write_text("extra file")
    
    request = HandObjectContactRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )
    
    validate_request(request)

def test_validate_request_rejects_existing_input_that_is_not_file_or_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "special-input"
    output_dir = tmp_path / "results"
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = output_dir,
    )
    
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir
    
    def fake_exists(path: Path) -> bool:
        if path == input_path:
            return True
        
        return original_exists(path)

    def fake_is_file(path: Path) -> bool:
        if path == input_path:
            return False

        return original_is_file(path)
    
    def fake_is_dir(path: Path) -> bool:
        if path == input_path:
            return False

        return original_is_dir(path)
    
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)
    
    with pytest.raises(
        HandObjectContactInputError,
        match = (
            "Input path must be a supported image file or a directory "
            "containing supported image files."
        )
    ):
        validate_request(request)
