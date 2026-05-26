from pathlib import Path

import pytest

from egomodelkit.models.adl_recognition import (
    AdlRecognitionInputError,
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)


def test_validate_adl_recognition_accepts_all_preds_pickle(tmp_path: Path) -> None:
    input_path = tmp_path / "all_preds.pkl"
    input_path.write_bytes(b"fake-pickle")
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    validate_adl_recognition_request(request)

def test_validate_adl_recognition_accepts_video_file(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"fake-video")
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    validate_adl_recognition_request(request)

def test_validate_adl_recognition_accepts_video_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    
    (input_dir / "clip.mp4").write_bytes(b"fake-video")
    
    request = AdlRecognitionRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )
    
    validate_adl_recognition_request(request)

def test_validate_adl_recognition_rejects_empty_video_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    
    request = AdlRecognitionRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )
    
    with pytest.raises(
        AdlRecognitionInputError,
        match = "does not contain any MP4 video files",
    ):
        validate_adl_recognition_request(request)

def test_validate_adl_recognition_rejects_output_file(tmp_path: Path) -> None:
    input_path = tmp_path / "all_preds.pkl"
    input_path.write_bytes(b"fake-pickle")
    
    output_path = tmp_path / "results.txt"
    output_path.write_text("not a directory")
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = output_path,
    )
    
    with pytest.raises(
        AdlRecognitionInputError,
        match = "Output path exists but is not a directory",
    ):
        validate_adl_recognition_request(request)

def test_validate_adl_recognition_rejects_input_path_that_does_not_exist(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "fake_path"
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    with pytest.raises(
        AdlRecognitionInputError,
        match = "Input path does not exist",
    ):
        validate_adl_recognition_request(request)

def test_validate_adl_recognition_rejects_directory_without_supported_input(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not a video")
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    with pytest.raises(
        AdlRecognitionInputError,
        match = (
            "Input must be an EgoVizML all_preds.pkl file, an MP4 video file, "
            "or a directory containing MP4 video files."
        )
    ):
        validate_adl_recognition_request(request)

def test_validate_adl_recognition_rejects_existing_input_that_is_not_file_or_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "special-input"
    output_dir = tmp_path / "results"
    
    request = AdlRecognitionRequest(
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
        AdlRecognitionInputError,
        match = "Input path must be a file or directory."
    ):
        validate_adl_recognition_request(request)

def test_validate_adl_recognition_rejects_directory_named_like_video(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()

    fake_video_dir = input_dir / "clip.mp4"
    fake_video_dir.mkdir()

    request = AdlRecognitionRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )

    with pytest.raises(
        AdlRecognitionInputError,
        match = "does not contain any MP4 video files",
    ):
        validate_adl_recognition_request(request)
