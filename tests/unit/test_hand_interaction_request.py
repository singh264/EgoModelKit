from pathlib import Path

import pytest

from egomodelkit.models.hand_interaction import (
    HAND_INTERACTION_DRY_RUN_VALIDATION_MESSAGE,
    HAND_INTERACTION_MODEL_ID,
    HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES,
    HandInteractionInputError,
    HandInteractionRequest,
    validate_hand_interaction_request,
)


def _request(
    input_path: Path,
    output_dir: Path,
    dominant_hand: str = "right",
) -> HandInteractionRequest:
    return HandInteractionRequest(
        input_path=input_path,
        output_dir=output_dir,
        dominant_hand=dominant_hand,  # type: ignore[arg-type]
    )


def test_hand_interaction_public_constants() -> None:
    assert HAND_INTERACTION_MODEL_ID == "hand-interaction"
    assert HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES == frozenset({".mp4"})
    assert HAND_INTERACTION_DRY_RUN_VALIDATION_MESSAGE == (
        "Dry run: hand-interaction request is valid."
    )


def test_validate_hand_interaction_accepts_single_video_and_default_hand(tmp_path: Path) -> None:
    video = tmp_path / "clip.MP4"
    video.write_bytes(b"video")
    request = HandInteractionRequest(input_path=video, output_dir=tmp_path / "results")

    validate_hand_interaction_request(request)

    assert request.dominant_hand == "right"


def test_validate_hand_interaction_accepts_nonrecursive_video_directory(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clip.mp4").write_bytes(b"video")
    nested = videos / "nested"
    nested.mkdir()
    (nested / "ignored.mp4").write_bytes(b"video")

    validate_hand_interaction_request(
        _request(videos, tmp_path / "results", dominant_hand="left")
    )


def test_validate_hand_interaction_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(HandInteractionInputError, match="Input path does not exist"):
        validate_hand_interaction_request(_request(tmp_path / "missing.mp4", tmp_path / "out"))


def test_validate_hand_interaction_rejects_unsupported_file_and_all_preds(tmp_path: Path) -> None:
    for name in ("frame.jpg", "clip.mov", "all_preds.pkl"):
        input_path = tmp_path / name
        input_path.write_bytes(b"unsupported")
        with pytest.raises(HandInteractionInputError, match="Input must be an MP4 video"):
            validate_hand_interaction_request(_request(input_path, tmp_path / "out"))


def test_validate_hand_interaction_rejects_empty_or_nested_only_directory(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    nested = videos / "nested"
    nested.mkdir()
    (nested / "clip.mp4").write_bytes(b"video")

    with pytest.raises(HandInteractionInputError, match="does not contain any MP4"):
        validate_hand_interaction_request(_request(videos, tmp_path / "out"))


def test_validate_hand_interaction_ignores_directory_named_like_video(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clip.mp4").mkdir()

    with pytest.raises(HandInteractionInputError, match="does not contain any MP4"):
        validate_hand_interaction_request(_request(videos, tmp_path / "out"))


def test_validate_hand_interaction_rejects_invalid_dominant_hand(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    with pytest.raises(HandInteractionInputError, match="Dominant hand must be 'left' or 'right'"):
        validate_hand_interaction_request(
            _request(video, tmp_path / "out", dominant_hand="ambidextrous")
        )


def test_validate_hand_interaction_rejects_output_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "results"
    output.write_text("file", encoding="utf-8")

    with pytest.raises(
        HandInteractionInputError,
        match="Output path exists but is not a directory",
    ):
        validate_hand_interaction_request(_request(video, output))


def test_validate_hand_interaction_rejects_special_existing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "special"
    request = _request(input_path, tmp_path / "out")
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir

    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: True if path == input_path else original_exists(path),
    )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path == input_path else original_is_file(path),
    )
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: False if path == input_path else original_is_dir(path),
    )

    with pytest.raises(HandInteractionInputError, match="Input path must be an MP4"):
        validate_hand_interaction_request(request)
