from pathlib import Path

import pytest

from egomodelkit.models.adl_recognition import (
    AdlRecognitionRequest,
)
from egomodelkit.runtime.adl_recognition import (
    AdlRecognitionRuntimeError,
    run_adl_recognition,
)


def test_run_adl_recognition_reports_progress_messages(tmp_path: Path) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake-video")

    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )

    messages: list[str] = []

    with pytest.raises(
        AdlRecognitionRuntimeError,
        match = "adl-recognition runtime is not available yet."
    ):
        run_adl_recognition(
            request,
            command_runner = lambda command: 0,
            progress = messages.append,
        )

    assert "Validating adl-recognition request." in messages
    assert "Checking host runtime prerequisites." in messages
    assert any(message.startswith("Python ") for message in messages)
    assert "Docker daemon is available." in messages
