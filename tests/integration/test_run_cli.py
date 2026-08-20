from pathlib import Path

import pytest
from typer.testing import CliRunner

from egomodelkit.cli import CLI_RUNTIME_ERROR_EXIT_CODE, CLI_UNSUPPORTED_MODEL_EXIT_CODE, app
from egomodelkit.runtime.adl_recognition import AdlRecognitionRuntimeError
from egomodelkit.runtime.disk_space import DiskSpacePreflightError

runner = CliRunner()


@pytest.fixture(autouse=True)
def _avoid_machine_preflight_for_cli_integration_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep request/dispatch tests independent of the host Docker installation."""
    monkeypatch.setattr(
        "egomodelkit.cli.ensure_host_runtime_ready",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "egomodelkit.cli.ensure_sufficient_disk_space",
        lambda **_kwargs: None,
    )

def test_run_rejects_internal_hand_object_contact_model(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")

    result = runner.invoke(
        app,
        [
            "run",
            "hand-object-contact",
            "--input",
            str(image_path),
            "--output",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )

    assert result.exit_code == CLI_UNSUPPORTED_MODEL_EXIT_CODE
    assert "Unsupported model: hand-object-contact" in result.output


def test_run_rejects_unknown_model(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")

    result = runner.invoke(
        app,
        [
            "run",
            "unknown-model",
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )

    assert result.exit_code == CLI_UNSUPPORTED_MODEL_EXIT_CODE
    assert "Unsupported model" in result.output


def test_run_rejects_unsupported_adl_video_suffix(tmp_path: Path) -> None:
    unsupported_input = tmp_path / "clip.avi"
    unsupported_input.write_text("not a supported video suffix")

    result = runner.invoke(
        app,
        [
            "run",
            "adl-recognition",
            "--input",
            str(unsupported_input),
            "--output",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )

    assert result.exit_code == CLI_RUNTIME_ERROR_EXIT_CODE
    assert "Error:" in result.output
    assert "MP4 video" in result.output


def test_run_rejects_invalid_input_before_creating_output_scaffold(
    tmp_path: Path,
) -> None:
    unsupported_input = tmp_path / "clip.avi"
    unsupported_input.write_text("not a supported video suffix")
    output_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "adl-recognition",
            "--input",
            str(unsupported_input),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == CLI_RUNTIME_ERROR_EXIT_CODE
    assert "Error:" in result.output
    assert "MP4 video" in result.output
    assert not output_dir.exists()


def test_run_dry_run_accepts_adl_recognition_directory(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    
    (input_dir / "clip.mp4").write_bytes(b"fake-video")
    
    output_dir = tmp_path / "results"
    
    result = runner.invoke(
        app,
        [
            "run",
            "adl-recognition",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--dry-run",
        ]
    )
    
    assert result.exit_code == 0
    assert "Dry run: adl-recognition request is valid." in result.output

def test_run_reports_adl_recognition_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    
    (input_dir / "video.mp4").write_bytes(b"fake-video")
    
    output_dir = tmp_path / "results"

    def fake_run_adl_recognition(
        request,
        *,
        command_runner,
        streaming_command_runner,
        progress,
    ) -> list[str]:
        raise AdlRecognitionRuntimeError("simulated ADL failure")

    monkeypatch.setattr(
        "egomodelkit.runtime.adl_recognition.run_adl_recognition",
        fake_run_adl_recognition,
    )
    
    result = runner.invoke(
        app,
        [
            "run",
            "adl-recognition",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == CLI_RUNTIME_ERROR_EXIT_CODE
    assert "simulated ADL failure" in result.output

def test_run_reports_output_finalization_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "results"

    def fake_run_adl_recognition(
        request,
        *,
        command_runner,
        streaming_command_runner,
        progress,
    ) -> list[str]:
        del request, command_runner, streaming_command_runner, progress
        return ["docker", "run"]

    def fail_finalization(**kwargs) -> None:
        del kwargs
        raise RuntimeError("finalization failed")

    monkeypatch.setattr(
        "egomodelkit.runtime.adl_recognition.run_adl_recognition",
        fake_run_adl_recognition,
    )
    monkeypatch.setattr(
        "egomodelkit.cli.finalize_runtime_outputs",
        fail_finalization,
    )

    result = runner.invoke(
        app,
        [
            "run",
            "adl-recognition",
            "--input",
            str(video_path),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == CLI_RUNTIME_ERROR_EXIT_CODE
    assert "Error: finalization failed" in result.output
    assert result.exception is not None
    assert not isinstance(result.exception, RuntimeError)

def test_run_executes_adl_recognition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake-video")
    
    output_dir = tmp_path / "results"
    
    captured: dict[str, object] = {}
    
    def fake_run_with_output_contract(*, model_id, request) -> Path:
        captured["model_id"] = model_id
        captured["request"] = request

        return output_dir / "run-test"

    monkeypatch.setattr(
        "egomodelkit.cli._run_model_with_output_contract",
        fake_run_with_output_contract,
    )
    
    result = runner.invoke(
        app,
        [
          "run",
          "adl-recognition",
          "--input",
          str(input_path),
          "--output",
          str(output_dir)
        ],
    )
    
    assert result.exit_code == 0
    assert "Completed: adl-recognition" in result.output
    assert f"Outputs: {output_dir / 'run-test'}" in result.output
    assert captured["model_id"] == "adl-recognition"
    assert "request" in captured


def test_run_rechecks_disk_space_without_cleanup_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "results"
    calls: list[dict[str, object]] = []

    def failing_disk_space_check(**kwargs) -> None:
        calls.append(kwargs)
        raise DiskSpacePreflightError(
            "Insufficient disk space before model execution."
        )

    monkeypatch.setattr(
        "egomodelkit.cli.ensure_sufficient_disk_space",
        failing_disk_space_check,
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.hand_interaction.run_hand_interaction",
        lambda *_args, **_kwargs: pytest.fail(
            "Model runtime must not start after a disk-space failure."
        ),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "hand-interaction",
            "--input",
            str(video),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == CLI_RUNTIME_ERROR_EXIT_CODE
    assert "Insufficient disk space before model execution." in result.output
    assert len(calls) == 1
    assert calls[0]["cleanup_stale_images"] is True
    assert not output.exists()


def test_run_dry_run_accepts_hand_interaction_video_and_directory(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    result = runner.invoke(
        app,
        [
            "run",
            "hand-interaction",
            "--input",
            str(video),
            "--output",
            str(tmp_path / "single-results"),
            "--dominant-hand",
            "left",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry run: hand-interaction request is valid." in result.output
    assert "Dominant hand: left" in result.output

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"video")
    result = runner.invoke(
        app,
        [
            "run",
            "hand-interaction",
            "--input",
            str(videos),
            "--output",
            str(tmp_path / "directory-results"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0


def test_run_executes_hand_interaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "results"
    captured: dict[str, object] = {}

    def fake_run(*, model_id, request) -> Path:
        captured["model_id"] = model_id
        captured["request"] = request
        return output / "run-test"

    monkeypatch.setattr(
        "egomodelkit.cli._run_model_with_output_contract",
        fake_run,
    )
    result = runner.invoke(
        app,
        [
            "run",
            "hand-interaction",
            "--input",
            str(video),
            "--output",
            str(output),
            "--dominant-hand",
            "left",
        ],
    )
    assert result.exit_code == 0
    assert "Completed: hand-interaction" in result.output
    assert f"Outputs: {output / 'run-test'}" in result.output
    assert captured["request"].dominant_hand == "left"


def test_shared_output_contract_dispatches_hand_interaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit import cli
    from egomodelkit.models.adl_recognition import AdlRecognitionRequest
    from egomodelkit.models.hand_interaction import HandInteractionRequest

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    captured: dict[str, object] = {}

    def fake_runtime(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return [["docker", "run"]]

    monkeypatch.setattr(
        "egomodelkit.runtime.hand_interaction.run_hand_interaction",
        fake_runtime,
    )
    monkeypatch.setattr(cli, "finalize_runtime_outputs", lambda **_kwargs: None)
    run_dir = cli._run_model_with_output_contract(
        model_id="hand-interaction",
        request=HandInteractionRequest(
            input_path=video,
            output_dir=tmp_path / "results",
            dominant_hand="left",
        ),
    )
    assert run_dir.is_dir()
    assert captured["request"].output_dir == run_dir
    assert captured["request"].dominant_hand == "left"

    with pytest.raises(TypeError, match="Hand interaction requires"):
        cli._run_model_with_output_contract(
            model_id="hand-interaction",
            request=AdlRecognitionRequest(input_path=video, output_dir=tmp_path / "bad"),
        )


def test_shared_output_contract_rejects_unknown_model_directly(tmp_path: Path) -> None:
    from egomodelkit import cli
    from egomodelkit.models.hand_interaction import HandInteractionRequest

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    request = HandInteractionRequest(input_path=video, output_dir=tmp_path / "results")

    with pytest.raises(ValueError, match="Unsupported model id"):
        cli._run_model_with_output_contract(model_id="unknown", request=request)
