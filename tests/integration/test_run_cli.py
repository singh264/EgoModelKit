from pathlib import Path

import pytest
from typer.testing import CliRunner

from egomodelkit.cli import CLI_RUNTIME_ERROR_EXIT_CODE, CLI_UNSUPPORTED_MODEL_EXIT_CODE, app
from egomodelkit.models.hand_object_contact import DRY_RUN_VALIDATION_MESSAGE

runner = CliRunner()

def test_run_dry_run_accepts_hand_object_contact(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    
    output_dir = tmp_path / "results"
    
    result = runner.invoke(
        app,
        [
            "run",
            "hand-object-contact",
            "--input",
            str(image_path),
            "--output",
            str(output_dir),
            "--dry-run",
        ],
    )
    
    assert result.exit_code == 0
    assert DRY_RUN_VALIDATION_MESSAGE in result.output
    assert str(image_path) in result.output
    assert str(output_dir) in result.output

def test_run_rejects_unknown_model(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    
    result = runner.invoke(
        app,
        [
            "run",
            "unknown-model",
            "--input",
            str(image_path),
            "--output",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )
    
    assert result.exit_code == CLI_UNSUPPORTED_MODEL_EXIT_CODE
    assert "Unsupported model" in result.output
    
def test_run_rejects_unsupported_input_image_suffix(tmp_path: Path) -> None:
    unsupported_input = tmp_path / "frame.gif"
    unsupported_input.write_text("not a supported image suffix")
        
    result = runner.invoke(
        app,
        [
            "run",
            "hand-object-contact",
            "--input",
            str(unsupported_input),
            "--output",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )
    
    assert result.exit_code == CLI_RUNTIME_ERROR_EXIT_CODE
    assert "Error:" in result.output
    assert ".gif" in result.output

def test_run_executes_hand_object_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    
    output_dir = tmp_path / "results"
    
    captured: dict[str, object] = {}
    
    def fake_run_hand_object_contact(request) -> list[str]:
        captured["request"] = request
        
        return ["docker", "run"]

    monkeypatch.setattr(
        "egomodelkit.cli.run_hand_object_contact",
        fake_run_hand_object_contact,
    )
    
    result = runner.invoke(
        app,
        [
          "run",
          "hand-object-contact",
          "--input",
          str(image_path),
          "--output",
          str(output_dir)
        ],
    )
    
    assert result.exit_code == 0
    assert "Completed: hand-object-contact" in result.output
    assert "request" in captured
