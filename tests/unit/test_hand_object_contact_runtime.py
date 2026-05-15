import subprocess
from pathlib import Path

import pytest

import egomodelkit.runtime.hand_object_contact as runtime_module
from egomodelkit.models.hand_object_contact import (
    HandObjectContactRequest,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_RUNTIME_SPEC,
    HandObjectContactRuntimeError,
    _subprocess_runner,
    build_run_command,
    ensure_runtime_image,
    run_hand_object_contact,
)


def test_ensure_runtime_image_skips_build_when_image_exists() -> None:
    calls: list[list[str]] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        return 0

    ensure_runtime_image(command_runner = runner)

    assert calls == [
        [
            DEFAULT_RUNTIME_SPEC.docker_executable,
            "image",
            "inspect",
            DEFAULT_RUNTIME_SPEC.image_tag,
        ],
    ]
    
def test_ensure_runtime_image_builds_when_image_is_missing() -> None:
    calls: list[list[str]] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        
        if command[1:3] == ["image", "inspect"]:
            return 1

        return 0

    ensure_runtime_image(command_runner = runner)
    
    inspect_command = calls[0]
    build_command = calls[1]
    
    assert inspect_command == [
        DEFAULT_RUNTIME_SPEC.docker_executable,
        "image",
        "inspect",
        DEFAULT_RUNTIME_SPEC.image_tag,
    ]
    
    assert build_command[:2] == [
        DEFAULT_RUNTIME_SPEC.docker_executable,
        "build"
    ]
    
    assert (
        f"SHAN_COMMIT_SHA={DEFAULT_RUNTIME_SPEC.shan_commit_sha}"
        in build_command
    )
    
    assert (
        f"CHECKPOINT_STEP={DEFAULT_RUNTIME_SPEC.checkpoint_step}"
        in build_command
    )
    
    assert (
        f"CHECKPOINT_FILENAME={DEFAULT_RUNTIME_SPEC.checkpoint_filename}"
        in build_command
    )

def test_ensure_runtime_image_reports_build_failure() -> None:
    def runner(command: list[str]) -> int:
        if command[1:3] == ["image", "inspect"]:
            return 1
        
        return 17

    with pytest.raises(
        HandObjectContactRuntimeError,
        match = "runtime image build failed",
    ):
        ensure_runtime_image(command_runner = runner)

def test_build_run_command_mounts_input_and_output(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image")
    
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    
    request = HandObjectContactRequest(
        input_path = image_path,
        output_dir = output_dir,
    )
    
    command = build_run_command(request)
    
    expected_container_input = str(
        DEFAULT_RUNTIME_SPEC.container_input_dir / image_path.name
    )
    
    expected_container_output = str(
        DEFAULT_RUNTIME_SPEC.container_output_dir
    )
    
    assert command[:5] == [
        DEFAULT_RUNTIME_SPEC.docker_executable,
        "run",
        "--rm",
        "--gpus",
        "all",
    ]
    
    assert DEFAULT_RUNTIME_SPEC.image_tag in command
    
    assert (
        f"{tmp_path.resolve()}:{DEFAULT_RUNTIME_SPEC.container_input_dir}:ro"
        in command
    )
    
    assert (
        f"{output_dir.resolve()}:{DEFAULT_RUNTIME_SPEC.container_output_dir}"
        in command
    )
    
    assert command[-4:] == [
        "--input-path",
        expected_container_input,
        "--output-dir",
        expected_container_output,
    ]

def test_run_hand_object_contact_executes_hidden_runtime(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    
    output_dir = tmp_path / "results"
    
    request = HandObjectContactRequest(
        input_path = image_path,
        output_dir = output_dir,
    )
    
    calls: list[list[str]] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        return 0
    
    command = run_hand_object_contact(
        request,
        command_runner = runner,
    )
    
    assert output_dir.is_dir()
    assert command[0] == DEFAULT_RUNTIME_SPEC.docker_executable
    
    assert calls[0] == [
        DEFAULT_RUNTIME_SPEC.docker_executable,
        "image",
        "inspect",
        DEFAULT_RUNTIME_SPEC.image_tag,
    ]
    
    assert calls[1] == command

def test_run_hand_object_contact_reports_runtime_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    
    request = HandObjectContactRequest(
        input_path = image_path,
        output_dir = tmp_path / "results",
    )
    
    def runner(command: list[str]) -> int:
        if command[1:3] == ["image", "inspect"]:
            return 0

        return 17

    with pytest.raises(
        HandObjectContactRuntimeError,
        match="inference runtime failed"
    ):
        run_hand_object_contact(
            request,
            command_runner = runner,
        )

def test_subprocess_runner_returns_subprocess_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ["docker", "image", "inspect", "egomodelkit-hand-object-contact:dev"]
    
    def fake_run(
        received_command: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert received_command == command
        assert check is False
        
        return subprocess.CompletedProcess(
            args = received_command,
            returncode = 17
        )

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    
    assert _subprocess_runner(command) == 17