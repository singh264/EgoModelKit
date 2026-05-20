import subprocess
from pathlib import Path

import pytest

import egomodelkit.runtime.hand_object_contact as runtime_module
from egomodelkit.models.hand_object_contact import (
    HandObjectContactRequest,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_RUNTIME_SPEC,
    SHAN_FORK_COMMIT_SHA,
    SHAN_FORK_REPOSITORY_URL,
    HandObjectContactRuntimeError,
    _subprocess_runner,
    build_run_command,
    ensure_runtime_image,
    run_hand_object_contact,
)


def test_ensure_runtime_image_skips_build_when_image_exists() -> None:
    calls: list[list[str]] = []
    messages: list[str] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        return 0

    ensure_runtime_image(
        command_runner = runner,
        progress = messages.append,
    )

    assert calls == [
        [
            DEFAULT_RUNTIME_SPEC.docker_executable,
            "image",
            "inspect",
            DEFAULT_RUNTIME_SPEC.image_tag,
        ],
    ]
    
    assert "Checking packaged Shan runtime image." in messages
    assert "Packaged Shan runtime image is already available." in messages

def test_ensure_runtime_image_builds_when_image_is_missing() -> None:
    calls: list[list[str]] = []
    messages: list[str] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        
        if command[1:3] == ["image", "inspect"]:
            return 1

        return 0

    ensure_runtime_image(
        command_runner = runner,
        progress = messages.append,
    )
    
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
    
    assert (
        f"SHAN_REPOSITORY_URL={SHAN_FORK_REPOSITORY_URL}"
        in build_command
    )
    
    assert (
        f"SHAN_COMMIT_SHA={SHAN_FORK_COMMIT_SHA}"
        in build_command
    )

    assert any("preparing it now" in message for message in messages)
    assert "Packaged Shan runtime image is ready." in messages

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
        f"{image_path.resolve()}:{expected_container_input}:ro"
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
        "version",
        "--format",
        "{{.Server.Version}}",
    ]

    assert calls[1] == [
        DEFAULT_RUNTIME_SPEC.docker_executable,
        "image",
        "inspect",
        DEFAULT_RUNTIME_SPEC.image_tag,
    ]
    
    assert calls[2] == command

def test_run_hand_object_contact_reports_runtime_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    
    request = HandObjectContactRequest(
        input_path = image_path,
        output_dir = tmp_path / "results",
    )
    
    def runner(command: list[str]) -> int:
        if command[1:3] == ["version", "--format"]:
            return 0

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

def test_run_hand_object_contact_reports_progress_messages(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")

    request = HandObjectContactRequest(
        input_path = image_path,
        output_dir = tmp_path / "results",
    )

    messages: list[str] = []

    run_hand_object_contact(
        request,
        command_runner = lambda command: 0,
        progress = messages.append,
    )

    assert "Validating hand-object-contact request." in messages
    assert "Checking host runtime prerequisites." in messages
    assert any(message.startswith("Python ") for message in messages)
    assert "Docker daemon is available." in messages
    assert any(message.startswith("Using output directory:") for message in messages)
    assert "Checking packaged Shan runtime image." in messages
    assert "Starting Shan hand-object-contact inference." in messages
    assert "Shan hand-object-contact inference completed." in messages

def test_runtime_spec_uses_pinned_shan_fork() -> None:
    assert DEFAULT_RUNTIME_SPEC.shan_repository_url == SHAN_FORK_REPOSITORY_URL
    assert DEFAULT_RUNTIME_SPEC.shan_commit_sha == SHAN_FORK_COMMIT_SHA

def test_build_run_command_supports_directory_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    
    (input_dir / "frame_001.jpg").write_bytes(b"fake-image")
    
    request = HandObjectContactRequest(
        input_path = input_dir,
        output_dir = tmp_path / "results",
    )
    
    command = build_run_command(request)
        
    assert (
        f"{input_dir.resolve()}:{DEFAULT_RUNTIME_SPEC.container_input_dir}:ro"
        in command
    )
    
    input_arg_index = command.index("--input-path")
    
    assert command[input_arg_index + 1] == str(
        DEFAULT_RUNTIME_SPEC.container_input_dir
    )
