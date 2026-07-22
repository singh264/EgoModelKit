from dataclasses import replace
from pathlib import Path

import pytest

import egomodelkit.runtime.hand_interaction as runtime
from egomodelkit.models.hand_interaction import HandInteractionRequest
from egomodelkit.progress import external_progress_line
from egomodelkit.runtime.hand_interaction import (
    DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
    HandInteractionRuntimeError,
    build_extract_run_command,
    build_output_ownership_repair_command,
    ensure_runtime_image,
    hand_interaction_image_identity,
    run_hand_interaction,
)


def _request(
    tmp_path: Path,
    *,
    directory: bool = False,
    dominant_hand: str = "right",
) -> HandInteractionRequest:
    if directory:
        input_path = tmp_path / "videos"
        input_path.mkdir()
        (input_path / "clip.mp4").write_bytes(b"video")
    else:
        input_path = tmp_path / "clip.mp4"
        input_path.write_bytes(b"video")
    return HandInteractionRequest(
        input_path=input_path,
        output_dir=tmp_path / "results",
        dominant_hand=dominant_hand,  # type: ignore[arg-type]
    )


def test_image_identity_is_content_addressed() -> None:
    identity = hand_interaction_image_identity()
    assert identity.repository == "egomodelkit-hand-interaction"
    assert identity.runtime_name == "hand-interaction"
    assert identity.tag == DEFAULT_HAND_INTERACTION_RUNTIME_SPEC.image_tag
    assert identity.fingerprint[:12] in identity.tag


def test_ensure_runtime_image_skips_existing_image() -> None:
    calls: list[list[str]] = []
    messages: list[str] = []
    ensure_runtime_image(
        command_runner=lambda command: calls.append(command) or 0,
        progress=messages.append,
    )
    assert calls == [
        ["docker", "image", "inspect", DEFAULT_HAND_INTERACTION_RUNTIME_SPEC.image_tag]
    ]
    assert messages[-1] == "Packaged hand-interaction runtime image is already available."


def test_ensure_runtime_image_builds_and_removes_stale_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    stale_calls: list[dict[str, object]] = []
    messages: list[str] = []

    def command_runner(command: list[str]) -> int:
        calls.append(command)
        return 1 if command[1:3] == ["image", "inspect"] else 0

    monkeypatch.setattr(
        runtime,
        "remove_stale_runtime_images",
        lambda **kwargs: stale_calls.append(kwargs),
    )
    ensure_runtime_image(command_runner=command_runner, progress=messages.append)

    assert calls[1][:2] == ["docker", "build"]
    assert "Dockerfile" in calls[1][3]
    assert DEFAULT_HAND_INTERACTION_RUNTIME_SPEC.image_tag in calls[1]
    assert stale_calls[0]["docker_executable"] == "docker"
    assert messages[-1] == "Packaged hand-interaction runtime image is ready."


def test_ensure_runtime_image_uses_streaming_build_and_reports_failure() -> None:
    streamed: list[list[str]] = []

    def stream(command: list[str], progress) -> int:
        streamed.append(command)
        progress("build output")
        return 17

    with pytest.raises(HandInteractionRuntimeError, match="image build failed with exit code 17"):
        ensure_runtime_image(
            command_runner=lambda _command: 1,
            streaming_command_runner=stream,
        )
    assert streamed[0][:2] == ["docker", "build"]


def test_extract_and_ownership_commands_include_bandini_configuration(tmp_path: Path) -> None:
    request = _request(tmp_path, dominant_hand="left")
    spec = replace(
        DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
        video_processing_config=replace(
            DEFAULT_HAND_INTERACTION_RUNTIME_SPEC.video_processing_config,
            dominant_hand="left",
        ),
    )
    command = build_extract_run_command(request, runtime_spec=spec)

    assert "--gpus" not in command
    assert "--processing-fps" in command
    assert command[command.index("--processing-fps") + 1] == "30"
    assert command[command.index("--resize-width") + 1] == "720"
    assert command[command.index("--resize-height") + 1] == "405"
    assert command[command.index("--pooling-window-seconds") + 1] == "1.0"
    assert command[command.index("--interaction-contact-state-threshold") + 1] == "3"
    assert command[command.index("--dominant-hand") + 1] == "left"
    assert f"{request.input_path.resolve()}:/workspace/input/clip.mp4:ro" in command

    repair = build_output_ownership_repair_command(request.output_dir, runtime_spec=spec)
    assert repair[:4] == ["docker", "run", "--rm", "--entrypoint"]
    assert repair[-3:] == ["-R", f"{spec.host_uid}:{spec.host_gid}", "/workspace/output"]


def test_extract_command_supports_directory_input(tmp_path: Path) -> None:
    request = _request(tmp_path, directory=True)
    command = build_extract_run_command(request)
    assert f"{request.input_path.resolve()}:/workspace/input:ro" in command
    assert command[command.index("--input-path") + 1] == "/workspace/input"


def test_run_hand_interaction_extracts_then_reuses_hoc_for_each_subclip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, dominant_hand="left")
    messages: list[str] = []
    hoc_requests: list[object] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(runtime, "ensure_host_runtime_ready", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "ensure_runtime_image", lambda **_kwargs: None)

    def command_runner(command: list[str]) -> int:
        commands.append(command)
        if "--work-dir-name" in command:
            frames = request.output_dir / "hand_interaction_work" / "extracted_frames"
            for name, count in (("video001--1", 2), ("video001--2", 1)):
                clip = frames / name
                clip.mkdir(parents=True)
                for index in range(count):
                    (clip / f"frame_{index:06d}.jpg").write_bytes(b"image")
        return 0

    def fake_hoc(hoc_request, **kwargs):
        hoc_requests.append(hoc_request)
        progress = kwargs["progress"]
        progress(external_progress_line("hand_object_image_processed", current=1, total=2))
        return ["docker", "run", "hoc"]

    monkeypatch.setattr(runtime, "run_hand_object_contact", fake_hoc)
    executed = run_hand_interaction(
        request,
        command_runner=command_runner,
        progress=messages.append,
    )

    assert [item.input_path.name for item in hoc_requests] == ["video001--1", "video001--2"]
    assert [item.output_dir.name for item in hoc_requests] == ["video001--1", "video001--2"]
    assert all(item.output_dir.parent.name == "shan_outputs" for item in hoc_requests)
    assert len(executed) == 6
    assert all("detic" not in " ".join(command).lower() for command in executed)
    assert all("process_all_preds" not in " ".join(command) for command in executed)
    assert any('"kind": "hand_interaction_hoc_frame_processed"' in message for message in messages)
    assert messages[-1] == "hand-interaction inference completed."
    extraction = executed[0]
    assert extraction[extraction.index("--dominant-hand") + 1] == "left"


def test_run_hand_interaction_uses_streaming_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    streamed: list[list[str]] = []
    monkeypatch.setattr(runtime, "ensure_host_runtime_ready", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "ensure_runtime_image", lambda **_kwargs: None)

    def streaming(command: list[str], _progress) -> int:
        streamed.append(command)
        if "--work-dir-name" in command:
            clip = request.output_dir / "hand_interaction_work" / "extracted_frames" / "video001--1"
            clip.mkdir(parents=True)
            (clip / "frame.jpg").write_bytes(b"image")
        return 0

    monkeypatch.setattr(runtime, "run_hand_object_contact", lambda *_args, **_kwargs: [])
    run_hand_interaction(request, streaming_command_runner=streaming)
    assert len(streamed) == 3


def test_run_hand_interaction_reports_extraction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(runtime, "ensure_host_runtime_ready", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "ensure_runtime_image", lambda **_kwargs: None)

    with pytest.raises(
        HandInteractionRuntimeError,
        match="frame extraction failed with exit code 9",
    ):
        run_hand_interaction(request, command_runner=lambda _command: 9)


def test_run_hand_interaction_requires_frame_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(runtime, "ensure_host_runtime_ready", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "ensure_runtime_image", lambda **_kwargs: None)

    with pytest.raises(HandInteractionRuntimeError, match="no frame directories"):
        run_hand_interaction(request, command_runner=lambda _command: 0)


def test_run_hand_interaction_requires_frame_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(runtime, "ensure_host_runtime_ready", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "ensure_runtime_image", lambda **_kwargs: None)
    frames = request.output_dir / "hand_interaction_work" / "extracted_frames" / "video001--1"
    frames.mkdir(parents=True)
    (frames / "not-a-frame.txt").write_text("x", encoding="utf-8")

    with pytest.raises(HandInteractionRuntimeError, match="no frame directories"):
        run_hand_interaction(request, command_runner=lambda _command: 0)


def test_progress_adapter_forwards_plain_messages_and_parses_payloads() -> None:
    messages: list[str] = []
    reporter = runtime._global_frame_progress(messages.append, offset=10, total=20)
    reporter("plain HOC message")
    reporter(external_progress_line("hand_object_image_processed", current="2", total=3))
    reporter(external_progress_line("hand_object_image_processed", current=2.9, total=3))
    reporter(external_progress_line("hand_object_image_processed", current=None, total=3))

    assert messages[0] == "plain HOC message"
    assert '"current": 12' in messages[1]
    assert '"current": 12' in messages[2]
    assert '"current": 10' in messages[3]


def test_append_executed_command_result_handles_empty_flat_and_nested() -> None:
    commands: list[list[str]] = []
    runtime._append_executed_command_result(commands, [])
    runtime._append_executed_command_result(commands, ["docker", "run"])
    runtime._append_executed_command_result(commands, [["one"], ["two"]])
    assert commands == [["docker", "run"], ["one"], ["two"]]


def test_runtime_check_overrides_and_frame_helpers(tmp_path: Path) -> None:
    def locator(_name: str) -> str:
        return "/docker"

    def detector() -> str:
        return "Linux"

    assert runtime._runtime_check_overrides(
        executable_locator=locator,
        platform_detector=detector,
    ) == {"executable_locator": locator, "platform_detector": detector}
    assert runtime._runtime_check_overrides(
        executable_locator=None,
        platform_detector=None,
    ) == {}

    output = tmp_path / "out"
    extracted = output / "hand_interaction_work" / "extracted_frames"
    invalid = extracted / "invalid"
    valid = extracted / "valid"
    invalid.mkdir(parents=True)
    valid.mkdir()
    (invalid / "frame.png").write_bytes(b"x")
    (valid / "frame.JPG").write_bytes(b"x")
    assert runtime._subclip_frame_dirs(
        output,
        runtime_spec=DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
    ) == [valid]
    assert runtime._frame_count(valid) == 1
    assert runtime._shan_output_dir(
        output,
        "clip",
        runtime_spec=DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
    ).name == "clip"


def test_run_stage_streaming_failure() -> None:
    with pytest.raises(HandInteractionRuntimeError, match="custom failed with exit code 4"):
        runtime._run_stage(
            ["command"],
            command_runner=lambda _command: 0,
            streaming_command_runner=lambda _command, _progress: 4,
            stage_name="custom",
            progress=lambda _message: None,
        )


def test_run_hand_interaction_rejects_empty_discovered_frame_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    empty_frames = tmp_path / "empty-frames"
    empty_frames.mkdir()
    monkeypatch.setattr(runtime, "ensure_host_runtime_ready", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "ensure_runtime_image", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_subclip_frame_dirs", lambda *_args, **_kwargs: [empty_frames])

    with pytest.raises(HandInteractionRuntimeError, match="no frame images"):
        run_hand_interaction(request, command_runner=lambda _command: 0)
