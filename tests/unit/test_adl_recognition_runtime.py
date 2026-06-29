from dataclasses import replace
from pathlib import Path

import pytest

import egomodelkit.runtime.adl_recognition as adl_runtime
from egomodelkit.models.adl_recognition import (
    AdlRecognitionRequest,
)
from egomodelkit.runtime.adl_recognition import (
    DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    AdlRecognitionRuntimeError,
    _append_executed_command_result,
    _detic_resource_dir,
    _subclip_frame_dirs,
    build_core_run_command,
    build_detic_run_command,
    build_output_ownership_repair_command,
    ensure_core_runtime_image,
    ensure_detic_runtime_image,
    run_adl_recognition,
)
from egomodelkit.runtime.external_code import (
    DETECTRON2_PIN,
    DETIC_PIN,
    DETIC_WEIGHTS_PIN,
    EGOVIZML_PIN,
    HAND_OBJECT_DETECTOR_PIN,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
)


def _test_runtime_spec(docker_executable: str = "docker"):
    return replace(
        DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
        docker_executable = docker_executable,
        host_uid = 1000,
        host_gid = 1001,
        hand_object_contact_runtime_spec = replace(
            DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
            docker_executable = docker_executable,
        )
    )

def test_default_adl_runtime_spec_exposes_external_code_pins() -> None:
    runtime_spec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC
    
    assert runtime_spec.egovizml_repository_url == EGOVIZML_PIN.fork_repository_url
    assert runtime_spec.egovizml_commit_sha == EGOVIZML_PIN.commit_sha
    
    assert runtime_spec.detic_repository_url == DETIC_PIN.fork_repository_url
    assert runtime_spec.detic_commit_sha == DETIC_PIN.commit_sha
    
    assert runtime_spec.detectron2_repository_url == DETECTRON2_PIN.fork_repository_url
    assert runtime_spec.detectron2_commit_sha == DETECTRON2_PIN.commit_sha
    
    assert runtime_spec.detic_weights_url == DETIC_WEIGHTS_PIN.source_url
    assert runtime_spec.detic_weights_filename == DETIC_WEIGHTS_PIN.filename
    
    assert (
        runtime_spec.hand_object_contact_runtime_spec.shan_repository_url
        == HAND_OBJECT_DETECTOR_PIN.fork_repository_url
    )
    
    assert (
        runtime_spec.hand_object_contact_runtime_spec.shan_commit_sha
        == HAND_OBJECT_DETECTOR_PIN.commit_sha
    )

def test_ensure_core_runtime_image_builds_when_missing() -> None:
    calls: list[list[str]] = []
    messages: list[str] = []

    def runner(command: list[str]) -> int:
        calls.append(command)

        if command[1:3] == ["image", "inspect"]:
            return 1

        return 0

    runtime_spec = _test_runtime_spec()
    
    ensure_core_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = runner,
        progress = messages.append,
    )
    
    assert calls[0] == [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        runtime_spec.core_image_tag,
    ]
    
    build_command = calls[1]
    
    assert build_command[:2] == [
        runtime_spec.docker_executable,
        "build",
    ]
    
    assert f"EGOVIZML_REPOSITORY_URL={runtime_spec.egovizml_repository_url}" in build_command
    assert f"EGOVIZML_COMMIT_SHA={runtime_spec.egovizml_commit_sha}" in build_command
    
    assert (
        f"org.egomodelkit.provenance.code.egovizml.commit-sha={EGOVIZML_PIN.commit_sha}"
        in build_command
    )
    
    assert "Packaged adl-recognition core runtime image is ready." in messages

def test_ensure_detic_runtime_image_builds_when_missing() -> None:
    calls: list[list[str]] = []
    messages: list[str] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        
        if command[1:3] == ["image", "inspect"]:
            return 1

        return 0
    
    runtime_spec = _test_runtime_spec()
    
    ensure_detic_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = runner,
        progress = messages.append,
    )
    
    assert calls[0] == [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        runtime_spec.detic_image_tag,
    ]
    
    build_command = calls[1]
    
    assert build_command[:2] == [
        runtime_spec.docker_executable,
        "build",
    ]
    
    assert f"EGOVIZML_REPOSITORY_URL={runtime_spec.egovizml_repository_url}" in build_command
    assert f"EGOVIZML_COMMIT_SHA={runtime_spec.egovizml_commit_sha}" in build_command
    assert f"DETIC_REPOSITORY_URL={runtime_spec.detic_repository_url}" in build_command
    assert f"DETIC_COMMIT_SHA={runtime_spec.detic_commit_sha}" in build_command
    assert f"DETECTRON2_REPOSITORY_URL={runtime_spec.detectron2_repository_url}" in build_command
    assert f"DETECTRON2_COMMIT_SHA={runtime_spec.detectron2_commit_sha}" in build_command
    assert f"DETIC_WEIGHTS_URL={runtime_spec.detic_weights_url}" in build_command
    assert f"PYTORCH_VERSION={runtime_spec.pytorch_version}" in build_command
    assert f"TORCHVISION_VERSION={runtime_spec.torchvision_version}" in build_command
    assert f"DETIC_WEIGHTS_URL={DETIC_WEIGHTS_PIN.source_url}" in build_command
    assert f"DETIC_WEIGHTS_FILENAME={DETIC_WEIGHTS_PIN.filename}" in build_command
    
    assert (
        f"org.egomodelkit.provenance.code.egovizml.commit-sha={EGOVIZML_PIN.commit_sha}"
        in build_command
    )
    
    assert (
        f"org.egomodelkit.provenance.code.detic.commit-sha={DETIC_PIN.commit_sha}"
        in build_command
    )
    
    assert (
        f"org.egomodelkit.provenance.code.detectron2.commit-sha={DETECTRON2_PIN.commit_sha}"
        in build_command
    )
    
    assert (
        f"org.egomodelkit.provenance.asset.{DETIC_WEIGHTS_PIN.asset_id}.source-url="
        f"{DETIC_WEIGHTS_PIN.source_url}"
        in build_command
    )
    
    assert (
        f"org.egomodelkit.provenance.asset.{DETIC_WEIGHTS_PIN.asset_id}.download-tool="
        f"{DETIC_WEIGHTS_PIN.download_tool}"
        in build_command 
    )
    
    assert "Packaged adl-recognition Detic runtime image is ready." in messages
    
def test_build_core_command_mounts_input_and_output(tmp_path: Path) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake-video")
    
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = output_dir,
    )
    
    runtime_spec = _test_runtime_spec()
    
    command = build_core_run_command(
        request,
        stage = "extract",
        runtime_spec = runtime_spec,
    )
    
    expected_container_input = runtime_spec.container_input_dir / input_path.name
    
    assert command[:3] == [
        runtime_spec.docker_executable,
        "run",
        "--rm",
    ]
    
    assert f"{input_path.resolve()}:{expected_container_input}:ro" in command
    assert f"{output_dir.resolve()}:{runtime_spec.container_output_dir}" in command
    assert runtime_spec.core_image_tag in command
    assert command[command.index("--stage") + 1] == "extract"
    assert command[command.index("--input-path") + 1] == str(expected_container_input)

def test_build_core_run_command_mounts_directory_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    
    (input_dir / "video.mp4").write_bytes(b"fake-video")
    
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    
    request = AdlRecognitionRequest(
        input_path = input_dir,
        output_dir = output_dir,
    )
    
    runtime_spec = _test_runtime_spec()
    
    command = build_core_run_command(
        request,
        stage = "extract",
        runtime_spec = runtime_spec,
    )
    
    assert f"{input_dir.resolve()}:{runtime_spec.container_input_dir}:ro" in command
    assert command[command.index("--input-path") + 1] == str(runtime_spec.container_input_dir)

def test_build_detic_run_command_mounts_frame_and_output_dirs(tmp_path: Path) -> None:
    frame_dir = tmp_path / "video001--1"
    frame_dir.mkdir()
    
    output_dir = tmp_path / "detic"
    output_dir.mkdir()
    
    runtime_spec = _test_runtime_spec()
    
    command = build_detic_run_command(
        frame_dir = frame_dir,
        output_dir = output_dir,
        runtime_spec = runtime_spec,
    )
    
    assert command[:5] == [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--gpus",
        "all",
    ]
    
    assert f"{frame_dir.resolve()}:/workspace/input:ro" in command
    assert f"{output_dir.resolve()}:/workspace/output" in command
    assert runtime_spec.detic_image_tag in command
    assert command[command.index("--confidence-threshold") + 1] == str(
        runtime_spec.detic_confidence_threshold
    )
    
def test_run_adl_recognition_from_all_preds_skips_video_model_stages(tmp_path: Path) -> None:
    all_preds_path = tmp_path / "all_preds.pkl"
    all_preds_path.write_bytes(b"fake-pickle")
    
    output_dir = tmp_path / "results"
    
    request = AdlRecognitionRequest(
        input_path = all_preds_path,
        output_dir = output_dir,
    )
    
    runtime_spec = _test_runtime_spec()
    calls: list[list[str]] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        return 0

    commands = run_adl_recognition(
        request,
        runtime_spec = runtime_spec,
        command_runner = runner,
    )
    
    assert output_dir.is_dir()
    assert len(commands) == 2
    
    prediction_command = commands[0]
    ownership_repair_command = commands[1]
        
    assert prediction_command[:3] == [
        "docker",
        "run",
        "--rm",
    ]
    
    assert prediction_command[prediction_command.index("--stage") + 1] == "predict"
    
    assert ownership_repair_command[:5] == [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "chown",
    ]
    
    assert ownership_repair_command[
        ownership_repair_command.index("-R") + 1
    ] == "1000:1001"
    
    assert str(runtime_spec.container_output_dir) == ownership_repair_command[-1]
    
    assert not any(runtime_spec.detic_image_tag in command for command in calls)
    assert not any(
        runtime_spec.hand_object_contact_runtime_spec.image_tag in command
        for command in calls
    )

def test_run_adl_recognition_from_video_orchestrating_existing_runtimes(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake-video")

    output_dir = tmp_path / "results"

    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = output_dir,
    )
   
    runtime_spec = _test_runtime_spec()
    calls: list[list[str]] = []
   
    def runner(commands: list[str]) -> int:
        calls.append(commands)
    
        if "--stage" in commands and commands[commands.index("--stage") + 1] == "extract":
            frame_dir = (
                output_dir
                / runtime_spec.work_dir_name
                / runtime_spec.egoviz_data_dir_name
                / runtime_spec.staged_adl_dir_name
                / "subclips"
                / "video001--1"
            )
           
            frame_dir.mkdir(parents = True, exist_ok = True)
            (frame_dir / "frame_0.jpg").write_bytes(b"fake-frame")
        
        return 0

    commands = run_adl_recognition(
        request,
        runtime_spec = runtime_spec,
        command_runner = runner,
    )
    
    ownership_repair_commands = [
        command
        for command in commands
        if "--entrypoint" in command
        and command[command.index("--entrypoint") + 1] == "chown"
    ]
    
    assert len(ownership_repair_commands) == 4
    
    for command in ownership_repair_commands:
        assert command[command.index("-R") + 1] == "1000:1001"
        assert command[-1] == str(runtime_spec.container_output_dir)
    
    extract_call_index = next(
        index
        for index, command in enumerate(calls)
        if "--stage" in command
        and command[command.index("--stage") + 1] == "extract"
    )
    
    detic_inspect_index = next(
        index
        for index, command in enumerate(calls)
        if command[:3] == [
            runtime_spec.docker_executable,
            "image",
            "inspect",
        ]
        and command[3] == runtime_spec.detic_image_tag
    )
    
    assert extract_call_index < detic_inspect_index
    
    stages = [
        command[command.index("--stage") + 1]
        for command in commands
        if "--stage" in command
    ]
    
    assert stages == ["extract", "finalize"]
    
    assert any(
        runtime_spec.hand_object_contact_runtime_spec.image_tag in command
        for command in commands
    )
    
    assert any(runtime_spec.detic_image_tag in command for command in commands)

def test_run_adl_recognition_reports_stage_failure(tmp_path: Path) -> None:
    all_preds_path = tmp_path / "all_preds.pkl"
    all_preds_path.write_bytes(b"fake-pickle")
    
    request = AdlRecognitionRequest(
        input_path = all_preds_path,
        output_dir = tmp_path / "results",
    )
    
    runtime_spec = _test_runtime_spec()
    
    def runner(command: list[str]) -> int:
        if "--stage" in command:
            return 17
        
        return 0
    
    with pytest.raises(
        AdlRecognitionRuntimeError,
        match = "adl-recognition prediction failed with exit code 17",
    ):
        run_adl_recognition(
            request,
            runtime_spec = runtime_spec,
            command_runner = runner,
        )

def test_detic_patch_file_is_inside_detic_build_context() -> None:
    detic_resource_dir = _detic_resource_dir()
    
    assert (detic_resource_dir / "Dockerfile").is_file()
    assert (detic_resource_dir / "patch_egovizml_run_detic.py").is_file()

def test_ensure_core_runtime_image_skips_build_when_available() -> None:
    calls: list[list[str]] = []
    messages: list[str] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        
        return 0
    
    runtime_spec = _test_runtime_spec()
    
    ensure_core_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = runner,
        progress = messages.append,
    )
    
    assert calls == [
        [
            runtime_spec.docker_executable,
            "image",
            "inspect",
            runtime_spec.core_image_tag,
        ]
    ]
    
    assert (
        "Packaged adl-recognition core runtime image is already available."
        in messages
    )
    
def test_ensure_detic_runtime_image_reports_build_failure() -> None:
    calls: list[list[str]] = []
    
    def runner(command: list[str]) -> int:
        calls.append(command)
        
        if command[1:3] == ["image", "inspect"]:
            return 1
    
        return 19
    
    runtime_spec = _test_runtime_spec()
    
    with pytest.raises(
        AdlRecognitionRuntimeError,
        match = "adl-recognition Detic runtime image build failed with exit code 19"
    ):
        ensure_detic_runtime_image(
            runtime_spec = runtime_spec,
            command_runner = runner,
        )

    assert calls[0] == [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        runtime_spec.detic_image_tag,
    ]
    
    assert calls[1][:2] == [
        runtime_spec.docker_executable,
        "build",
    ]

def test_run_adl_recognition_reports_missing_extracted_frames(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake-video")
    
    request = AdlRecognitionRequest(
        input_path = input_path,
        output_dir = tmp_path / "results",
    )
    
    runtime_spec = _test_runtime_spec()
    
    def runner(command: list[str]) -> int:
        return 0
    
    with pytest.raises(
        AdlRecognitionRuntimeError,
        match = "ADL frame extraction completed but no subclip frame directories were found",
    ):
        run_adl_recognition(
            request,
            runtime_spec = runtime_spec,
            command_runner = runner,
        )

def test_subclip_frame_dirs_returns_only_dirs_containing_jpg_frames(
    tmp_path: Path,
) -> None:
    runtime_spec = _test_runtime_spec()
    
    subclips_dir = (
        tmp_path
        / runtime_spec.work_dir_name
        / runtime_spec.egoviz_data_dir_name
        / runtime_spec.staged_adl_dir_name
        / "subclips"
    )
    
    valid_later = subclips_dir / "video002--1"
    valid_later.mkdir(parents = True)
    (valid_later / "frame_0.JPG").write_bytes(b"fake_frame")
    
    valid_earlier = subclips_dir / "video001--1"
    valid_earlier.mkdir()
    (valid_earlier / "frame_0.jpg").write_bytes(b"fake-frame")
    
    empty_dir = subclips_dir / "empty"
    empty_dir.mkdir()
    
    no_jpg_dir = subclips_dir / "png-only"
    no_jpg_dir.mkdir()
    (no_jpg_dir / "frame_0.png").write_bytes(b"fake_frame")
    
    (subclips_dir / "root_frame.jpg").write_bytes(b"fake-frame")
    
    frame_dirs = _subclip_frame_dirs(
        tmp_path,
        runtime_spec = runtime_spec,
    )
    
    assert [path.name for path in frame_dirs] == [
        "video001--1",
        "video002--1",
    ]

def test_append_executed_command_result_ignores_empty_result() -> None:
    executed_commands: list[list[str]] = [["existing"]]
    
    _append_executed_command_result(
        executed_commands,
        [],
    )
    
    assert executed_commands == [["existing"]]

def test_append_executed_command_result_extends_nested_results() -> None:
    executed_commands: list[list[str]] = [["existing"]]
    
    _append_executed_command_result(
        executed_commands,
        [
            ["docker", "run", "first"],
            ["docker", "run", "second"]
        ],
    )
    
    assert executed_commands == [
        ["existing"],
        ["docker", "run", "first"],
        ["docker", "run", "second"],
    ]

def test_build_output_ownership_repair_command_restores_host_ownership(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    
    runtime_spec = _test_runtime_spec()
    
    command = build_output_ownership_repair_command(
        output_dir,
        runtime_spec = runtime_spec,
    )
    
    assert command == [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--entrypoint",
        "chown",
        "-v",
        f"{output_dir.resolve()}:{runtime_spec.container_output_dir}",
        runtime_spec.core_image_tag,
        "-R",
        "1000:1001",
        str(runtime_spec.container_output_dir),
    ]

def test_detic_dockerfile_configures_detectron2_import_smoke_test() -> None:
    dockerfile_text = (_detic_resource_dir() / "Dockerfile").read_text()
    
    assert "python3.8 -m pip install --no-build-isolation -e ." in dockerfile_text
    assert "PYTHONPATH" in dockerfile_text
    assert "/opt/detectron2" in dockerfile_text
    assert "/opt/Detic" in dockerfile_text
    assert "/opt/EgoVizML" in dockerfile_text
    
    assert "from detectron2.config import get_cfg" in dockerfile_text
    assert "from detectron2.layers import nms" in dockerfile_text
    assert "detectron2 import ok" in dockerfile_text
    
    assert "from detectron.config" not in dockerfile_text
    assert dockerfile_text.index("Pillow==9.5.0") < dockerfile_text.index(
        "detectron2 import ok"
    )

def test_detic_dockerfile_downloads_weights_with_gdown() -> None:
    dockerfile_text = (_detic_resource_dir() / "Dockerfile").read_text()
    
    assert "gdown==5.2.0" in dockerfile_text
    assert "gdown --fuzzy" in dockerfile_text
    assert "DETIC_WEIGHTS_FILENAME" in dockerfile_text
    assert "test -s" in dockerfile_text

def test_ensure_core_runtime_image_reports_build_failure() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> int:
        calls.append(command)

        if command[1:3] == ["image", "inspect"]:
            return 1

        return 23

    runtime_spec = _test_runtime_spec()

    with pytest.raises(
        AdlRecognitionRuntimeError,
        match = "adl-recognition core runtime image build failed with exit code 23",
    ):
        ensure_core_runtime_image(
            runtime_spec = runtime_spec,
            command_runner = runner,
        )

    assert calls[0] == [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        runtime_spec.core_image_tag,
    ]
    
    assert calls[1][:2] == [runtime_spec.docker_executable, "build"]
    
def test_ensure_core_runtime_image_uses_streaming_runner_when_building() -> None:
    messages: list[str] = []
    streamed_commands: list[list[str]] = []

    def inspect_runner(command: list[str]) -> int:
        assert command[1:3] == ["image", "inspect"]
        return 1

    def streaming_runner(command: list[str], progress) -> int:
        streamed_commands.append(command)
        progress("streamed build output")

        return 0

    runtime_spec = _test_runtime_spec()

    ensure_core_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = inspect_runner,
        streaming_command_runner = streaming_runner,
        progress = messages.append,
    )

    assert streamed_commands[0][:2] == [runtime_spec.docker_executable, "build"]
    assert "streamed build output" in messages
    assert "Packaged adl-recognition core runtime image is ready." in messages

def test_run_adl_recognition_reports_empty_extracted_frame_images(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake-video")

    output_dir = tmp_path / "results"
    request = AdlRecognitionRequest(input_path = input_path, output_dir = output_dir)
    runtime_spec = _test_runtime_spec()

    def runner(command: list[str]) -> int:
        if "--stage" in command and command[command.index("--stage") + 1] == "extract":
            frame_dir = (
                output_dir
                / runtime_spec.work_dir_name
                / runtime_spec.egoviz_data_dir_name
                / runtime_spec.staged_adl_dir_name
                / "subclips"
                / "video001--1"
            )
            
            frame_dir.mkdir(parents = True)
            (frame_dir / "frame_0.JPG").write_bytes(b"fake-frame")

        return 0

    with pytest.raises(
        AdlRecognitionRuntimeError,
        match = "ADL frame extraction completed but no extracted frame images were found",
    ):
        run_adl_recognition(
            request,
            runtime_spec = runtime_spec,
            command_runner = runner,
        )

def test_run_stage_uses_streaming_command_runner() -> None:
    messages: list[str] = []
    calls: list[list[str]] = []

    def streaming_runner(command: list[str], progress) -> int:
        calls.append(command)
        progress("streamed output")

        return 0

    adl_runtime._run_stage(
        ["docker", "run"],
        command_runner = lambda _command: 99,
        streaming_command_runner = streaming_runner,
        stage_name = "streaming stage",
        progress = messages.append,
    )

    assert calls == [["docker", "run"]]
    
    assert messages == [
        "Starting streaming stage.",
        "streamed output",
        "Finished streaming stage.",
    ]

def test_global_frame_progress_forwards_unrelated_and_offsets_matching_updates() -> None:
    messages: list[str] = []
    
    report = adl_runtime._global_frame_progress(
        messages.append,
        source_kind = "detic_frame_processed",
        offset = 20,
        total = 60,
    )

    report("plain runtime output")
    report('EGOMODELKIT_PROGRESS {"kind": "hand_object_image_processed", "current": 1}')
    report('EGOMODELKIT_PROGRESS {"kind": "detic_frame_processed", "current": "5"}')

    assert messages[:2] == [
        "plain runtime output",
        'EGOMODELKIT_PROGRESS {"kind": "hand_object_image_processed", "current": 1}',
    ]
    
    assert messages[2] == (
        'EGOMODELKIT_PROGRESS {"current": 25, '
        '"kind": "detic_frame_processed", "total": 60}'
    )
    
def test_adl_payload_int_covers_float_string_and_missing_values() -> None:
    payload = {
        "integer": 3,
        "floating": 4.9,
        "digits": "5",
        "letters": "five",
    }

    assert adl_runtime._payload_int(payload, "integer") == 3
    assert adl_runtime._payload_int(payload, "floating") == 4
    assert adl_runtime._payload_int(payload, "digits") == 5
    assert adl_runtime._payload_int(payload, "letters") == 0
    assert adl_runtime._payload_int(payload, "missing") == 0
