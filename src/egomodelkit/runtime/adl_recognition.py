""" Hidden runtime execution for ADL recognition inference. """

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from egomodelkit.models.adl_recognition import (
    COMBINED_PREDS_FILENAME,
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)
from egomodelkit.models.hand_object_contact import HandObjectContactRequest
from egomodelkit.runtime.commands import subprocess_runner
from egomodelkit.runtime.external_code import (
    DETECTRON2_PIN,
    DETIC_PIN,
    DETIC_WEIGHTS_PIN,
    EGOVIZML_PIN,
    docker_asset_label_arguments,
    docker_code_label_arguments,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    HandObjectContactRuntimeSpec,
    run_hand_object_contact,
)
from egomodelkit.runtime.preflight import (
    ProgressReporter,
    ensure_host_runtime_ready,
)

CommandRunner = Callable[[list[str]], int]
AdlRecognitionStage = Literal["extract", "predict", "finalize"]

EGOVIZML_REPOSITORY_URL: Final[str] = EGOVIZML_PIN.fork_repository_url
DETIC_REPOSITORY_URL: Final[str] = DETIC_PIN.fork_repository_url
DETECTRON2_REPOSITORY_URL: Final[str] = DETECTRON2_PIN.fork_repository_url

EGOVIZML_COMMIT_SHA: Final[str] = EGOVIZML_PIN.commit_sha
DETIC_COMMIT_SHA: Final[str] = DETIC_PIN.commit_sha
DETECTRON2_COMMIT_SHA: Final[str] = DETECTRON2_PIN.commit_sha

DETIC_WEIGHTS_URL: Final[str] = DETIC_WEIGHTS_PIN.source_url
DETIC_WEIGHTS_FILENAME: Final[str] = DETIC_WEIGHTS_PIN.filename

@dataclass(frozen = True, slots = True)
class AdlRecognitionRuntimeSpec:
    """ Build and execution settings for the hidden adl-recognition runtime. """

    docker_executable: str
    core_image_tag: str
    detic_image_tag: str
    container_input_dir: PurePosixPath
    container_output_dir: PurePosixPath
    work_dir_name: str
    egoviz_data_dir_name: str
    staged_adl_dir_name: str
    
    egovizml_repository_url: str
    egovizml_commit_sha: str
    detic_repository_url: str
    detic_commit_sha: str
    detectron2_repository_url: str
    detectron2_commit_sha: str
    detic_weights_url: str
    detic_weights_filename: str
    
    pytorch_version: str
    torchvision_version: str
    torchaudio_version: str
    pytorch_cuda_index_url: str
    
    detic_confidence_threshold: float
    detic_num_workers: int
    subclip_length_seconds: int
    subclip_fps: int
    frame_fps: int
    active_iou: float
    
    hand_object_contact_runtime_spec: HandObjectContactRuntimeSpec
    
    host_uid: int
    host_gid: int

DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC: Final[AdlRecognitionRuntimeSpec] = (
    AdlRecognitionRuntimeSpec(
        docker_executable = "docker",
        core_image_tag = "egomodelkit-adl-recognition-core:dev",
        detic_image_tag = "egomodelkit-adl-recognition-detic:dev",
        container_input_dir = PurePosixPath("/workspace/input"),
        container_output_dir = PurePosixPath("/workspace/output"),
        work_dir_name = "adl_recognition_work",
        egoviz_data_dir_name = "egoviz_data",
        # process_all_preds.py currently loops over EgoVizML's known ADL folders.
        # This folder is a staging bucket for inference, not a ground-truth label.
        staged_adl_dir_name = "meal-preparation-cleanup",
        egovizml_repository_url = EGOVIZML_REPOSITORY_URL,
        egovizml_commit_sha = EGOVIZML_COMMIT_SHA,
        detic_repository_url = DETIC_REPOSITORY_URL,
        detic_commit_sha = DETIC_COMMIT_SHA,
        detectron2_repository_url = DETECTRON2_REPOSITORY_URL,
        detectron2_commit_sha = DETECTRON2_COMMIT_SHA,
        detic_weights_url = DETIC_WEIGHTS_URL,
        detic_weights_filename = DETIC_WEIGHTS_FILENAME,
        pytorch_version = "1.10.0+cu113",
        torchvision_version = "0.11.1+cu113",
        torchaudio_version = "0.10.0+cu113",
        pytorch_cuda_index_url = "https://download.pytorch.org/whl/cu113",
        detic_confidence_threshold = 0.3,
        detic_num_workers = 1,
        subclip_length_seconds = 10,
        subclip_fps = 10,
        frame_fps = 2,
        active_iou = 0.75,
        hand_object_contact_runtime_spec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
        host_uid = os.getuid(),
        host_gid = os.getgid(),
    )
)

class AdlRecognitionRuntimeError(RuntimeError):
    """ Raised when ADL recognition runtime execution fails. """

def _ignore_progress(_: str) -> None:
    """ Default no-op progress reporter. """

def _container_resource_dir() -> Path:
    return Path(
        str(
            files("egomodelkit").joinpath(
                "resources/containers/adl_recognition"
            )
        )
    )

def _detic_resource_dir() -> Path:
    return _container_resource_dir() / "detic"

def _core_docker_build_arguments(runtime_spec: AdlRecognitionRuntimeSpec) -> list[str]:
    return [
        *docker_code_label_arguments(EGOVIZML_PIN),
        "--build-arg",
        f"EGOVIZML_REPOSITORY_URL={runtime_spec.egovizml_repository_url}",
        "--build-arg",
        f"EGOVIZML_COMMIT_SHA={runtime_spec.egovizml_commit_sha}",
    ]

def _detic_docker_build_arguments(runtime_spec: AdlRecognitionRuntimeSpec) -> list[str]:
    return [
        *docker_code_label_arguments(EGOVIZML_PIN, DETIC_PIN, DETECTRON2_PIN),
        *docker_asset_label_arguments(DETIC_WEIGHTS_PIN),
        "--build-arg",
        f"EGOVIZML_REPOSITORY_URL={runtime_spec.egovizml_repository_url}",
        "--build-arg",
        f"EGOVIZML_COMMIT_SHA={runtime_spec.egovizml_commit_sha}",
        "--build-arg",
        f"DETIC_REPOSITORY_URL={runtime_spec.detic_repository_url}",
        "--build-arg",
        f"DETIC_COMMIT_SHA={runtime_spec.detic_commit_sha}",
        "--build-arg",
        f"DETECTRON2_REPOSITORY_URL={runtime_spec.detectron2_repository_url}",
        "--build-arg",
        f"DETECTRON2_COMMIT_SHA={runtime_spec.detectron2_commit_sha}",
        "--build-arg",
        f"DETIC_WEIGHTS_URL={runtime_spec.detic_weights_url}",
        "--build-arg",
        f"DETIC_WEIGHTS_FILENAME={runtime_spec.detic_weights_filename}",
        "--build-arg",
        f"PYTORCH_VERSION={runtime_spec.pytorch_version}",
        "--build-arg",
        f"TORCHVISION_VERSION={runtime_spec.torchvision_version}",
        "--build-arg",
        f"TORCHAUDIO_VERSION={runtime_spec.torchaudio_version}",
        "--build-arg",
        f"PYTORCH_CUDA_INDEX_URL={runtime_spec.pytorch_cuda_index_url}",
    ]

def ensure_core_runtime_image(
    *,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """ Build the ADL core image only when it is missing. """
    _ensure_runtime_image(
        image_tag = runtime_spec.core_image_tag,
        dockerfile_path = _container_resource_dir() / "Dockerfile",
        context_dir = _container_resource_dir(),
        build_arguments = _core_docker_build_arguments(runtime_spec),
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress,
        runtime_name = "adl-recognition core",
    )

def ensure_detic_runtime_image(
    *,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """ Build the ADL Detic image only when it is missing. """
    _ensure_runtime_image(
        image_tag = runtime_spec.detic_image_tag,
        dockerfile_path = _detic_resource_dir() / "Dockerfile",
        context_dir = _detic_resource_dir(),
        build_arguments = _detic_docker_build_arguments(runtime_spec),
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress,
        runtime_name = "adl-recognition Detic",
    )

def _ensure_runtime_image(
    *,
    image_tag: str,
    dockerfile_path: Path,
    context_dir: Path,
    build_arguments: list[str],
    runtime_spec: AdlRecognitionRuntimeSpec,
    command_runner: CommandRunner,
    progress: ProgressReporter,
    runtime_name: str,
) -> None:
    progress(f"Checking packaged {runtime_name} runtime image.")
    
    inspect_command = [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        image_tag,
    ]
    
    if command_runner(inspect_command) == 0:
        progress(f"Packaged {runtime_name} runtime image is already available.")
        return

    progress(
        f"Packaged {runtime_name} runtime image is missing; preparing it now. "
        "The first run may take longer."
    )
    
    build_command = [
        runtime_spec.docker_executable,
        "build",
        "-f",
        str(dockerfile_path),
        "-t",
        image_tag,
        *build_arguments,
        str(context_dir),
    ]
    
    exit_code = command_runner(build_command)
    
    if exit_code != 0:
        raise AdlRecognitionRuntimeError(
            f"{runtime_name} runtime image build failed with exit code {exit_code}."
        )
    
    progress(f"Packaged {runtime_name} runtime image is ready.")

def build_core_run_command(
    request: AdlRecognitionRequest,
    *,
    stage: AdlRecognitionStage,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
) -> list[str]:
    """ Build a Docker command for one ADL core runtime stage. """
    output_dir = request.output_dir.resolve()
    
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "-v",
        _input_mount_argument(request, runtime_spec = runtime_spec),
        "-v",
        f"{output_dir}:{runtime_spec.container_output_dir}",
        runtime_spec.core_image_tag,
        "--stage",
        stage,
        "--input-path",
        str(_container_input_path(request, runtime_spec = runtime_spec)),
        "--output-dir",
        str(runtime_spec.container_output_dir),
        "--work-dir-name",
        runtime_spec.work_dir_name,
        "--egoviz-data-dir-name",
        runtime_spec.egoviz_data_dir_name,
        "--adl-dir-name",
        runtime_spec.staged_adl_dir_name,
        "--subclip-length",
        str(runtime_spec.subclip_length_seconds),
        "--fps",
        str(runtime_spec.subclip_fps),
        "--frame-fps",
        str(runtime_spec.frame_fps),
        "--active-iou",
        str(runtime_spec.active_iou),
    ]

def build_detic_run_command(
    *,
    frame_dir: Path,
    output_dir: Path,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
) -> list[str]:
    """ Build a Docker command for Detic inference over one frame directory. """
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{frame_dir.resolve()}:/workspace/input:ro",
        "-v",
        f"{output_dir.resolve()}:/workspace/output",
        runtime_spec.detic_image_tag,
        "--input-dir",
        "/workspace/input",
        "--output-dir",
        "/workspace/output",
        "--no-images",
        "--confidence-threshold",
        str(runtime_spec.detic_confidence_threshold),
        "--num-workers",
        str(runtime_spec.detic_num_workers),
    ]

def build_output_ownership_repair_command(
    output_dir: Path,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
) -> list[str]:
    """ Build a Docker command that restores host ownership of mounted output. """
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--entrypoint",
        "chown",
        "-v",
        f"{output_dir.resolve()}:{runtime_spec.container_output_dir}",
        runtime_spec.core_image_tag,
        "-R",
        f"{runtime_spec.host_uid}:{runtime_spec.host_gid}",
        str(runtime_spec.container_output_dir),
    ]

def _repair_output_ownership(
    output_dir: Path,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
    command_runner: CommandRunner,
    progress: ProgressReporter,
) -> list[str]:
    command = build_output_ownership_repair_command(
        output_dir,
        runtime_spec = runtime_spec,
    )
    
    _run_stage(
        command,
        command_runner = command_runner,
        stage_name = "ADL output ownership repair",
        progress = progress,
    )
    
    return command

def run_adl_recognition(
    request: AdlRecognitionRequest,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> list[list[str]]:
    """ Run ADL recognition behind EgoModelKit's run command. """
    progress("Validating adl-recognition request.")
    validate_adl_recognition_request(request)
    
    ensure_host_runtime_ready(
        docker_executable = runtime_spec.docker_executable,
        command_runner = command_runner,
        progress = progress,
    )
    
    request.output_dir.mkdir(parents = True, exist_ok = True)
    progress(f"Using output directory: {request.output_dir}")
    
    ensure_core_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress,
    )
    
    executed_commands: list[list[str]] = []
    
    if _is_combined_predictions_file(request.input_path):
        command = build_core_run_command(
            request,
            stage = "predict",
            runtime_spec = runtime_spec,
        )

        _run_stage(
            command,
            command_runner = command_runner,
            stage_name = "adl-recognition prediction",
            progress = progress,
        )
        
        executed_commands.append(command)
        
        ownership_repair_command = _repair_output_ownership(
            request.output_dir,
            runtime_spec = runtime_spec,
            command_runner = command_runner,
            progress = progress,
        )
        
        executed_commands.append(ownership_repair_command)
        
        progress("adl-recognition inference completed.")

        return executed_commands
    
    extract_command = build_core_run_command(
        request,
        stage = "extract",
        runtime_spec = runtime_spec,
    )
    
    _run_stage(
        extract_command,
        command_runner = command_runner,
        stage_name = "ADL video frame extraction",
        progress = progress,
    )
    
    executed_commands.append(extract_command)
    
    ownership_repair_command = _repair_output_ownership(
        request.output_dir,
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress,
    )
    
    executed_commands.append(ownership_repair_command)
    
    frame_dirs = _subclip_frame_dirs(
        request.output_dir,
        runtime_spec = runtime_spec,
    )
    
    if not frame_dirs:
        raise AdlRecognitionRuntimeError(
            "ADL frame extraction completed but no subclip frame directories were found."
        )
    
    ensure_detic_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress,
    )
    
    for frame_dir in frame_dirs:
        shan_output_dir = _shan_output_dir(
            request.output_dir,
            frame_dir.name,
            runtime_spec = runtime_spec,
        )
        
        shan_command_result = run_hand_object_contact(
            HandObjectContactRequest(
                input_path = frame_dir,
                output_dir = shan_output_dir,
            ),
            runtime_spec = runtime_spec.hand_object_contact_runtime_spec,
            command_runner = command_runner,
            progress = progress,
        )
        
        _append_executed_command_result(
            executed_commands,
            shan_command_result,
        )
        
        ownership_repair_command = _repair_output_ownership(
            request.output_dir,
            runtime_spec = runtime_spec,
            command_runner = command_runner,
            progress = progress,
        )
        
        executed_commands.append(ownership_repair_command)
                
        detic_output_dir = _detic_output_dir(
            request.output_dir,
            frame_dir.name,
            runtime_spec = runtime_spec
        )
        
        detic_output_dir.mkdir(parents = True, exist_ok = True)
        
        detic_command = build_detic_run_command(
            frame_dir = frame_dir,
            output_dir = detic_output_dir,
            runtime_spec = runtime_spec,
        )
        
        _run_stage(
            detic_command,
            command_runner = command_runner,
            stage_name = f"Detic inference for {frame_dir.name}",
            progress = progress,
        )
        
        executed_commands.append(detic_command)
        
        ownership_repair_command = _repair_output_ownership(
            request.output_dir,
            runtime_spec = runtime_spec,
            command_runner = command_runner,
            progress = progress,
        )
        
        executed_commands.append(ownership_repair_command)
    
    finalize_command = build_core_run_command(
        request,
        stage = "finalize",
        runtime_spec = runtime_spec,
    )
    
    _run_stage(
        finalize_command,
        command_runner = command_runner,
        stage_name = "ADL prediction finalization",
        progress = progress,
    )
    
    executed_commands.append(finalize_command)
    
    ownership_repair_command = _repair_output_ownership(
        request.output_dir,
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress,
    )
    
    executed_commands.append(ownership_repair_command)
    
    progress("adl-recognition inference completed.")
    
    return executed_commands

def _run_stage(
    command: list[str],
    *,
    command_runner: CommandRunner,
    stage_name: str,
    progress: ProgressReporter,
) -> None:
    progress(f"Starting {stage_name}.")
    
    exit_code = command_runner(command)
    
    if exit_code != 0:
        raise AdlRecognitionRuntimeError(
            f"{stage_name} failed with exit code {exit_code}."
        )
        
    progress(f"Finished {stage_name}.")

def _append_executed_command_result(
    executed_commands: list[list[str]],
    command_result: list[str] | list[list[str]],
) -> None:
    if not command_result:
        return
    
    if all(isinstance(part, str) for part in command_result):
        executed_commands.append(command_result)
        return
    
    executed_commands.extend(command_result)

def _is_combined_predictions_file(input_path: Path) -> bool:
    return (
        input_path.is_file() and
        input_path.name == COMBINED_PREDS_FILENAME
    )

def _container_input_path(
    request: AdlRecognitionRequest,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
) -> PurePosixPath:
    if request.input_path.is_dir():
        return runtime_spec.container_input_dir
    
    return runtime_spec.container_input_dir / request.input_path.name

def _input_mount_argument(
    request: AdlRecognitionRequest,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
) -> str:
    container_input_path = _container_input_path(
        request,
        runtime_spec = runtime_spec,
    )
    
    return f"{request.input_path.resolve()}:{container_input_path}:ro"

def _egoviz_data_root(
    output_dir: Path,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
) -> Path:
    return (
        output_dir
        / runtime_spec.work_dir_name
        / runtime_spec.egoviz_data_dir_name
    )

def _staged_adl_dir(
    output_dir: Path,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
) -> Path:
    return _egoviz_data_root(
        output_dir,
        runtime_spec = runtime_spec,
    ) / runtime_spec.staged_adl_dir_name

def _subclip_frame_dirs(
    output_dir: Path,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec
) -> list[Path]:
    subclips_dir = _staged_adl_dir(
        output_dir,
        runtime_spec = runtime_spec,
    ) / "subclips"
    
    if not subclips_dir.exists():
        return []
    
    return sorted(
        child
        for child in subclips_dir.iterdir()
        if child.is_dir()
        and any(frame.suffix.lower() == ".jpg" for frame in child.iterdir())
    )

def _shan_output_dir(
    output_dir: Path,
    clip_name: str,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
) -> Path:
    return (
        _staged_adl_dir(output_dir, runtime_spec = runtime_spec)
        / "subclips_shan"
        / clip_name
    )

def _detic_output_dir(
    output_dir: Path,
    clip_name: str,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec,
) -> Path:
    return (
        _staged_adl_dir(output_dir, runtime_spec = runtime_spec)
        / "detic_raw"
        / clip_name
    )
