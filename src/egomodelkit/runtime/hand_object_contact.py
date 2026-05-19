""" Hidden runtime execution for Shan hand-object-contact inference. """

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Final

from egomodelkit.models.hand_object_contact import (
    HandObjectContactRequest,
    validate_request,
)
from egomodelkit.runtime.preflight import (
    ProgressReporter,
    ensure_host_runtime_ready,
)

CommandRunner = Callable[[list[str]], int]

SHAN_FORK_REPOSITORY_URL: Final[str] = "https://github.com/singh264/hand_object_detector"
SHAN_FORK_COMMIT_SHA: Final[str] = "70146cabaeffb41ecc02e6edb605fc021dbdb555"

@dataclass(frozen = True, slots = True)
class HandObjectContactRuntimeSpec:
    """ Build and execution settings for the hidden Shan runtime. """
    
    docker_executable: str
    image_tag: str
    container_input_dir: PurePosixPath
    container_output_dir: PurePosixPath
    
    shan_repository_url: str
    shan_commit_sha: str
    
    checkpoint_google_drive_file_id: str
    checkpoint_session: int
    checkpoint_epoch: int
    checkpoint_step: int
    
    shan_network_name: str
    shan_dataset_name: str
    shan_load_dir: str
    
    @property
    def checkpoint_filename(self) -> str:
        """ Return the Shan checkpoint filename expected by demo.py. """ 

        return (
            "faster_rcnn_"
            f"{self.checkpoint_session}_"
            f"{self.checkpoint_epoch}_"
            f"{self.checkpoint_step}.pth"
        )

    @property
    def shan_model_subdir(self) -> str:
        """ Return the checkpoint subdirectory expected by demo.py. """
        
        return (
            f"{self.shan_network_name}_handobj_100K/{self.shan_dataset_name}"    
        )

DEFAULT_RUNTIME_SPEC: Final[HandObjectContactRuntimeSpec] = (
    HandObjectContactRuntimeSpec(
        docker_executable = "docker",
        image_tag = "egomodelkit-hand-object-contact:dev",
        container_input_dir = PurePosixPath("/workspace/input"),
        container_output_dir = PurePosixPath("/workspace/output"),
        shan_repository_url = SHAN_FORK_REPOSITORY_URL,
        shan_commit_sha = SHAN_FORK_COMMIT_SHA,
        checkpoint_google_drive_file_id = "1b_BkGgmYAe8VNbsFeljrSP7V1Jd8E82v",
        checkpoint_session = 1,
        checkpoint_epoch = 8,
        checkpoint_step = 132028,
        shan_network_name = "res101",
        shan_dataset_name="pascal_voc",
        shan_load_dir = "models",
    )
)

class HandObjectContactRuntimeError(RuntimeError):
    """ Raised when the hidden Shan runtime fails. """

def _ignore_progress(_: str) -> None:
    """ Default no-op progress reporter. """

def _subprocess_runner(command: list[str]) -> int:
    completed = subprocess.run(command, check = False)
    
    return completed.returncode

def _container_resource_dir() -> Path:
    return Path(
        str(
            files("egomodelkit").joinpath("resources/containers/hand_object_contact")
        )
    )

def _docker_build_arguments(runtime_spec: HandObjectContactRuntimeSpec) -> list[str]:
    return [
        "--build-arg",
        f"SHAN_REPOSITORY_URL={runtime_spec.shan_repository_url}",
        "--build-arg",
        f"SHAN_COMMIT_SHA={runtime_spec.shan_commit_sha}",
        "--build-arg",
        f"CHECKPOINT_GOOGLE_DRIVE_FILE_ID={runtime_spec.checkpoint_google_drive_file_id}",
        "--build-arg",
        f"CHECKPOINT_SESSION={runtime_spec.checkpoint_session}",
        "--build-arg",
        f"CHECKPOINT_EPOCH={runtime_spec.checkpoint_epoch}",
        "--build-arg",
        f"CHECKPOINT_STEP={runtime_spec.checkpoint_step}",
        "--build-arg",
        f"CHECKPOINT_FILENAME={runtime_spec.checkpoint_filename}",
        "--build-arg",
        f"SHAN_NETWORK_NAME={runtime_spec.shan_network_name}",
        "--build-arg",
        f"SHAN_DATASET_NAME={runtime_spec.shan_dataset_name}",
        "--build-arg",
        f"SHAN_MODEL_SUBDIR={runtime_spec.shan_model_subdir}",
        "--build-arg",
        f"SHAN_LOAD_DIR={runtime_spec.shan_load_dir}",
    ]
    
def ensure_runtime_image(
    *,
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_RUNTIME_SPEC,
    command_runner: CommandRunner = _subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """ Build the hidden Shan runtime image only when missing. """
    progress("Checking packaged Shan runtime image.")

    inspect_command = [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        runtime_spec.image_tag,
    ]
    
    if command_runner(inspect_command) == 0:
        progress("Packaged Shan runtime image is already available.")
        return

    progress(
        "Packaged Shan runtime image is missing; preparing it now. "
        "The first run may take longer."
    )

    container_dir = _container_resource_dir()
    dockerfile_path = container_dir / "Dockerfile"
    
    build_command = [
        runtime_spec.docker_executable,
        "build",
        "-f",
        str(dockerfile_path),
        "-t",
        runtime_spec.image_tag,
        *_docker_build_arguments(runtime_spec),
        str(container_dir),
    ]
    
    exit_code = command_runner(build_command)
    
    if exit_code != 0:
        raise HandObjectContactRuntimeError(
            f"hand-object-contact runtime image build failed with exit code {exit_code}."
        )

    progress("Packaged Shan runtime image is ready.")

def build_run_command(
    request: HandObjectContactRequest,
    *,
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_RUNTIME_SPEC,
) -> list[str]:
    """ Build the hidden runtime execution command. """
    input_parent = request.input_path.resolve().parent
    output_dir = request.output_dir.resolve()
    
    container_input_path = (
        runtime_spec.container_input_dir / request.input_path.name
    )
    
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{input_parent}:{runtime_spec.container_input_dir}:ro",
        "-v",
        f"{output_dir}:{runtime_spec.container_output_dir}",
        runtime_spec.image_tag,
        "--input-path",
        str(container_input_path),
        "--output-dir",
        str(runtime_spec.container_output_dir),
    ]
    
def run_hand_object_contact(
    request: HandObjectContactRequest,
    *,
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_RUNTIME_SPEC,
    command_runner: CommandRunner = _subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> list[str]:
    """ Run Shan hand-object-contact behind EgoModelKit's run command. """
    progress("Validating hand-object-contact request.")
    validate_request(request)

    ensure_host_runtime_ready(
        docker_executable = runtime_spec.docker_executable,
        command_runner = command_runner,
        progress = progress,
    )

    request.output_dir.mkdir(parents = True, exist_ok = True)
    progress(f"Using output directory: {request.output_dir}")
    
    ensure_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        progress = progress
    )
    
    run_command = build_run_command(
        request,
        runtime_spec = runtime_spec,
    )
    
    progress("Starting Shan hand-object-contact inference.")
    exit_code = command_runner(run_command)
    
    if exit_code != 0:
        raise HandObjectContactRuntimeError(
            f"hand-object-contact inference runtime failed with exit code {exit_code}."
        )

    progress("Shan hand-object-contact inference completed.")
    return run_command
