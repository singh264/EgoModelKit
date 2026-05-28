""" Hidden runtime execution for hand-object-contact inference. """

from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Final

from egomodelkit.models.hand_object_contact import (
    HandObjectContactRequest,
    validate_hand_object_contact_request,
)
from egomodelkit.runtime.commands import subprocess_runner
from egomodelkit.runtime.external_code import (
    HAND_OBJECT_DETECTOR_PIN,
    HAND_OBJECT_DETECTOR_WEIGHTS_PIN,
    docker_asset_label_arguments,
    docker_code_label_arguments,
)
from egomodelkit.runtime.preflight import (
    ProgressReporter,
    ensure_host_runtime_ready,
)

CommandRunner = Callable[[list[str]], int]

SHAN_FORK_REPOSITORY_URL: Final[str] = HAND_OBJECT_DETECTOR_PIN.fork_repository_url
SHAN_FORK_COMMIT_SHA: Final[str] = HAND_OBJECT_DETECTOR_PIN.commit_sha

CHECKPOINT_SOURCE_URL: Final[str] = HAND_OBJECT_DETECTOR_WEIGHTS_PIN.source_url
CHECKPOINT_FILENAME: Final[str] = HAND_OBJECT_DETECTOR_WEIGHTS_PIN.filename

@dataclass(frozen = True, slots = True)
class HandObjectContactRuntimeSpec:
    """ Build and execution settings for the hidden hand-object-contact runtime. """
    
    docker_executable: str
    image_tag: str
    container_input_dir: PurePosixPath
    container_output_dir: PurePosixPath
    
    shan_repository_url: str
    shan_commit_sha: str
    
    checkpoint_source_url: str
    checkpoint_filename: str
    checkpoint_session: int
    checkpoint_epoch: int
    checkpoint_step: int
    
    shan_network_name: str
    shan_dataset_name: str
    shan_load_dir: str

    @property
    def shan_model_subdir(self) -> str:
        """ Return the checkpoint subdirectory expected by demo.py. """
        
        return (
            f"{self.shan_network_name}_handobj_100K/{self.shan_dataset_name}"    
        )

DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC: Final[HandObjectContactRuntimeSpec] = (
    HandObjectContactRuntimeSpec(
        docker_executable = "docker",
        image_tag = "egomodelkit-hand-object-contact:dev",
        container_input_dir = PurePosixPath("/workspace/input"),
        container_output_dir = PurePosixPath("/workspace/output"),
        shan_repository_url = SHAN_FORK_REPOSITORY_URL,
        shan_commit_sha = SHAN_FORK_COMMIT_SHA,
        checkpoint_source_url = CHECKPOINT_SOURCE_URL,
        checkpoint_filename = CHECKPOINT_FILENAME,
        checkpoint_session = 1,
        checkpoint_epoch = 8,
        checkpoint_step = 132028,
        shan_network_name = "res101",
        shan_dataset_name="pascal_voc",
        shan_load_dir = "models",
    )
)

class HandObjectContactRuntimeError(RuntimeError):
    """ Raised when hand-object-contact runtime execution fails. """

def _ignore_progress(_: str) -> None:
    """ Default no-op progress reporter. """

def _container_resource_dir() -> Path:
    return Path(
        str(
            files("egomodelkit").joinpath("resources/containers/hand_object_contact")
        )
    )

def _docker_build_arguments(runtime_spec: HandObjectContactRuntimeSpec) -> list[str]:
    return [
        *docker_code_label_arguments(HAND_OBJECT_DETECTOR_PIN),
        *docker_asset_label_arguments(HAND_OBJECT_DETECTOR_WEIGHTS_PIN),
        "--build-arg",
        f"SHAN_REPOSITORY_URL={runtime_spec.shan_repository_url}",
        "--build-arg",
        f"SHAN_COMMIT_SHA={runtime_spec.shan_commit_sha}",
        "--build-arg",
        f"CHECKPOINT_SOURCE_URL={runtime_spec.checkpoint_source_url}",
        "--build-arg",
        f"CHECKPOINT_FILENAME={runtime_spec.checkpoint_filename}",
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
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """ Build the hidden hand-object-contact runtime image only when missing. """
    progress("Checking packaged hand-object-contact runtime image.")

    inspect_command = [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        runtime_spec.image_tag,
    ]
    
    if command_runner(inspect_command) == 0:
        progress("Packaged hand-object-contact runtime image is already available.")
        return

    progress(
        "Packaged hand-object-contact runtime image is missing; preparing it now. "
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

    progress("Packaged hand-object-contact runtime image is ready.")

def build_run_command(
    request: HandObjectContactRequest,
    *,
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
) -> list[str]:
    """ Build the hidden runtime execution command. """
    output_dir = request.output_dir.resolve()
    
    container_input_path = _container_input_path(
        request,
        runtime_spec = runtime_spec,
    )
    
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        _input_mount_argument(
            request,
            runtime_spec = runtime_spec,
        ),
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
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> list[str]:
    """ Run hand-object-contact behind EgoModelKit's run command. """
    progress("Validating hand-object-contact request.")
    validate_hand_object_contact_request(request)

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
    
    progress("Starting hand-object-contact inference.")
    exit_code = command_runner(run_command)
    
    if exit_code != 0:
        raise HandObjectContactRuntimeError(
            f"hand-object-contact inference runtime failed with exit code {exit_code}."
        )

    progress("hand-object-contact inference completed.")
    return run_command

def _container_input_path(
    request: HandObjectContactRequest,
    *,
    runtime_spec: HandObjectContactRuntimeSpec,
) -> PurePosixPath:
    """ Return the input path that the container entrypoint should receive. """
    if request.input_path.is_dir():
        return runtime_spec.container_input_dir

    return runtime_spec.container_input_dir / request.input_path.name

def _input_mount_argument(
    request: HandObjectContactRequest,
    *,
    runtime_spec: HandObjectContactRuntimeSpec,
) -> str:
    """ return the Docker read-only mount for file or directory input. """
    container_input_path = _container_input_path(
        request,
        runtime_spec = runtime_spec
    )
    
    return (
        f"{request.input_path.resolve()}:"
        f"{container_input_path}:ro"
    )
