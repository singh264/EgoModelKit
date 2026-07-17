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
from egomodelkit.runtime.commands import (
    CommandResult,
    capturing_subprocess_runner,
    subprocess_runner,
)
from egomodelkit.runtime.docker_images import (
    DockerImageIdentity,
    build_runtime_image_identity,
    remove_stale_runtime_images,
)
from egomodelkit.runtime.external_code import (
    HAND_OBJECT_DETECTOR_PIN,
    HAND_OBJECT_DETECTOR_WEIGHTS_PIN,
    docker_asset_label_arguments,
    docker_code_label_arguments,
)
from egomodelkit.runtime.preflight import (
    ExecutableLocator,
    PlatformDetector,
    ProgressReporter,
    ensure_host_runtime_ready,
)

CommandRunner = Callable[[list[str]], int]
CaptureRunner = Callable[[list[str]], CommandResult]
StreamingCommandRunner = Callable[[list[str], ProgressReporter], int]

SHAN_FORK_REPOSITORY_URL: Final[str] = HAND_OBJECT_DETECTOR_PIN.fork_repository_url
SHAN_FORK_COMMIT_SHA: Final[str] = HAND_OBJECT_DETECTOR_PIN.commit_sha

CHECKPOINT_SOURCE_URL: Final[str] = HAND_OBJECT_DETECTOR_WEIGHTS_PIN.source_url
CHECKPOINT_FILENAME: Final[str] = HAND_OBJECT_DETECTOR_WEIGHTS_PIN.filename

@dataclass(frozen = True, slots = True)
class HandObjectContactRuntimeSpec:
    """ Build and execution settings for the hidden hand-object-contact runtime. """
    
    docker_executable: str
    image_repository: str
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
    
    pytorch_version: str
    torchvision_version: str
    torchaudio_version: str
    pytorch_cuda_index_url: str
    torch_cuda_arch_list: str

    @property
    def image_tag(self) -> str:
        """ Return the content-addressed Docker image tag for this spec. """
        return hand_object_contact_image_identity(self).tag

    @property
    def shan_model_subdir(self) -> str:
        """ Return the checkpoint subdirectory expected by demo.py. """
        
        return (
            f"{self.shan_network_name}_handobj_100K/{self.shan_dataset_name}"    
        )

DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC: Final[HandObjectContactRuntimeSpec] = (
    HandObjectContactRuntimeSpec(
        docker_executable = "docker",
        image_repository = "egomodelkit-hand-object-contact",
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
        pytorch_version = "2.11.0",
        torchvision_version = "0.26.0",
        torchaudio_version = "2.11.0",
        pytorch_cuda_index_url = "https://download.pytorch.org/whl/cu128",
        torch_cuda_arch_list = "7.5;8.0;8.6;8.9;9.0;12.0+PTX",
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
        f"SHAN_NETWORK_NAME={runtime_spec.shan_network_name}",
        "--build-arg",
        f"SHAN_DATASET_NAME={runtime_spec.shan_dataset_name}",
        "--build-arg",
        f"SHAN_MODEL_SUBDIR={runtime_spec.shan_model_subdir}",
        "--build-arg",
        f"SHAN_LOAD_DIR={runtime_spec.shan_load_dir}",
        "--build-arg",
        f"PYTORCH_VERSION={runtime_spec.pytorch_version}",
        "--build-arg",
        f"TORCHVISION_VERSION={runtime_spec.torchvision_version}",
        "--build-arg",
        f"TORCHAUDIO_VERSION={runtime_spec.torchaudio_version}",
        "--build-arg",
        f"PYTORCH_CUDA_INDEX_URL={runtime_spec.pytorch_cuda_index_url}",
        "--build-arg",
        f"TORCH_CUDA_ARCH_LIST={runtime_spec.torch_cuda_arch_list}",
    ]
    
def hand_object_contact_image_identity(
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
) -> DockerImageIdentity:
    """ Return the deterministic identity for the HOC runtime image. """
    return build_runtime_image_identity(
        runtime_name = "hand-object-contact",
        repository = runtime_spec.image_repository,
        context_dir = _container_resource_dir(),
        build_arguments = _docker_build_arguments(runtime_spec),
    )


def ensure_runtime_image(
    *,
    runtime_spec: HandObjectContactRuntimeSpec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    streaming_command_runner: StreamingCommandRunner | None = None,
    capture_runner: CaptureRunner = capturing_subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """ Build the hidden hand-object-contact runtime image only when missing. """
    progress("Checking packaged hand-object-contact runtime image.")
    image_identity = hand_object_contact_image_identity(runtime_spec)

    inspect_command = [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        image_identity.tag,
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
        image_identity.tag,
        *image_identity.label_arguments,
        *_docker_build_arguments(runtime_spec),
        str(container_dir),
    ]
    
    if streaming_command_runner is None:
        exit_code = command_runner(build_command)
    else:
        exit_code = streaming_command_runner(build_command, progress)
    
    if exit_code != 0:
        raise HandObjectContactRuntimeError(
            f"hand-object-contact runtime image build failed with exit code {exit_code}."
        )

    progress("Packaged hand-object-contact runtime image is ready.")
    remove_stale_runtime_images(
        docker_executable = runtime_spec.docker_executable,
        current_image = image_identity,
        capture_runner = capture_runner,
        progress = progress,
    )

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
    streaming_command_runner: StreamingCommandRunner | None = None,
    capture_runner: CaptureRunner = capturing_subprocess_runner,
    executable_locator: ExecutableLocator | None = None,
    platform_detector: PlatformDetector | None = None,
    progress: ProgressReporter = _ignore_progress,
) -> list[str]:
    """ Run hand-object-contact behind EgoModelKit's run command. """
    progress("Validating hand-object-contact request.")
    validate_hand_object_contact_request(request)

    runtime_check_kwargs = _runtime_check_overrides(
        executable_locator = executable_locator,
        platform_detector = platform_detector,
    )

    ensure_host_runtime_ready(
        docker_executable = runtime_spec.docker_executable,
        command_runner = command_runner,
        require_linux_nvidia_gpu = True,
        progress = progress,
        **runtime_check_kwargs,
    )

    request.output_dir.mkdir(parents = True, exist_ok = True)
    progress(f"Using output directory: {request.output_dir}")
    
    ensure_runtime_image(
        runtime_spec = runtime_spec,
        command_runner = command_runner,
        streaming_command_runner = streaming_command_runner,
        capture_runner = capture_runner,
        progress = progress
    )
    
    run_command = build_run_command(
        request,
        runtime_spec = runtime_spec,
    )
    
    progress("Starting hand-object-contact inference.")
    
    if streaming_command_runner is None:
        exit_code = command_runner(run_command)
    else:
        exit_code = streaming_command_runner(run_command, progress)

    if exit_code != 0:
        raise HandObjectContactRuntimeError(
            f"hand-object-contact inference runtime failed with exit code {exit_code}."
        )

    progress("hand-object-contact inference completed.")
    return run_command

def _runtime_check_overrides(
    *,
    executable_locator: ExecutableLocator | None,
    platform_detector: PlatformDetector | None,
) -> dict[str, ExecutableLocator | PlatformDetector]:
    """ Return optional host-runtime check dependencies for tests. """
    overrides: dict[str, ExecutableLocator | PlatformDetector] = {}

    if executable_locator is not None:
        overrides["executable_locator"] = executable_locator

    if platform_detector is not None:
        overrides["platform_detector"] = platform_detector

    return overrides

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
