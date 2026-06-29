""" Host prerequistes checks for EgoModelKit runtime execution. """

import platform
import shutil
import sys
from collections.abc import Callable
from typing import Final

CommandRunner = Callable[[list[str]], int]
ExecutableLocator = Callable[[str], str | None]
ProgressReporter = Callable[[str], None]
PlatformDetector = Callable[[], str]

MINIMUM_SUPPORTED_PYTHON_VERSION: Final[tuple[int, int]] = (3, 10)
SUPPORTED_GPU_HOST_PLATFORM: Final[str] = "Linux"

DOCKER_DAEMON_PROBE_SUFFIX: Final[tuple[str, ...]] = (
    "version",
    "--format",
    "{{.Server.Version}}",
)

NVIDIA_DOCKER_GPU_PROBE_SUFFIX: Final[tuple[str, ...]] = (
    "run",
    "--rm",
    "--gpus",
    "all",
    "nvidia/cuda:11.3.1-base-ubuntu20.04",
    "nvidia-smi",
)

class HostPrerequisiteError(RuntimeError):
    """ Raised when the host machine cannot run a packaged model runtime. """

def _ignore_progress(_: str) -> None:
    """ Default no-op progress reporter. """

def _format_python_version(version: tuple[int, int, int]) -> str:
    """ Return a human-readable Python version string. """
    return ".".join(str(part) for part in version)

def ensure_host_runtime_ready(
    *,
    docker_executable: str,
    command_runner: CommandRunner,
    executable_locator: ExecutableLocator = shutil.which,
    python_version: tuple[int, int, int] | None = None,
    platform_detector: PlatformDetector = platform.system,
    require_linux_nvidia_gpu: bool = False,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """ Validate host requirements needed by the current run path. """
    detected_python_version = (
        python_version
        if python_version is not None
        else tuple(sys.version_info[:3])
    )
    
    progress("Checking host runtime prerequisites.")
    progress(f"Python {_format_python_version(detected_python_version)} detected.")
    
    if detected_python_version[:2] < MINIMUM_SUPPORTED_PYTHON_VERSION:
        required_version = ".".join(str(part) for part in MINIMUM_SUPPORTED_PYTHON_VERSION)
        detected_version = _format_python_version(detected_python_version)
        
        raise HostPrerequisiteError(
            f"Python {required_version} or newer is required; "
            f"detected Python {detected_version}."
        )
    
    if require_linux_nvidia_gpu:
        detected_platform = platform_detector()
        progress(f"Host platform detected: {detected_platform}.")

        if detected_platform != SUPPORTED_GPU_HOST_PLATFORM:
            raise HostPrerequisiteError(
                "EgoModelKit model runs require a Linux host with an NVIDIA GPU; "
                f"detected {detected_platform}. Use a Linux NVIDIA GPU machine "
                "for dry runs and model runs."
            )
        
    docker_path = executable_locator(docker_executable)
    
    if docker_path is None:
        raise HostPrerequisiteError(
            f"Docker executable '{docker_executable}' was not found on PATH. "
            "Install Docker, then rerun the same egomodelkit command."
        )
    
    progress(f"Docker executable found: {docker_path}")
    
    daemon_probe_command = [
        docker_executable,
        *DOCKER_DAEMON_PROBE_SUFFIX,
    ]
    
    if command_runner(daemon_probe_command) != 0:
        raise HostPrerequisiteError(
            "Docker is installed, but its daemon is not available or "
            "the current user cannot access it. Start Docker or fix Docker "
            "permissions, then rerun the same egomodelkit command."
        )

    progress("Docker daemon is available.")

    if require_linux_nvidia_gpu:
        progress("Checking Docker NVIDIA GPU runtime.")
        
        gpu_probe_command = [
            docker_executable,
            *NVIDIA_DOCKER_GPU_PROBE_SUFFIX,
        ]

        if command_runner(gpu_probe_command) != 0:
            raise HostPrerequisiteError(
                "Docker is available, but the NVIDIA GPU runtime is not available. "
                "Install NVIDIA drivers and the NVIDIA Container Toolkit on a "
                "Linux GPU machine, then rerun the same EgoModelKit command."
            )

        progress("Docker NVIDIA GPU runtime is available.")
