"""Host prerequisite checks and automatic Docker runtime recovery."""

from __future__ import annotations

import base64
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

from egomodelkit.runtime.host_platform import is_wsl, wsl_distribution_name

CommandRunner = Callable[[list[str]], int]
ExecutableLocator = Callable[[str], str | None]
ProgressReporter = Callable[[str], None]
PlatformDetector = Callable[[], str]
WslDetector = Callable[[], bool]
SleepFunction = Callable[[float], None]
RuntimeRecoverer = Callable[
    [str, bool, str, bool, CommandRunner, ProgressReporter],
    bool,
]

MINIMUM_SUPPORTED_PYTHON_VERSION: Final[tuple[int, int]] = (3, 10)
SUPPORTED_GPU_HOST_PLATFORM: Final[str] = "Linux"
DOCKER_RECOVERY_ATTEMPTS: Final[int] = 45
DOCKER_RECOVERY_INTERVAL_SECONDS: Final[float] = 2.0

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
    "nvidia/cuda:12.8.1-base-ubuntu22.04",
    "nvidia-smi",
)


class HostPrerequisiteError(RuntimeError):
    """Raised when the host machine cannot run a packaged model runtime."""


def _ignore_progress(_: str) -> None:
    """Default no-op progress reporter."""


def _format_python_version(version: tuple[int, int, int]) -> str:
    """Return a human-readable Python version string."""
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
    wsl_detector: WslDetector = is_wsl,
    allow_runtime_recovery: bool | None = None,
    runtime_recoverer: RuntimeRecoverer | None = None,
    sleep: SleepFunction = time.sleep,
    recovery_attempts: int = DOCKER_RECOVERY_ATTEMPTS,
) -> None:
    """Validate host requirements and recover a stopped Docker runtime automatically."""
    detected_python_version = (
        python_version if python_version is not None else tuple(sys.version_info[:3])
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

    detected_platform = platform_detector()
    running_in_wsl = wsl_detector()

    if require_linux_nvidia_gpu:
        progress(f"Host platform detected: {detected_platform}.")
        if detected_platform != SUPPORTED_GPU_HOST_PLATFORM:
            raise HostPrerequisiteError(
                "EgoModelKit model runs require a Linux host with an NVIDIA GPU; "
                f"detected {detected_platform}. Use Linux directly or Windows through "
                "a WSL2 distribution with Docker Desktop GPU access."
            )

    docker_path = executable_locator(docker_executable)
    docker_ready = _docker_daemon_is_ready(
        docker_executable=docker_executable,
        command_runner=command_runner,
        docker_path=docker_path,
    )

    recovery_enabled = (
        allow_runtime_recovery
        if allow_runtime_recovery is not None
        else executable_locator is shutil.which
    )

    if not docker_ready and recovery_enabled:
        progress("Docker is not ready; attempting automatic recovery.")
        recoverer = runtime_recoverer or _recover_docker_runtime
        recovery_started = recoverer(
            detected_platform,
            running_in_wsl,
            docker_executable,
            docker_path is not None,
            command_runner,
            progress,
        )

        if recovery_started:
            docker_path, docker_ready = _wait_for_docker_ready(
                docker_executable=docker_executable,
                command_runner=command_runner,
                executable_locator=executable_locator,
                sleep=sleep,
                recovery_attempts=recovery_attempts,
            )

    if docker_path is None:
        if running_in_wsl:
            raise HostPrerequisiteError(_wsl_docker_recovery_error(docker_executable))
        raise HostPrerequisiteError(
            f"Docker executable '{docker_executable}' was not found on PATH. "
            "The machine-level EgoModelKit installation must be repaired by a maintainer."
        )

    progress(f"Docker executable found: {docker_path}")

    if not docker_ready:
        if running_in_wsl:
            raise HostPrerequisiteError(_wsl_docker_recovery_error(docker_executable))
        raise HostPrerequisiteError(
            "Docker is installed, but EgoModelKit could not start or reconnect to its "
            "daemon automatically. The machine-level Docker installation must be "
            "repaired by a maintainer."
        )

    progress("Docker daemon is available.")

    if require_linux_nvidia_gpu:
        progress("Checking Docker NVIDIA GPU runtime.")
        gpu_probe_command = [docker_executable, *NVIDIA_DOCKER_GPU_PROBE_SUFFIX]
        if command_runner(gpu_probe_command) != 0:
            if running_in_wsl:
                raise HostPrerequisiteError(
                    "Docker recovered successfully, but GPU access is unavailable in WSL2. "
                    "The machine-level Windows NVIDIA/WSL/Docker installation must be "
                    "repaired by a maintainer."
                )
            raise HostPrerequisiteError(
                "Docker recovered successfully, but the NVIDIA container runtime is "
                "unavailable. The machine-level NVIDIA Docker installation must be "
                "repaired by a maintainer."
            )
        progress("Docker NVIDIA GPU runtime is available.")


def _docker_daemon_is_ready(
    *,
    docker_executable: str,
    command_runner: CommandRunner,
    docker_path: str | None,
) -> bool:
    if docker_path is None:
        return False
    return command_runner([docker_executable, *DOCKER_DAEMON_PROBE_SUFFIX]) == 0


def _wait_for_docker_ready(
    *,
    docker_executable: str,
    command_runner: CommandRunner,
    executable_locator: ExecutableLocator,
    sleep: SleepFunction,
    recovery_attempts: int,
) -> tuple[str | None, bool]:
    """Wait until the Docker CLI and daemon are both available."""
    attempts = max(1, recovery_attempts)
    for attempt in range(attempts):
        docker_path = executable_locator(docker_executable)
        if _docker_daemon_is_ready(
            docker_executable=docker_executable,
            command_runner=command_runner,
            docker_path=docker_path,
        ):
            return docker_path, True
        if attempt + 1 < attempts:
            sleep(DOCKER_RECOVERY_INTERVAL_SECONDS)
    return executable_locator(docker_executable), False


def _recover_docker_runtime(
    detected_platform: str,
    running_in_wsl: bool,
    docker_executable: str,
    docker_cli_available: bool,
    command_runner: CommandRunner,
    progress: ProgressReporter,
) -> bool:
    """Start or restart the host Docker implementation without user interaction."""
    if running_in_wsl:
        return _recover_wsl_docker_desktop(
            docker_cli_available=docker_cli_available,
            progress=progress,
        )

    if detected_platform == "Darwin":
        return _start_macos_docker_desktop(progress=progress)

    if detected_platform == "Linux":
        return _start_linux_docker_runtime(
            command_runner=command_runner,
            progress=progress,
        )

    return False


def _resolve_windows_executable(candidates: tuple[str, ...]) -> str | None:
    """Resolve a Windows executable from WSL without relying on PATH export."""
    for candidate in candidates:
        if "/" in candidate:
            if Path(candidate).is_file():
                return candidate
            continue

        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

    return None


def _windows_docker_cli_candidates() -> tuple[str, ...]:
    """Return system-wide, current per-user, and legacy Docker CLI locations."""
    candidates = [
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe",
    ]
    users_root = Path("/mnt/c/Users")
    if users_root.is_dir():
        patterns = (
            "*/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe",
            "*/AppData/Local/Programs/Docker/Docker/resources/bin/docker.exe",
        )
        for pattern in patterns:
            candidates.extend(str(path) for path in sorted(users_root.glob(pattern)))
    return tuple(candidates)


def _windows_docker_desktop_candidates(
    *, windows_docker_cli: str | None = None,
) -> tuple[str, ...]:
    """Return system-wide, current per-user, and legacy Desktop locations."""
    candidates: list[str] = []
    if windows_docker_cli is not None:
        cli_path = Path(windows_docker_cli)
        try:
            install_root = cli_path.parents[2]
        except IndexError:
            pass
        else:
            candidates.extend(
                [
                    str(install_root / "Docker Desktop.exe"),
                    str(install_root / "frontend" / "Docker Desktop.exe"),
                ]
            )

    candidates.extend(
        [
            "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe",
            "/mnt/c/Program Files/Docker/Docker/frontend/Docker Desktop.exe",
        ]
    )

    users_root = Path("/mnt/c/Users")
    if users_root.is_dir():
        patterns = (
            "*/AppData/Local/Programs/DockerDesktop/Docker Desktop.exe",
            "*/AppData/Local/Programs/DockerDesktop/frontend/Docker Desktop.exe",
            "*/AppData/Local/Programs/Docker/Docker/Docker Desktop.exe",
            "*/AppData/Local/Programs/Docker/Docker/frontend/Docker Desktop.exe",
        )
        for pattern in patterns:
            candidates.extend(str(path) for path in sorted(users_root.glob(pattern)))

    return tuple(dict.fromkeys(candidates))


def _windows_path_to_wsl_path(path_text: str) -> str | None:
    """Convert a simple drive-letter Windows path to its default WSL mount."""
    normalized = path_text.strip().strip('"').replace("\\", "/")
    if len(normalized) < 3 or normalized[1:3] != ":/":
        return None
    drive = normalized[0].lower()
    return f"/mnt/{drive}/{normalized[3:]}"


def _discover_windows_executable(executable_name: str) -> str | None:
    """Ask Windows where.exe for an executable and convert its path for WSL."""
    where_executable = _resolve_windows_executable(
        (
            "where.exe",
            "/mnt/c/Windows/System32/where.exe",
        )
    )
    if where_executable is None:
        return None

    try:
        completed = subprocess.run(
            [where_executable, executable_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        candidate = _windows_path_to_wsl_path(line)
        if candidate is not None and Path(candidate).is_file():
            return candidate
    return None


def _run_quietly(command: list[str]) -> int | None:
    """Run a short Windows interoperability command and return its exit code."""
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return None


def _wsl_path_to_windows_path(path_text: str) -> str | None:
    """Convert a path under /mnt/<drive> to a Windows drive-letter path."""
    path = Path(path_text)
    parts = path.parts
    if len(parts) < 4 or parts[1] != "mnt" or len(parts[2]) != 1:
        return None
    drive = parts[2].upper()
    remainder = "\\".join(parts[3:])
    return f"{drive}:\\{remainder}"


def _powershell_encoded_command(script: str) -> str:
    """Encode a PowerShell command without shell-quoting ambiguities."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _windows_interop_is_ready() -> bool:
    """Return whether WSL can execute Windows programs through WSLInterop."""
    cmd_executable = _resolve_windows_executable(
        (
            "cmd.exe",
            "/mnt/c/Windows/System32/cmd.exe",
        )
    )
    if cmd_executable is None:
        return False
    return _run_quietly([cmd_executable, "/d", "/c", "exit", "0"]) == 0


def _repair_wsl_interop() -> bool:
    """Restore WSLInterop's binfmt registration without prompting for input."""
    repair_script = r"""
set -eu
if [ ! -d /proc/sys/fs/binfmt_misc ]; then
    exit 1
fi
if [ ! -e /proc/sys/fs/binfmt_misc/register ]; then
    mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
fi
if [ -e /proc/sys/fs/binfmt_misc/WSLInterop ]; then
    printf '1' > /proc/sys/fs/binfmt_misc/WSLInterop
else
    printf ':WSLInterop:M::MZ::/init:PF' > /proc/sys/fs/binfmt_misc/register
fi
""".strip()

    commands = (
        ["sh", "-c", repair_script],
        ["sudo", "-n", "sh", "-c", repair_script],
    )
    return any(_run_quietly(command) == 0 for command in commands)


def _ensure_windows_interop(*, progress: ProgressReporter) -> bool:
    """Ensure Windows executables can be launched from the active WSL session."""
    if _windows_interop_is_ready():
        return True

    progress(
        "Windows interoperability is unavailable in WSL; attempting automatic "
        "recovery."
    )
    if _repair_wsl_interop() and _windows_interop_is_ready():
        progress("Windows interoperability was recovered automatically.")
        return True

    progress(
        "Windows interoperability could not be recovered automatically; Windows "
        "applications cannot be launched from this WSL session."
    )
    return False


def _launch_windows_process(
    executable: str,
    *,
    progress: ProgressReporter = _ignore_progress,
) -> bool:
    """Launch a Windows GUI process from WSL without waiting for it to exit."""
    windows_path = _wsl_path_to_windows_path(executable)

    powershell_executable = _resolve_windows_executable(
        (
            "powershell.exe",
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        )
    )
    if powershell_executable is not None and windows_path is not None:
        escaped_path = windows_path.replace("'", "''")
        encoded_command = _powershell_encoded_command(
            f"Start-Process -FilePath '{escaped_path}'; exit 0"
        )
        result = _run_quietly(
            [
                powershell_executable,
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_command,
            ]
        )
        if result == 0:
            return True
        progress(
            "PowerShell could not launch Docker Desktop "
            f"(exit code: {result if result is not None else 'unavailable'})."
        )

    cmd_executable = _resolve_windows_executable(
        (
            "cmd.exe",
            "/mnt/c/Windows/System32/cmd.exe",
        )
    )
    if cmd_executable is not None and windows_path is not None:
        command_text = f'start "" "{windows_path}"'
        result = _run_quietly(
            [cmd_executable, "/d", "/s", "/c", command_text]
        )
        if result == 0:
            return True
        progress(
            "Windows Command Prompt could not launch Docker Desktop "
            f"(exit code: {result if result is not None else 'unavailable'})."
        )

    try:
        subprocess.Popen(
            [executable],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        progress(f"Direct Docker Desktop launch failed: {error}.")
        return False
    return True


def _recover_wsl_docker_desktop(
    *,
    docker_cli_available: bool,
    progress: ProgressReporter,
) -> bool:
    """Set the active distro as default and start/restart Docker Desktop."""
    distro = wsl_distribution_name()

    if not _ensure_windows_interop(progress=progress):
        return False

    wsl_executable = _resolve_windows_executable(
        (
            "wsl.exe",
            "/mnt/c/Windows/System32/wsl.exe",
        )
    )
    if wsl_executable is not None:
        configured = _run_quietly([wsl_executable, "--set-default", distro]) == 0
        if configured:
            progress(f"Configured {distro} as the default WSL distribution.")

    windows_docker_cli = _resolve_windows_executable(
        _windows_docker_cli_candidates()
    ) or _discover_windows_executable("docker.exe")

    if windows_docker_cli is not None:
        status_code = _run_quietly([windows_docker_cli, "desktop", "status"])
        primary_operation = (
            "restart" if status_code == 0 or docker_cli_available else "start"
        )
        fallback_operation = "start" if primary_operation == "restart" else "restart"

        for operation in (primary_operation, fallback_operation):
            if _run_quietly(
                [windows_docker_cli, "desktop", operation]
            ) == 0:
                progress("Docker Desktop recovery was started automatically.")
                return True

    desktop_executable = _resolve_windows_executable(
        _windows_docker_desktop_candidates(
            windows_docker_cli=windows_docker_cli,
        )
    )
    if desktop_executable is not None:
        progress(f"Located Docker Desktop launcher: {desktop_executable}")
        if _launch_windows_process(desktop_executable, progress=progress):
            progress("Docker Desktop recovery was started automatically.")
            return True
        progress(
            "Docker Desktop was located, but Windows could not launch it from the "
            "active WSL session."
        )
        return False

    progress(
        "Docker Desktop was not found in the Windows system-wide or per-user "
        "installation locations."
    )
    return False


def _start_macos_docker_desktop(*, progress: ProgressReporter) -> bool:
    try:
        completed = subprocess.run(
            ["open", "-gj", "-a", "Docker"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False

    if completed.returncode == 0:
        progress("Docker Desktop recovery was started automatically.")
        return True
    return False


def _start_linux_docker_runtime(
    *,
    command_runner: CommandRunner,
    progress: ProgressReporter,
) -> bool:
    commands = (
        ["systemctl", "--user", "start", "docker-desktop"],
        ["systemctl", "--user", "start", "docker"],
        ["sudo", "-n", "systemctl", "start", "docker"],
    )
    for command in commands:
        if command_runner(command) == 0:
            progress("Docker runtime recovery was started automatically.")
            return True
    return False


def _wsl_docker_recovery_error(docker_executable: str) -> str:
    distro = wsl_distribution_name()
    return (
        f"EgoModelKit could not automatically connect Docker Desktop to {distro} "
        f"through '{docker_executable}'. The machine-level Docker Desktop/WSL2 "
        "installation must be repaired by a maintainer."
    )
