#!/usr/bin/env python3
"""Install, update, repair, open, stop, or uninstall EgoModelKit.

Users run this file directly with Python on Windows or as an executable
program on Linux. The script checks the computer, chooses the required action
automatically, opens a small progress window, creates a permanent desktop
shortcut, and provides confirmed stop and uninstall controls.

Machine-level prerequisites are documented in egomodelkit-machine-setup.pdf
and are never installed with administrator or sudo privileges by this script.
"""

from __future__ import annotations

import getpass
import http.client
import json
import logging
import os
import platform
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Maintainer configuration
# ---------------------------------------------------------------------------
# Change REPOSITORY_BRANCH to a validated release tag when one is available.
# Following "main" is convenient during development, but a release tag is more
# reproducible for research deployments.
REPOSITORY_URL = "https://github.com/singh264/EgoModelKit.git"
REPOSITORY_BRANCH = "main"

PREFERRED_WSL_DISTRO = "Ubuntu-24.04"
GUI_HOST = "127.0.0.1"
GUI_BASE_PORT = 7860
GUI_PORT_SPAN = 500
GUI_READY_TIMEOUT_SECONDS = 120
DOCKER_READY_TIMEOUT_SECONDS = 180
GPU_PROBE_IMAGE = "nvidia/cuda:12.8.1-base-ubuntu22.04"
MINIMUM_PYTHON = (3, 10)
VALIDATED_WINDOWS_BUILD_MINIMUM = 22000
VALIDATED_LINUX_ID = "ubuntu"
VALIDATED_LINUX_VERSION_PREFIX = "24.04"
KEEP_RELEASES = 3

# User-local frontend build toolchain. The launcher prefers a compatible
# Linux Node.js already present in the selected environment. When none exists,
# it downloads this pinned official build under the current user's EgoModelKit
# data directory and verifies its SHA-256 digest before use. No sudo/admin
# installation and no Windows Node.js executable inside WSL are required.
NODE_VERSION = "22.23.1"
NODE_X64_SHA256 = "9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578"
NODE_ARM64_SHA256 = "0294e8b915ab75f92c7513d2fcb830ae06e10684e6c603e99a87dbf8835389c1"

APP_NAME = "EgoModelKit"
SCRIPT_NAME = "EMKSetup.py"
GUIDE_NAME = "egomodelkit-machine-setup.pdf"

# A small marker placed in logs and state files.  Increment when installer state
# formats or important launcher behaviour changes.
SETUP_SCRIPT_VERSION = "1.1.9"


# ---------------------------------------------------------------------------
# Data models and errors
# ---------------------------------------------------------------------------
class SetupError(RuntimeError):
    """An actionable setup or launch failure."""

    def __init__(self, message: str, *, repair_wsl_may_help: bool = False) -> None:
        super().__init__(message)
        self.repair_wsl_may_help = repair_wsl_may_help


@dataclass(slots=True)
class CommandResult:
    """Captured command result."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(slots=True)
class CheckResult:
    """One visible prerequisite check."""

    name: str
    status: str  # pass, fail, warn
    detail: str = ""


@dataclass(slots=True)
class AppPaths:
    """Host-side paths used by this script."""

    app_dir: Path
    log_dir: Path
    installed_script: Path
    installed_guide: Path
    state_file: Path
    process_log: Path
    icon_file: Path


@dataclass(slots=True)
class InstallResult:
    """Linux-side active-installation information."""

    commit: str
    repo_dir: str
    venv_python: str
    egomodelkit_executable: str
    updated: bool
    update_warning: str = ""


@dataclass(slots=True)
class LaunchResult:
    """Result of reusing or starting the local GUI."""

    url: str
    reused: bool
    update_pending_restart: bool


@dataclass(slots=True)
class WorkerOutcome:
    """Result sent from the background worker to the Tk GUI."""

    success: bool
    message: str
    gui_url: str | None = None
    repair_wsl_may_help: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    gui_running: bool = False
    update_pending_restart: bool = False


# ---------------------------------------------------------------------------
# Host paths, logging, and safe command helpers
# ---------------------------------------------------------------------------
class WorkflowLock:
    """Atomic per-user guard against concurrent setup/launch races.

    The lock is a user-owned directory, so it works without platform-specific
    packages on Windows and Linux.  A lock older than six hours is considered
    stale after a crash.  A second launcher never starts another GUI server; it
    reports that the first setup/launch is still running.
    """

    def __init__(self, paths: AppPaths) -> None:
        self.path = paths.app_dir / ".workflow-lock"
        self.acquired = False

    def __enter__(self) -> WorkflowLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
            self.acquired = True
        except FileExistsError:
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError:
                age = 0
            if age > 6 * 60 * 60:
                logging.warning("Removing stale workflow lock: %s", self.path)
                shutil.rmtree(self.path, ignore_errors=True)
                try:
                    self.path.mkdir()
                    self.acquired = True
                except FileExistsError as exc:
                    raise SetupError(
                        "Another EgoModelKit setup or launch is already running for this user."
                    ) from exc
            else:
                raise SetupError(
                    "Another EgoModelKit setup or launch is already running for this user. "
                    "Wait for its progress window to finish, then retry."
                ) from None
        try:
            (self.path / "owner.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "startedAt": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            logging.exception("Could not write workflow lock metadata")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)

def is_windows() -> bool:
    return os.name == "nt"


def is_linux() -> bool:
    return platform.system() == "Linux" and not is_windows()


def get_paths() -> AppPaths:
    """Return per-user host paths without using administrator locations."""
    if is_windows():
        local_app_data = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        app_dir = local_app_data / APP_NAME
        log_dir = app_dir / "logs"
    else:
        data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        app_dir = data_home / APP_NAME
        log_dir = state_home / APP_NAME / "logs"

    return AppPaths(
        app_dir=app_dir,
        log_dir=log_dir,
        installed_script=app_dir / SCRIPT_NAME,
        installed_guide=app_dir / GUIDE_NAME,
        state_file=app_dir / "launcher-state.json",
        process_log=log_dir / "egomodelkit-gui.log",
        icon_file=app_dir / "egomodelkit.svg",
    )


def configure_logging(paths: AppPaths) -> Path:
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = paths.log_dir / f"emksetup-{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("EMKSetup.py version %s", SETUP_SCRIPT_VERSION)
    logging.info("Mode host: %s %s", platform.system(), platform.release())
    prune_old_logs(paths.log_dir)
    return log_path


def prune_old_logs(log_dir: Path, keep: int = 20) -> None:
    try:
        logs = sorted(
            log_dir.glob("emksetup-*.log"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for old in logs[keep:]:
            old.unlink(missing_ok=True)
    except OSError:
        logging.exception("Could not prune old setup logs")


def _decode_process_output(data: bytes) -> str:
    """Decode output from Windows tools that may emit UTF-16 when redirected."""
    if not data:
        return ""
    if b"\x00" in data[:200]:
        try:
            return data.decode("utf-16-le", errors="replace").replace("\ufeff", "")
        except UnicodeError:
            pass
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_command(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    hide_window: bool = True,
) -> CommandResult:
    """Run a command without a shell and capture diagnostic output."""
    command = [str(part) for part in args]
    logging.info("Running command: %s", shlex.join(command))

    startupinfo = None
    creationflags = 0
    if is_windows() and hide_window:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            input=stdin_text.encode("utf-8") if stdin_text is not None else None,
            capture_output=True,
            timeout=timeout,
            check=False,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        return CommandResult(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_process_output(exc.stdout or b"")
        stderr = _decode_process_output(exc.stderr or b"")
        return CommandResult(command, 124, stdout, stderr or "Command timed out.")
    except OSError as exc:
        return CommandResult(command, 126, "", str(exc))

    stdout = _decode_process_output(completed.stdout)
    stderr = _decode_process_output(completed.stderr)
    if stdout.strip():
        logging.info("stdout: %s", stdout.strip())
    if stderr.strip():
        logging.info("stderr: %s", stderr.strip())
    logging.info("Exit code: %s", completed.returncode)
    return CommandResult(command, completed.returncode, stdout, stderr)


def quote_bash(value: str) -> str:
    return shlex.quote(value)


def run_linux_shell(
    script: str,
    *,
    distro: str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """Run a Bash script natively on Linux or inside the selected WSL distro."""
    if is_windows():
        if not distro:
            raise SetupError("No WSL distribution was selected.")
        return run_command(
            ["wsl.exe", "-d", distro, "--", "bash", "-lc", script],
            timeout=timeout,
        )
    return run_command(["bash", "-lc", script], timeout=timeout)


def run_linux_script(
    script: str,
    *,
    arguments: Sequence[str] = (),
    distro: str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """Run a multiline Bash script through standard input.

    Passing the script through stdin avoids Windows-to-WSL command-line quoting
    and expansion of heredoc contents by an outer ``bash -lc`` process.
    """
    bash_command = ["bash", "-s", "--", *(str(value) for value in arguments)]
    if is_windows():
        if not distro:
            raise SetupError("No WSL distribution was selected.")
        command = ["wsl.exe", "-d", distro, "--", *bash_command]
    else:
        command = bash_command
    return run_command(command, timeout=timeout, stdin_text=script)


def require_ok(result: CommandResult, message: str) -> CommandResult:
    if not result.ok:
        detail = (result.stderr or result.stdout).strip()
        if detail:
            message = f"{message}\n\nTechnical detail: {detail[-1200:]}"
        raise SetupError(message)
    return result


# ---------------------------------------------------------------------------
# GUI event reporting
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[str, str], None]


def noop_progress(_message: str, _status: str = "info") -> None:
    return


def report(progress: ProgressCallback, message: str, status: str = "info") -> None:
    logging.info("[%s] %s", status.upper(), message)
    progress(message, status)


# ---------------------------------------------------------------------------
# Prerequisite discovery and checks
# ---------------------------------------------------------------------------
def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values


def windows_build_number() -> int | None:
    if not is_windows():
        return None
    try:
        return int(platform.version().split(".")[-1])
    except (ValueError, IndexError):
        return None


def parse_wsl_list_verbose(text: str) -> list[tuple[str, str, int]]:
    """Parse `wsl --list --verbose` across localized spacing and default markers."""
    rows: list[tuple[str, str, int]] = []
    cleaned = text.replace("\x00", "")
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("name"):
            continue
        line = line.lstrip("* ").strip()
        # The final column is the WSL version.  Treat the middle status column
        # as opaque so this still works when Windows localizes "Running" and
        # "Stopped".  Distribution names normally contain no whitespace.
        parts = line.rsplit(maxsplit=2)
        if len(parts) == 3 and parts[2].isdigit():
            rows.append((parts[0].strip(), parts[1], int(parts[2])))
    return rows


def select_wsl_distro() -> tuple[str | None, list[tuple[str, str, int]]]:
    result = run_command(["wsl.exe", "--list", "--verbose"], timeout=30)
    if not result.ok:
        return None, []
    rows = parse_wsl_list_verbose(result.stdout)
    names = [row[0] for row in rows]
    if PREFERRED_WSL_DISTRO in names:
        return PREFERRED_WSL_DISTRO, rows
    ubuntu = [name for name in names if name.lower().startswith("ubuntu")]
    if len(ubuntu) == 1:
        return ubuntu[0], rows
    return None, rows


def find_docker_desktop() -> Path | None:
    if not is_windows():
        return None
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Docker/Docker/Docker Desktop.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs/Docker/Docker/Docker Desktop.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs/DockerDesktop/Docker Desktop.exe",
    ]
    return next((path for path in candidates if path.is_file()), None)


def wsl_has_command(distro: str, command: str) -> bool:
    result = run_linux_shell(
        f"command -v {quote_bash(command)} >/dev/null 2>&1",
        distro=distro,
        timeout=20,
    )
    return result.ok


def linux_python_version(distro: str | None = None) -> tuple[int, int] | None:
    result = run_linux_shell(
        "python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")'",
        distro=distro,
        timeout=20,
    )
    if not result.ok:
        return None
    match = re.search(r"(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def tkinter_is_available() -> bool:
    try:
        import tkinter  # noqa: F401

        return True
    except Exception:
        return False


def host_nvidia_check() -> tuple[bool, str]:
    executable = shutil.which("nvidia-smi")
    if not executable and is_windows():
        candidates = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation/NVSMI/nvidia-smi.exe",
        ]
        executable = next((str(path) for path in candidates if path.is_file()), None)
    if not executable:
        return False, "nvidia-smi was not found."
    result = run_command(
        [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
        timeout=30,
    )
    return result.ok, (result.stdout.strip() or result.stderr.strip())


def linux_or_wsl_nvidia_check(distro: str | None) -> tuple[bool, str]:
    command = (
        "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L; "
        "elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then /usr/lib/wsl/lib/nvidia-smi -L; "
        "else exit 127; fi"
    )
    result = run_linux_shell(command, distro=distro, timeout=30)
    return result.ok, (result.stdout.strip() or result.stderr.strip())


def docker_info(distro: str | None) -> CommandResult:
    return run_linux_shell(
        "docker info --format '{{.ServerVersion}}'",
        distro=distro,
        timeout=30,
    )


def docker_gpu_probe(distro: str | None, *, allow_pull: bool) -> CommandResult:
    if not allow_pull:
        present = run_linux_shell(
            f"docker image inspect {quote_bash(GPU_PROBE_IMAGE)} >/dev/null 2>&1",
            distro=distro,
            timeout=30,
        )
        if not present.ok:
            return CommandResult(
                ["docker", "image", "inspect", GPU_PROBE_IMAGE],
                3,
                "",
                "GPU probe image is not cached; setup will download it before launch.",
            )
    return run_linux_shell(
        "docker run --rm --gpus all "
        + quote_bash(GPU_PROBE_IMAGE)
        + " nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
        distro=distro,
        timeout=240,
    )


def collect_prerequisite_checks(
    *,
    progress: ProgressCallback = noop_progress,
) -> tuple[list[CheckResult], str | None]:
    """Collect prerequisite checks without installing machine components."""
    checks: list[CheckResult] = []
    distro: str | None = None

    def add(name: str, status: str, detail: str = "") -> None:
        checks.append(CheckResult(name, status, detail))
        symbol = "OK" if status == "pass" else "WARNING" if status == "warn" else "MISSING"
        report(progress, f"{symbol}: {name}{(': ' + detail) if detail else ''}", status)

    report(progress, "Checking this computer...", "info")

    if sys.version_info[:2] >= MINIMUM_PYTHON:
        add("Host Python", "pass", platform.python_version())
    else:
        add(
            "Host Python",
            "fail",
            f"Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer is required.",
        )

    if tkinter_is_available():
        add("Python graphical support (Tkinter)", "pass")
    else:
        add("Python graphical support (Tkinter)", "fail", "Tkinter is unavailable.")

    if is_windows():
        build = windows_build_number()
        if build is not None and build >= VALIDATED_WINDOWS_BUILD_MINIMUM:
            add("Validated Windows baseline", "pass", f"Windows build {build}")
        else:
            add(
                "Validated Windows baseline",
                "warn",
                "Detected Windows build "
                f"{build or 'unknown'}, outside the validated Windows 11 baseline. "
                "Continuing with best-effort functional checks.",
            )

        if shutil.which("wsl.exe"):
            add("WSL command", "pass")
        else:
            add("WSL command", "fail", "wsl.exe was not found.")
            return checks, None

        distro, rows = select_wsl_distro()
        if distro:
            version = next((row[2] for row in rows if row[0] == distro), None)
            if version == 2:
                add("Ubuntu WSL 2 distribution", "pass", distro)
            else:
                add("Ubuntu WSL 2 distribution", "fail", f"{distro} is WSL {version}")
        else:
            available = ", ".join(row[0] for row in rows) or "none"
            add(
                "Ubuntu WSL 2 distribution",
                "fail",
                f"Expected {PREFERRED_WSL_DISTRO}; detected: {available}",
            )
            return checks, None

        desktop = find_docker_desktop()
        if desktop:
            add("Docker Desktop", "pass", str(desktop))
        else:
            add("Docker Desktop", "fail", "Docker Desktop was not found.")

        host_gpu_ok, host_gpu_detail = host_nvidia_check()
        add("Windows NVIDIA driver", "pass" if host_gpu_ok else "fail", host_gpu_detail)
        if host_gpu_ok:
            add(
                "GPU age/model policy",
                "pass",
                "The launcher does not reject NVIDIA hardware by model or age; "
                "functional container and model execution determine compatibility.",
            )
    elif is_linux():
        os_release = parse_os_release()
        linux_id = os_release.get("ID", "")
        version = os_release.get("VERSION_ID", "")
        if linux_id == VALIDATED_LINUX_ID and version.startswith(VALIDATED_LINUX_VERSION_PREFIX):
            add("Validated Linux baseline", "pass", f"Ubuntu {version}")
        else:
            detected = f"{linux_id or 'unknown'} {version or ''}".strip()
            add(
                "Validated Linux baseline",
                "warn",
                f"Detected {detected}, outside the validated Ubuntu 24.04 baseline. "
                "Continuing with best-effort functional checks.",
            )
        host_gpu_ok, host_gpu_detail = host_nvidia_check()
        add("Linux NVIDIA driver", "pass" if host_gpu_ok else "fail", host_gpu_detail)
        if host_gpu_ok:
            add(
                "GPU age/model policy",
                "pass",
                "The launcher does not reject NVIDIA hardware by model or age; "
                "functional container and model execution determine compatibility.",
            )
    else:
        add("Supported operating system", "fail", platform.system())
        return checks, None

    linux_python = linux_python_version(distro)
    if linux_python and linux_python >= MINIMUM_PYTHON:
        add("Linux Python", "pass", ".".join(map(str, linux_python)))
    else:
        add("Linux Python", "fail", "python3 3.10+ was not found.")

    command_checks = ["git", "python3"]
    for command in command_checks:
        exists = (
            wsl_has_command(distro, command)
            if is_windows()
            else shutil.which(command) is not None
        )
        add(f"Linux command: {command}", "pass" if exists else "fail")

    if chrome_candidates():
        add("Google Chrome", "pass", str(chrome_candidates()[0]))
    else:
        add(
            "Google Chrome",
            "warn",
            "Chrome was not found. The launcher can try the default browser, but the "
            f"validated deployment uses Chrome; see {GUIDE_NAME}.",
        )

    venv_check = run_linux_shell(
        "python3 -c 'import venv'",
        distro=distro,
        timeout=20,
    )
    add("Python virtual environments", "pass" if venv_check.ok else "fail")

    gpu_ok, gpu_detail = linux_or_wsl_nvidia_check(distro)
    add("GPU visible in WSL/Linux", "pass" if gpu_ok else "fail", gpu_detail)

    docker = docker_info(distro)
    if docker.ok:
        add("Docker daemon and user access", "pass", docker.stdout.strip())
        probe = docker_gpu_probe(distro, allow_pull=False)
        if probe.ok:
            add("NVIDIA GPU visible in Docker", "pass", probe.stdout.strip())
        elif probe.returncode == 3:
            add(
                "NVIDIA GPU visible in Docker",
                "warn",
                "Not tested because the probe image is not cached; setup will test it.",
            )
        else:
            add("NVIDIA GPU visible in Docker", "fail", probe.stderr.strip())
    else:
        add(
            "Docker daemon and user access",
            "fail",
            (docker.stderr or docker.stdout).strip() or "Docker is not ready.",
        )

    return checks, distro


def failed_checks(checks: Iterable[CheckResult]) -> list[CheckResult]:
    return [item for item in checks if item.status == "fail"]


# ---------------------------------------------------------------------------
# Docker startup and optional WSL repair
# ---------------------------------------------------------------------------
def start_docker_desktop(progress: ProgressCallback) -> None:
    desktop = find_docker_desktop()
    if desktop is None:
        raise SetupError(
            "Docker Desktop is not installed. Complete the Windows prerequisites in "
            "egomodelkit-machine-setup.pdf."
        )
    report(progress, "Starting Docker Desktop...", "info")
    try:
        subprocess.Popen(
            [str(desktop)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        raise SetupError(f"Docker Desktop could not be started: {exc}") from exc


def ensure_docker_ready(distro: str | None, progress: ProgressCallback) -> None:
    """Start Docker Desktop on Windows and wait; never restart WSL automatically."""
    initial = docker_info(distro)
    if initial.ok:
        report(progress, "Docker is available.", "pass")
    elif is_windows():
        start_docker_desktop(progress)
        deadline = time.monotonic() + DOCKER_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            current = docker_info(distro)
            if current.ok:
                report(progress, "Docker Desktop and WSL integration are ready.", "pass")
                break
            time.sleep(3)
        else:
            raise SetupError(
                "Docker Desktop started, but Docker is still unavailable inside the "
                "Ubuntu WSL distribution. Check Docker Desktop > Settings > General > Use "
                "the WSL 2 based engine and Settings > Resources > WSL Integration. You "
                "may also use the explicit Restart WSL repair option; it can close active "
                "WSL sessions.",
                repair_wsl_may_help=True,
            )
    else:
        raise SetupError(
            "Docker is installed but its daemon is unavailable or this user lacks permission. "
            "A sudo-capable maintainer must start Docker and configure the docker group; "
            "this launcher does not use sudo."
        )

    report(progress, "Checking NVIDIA GPU access inside Docker...", "info")
    probe = docker_gpu_probe(distro, allow_pull=True)
    if not probe.ok:
        detail = (probe.stderr or probe.stdout).strip()
        raise SetupError(
            "Docker is running, but the NVIDIA GPU container check failed. Complete the "
            "GPU/container-runtime "
            f"prerequisites in {GUIDE_NAME}.\n\nTechnical detail: {detail[-1200:]}"
        )
    report(progress, "GPU container support is available.", "pass")




# ---------------------------------------------------------------------------
# Self-installation and desktop shortcuts
# ---------------------------------------------------------------------------
def write_default_icon(path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
<rect width="256" height="256" rx="40" fill="#1f2937"/>
<path d="M42 58h172v32H82v30h116v31H82v31h132v32H42z" fill="#f9fafb"/>
<circle cx="204" cy="55" r="20" fill="#38bdf8"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def install_self(paths: AppPaths, progress: ProgressCallback) -> None:
    """Copy this file and an adjacent guide into user-owned application data."""
    paths.app_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    source = Path(__file__).resolve()
    destination_is_source = (
        paths.installed_script.exists()
        and source == paths.installed_script.resolve()
    )
    if not destination_is_source:
        shutil.copy2(source, paths.installed_script)
    if not is_windows():
        paths.installed_script.chmod(0o755)

    source_guide = source.with_name(GUIDE_NAME)
    if source_guide.is_file():
        guide_is_installed_copy = (
            paths.installed_guide.exists()
            and source_guide.samefile(paths.installed_guide)
        )
        if not guide_is_installed_copy:
            shutil.copy2(source_guide, paths.installed_guide)
    elif not paths.installed_guide.is_file():
        logging.warning("Setup guide was not found next to the setup script: %s", source_guide)

    write_default_icon(paths.icon_file)
    report(progress, "Setup launcher installed for this user.", "pass")


def find_pythonw() -> Path:
    executable = Path(sys.executable)
    candidates = [
        executable.with_name("pythonw.exe"),
        Path(sys.prefix) / "pythonw.exe",
    ]
    return next((path for path in candidates if path.is_file()), executable)


def powershell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_windows_shortcut(paths: AppPaths) -> Path:
    pythonw = find_pythonw()
    script = str(paths.installed_script)
    ps = f"""
$desktop = [Environment]::GetFolderPath('Desktop')
if ([string]::IsNullOrWhiteSpace($desktop)) {{
    throw 'Windows desktop folder could not be resolved.'
}}
$linkPath = Join-Path $desktop 'EgoModelKit.lnk'
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($linkPath)
$link.TargetPath = {powershell_single_quote(str(pythonw))}
$link.Arguments = {powershell_single_quote('"' + script + '"')}
$link.WorkingDirectory = {powershell_single_quote(str(Path.home()))}
$link.Description = 'Launch EgoModelKit'
$link.IconLocation = {powershell_single_quote(str(pythonw) + ',0')}
$link.Save()
Write-Output $linkPath
""".strip()
    result = run_command(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
        timeout=60,
    )
    require_ok(result, "The Windows desktop shortcut could not be created.")
    return Path(result.stdout.strip().splitlines()[-1])


def desktop_exec_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def linux_desktop_directory() -> Path | None:
    result = run_command(["xdg-user-dir", "DESKTOP"], timeout=10)
    if result.ok and result.stdout.strip():
        path = Path(result.stdout.strip()).expanduser()
        if path.is_dir():
            return path
    fallback = Path.home() / "Desktop"
    return fallback if fallback.is_dir() else None


def create_linux_shortcut(paths: AppPaths) -> Path:
    python = shutil.which("python3") or sys.executable
    app_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "applications"
    app_dir.mkdir(parents=True, exist_ok=True)
    desktop_file = app_dir / "egomodelkit.desktop"
    content = f"""[Desktop Entry]
Type=Application
Version=1.0
Name=EgoModelKit
Comment=Launch the local EgoModelKit research application
Exec={desktop_exec_quote(str(python))} {desktop_exec_quote(str(paths.installed_script))}
Icon={paths.icon_file}
Terminal=false
Categories=Science;Education;
StartupNotify=true
Actions=Stop;Uninstall;

[Desktop Action Stop]
Name=Stop EgoModelKit
Exec={desktop_exec_quote(str(python))} {desktop_exec_quote(str(paths.installed_script))} --stop

[Desktop Action Uninstall]
Name=Uninstall EgoModelKit
Exec={desktop_exec_quote(str(python))} {desktop_exec_quote(str(paths.installed_script))} --uninstall
"""
    desktop_file.write_text(content, encoding="utf-8")
    desktop_file.chmod(0o755)

    desktop = linux_desktop_directory()
    if desktop:
        desktop_copy = desktop / "EgoModelKit.desktop"
        shutil.copy2(desktop_file, desktop_copy)
        desktop_copy.chmod(0o755)
        # GNOME may require trust metadata; failure is harmless and file remains usable.
        run_command(["gio", "set", str(desktop_copy), "metadata::trusted", "true"], timeout=10)
    return desktop_file


def create_shortcut(paths: AppPaths, progress: ProgressCallback) -> None:
    shortcut = create_windows_shortcut(paths) if is_windows() else create_linux_shortcut(paths)
    report(progress, f"Desktop shortcut created: {shortcut}", "pass")


# ---------------------------------------------------------------------------
# Transactional Linux-side Git and Python installation
# ---------------------------------------------------------------------------
def ensure_install_script() -> str:
    """Return the Bash transaction used on native Linux and WSL.

    A candidate commit receives its own repository checkout and virtual
    environment.  The active symlink changes only after installation and import
    checks pass.  Therefore a failed update leaves the prior release untouched.
    """
    return r'''
set -Eeuo pipefail

APP_ROOT="$HOME/.local/share/EgoModelKit"
MIRROR="$APP_ROOT/git-cache.git"
RELEASES="$APP_ROOT/releases"
ACTIVE_LINK="$APP_ROOT/repo"
ACTIVE_MARKER="$APP_ROOT/active-commit.txt"
REPO_URL="${1:?missing repository URL}"
REPO_REF="${2:-main}"
KEEP_RELEASES="${3:-3}"
NODE_VERSION="${4:?missing Node.js version}"
NODE_X64_SHA256="${5:?missing Node.js x64 checksum}"
NODE_ARM64_SHA256="${6:?missing Node.js arm64 checksum}"

mkdir -p "$APP_ROOT" "$RELEASES"

LOCK_DIR="$APP_ROOT/.install-lock"
if [ -d "$LOCK_DIR" ] && find "$LOCK_DIR" -maxdepth 0 -mmin +360 | grep -q .; then
    rm -rf "$LOCK_DIR"
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    printf 'EMK_ERROR=%s\n' "Another EgoModelKit installation or update is already running."
    exit 30
fi

# A Python virtual environment is not safely movable: generated console scripts
# contain the absolute path of the venv interpreter.  Build each candidate at
# its permanent release path and remove it on failure instead of moving it from
# a temporary .staging directory after installation.
candidate_in_progress=""
cleanup_install() {
    if [ -n "$candidate_in_progress" ]; then
        rm -rf -- "$candidate_in_progress"
    fi
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_install EXIT

active_commit=""
if [ -f "$ACTIVE_MARKER" ]; then
    active_commit="$(tr -d '\r\n' < "$ACTIVE_MARKER")"
fi

if [ ! -d "$MIRROR" ]; then
    git init --bare "$MIRROR" >/dev/null
fi
if git --git-dir="$MIRROR" remote get-url origin >/dev/null 2>&1; then
    git --git-dir="$MIRROR" remote set-url origin "$REPO_URL"
else
    git --git-dir="$MIRROR" remote add origin "$REPO_URL"
fi

fetch_warning=""
candidate_commit=""
if git --git-dir="$MIRROR" fetch --force --prune origin "$REPO_REF"; then
    candidate_commit="$(git --git-dir="$MIRROR" rev-parse FETCH_HEAD)"
else
    fetch_warning="Git update failed; continuing with the last successfully installed commit."
    if [ -n "$active_commit" ]; then
        candidate_commit="$active_commit"
    else
        printf 'EMK_ERROR=%s\n' \
            "The public repository could not be downloaded and no prior installation exists."
        exit 31
    fi
fi

updated=0
active_repo=""
if [ -L "$ACTIVE_LINK" ]; then
    active_repo="$(readlink -f "$ACTIVE_LINK" 2>/dev/null || true)"
fi

# Reuse the active release only when it represents the requested commit and its
# interpreter, import, and generated console entry point all still work.
if [ "$active_commit" = "$candidate_commit" ] \
   && [ -n "$active_repo" ] \
   && [ -x "$active_repo/.venv/bin/python" ] \
   && [ -x "$active_repo/.venv/bin/egomodelkit" ] \
   && [ -f "$active_repo/src/egomodelkit/web/dist/index.html" ] \
   && "$active_repo/.venv/bin/python" -c 'import egomodelkit' >/dev/null 2>&1 \
   && "$active_repo/.venv/bin/egomodelkit" --help >/dev/null 2>&1; then
    repo_dir="$active_repo"
    release_root="$(dirname "$repo_dir")"
    venv_dir="$repo_dir/.venv"
else
    # The timestamp and process ID make the immutable release directory unique,
    # including when repairing a broken installation of the same Git commit.
    release_id="$candidate_commit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    release_root="$RELEASES/$release_id"
    repo_dir="$release_root/repo"
    venv_dir="$repo_dir/.venv"
    candidate_in_progress="$release_root"

    rm -rf "$release_root"
    mkdir -p "$release_root"

    # Clone from the local mirror.  No source checkout is placed under /mnt/c.
    # The venv is created at its permanent path and is never renamed or moved.
    git clone --no-checkout "$MIRROR" "$repo_dir" >/dev/null
    git -C "$repo_dir" checkout --detach "$candidate_commit" >/dev/null

    # Resolve a Linux Node.js toolchain for the production React build. A
    # compatible system installation is reused. Otherwise, install a pinned
    # official Node.js binary under this user's application data. This avoids
    # sudo/admin access and avoids accidentally invoking Windows npm in WSL.
    web_dir="$repo_dir/src/egomodelkit/web"
    if [ ! -f "$web_dir/package.json" ]; then
        printf 'EMK_ERROR=%s\n' "The EgoModelKit frontend source was not found."
        exit 32
    fi

    node_bin_dir=""
    node_source=""
    if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
        system_node="$(command -v node)"
        system_npm="$(command -v npm)"
        case "$system_node:$system_npm" in
            /mnt/*|*:/mnt/*)
                # Ignore Windows Node.js/npm exposed through WSL interop.
                ;;
            *)
                if "$system_node" -e '
                    const [major, minor] = process.versions.node.split(".").map(Number);
                    const supported =
                        (major === 20 && minor >= 19) ||
                        (major === 22 && minor >= 12) ||
                        major > 22;
                    process.exit(supported ? 0 : 1);
                ' >/dev/null 2>&1; then
                    node_bin_dir="$(dirname "$system_node")"
                    node_source="system"
                fi
                ;;
        esac
    fi

    if [ -z "$node_bin_dir" ]; then
        machine_arch="$(uname -m)"
        case "$machine_arch" in
            x86_64|amd64)
                node_arch="x64"
                node_sha256="$NODE_X64_SHA256"
                ;;
            aarch64|arm64)
                node_arch="arm64"
                node_sha256="$NODE_ARM64_SHA256"
                ;;
            *)
                printf 'EMK_ERROR=%s\n' \
                    "Automatic Node.js setup does not support Linux architecture: $machine_arch"
                exit 33
                ;;
        esac

        node_name="node-v${NODE_VERSION}-linux-${node_arch}"
        node_tools="$APP_ROOT/tools/node"
        node_root="$node_tools/$node_name"
        node_archive="${node_name}.tar.xz"
        node_url="https://nodejs.org/download/release/v${NODE_VERSION}/${node_archive}"

        if [ ! -x "$node_root/bin/node" ] || [ ! -x "$node_root/bin/npm" ]; then
            mkdir -p "$node_tools"
            rm -rf -- "$node_root"
            printf 'EMK_INFO=%s\n' "Downloading the private Node.js frontend build toolchain."
            if ! python3 - \
                "$node_url" "$node_archive" "$node_sha256" \
                "$node_tools" "$node_name" <<'EMK_NODE_INSTALL'
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

url, archive_name, expected_sha256, tools_text, node_name = sys.argv[1:]
tools = Path(tools_text)
target = tools / node_name
tools.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(prefix=".node-download-", dir=tools) as temp_text:
    temp = Path(temp_text)
    archive = temp / archive_name
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "EgoModelKit-Setup"})
    with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Node.js download checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    extract_root = temp / "extract"
    extract_root.mkdir()
    with tarfile.open(archive, mode="r:xz") as bundle:
        bundle.extractall(extract_root, filter="data")

    extracted = extract_root / node_name
    if not (extracted / "bin" / "node").is_file() or not (extracted / "bin" / "npm").is_file():
        raise RuntimeError("The downloaded Node.js archive did not contain node and npm.")

    staging = tools / f".{node_name}.install-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.move(str(extracted), str(staging))
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
EMK_NODE_INSTALL
            then
                printf 'EMK_ERROR=%s%s\n' \
                    "The private Node.js frontend build toolchain could not be downloaded or " \
                    "verified. Check internet access and rerun setup."
                exit 34
            fi
        fi
        node_bin_dir="$node_root/bin"
        node_source="private"
    fi

    export PATH="$node_bin_dir:$PATH"
    export npm_config_cache="$APP_ROOT/cache/npm"
    mkdir -p "$npm_config_cache"

    if ! node -e '
        const [major, minor] = process.versions.node.split(".").map(Number);
        const supported =
            (major === 20 && minor >= 19) ||
            (major === 22 && minor >= 12) ||
            major > 22;
        process.exit(supported ? 0 : 1);
    '; then
        printf 'EMK_ERROR=%s\n' \
            "The resolved Linux Node.js version is not supported by the EgoModelKit frontend."
        exit 35
    fi
    if ! npm --version >/dev/null 2>&1; then
        printf 'EMK_ERROR=%s\n' "The resolved Linux npm executable could not run."
        exit 36
    fi
    resolved_node_version="$(node --version)"

    python3 -m venv "$venv_dir"
    "$venv_dir/bin/python" -m pip install --upgrade pip
    (
        cd "$repo_dir"
        "$venv_dir/bin/python" -m pip install -e ".[gui]"
    )

    # Build the production React interface once during installation. The normal
    # desktop launch then needs only the single `egomodelkit gui` server.
    (
        cd "$web_dir"
        npm ci
        npm run build
    )
    if [ ! -f "$web_dir/dist/index.html" ]; then
        printf 'EMK_ERROR=%s\n' "The frontend build did not produce dist/index.html."
        exit 37
    fi
    if ! find "$web_dir/dist/assets" -maxdepth 1 -type f -print -quit 2>/dev/null | grep -q .; then
        printf 'EMK_ERROR=%s\n' "The frontend build did not produce static assets."
        exit 38
    fi

    # Static smoke checks run through the permanent paths before activation.
    "$venv_dir/bin/python" -c 'import egomodelkit; print(egomodelkit.__file__)'
    "$venv_dir/bin/egomodelkit" --help >/dev/null
    printf '%s\n' "$candidate_commit" > "$release_root/.install-complete"

    candidate_in_progress=""
    updated=1
fi

# Activate only after the candidate has a verified executable.  Preserve an
# older installer/manual checkout if `repo` is a real directory rather than the
# managed active symlink used by this launcher.
if [ -e "$ACTIVE_LINK" ] && [ ! -L "$ACTIVE_LINK" ]; then
    legacy="$APP_ROOT/legacy-repo-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$ACTIVE_LINK" "$legacy"
    printf 'EMK_LEGACY_REPO=%s\n' "$legacy"
fi
rm -f "$APP_ROOT/.repo-next"
ln -s "$repo_dir" "$APP_ROOT/.repo-next"
mv -Tf "$APP_ROOT/.repo-next" "$ACTIVE_LINK"
printf '%s\n' "$candidate_commit" > "$ACTIVE_MARKER.tmp"
mv -f "$ACTIVE_MARKER.tmp" "$ACTIVE_MARKER"

# Keep the active release plus a small rollback window.  Never remove active.
if [ "$KEEP_RELEASES" -gt 0 ] 2>/dev/null; then
    mapfile -t old_releases < <(
        find "$RELEASES" -mindepth 1 -maxdepth 1 -type d ! -name '.staging-*' -printf '%T@ %p\n' \
          | sort -nr | awk -v keep="$KEEP_RELEASES" 'NR>keep {sub(/^[^ ]+ /, ""); print}'
    )
    for old in "${old_releases[@]:-}"; do
        [ -n "$old" ] || continue
        [ "$old" = "$release_root" ] && continue
        rm -rf -- "$old"
    done
fi

printf 'EMK_COMMIT=%s\n' "$candidate_commit"
printf 'EMK_REPO_DIR=%s\n' "$repo_dir"
printf 'EMK_VENV_PYTHON=%s\n' "$venv_dir/bin/python"
printf 'EMK_EXECUTABLE=%s\n' "$venv_dir/bin/egomodelkit"
printf 'EMK_UPDATED=%s\n' "$updated"
printf 'EMK_WARNING=%s\n' "$fetch_warning"
printf 'EMK_NODE_VERSION=%s\n' "${resolved_node_version:-not-required}"
printf 'EMK_NODE_SOURCE=%s\n' "${node_source:-existing-release}"
'''.strip()


def parse_key_value_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("EMK_") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def ensure_egomodelkit_install(
    distro: str | None,
    progress: ProgressCallback,
) -> InstallResult:
    report(progress, "Checking for EgoModelKit code updates...", "info")
    result = run_linux_script(
        ensure_install_script(),
        arguments=(
            REPOSITORY_URL,
            REPOSITORY_BRANCH,
            str(KEEP_RELEASES),
            NODE_VERSION,
            NODE_X64_SHA256,
            NODE_ARM64_SHA256,
        ),
        distro=distro,
        timeout=1800,
    )
    values = parse_key_value_lines(result.stdout)
    if not result.ok:
        error = values.get("EMK_ERROR") or (result.stderr or result.stdout).strip()
        raise SetupError(
            "EgoModelKit could not be installed or updated. The previous active "
            "installation was not replaced."
            + (f"\n\nTechnical detail: {error[-1500:]}" if error else "")
        )

    required = ["EMK_COMMIT", "EMK_REPO_DIR", "EMK_VENV_PYTHON", "EMK_EXECUTABLE"]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise SetupError(
            "The installation completed without returning expected state: " + ", ".join(missing)
        )

    updated = values.get("EMK_UPDATED") == "1"
    if updated:
        report(progress, f"Application installed at commit {values['EMK_COMMIT'][:12]}.", "pass")
    else:
        report(progress, f"Application is already current at {values['EMK_COMMIT'][:12]}.", "pass")
    node_version = values.get("EMK_NODE_VERSION", "")
    node_source = values.get("EMK_NODE_SOURCE", "")
    if node_version and node_version != "not-required":
        source_label = (
            "private user-level toolchain"
            if node_source == "private"
            else "Linux system toolchain"
        )
        report(progress, f"Frontend build used Node.js {node_version} ({source_label}).", "pass")

    warning = values.get("EMK_WARNING", "")
    if warning:
        report(progress, warning, "warn")
    legacy_repo = values.get("EMK_LEGACY_REPO", "")
    if legacy_repo:
        report(progress, f"Previous unmanaged repository preserved at {legacy_repo}.", "warn")

    return InstallResult(
        commit=values["EMK_COMMIT"],
        repo_dir=values["EMK_REPO_DIR"],
        venv_python=values["EMK_VENV_PYTHON"],
        egomodelkit_executable=values["EMK_EXECUTABLE"],
        updated=updated,
        update_warning=warning,
    )



# ---------------------------------------------------------------------------
# Process management, readiness, and browser opening
# ---------------------------------------------------------------------------
def predictable_port() -> int:
    """Choose a stable per-user port to reduce fast-user-switch collisions."""
    identity = f"{platform.node()}:{getpass.getuser()}".encode("utf-8", errors="replace")
    return GUI_BASE_PORT + (zlib.crc32(identity) % GUI_PORT_SPAN)


def gui_url(port: int) -> str:
    return f"http://{GUI_HOST}:{port}"


def http_get(path: str, port: int, timeout: float = 2.0) -> tuple[int, bytes] | None:
    try:
        connection = http.client.HTTPConnection(GUI_HOST, port, timeout=timeout)
        connection.request("GET", path)
        response = connection.getresponse()
        data = response.read(128 * 1024)
        connection.close()
        return response.status, data
    except (OSError, http.client.HTTPException):
        return None


def egomodelkit_is_ready(port: int) -> bool:
    models_response = http_get("/api/models", port)
    if models_response is None or models_response[0] != 200:
        return False
    try:
        payload = json.loads(models_response[1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, (list, dict)):
        return False

    root_response = http_get("/", port)
    if root_response is None or root_response[0] != 200:
        return False
    try:
        html = root_response[1].decode("utf-8", errors="strict").lower()
    except UnicodeError:
        return False
    return "<!doctype html" in html or '<div id="root"' in html


def backend_only_gui_is_running(port: int) -> bool:
    api = http_get("/api/models", port)
    root = http_get("/", port)
    return api is not None and api[0] == 200 and root is not None and root[0] == 404


def read_state(paths: AppPaths) -> dict[str, object]:
    """Read launcher state defensively; stale or invalid files are ignored."""
    try:
        payload = json.loads(paths.state_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def managed_port(paths: AppPaths) -> int:
    value = read_state(paths).get("port")
    if (
        isinstance(value, int)
        and 1 <= value <= 65535
        and (egomodelkit_is_ready(value) or backend_only_gui_is_running(value))
    ):
        return value
    return predictable_port()


def managed_commit(paths: AppPaths) -> str:
    value = read_state(paths).get("commit")
    return value if isinstance(value, str) else ""


def stop_gui_process(
    paths: AppPaths,
    distro: str | None,
    progress: ProgressCallback = noop_progress,
) -> bool:
    """Stop only the managed EgoModelKit GUI process for this user."""
    port = managed_port(paths)
    was_running = egomodelkit_is_ready(port) or backend_only_gui_is_running(port)
    if not was_running and not port_is_open(port):
        paths.state_file.unlink(missing_ok=True)
        report(progress, "EgoModelKit is not currently running.", "pass")
        return False

    report(progress, "Stopping the local EgoModelKit GUI...", "info")
    # Inspect /proc and signal exact matching PIDs instead of using pkill -f.
    # A broad command-line pattern can accidentally match the control shell that
    # contains the pattern itself, especially when invoked through WSL.
    stop_script = r'''
set -u
port="${1:?missing port}"
python3 - "$port" <<'PY_STOP'
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

PORT = sys.argv[1]


def process_arguments(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def is_egomodelkit_gui(arguments: list[str]) -> bool:
    if not arguments or "gui" not in arguments:
        return False
    has_entrypoint = any(Path(value).name == "egomodelkit" for value in arguments)
    has_module = any(
        arguments[index] == "-m" and arguments[index + 1] == "egomodelkit"
        for index in range(len(arguments) - 1)
    )
    if not (has_entrypoint or has_module):
        return False
    return any(
        value == f"--port={PORT}"
        or (value == "--port" and index + 1 < len(arguments) and arguments[index + 1] == PORT)
        for index, value in enumerate(arguments)
    )


def matching_pids() -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        if is_egomodelkit_gui(process_arguments(pid)):
            matches.append(pid)
    return matches


def signal_processes(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


pids = matching_pids()
signal_processes(pids, signal.SIGTERM)
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    remaining = matching_pids()
    if not remaining:
        break
    time.sleep(0.5)
else:
    signal_processes(remaining, signal.SIGKILL)

print(f"matched={len(pids)} remaining={len(matching_pids())}")
PY_STOP
'''.strip()
    result = run_linux_script(
        stop_script,
        arguments=(str(port),),
        distro=distro,
        timeout=30,
    )
    if not result.ok:
        detail = (result.stderr or result.stdout).strip()
        raise SetupError(
            "EgoModelKit could not be stopped. Open Logs or ask a maintainer for help."
            + (f"\n\nTechnical detail: {detail[-1200:]}" if detail else "")
        )

    if "matched=0" in result.stdout:
        # The predictable port may belong to another application. Never stop or
        # block uninstall on a process that is not an exact EgoModelKit GUI.
        paths.state_file.unlink(missing_ok=True)
        report(progress, "EgoModelKit is not currently running.", "pass")
        return False

    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and port_is_open(port):
        time.sleep(0.5)
    if port_is_open(port):
        raise SetupError(
            f"EgoModelKit did not release local port {port}. Open Logs or ask a "
            "maintainer for help."
        )

    paths.state_file.unlink(missing_ok=True)
    report(progress, "EgoModelKit has been stopped.", "pass")
    return True


def stop_backend_only_gui(port: int, distro: str | None) -> None:
    """Repair a prior backend-only process without affecting other programs."""
    paths = get_paths()
    state = read_state(paths)
    state["port"] = port
    write_state(paths, state)
    stop_gui_process(paths, distro)


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((GUI_HOST, port)) == 0


def write_state(paths: AppPaths, payload: dict[str, object]) -> None:
    paths.app_dir.mkdir(parents=True, exist_ok=True)
    temp = paths.state_file.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(paths.state_file)


def spawn_gui(
    install: InstallResult,
    distro: str | None,
    paths: AppPaths,
    port: int,
    progress: ProgressCallback,
) -> int:
    """Start one detached GUI process and capture backend output in a log."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = paths.process_log.open("a", encoding="utf-8", buffering=1)
    stamp = datetime.now(timezone.utc).isoformat()
    log_handle.write(f"\n===== GUI launch {stamp} commit {install.commit} port {port} =====\n")

    command_inside = (
        f"cd {quote_bash(install.repo_dir)} && "
        f"exec {quote_bash(install.egomodelkit_executable)} gui --port {port} --no-browser"
    )
    if is_windows():
        command = ["wsl.exe", "-d", str(distro), "--", "bash", "-lc", command_inside]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            startupinfo=startupinfo,
            close_fds=True,
        )
    else:
        command = ["bash", "-lc", command_inside]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    log_handle.close()
    write_state(
        paths,
        {
            "scriptVersion": SETUP_SCRIPT_VERSION,
            "pid": process.pid,
            "port": port,
            "commit": install.commit,
            "startedAt": stamp,
            "platform": platform.system(),
            "distro": distro,
        },
    )
    report(progress, "Starting the local EgoModelKit GUI...", "info")
    return process.pid


def wait_for_gui(port: int, pid: int, paths: AppPaths, progress: ProgressCallback) -> None:
    deadline = time.monotonic() + GUI_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if egomodelkit_is_ready(port):
            report(progress, "EgoModelKit is ready.", "pass")
            return
        time.sleep(1)
    tail = ""
    try:
        tail = "\n".join(
            paths.process_log.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-40:]
        )
    except OSError:
        pass
    raise SetupError(
        "The EgoModelKit GUI did not become reachable. Open Logs for backend details."
        + (f"\n\nRecent backend output:\n{tail[-2000:]}" if tail else "")
    )


def chrome_candidates() -> list[Path]:
    candidates: list[Path] = []
    if is_windows():
        roots = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("LOCALAPPDATA", "")),
        ]
        for root in roots:
            candidates.extend(
                [
                    root / "Google/Chrome/Application/chrome.exe",
                    root / "Chromium/Application/chrome.exe",
                ]
            )
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            resolved = shutil.which(name)
            if resolved:
                candidates.append(Path(resolved))
    return [path for path in candidates if path.is_file()]


def open_browser(url: str) -> None:
    for candidate in chrome_candidates():
        try:
            subprocess.Popen(
                [str(candidate), url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if is_windows() else 0,
                start_new_session=not is_windows(),
            )
            logging.info("Opened Chrome: %s", candidate)
            return
        except OSError:
            logging.exception("Could not open browser candidate %s", candidate)
    webbrowser.open(url, new=2)
    logging.info("Opened default browser")


def reuse_or_start_gui(
    install: InstallResult,
    distro: str | None,
    paths: AppPaths,
    progress: ProgressCallback,
) -> LaunchResult:
    port = predictable_port()
    url = gui_url(port)
    if egomodelkit_is_ready(port):
        running_commit = managed_commit(paths)
        update_pending = bool(running_commit and running_commit != install.commit)
        if not running_commit and install.updated:
            update_pending = True
        report(progress, "EgoModelKit is already running; reusing the existing GUI.", "pass")
        if update_pending:
            report(
                progress,
                "An update is installed but the running session still uses the previous build. "
                "Finish any model run, then stop and restart EgoModelKit to apply it.",
                "warn",
            )
        open_browser(url)
        return LaunchResult(
            url=url,
            reused=True,
            update_pending_restart=update_pending,
        )
    if backend_only_gui_is_running(port):
        report(progress, "Repairing an older backend-only GUI process...", "info")
        stop_backend_only_gui(port, distro)
    if port_is_open(port):
        raise SetupError(
            f"Local port {port} is already used by another program. EgoModelKit did "
            "not stop or alter that process. "
            "Close the conflicting program or review the logs."
        )
    pid = spawn_gui(install, distro, paths, port, progress)
    wait_for_gui(port, pid, paths, progress)
    open_browser(url)
    return LaunchResult(url=url, reused=False, update_pending_restart=False)


# ---------------------------------------------------------------------------
# Main workflows
# ---------------------------------------------------------------------------
def ensure_prerequisites_for_workflow(
    progress: ProgressCallback,
) -> tuple[list[CheckResult], str | None]:
    checks, distro = collect_prerequisite_checks(progress=progress)
    failures = failed_checks(checks)

    # Docker may legitimately be stopped at launch time.  On Windows it can be
    # started by this script, so remove only that one failure before deciding.
    actionable = failures[:]
    if is_windows() and find_docker_desktop() is not None:
        actionable = [item for item in actionable if item.name != "Docker daemon and user access"]

    if actionable:
        summary = "\n".join(f"- {item.name}: {item.detail}" for item in actionable)
        raise SetupError(
            f"This computer is missing required prerequisites. Complete {GUIDE_NAME}, "
            f"then retry.\n\n{summary}"
        )
    return checks, distro


def perform_automatic_workflow(
    paths: AppPaths,
    progress: ProgressCallback,
) -> WorkerOutcome:
    """Choose setup, update, repair, or launch from the current user state."""
    checks, distro = ensure_prerequisites_for_workflow(progress)

    # Re-copy the launcher and recreate the shortcut on every successful run.
    # This quietly repairs missing or outdated user-level launcher files.
    install_self(paths, progress)
    create_shortcut(paths, progress)

    ensure_docker_ready(distro, progress)
    install = ensure_egomodelkit_install(distro, progress)
    launch = reuse_or_start_gui(install, distro, paths, progress)

    if launch.update_pending_restart:
        message = (
            "An EgoModelKit update has been installed, but the currently running "
            "session was not interrupted. Finish or cancel any model run, click Stop "
            "EgoModelKit, and then click Restart EgoModelKit "
            "to use the updated interface and code."
        )
    else:
        message = "EgoModelKit is ready and has been opened in your browser."

    return WorkerOutcome(
        True,
        message,
        gui_url=launch.url,
        checks=checks,
        gui_running=True,
        update_pending_restart=launch.update_pending_restart,
    )


# ---------------------------------------------------------------------------
# Stop and uninstall workflows
# ---------------------------------------------------------------------------
def remove_shortcuts(paths: AppPaths) -> None:
    """Remove user-facing shortcuts without touching other applications."""
    if is_windows():
        ps = r'''
$desktop = [Environment]::GetFolderPath('Desktop')
Remove-Item (Join-Path $desktop 'EgoModelKit.lnk') -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $desktop 'Uninstall EgoModelKit.lnk') -Force -ErrorAction SilentlyContinue
'''.strip()
        result = run_command(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
            timeout=30,
        )
        if not result.ok:
            logging.warning("Windows shortcut cleanup returned %s", result.returncode)
        return

    app_file = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        / "applications/egomodelkit.desktop"
    )
    app_file.unlink(missing_ok=True)
    desktop = linux_desktop_directory()
    if desktop:
        (desktop / "EgoModelKit.desktop").unlink(missing_ok=True)
    updater = shutil.which("update-desktop-database")
    if updater:
        run_command([updater, str(app_file.parent)], timeout=15)


def remove_linux_side_install(distro: str | None) -> None:
    """Remove the native-Linux or WSL user installation and private toolchain."""
    result = run_linux_script(
        r'''
set -eu
app_root="$HOME/.local/share/EgoModelKit"
state_root="$HOME/.local/state/EgoModelKit"
rm -rf -- "$app_root" "$state_root"
'''.strip(),
        distro=distro,
        timeout=120,
    )
    if not result.ok:
        detail = (result.stderr or result.stdout).strip()
        raise SetupError(
            "The EgoModelKit user installation could not be fully removed."
            + (f"\n\nTechnical detail: {detail[-1200:]}" if detail else "")
        )


def close_log_handlers() -> None:
    """Close setup logs before Windows removes the containing directory."""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def schedule_windows_host_cleanup(paths: AppPaths) -> None:
    """Remove locked Windows launcher files after this process exits."""
    temp_dir = Path(os.environ.get("TEMP", Path.home()))
    cleanup = temp_dir / f"egomodelkit-uninstall-{os.getpid()}.cmd"
    script = (
        "@echo off\r\n"
        f":wait\r\ntasklist /FI \"PID eq {os.getpid()}\" 2>NUL | find \"{os.getpid()}\" >NUL\r\n"
        "if not errorlevel 1 (timeout /t 1 /nobreak >NUL & goto wait)\r\n"
        f'rmdir /s /q "{paths.app_dir}"\r\n'
        'del /q "%~f0"\r\n'
    )
    cleanup.write_text(script, encoding="utf-8", newline="")
    subprocess.Popen(
        ["cmd.exe", "/c", str(cleanup)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        close_fds=True,
    )


def uninstall_user_installation(paths: AppPaths, distro: str | None) -> None:
    """Remove EgoModelKit user files while preserving outputs and prerequisites."""
    stop_gui_process(paths, distro)
    remove_shortcuts(paths)
    remove_linux_side_install(distro)

    if is_windows():
        os.chdir(Path.home())
        close_log_handlers()
        schedule_windows_host_cleanup(paths)
    else:
        state_root = (
            Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            / APP_NAME
        )
        shutil.rmtree(state_root, ignore_errors=True)
        # Linux permits unlinking this running script from the user data folder.
        shutil.rmtree(paths.app_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# File/folder open helpers and graphical fallback
# ---------------------------------------------------------------------------
def open_path(path: Path) -> bool:
    try:
        if is_windows():
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        command = shutil.which("xdg-open") or shutil.which("gio")
        if not command:
            return False
        args = (
            [command, str(path)]
            if Path(command).name == "xdg-open"
            else [command, "open", str(path)]
        )
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        logging.exception("Could not open path: %s", path)
        return False


def show_fallback_error(message: str, paths: AppPaths) -> None:
    logging.error(message)
    if is_windows():
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "EgoModelKit Setup", 0x10)
            return
        except Exception:
            pass
    zenity = shutil.which("zenity")
    if zenity:
        run_command(
            [zenity, "--error", "--title=EgoModelKit Setup", f"--text={message}"],
            timeout=30,
        )
        return
    print(message, file=sys.stderr)
    print(f"Logs: {paths.log_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------
class SetupWindow:
    """Small graphical window for setup, launch, stop, and uninstall actions."""

    def __init__(self, paths: AppPaths, log_path: Path, initial_mode: str = "") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.paths = paths
        self.log_path = log_path
        self.root = tk.Tk()
        self.root.title("EgoModelKit")
        self.root.geometry("800x500")
        self.root.minsize(720, 420)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.last_outcome: WorkerOutcome | None = None
        self.running = False
        self.initial_mode = initial_mode

        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="EgoModelKit", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Checking, preparing, and opening the application.",
        ).pack(anchor="w", pady=(2, 14))

        self.status_var = tk.StringVar(value="Starting...")
        ttk.Label(outer, textvariable=self.status_var, wraplength=740).pack(anchor="w")
        self.progressbar = ttk.Progressbar(outer, mode="indeterminate")
        self.progressbar.pack(fill="x", pady=(8, 12))

        frame = ttk.Frame(outer)
        frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(frame, height=13, activestyle="none")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.detail_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.detail_var, wraplength=740).pack(
            anchor="w", pady=(10, 6)
        )

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(6, 0))
        self.open_button = ttk.Button(buttons, text="Open EgoModelKit", command=self.open_gui)
        self.logs_button = ttk.Button(buttons, text="Open Logs", command=self.open_logs)
        self.retry_button = ttk.Button(buttons, text="Retry", command=self.start)
        self.stop_button = ttk.Button(buttons, text="Stop EgoModelKit", command=self.confirm_stop)
        self.uninstall_button = ttk.Button(
            buttons,
            text="Uninstall EgoModelKit",
            command=self.confirm_uninstall,
        )
        self.wsl_button = ttk.Button(
            buttons,
            text="Restart WSL and Retry",
            command=self.restart_wsl_and_retry,
        )
        self.exit_button = ttk.Button(buttons, text="Exit", command=self.root.destroy)

        self.logs_button.pack(side="left")
        self.uninstall_button.pack(side="left", padx=(8, 0))
        self.exit_button.pack(side="right")
        self.root.after(100, self.poll_events)

    def add_line(self, message: str, status: str) -> None:
        prefix = {"pass": "OK", "fail": "X", "warn": "!", "info": "-"}.get(status, "-")
        self.listbox.insert("end", f"{prefix}  {message}")
        self.listbox.see("end")
        self.status_var.set(message)

    def progress_callback(self, message: str, status: str = "info") -> None:
        self.events.put(("progress", (message, status)))

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.last_outcome = None
        self.listbox.delete(0, "end")
        self.detail_var.set("")
        self.hide_action_buttons()
        self.set_management_buttons_enabled(False)
        self.progressbar.start(10)
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self) -> None:
        try:
            with WorkflowLock(self.paths):
                outcome = perform_automatic_workflow(self.paths, self.progress_callback)
        except SetupError as exc:
            logging.exception("Workflow failed")
            outcome = WorkerOutcome(
                False,
                str(exc),
                repair_wsl_may_help=exc.repair_wsl_may_help,
            )
        except Exception as exc:
            logging.exception("Unexpected workflow failure")
            outcome = WorkerOutcome(False, f"Unexpected setup error: {exc}")
        self.events.put(("done", outcome))

    def restart_wsl_and_retry(self) -> None:
        if self.running or not is_windows():
            return
        from tkinter import messagebox

        warning = (
            "Restarting WSL closes every active WSL distribution and may interrupt "
            "other terminals, containers, or model runs. Continue only when nobody "
            "is using WSL on this computer."
        )
        if not messagebox.askyesno("Restart WSL?", warning, parent=self.root):
            return
        result = run_command(["wsl.exe", "--shutdown"], timeout=60)
        if not result.ok:
            self.detail_var.set(
                "WSL could not be restarted. Open Logs or ask a maintainer for help."
            )
            return
        logging.warning("WSL was shut down after explicit user confirmation")
        self.start()

    def poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    message, status = payload  # type: ignore[misc]
                    self.add_line(str(message), str(status))
                elif kind == "done":
                    self.finish(payload)  # type: ignore[arg-type]
                elif kind == "stopped":
                    self.finish_stop(payload)  # type: ignore[arg-type]
                elif kind == "uninstalled":
                    self.finish_uninstall(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def hide_action_buttons(self) -> None:
        for button in (self.open_button, self.retry_button, self.stop_button, self.wsl_button):
            button.pack_forget()

    def set_management_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.uninstall_button.configure(state=state)
        self.stop_button.configure(state=state)

    def finish(self, outcome: WorkerOutcome) -> None:
        self.running = False
        self.last_outcome = outcome
        self.progressbar.stop()
        if outcome.update_pending_restart:
            self.status_var.set("Restart required to apply the update")
        else:
            self.status_var.set("Ready" if outcome.success else "Setup needs attention")
        self.detail_var.set(outcome.message)
        self.hide_action_buttons()
        self.set_management_buttons_enabled(True)
        if outcome.success and outcome.gui_url:
            self.open_button.pack(side="right", padx=(8, 0))
            if outcome.gui_running:
                self.stop_button.pack(side="right", padx=(8, 0))
        else:
            self.retry_button.configure(text="Retry")
            self.retry_button.pack(side="right", padx=(8, 0))
            if is_windows() and outcome.repair_wsl_may_help:
                self.wsl_button.pack(side="right", padx=(8, 0))

    def resolve_distro_for_management(self) -> str | None:
        if not is_windows():
            return None
        state_distro = read_state(self.paths).get("distro")
        if isinstance(state_distro, str) and state_distro:
            return state_distro
        distro, _rows = select_wsl_distro()
        return distro

    def confirm_stop(self) -> None:
        if self.running:
            return
        from tkinter import messagebox

        warning = (
            "Stopping EgoModelKit may interrupt an active model run. Continue only "
            "after the run has finished or has been cancelled. User-selected input and "
            "output folders are not deleted."
        )
        if not messagebox.askyesno("Stop EgoModelKit?", warning, parent=self.root):
            return
        self.running = True
        self.set_management_buttons_enabled(False)
        self.hide_action_buttons()
        self.progressbar.start(10)
        threading.Thread(target=self.stop_worker, daemon=True).start()

    def stop_worker(self) -> None:
        try:
            with WorkflowLock(self.paths):
                stop_gui_process(
                    self.paths,
                    self.resolve_distro_for_management(),
                    self.progress_callback,
                )
            outcome = WorkerOutcome(
                True,
                "EgoModelKit is stopped. Click Restart EgoModelKit to start the latest "
                "installed version.",
                gui_running=False,
            )
        except SetupError as exc:
            logging.exception("Stop failed")
            outcome = WorkerOutcome(False, str(exc))
        except Exception as exc:
            logging.exception("Unexpected stop failure")
            outcome = WorkerOutcome(False, f"Unexpected stop error: {exc}")
        self.events.put(("stopped", outcome))

    def confirm_uninstall(self) -> None:
        if self.running:
            return
        from tkinter import messagebox

        warning = (
            "Uninstalling removes this user's EgoModelKit code, private build tools, "
            "launcher files, logs, and shortcuts. It does not remove user-selected "
            "input/output folders, Docker, WSL, Python, Git, NVIDIA drivers, or shared "
            "Docker images. Any active model run will be interrupted."
        )
        if not messagebox.askyesno("Uninstall EgoModelKit?", warning, parent=self.root):
            return

        self.running = True
        self.set_management_buttons_enabled(False)
        self.hide_action_buttons()
        self.progressbar.start(10)
        threading.Thread(target=self.uninstall_worker, daemon=True).start()

    def uninstall_worker(self) -> None:
        try:
            distro = self.resolve_distro_for_management()
            with WorkflowLock(self.paths):
                uninstall_user_installation(self.paths, distro)
            message = (
                "EgoModelKit has been uninstalled for this user. User-selected input "
                "and output folders "
                "and machine prerequisites were preserved."
            )
            outcome = WorkerOutcome(True, message)
        except SetupError as exc:
            logging.exception("Uninstall failed")
            outcome = WorkerOutcome(False, str(exc))
        except Exception as exc:
            logging.exception("Unexpected uninstall failure")
            outcome = WorkerOutcome(False, f"Unexpected uninstall error: {exc}")
        self.events.put(("uninstalled", outcome))

    def finish_stop(self, outcome: WorkerOutcome) -> None:
        self.finish(outcome)
        if outcome.success:
            self.retry_button.configure(text="Restart EgoModelKit")
            self.retry_button.pack(side="right", padx=(8, 0))

    def finish_uninstall(self, outcome: WorkerOutcome) -> None:
        self.running = False
        self.progressbar.stop()
        self.status_var.set("Uninstalled" if outcome.success else "Uninstall needs attention")
        self.detail_var.set(outcome.message)
        self.hide_action_buttons()
        if outcome.success:
            self.uninstall_button.configure(state="disabled")
            self.logs_button.configure(state="disabled")
            if is_windows():
                self.root.after(8000, self.root.destroy)
        else:
            self.set_management_buttons_enabled(True)
            self.retry_button.configure(text="Retry")
            self.retry_button.pack(side="right", padx=(8, 0))

    def open_gui(self) -> None:
        if self.last_outcome and self.last_outcome.gui_url:
            open_browser(self.last_outcome.gui_url)

    def open_logs(self) -> None:
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        if not open_path(self.paths.log_dir):
            self.detail_var.set(f"Logs are stored at {self.paths.log_dir}")

    def run(self) -> None:
        if self.initial_mode == "--stop":
            self.root.after(150, self.confirm_stop)
        elif self.initial_mode == "--uninstall":
            self.root.after(150, self.confirm_uninstall)
        else:
            self.root.after(150, self.start)
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def maybe_reexec_with_pythonw() -> bool:
    """Use pythonw on Windows so the setup launcher does not leave a console open."""
    if not is_windows() or os.environ.get("EMK_PYTHONW_REEXEC") == "1":
        return False
    if Path(sys.executable).name.lower() == "pythonw.exe":
        return False
    pythonw = find_pythonw()
    if not pythonw.is_file() or pythonw.resolve() == Path(sys.executable).resolve():
        return False
    env = os.environ.copy()
    env["EMK_PYTHONW_REEXEC"] = "1"
    try:
        subprocess.Popen(
            [str(pythonw), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            close_fds=True,
        )
        return True
    except OSError:
        return False


def main() -> int:
    if maybe_reexec_with_pythonw():
        return 0

    paths = get_paths()
    log_path = configure_logging(paths)

    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    if len(sys.argv) > 2 or mode not in ("", "--stop", "--uninstall"):
        show_fallback_error(
            "Use the EgoModelKit launcher. Supported maintenance actions are --stop "
            "and --uninstall.",
            paths,
        )
        return 2

    if not (is_windows() or is_linux()):
        show_fallback_error("EgoModelKit setup supports Windows and Linux hosts only.", paths)
        return 2

    if not tkinter_is_available():
        show_fallback_error(
            f"Python graphical support (Tkinter) is required. Complete {GUIDE_NAME}, then retry.",
            paths,
        )
        return 2

    try:
        SetupWindow(paths, log_path, mode).run()
    except Exception as exc:
        logging.error("Fatal UI error: %s\n%s", exc, traceback.format_exc())
        show_fallback_error(f"EgoModelKit could not open its setup window: {exc}", paths)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
