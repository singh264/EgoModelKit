from types import SimpleNamespace

import pytest

from egomodelkit.runtime.preflight import (
    HostPrerequisiteError,
    _recover_wsl_docker_desktop,
    _start_linux_docker_runtime,
    _start_macos_docker_desktop,
    ensure_host_runtime_ready,
)


def test_host_preflight_accepts_supported_python_and_docker() -> None:
    commands: list[list[str]] = []
    messages: list[str] = []

    def runner(command: list[str]) -> int:
        commands.append(command)
        return 0

    ensure_host_runtime_ready(
        docker_executable="docker",
        command_runner=runner,
        executable_locator=lambda _executable: "/usr/bin/docker",
        python_version=(3, 12, 2),
        progress=messages.append,
    )

    assert commands == [["docker", "version", "--format", "{{.Server.Version}}"]]
    assert "Checking host runtime prerequisites." in messages
    assert "Python 3.12.2 detected." in messages
    assert "Docker executable found: /usr/bin/docker" in messages
    assert "Docker daemon is available." in messages


def test_host_preflight_rejects_unsupported_python() -> None:
    with pytest.raises(HostPrerequisiteError, match="Python 3.10 or newer is required"):
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=lambda _command: 0,
            executable_locator=lambda _executable: "/usr/bin/docker",
            python_version=(3, 9, 18),
        )


def test_host_preflight_rejects_missing_docker_without_test_side_effects() -> None:
    with pytest.raises(HostPrerequisiteError, match="was not found on PATH"):
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=lambda _command: 0,
            executable_locator=lambda _executable: None,
            python_version=(3, 12, 2),
            wsl_detector=lambda: False,
        )


def test_host_preflight_rejects_unavailable_daemon_when_recovery_is_disabled() -> None:
    with pytest.raises(HostPrerequisiteError, match="could not start or reconnect"):
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=lambda _command: 17,
            executable_locator=lambda _executable: "/usr/bin/docker",
            python_version=(3, 12, 2),
            allow_runtime_recovery=False,
            wsl_detector=lambda: False,
        )


def test_host_preflight_rejects_non_linux_gpu_hosts() -> None:
    calls: list[list[str]] = []

    with pytest.raises(HostPrerequisiteError, match="require a Linux host"):
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=lambda command: calls.append(command) or 0,
            executable_locator=lambda _executable: "/usr/bin/docker",
            python_version=(3, 12, 2),
            platform_detector=lambda: "Darwin",
            require_linux_nvidia_gpu=True,
        )

    assert calls == []


def test_host_preflight_rejects_missing_nvidia_container_runtime() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> int:
        commands.append(command)
        return 0 if command[:2] == ["docker", "version"] else 1

    with pytest.raises(HostPrerequisiteError, match="NVIDIA container runtime is unavailable"):
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=runner,
            executable_locator=lambda _executable: "/usr/bin/docker",
            python_version=(3, 12, 2),
            platform_detector=lambda: "Linux",
            require_linux_nvidia_gpu=True,
            wsl_detector=lambda: False,
        )

    assert commands[-1][:4] == ["docker", "run", "--rm", "--gpus"]


def test_host_preflight_accepts_linux_nvidia_runtime() -> None:
    commands: list[list[str]] = []
    messages: list[str] = []

    ensure_host_runtime_ready(
        docker_executable="docker",
        command_runner=lambda command: commands.append(command) or 0,
        executable_locator=lambda _executable: "/usr/bin/docker",
        python_version=(3, 12, 2),
        platform_detector=lambda: "Linux",
        require_linux_nvidia_gpu=True,
        progress=messages.append,
        wsl_detector=lambda: False,
    )

    assert commands[-1][-1] == "nvidia-smi"
    assert "Docker NVIDIA GPU runtime is available." in messages


def test_host_preflight_recovers_wsl_when_cli_is_missing() -> None:
    messages: list[str] = []
    recovery_calls: list[tuple[str, bool, str, bool]] = []
    locator_results = iter([None, "/usr/bin/docker"])

    def locator(_executable: str) -> str | None:
        return next(locator_results, "/usr/bin/docker")

    def recoverer(
        platform_name: str,
        running_in_wsl: bool,
        docker_executable: str,
        docker_cli_available: bool,
        _command_runner,
        _progress,
    ) -> bool:
        recovery_calls.append(
            (platform_name, running_in_wsl, docker_executable, docker_cli_available)
        )
        return True

    ensure_host_runtime_ready(
        docker_executable="docker",
        command_runner=lambda _command: 0,
        executable_locator=locator,
        python_version=(3, 12, 2),
        platform_detector=lambda: "Linux",
        progress=messages.append,
        wsl_detector=lambda: True,
        allow_runtime_recovery=True,
        runtime_recoverer=recoverer,
        sleep=lambda _seconds: None,
        recovery_attempts=2,
    )

    assert recovery_calls == [("Linux", True, "docker", False)]
    assert any("automatic recovery" in message for message in messages)
    assert "Docker daemon is available." in messages


def test_host_preflight_recovers_stopped_wsl_daemon() -> None:
    daemon_probe_count = 0
    recovery_calls: list[bool] = []

    def runner(command: list[str]) -> int:
        nonlocal daemon_probe_count
        if command[:2] == ["docker", "version"]:
            daemon_probe_count += 1
            return 1 if daemon_probe_count == 1 else 0
        return 0

    def recoverer(
        _platform_name,
        _running_in_wsl,
        _docker_executable,
        docker_cli_available,
        _command_runner,
        _progress,
    ) -> bool:
        recovery_calls.append(docker_cli_available)
        return True

    ensure_host_runtime_ready(
        docker_executable="docker",
        command_runner=runner,
        executable_locator=lambda _executable: "/usr/bin/docker",
        python_version=(3, 12, 2),
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: True,
        allow_runtime_recovery=True,
        runtime_recoverer=recoverer,
        sleep=lambda _seconds: None,
        recovery_attempts=2,
    )

    assert recovery_calls == [True]
    assert daemon_probe_count == 2


def test_host_preflight_reports_maintainer_error_after_wsl_recovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.wsl_distribution_name",
        lambda: "Ubuntu-24.04",
    )

    with pytest.raises(HostPrerequisiteError, match="repaired by a maintainer") as exc:
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=lambda _command: 1,
            executable_locator=lambda _executable: None,
            python_version=(3, 12, 2),
            platform_detector=lambda: "Linux",
            wsl_detector=lambda: True,
            allow_runtime_recovery=True,
            runtime_recoverer=lambda *_args: False,
        )

    assert "Ubuntu-24.04" in str(exc.value)
    assert "Settings" not in str(exc.value)


def test_host_preflight_reports_wsl_gpu_failure_to_maintainer() -> None:
    def runner(command: list[str]) -> int:
        return 0 if command[:2] == ["docker", "version"] else 1

    with pytest.raises(HostPrerequisiteError, match="GPU access is unavailable in WSL2"):
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=runner,
            executable_locator=lambda _executable: "/usr/bin/docker",
            python_version=(3, 12, 2),
            platform_detector=lambda: "Linux",
            require_linux_nvidia_gpu=True,
            wsl_detector=lambda: True,
            allow_runtime_recovery=False,
        )


def test_linux_recovery_uses_first_working_noninteractive_service_command() -> None:
    commands: list[list[str]] = []
    messages: list[str] = []

    def runner(command: list[str]) -> int:
        commands.append(command)
        return 0 if command[:3] == ["sudo", "-n", "systemctl"] else 1

    assert _start_linux_docker_runtime(
        command_runner=runner,
        progress=messages.append,
    ) is True
    assert commands == [
        ["systemctl", "--user", "start", "docker-desktop"],
        ["systemctl", "--user", "start", "docker"],
        ["sudo", "-n", "systemctl", "start", "docker"],
    ]
    assert messages


def test_macos_recovery_opens_docker_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("egomodelkit.runtime.preflight.subprocess.run", fake_run)

    assert _start_macos_docker_desktop(progress=lambda _message: None) is True
    assert calls == [["open", "-gj", "-a", "Docker"]]


def test_wsl_recovery_sets_default_distro_and_starts_desktop_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    messages: list[str] = []

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.wsl_distribution_name",
        lambda: "Ubuntu-24.04",
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._ensure_windows_interop",
        lambda **_kwargs: True,
    )

    def fake_resolve(candidates: tuple[str, ...]) -> str | None:
        if any(candidate.endswith("wsl.exe") for candidate in candidates):
            return "/mnt/c/Windows/System32/wsl.exe"
        if any(candidate.endswith("docker.exe") for candidate in candidates):
            return "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
        return None

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        if command[-2:] == ["desktop", "status"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._resolve_windows_executable",
        fake_resolve,
    )
    monkeypatch.setattr("egomodelkit.runtime.preflight.subprocess.run", fake_run)

    assert _recover_wsl_docker_desktop(
        docker_cli_available=False,
        progress=messages.append,
    ) is True
    assert calls == [
        [
            "/mnt/c/Windows/System32/wsl.exe",
            "--set-default",
            "Ubuntu-24.04",
        ],
        [
            "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe",
            "desktop",
            "status",
        ],
        [
            "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe",
            "desktop",
            "start",
        ],
    ]
    assert any("default WSL distribution" in message for message in messages)
    assert any("recovery was started" in message for message in messages)


def test_wsl_recovery_launches_desktop_executable_when_cli_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[list[str]] = []

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.wsl_distribution_name",
        lambda: "Ubuntu-24.04",
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._ensure_windows_interop",
        lambda **_kwargs: True,
    )

    def fake_resolve(candidates: tuple[str, ...]) -> str | None:
        if any(candidate.endswith("wsl.exe") for candidate in candidates):
            return "/mnt/c/Windows/System32/wsl.exe"
        if any(candidate.endswith("docker.exe") for candidate in candidates):
            return None
        return "/mnt/c/Users/test/AppData/Local/Programs/DockerDesktop/Docker Desktop.exe"

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._resolve_windows_executable",
        fake_resolve,
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._discover_windows_executable",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.subprocess.run",
        lambda _command, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._launch_windows_process",
        lambda executable, **_kwargs: launched.append([executable]) or True,
    )

    assert _recover_wsl_docker_desktop(
        docker_cli_available=False,
        progress=lambda _message: None,
    ) is True
    assert launched == [
        [
            "/mnt/c/Users/test/AppData/Local/Programs/"
            "DockerDesktop/Docker Desktop.exe"
        ]
    ]



def test_windows_interop_recovery_repairs_then_rechecks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.runtime.preflight import _ensure_windows_interop

    probe_results = iter([False, True])
    messages: list[str] = []

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._windows_interop_is_ready",
        lambda: next(probe_results),
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._repair_wsl_interop",
        lambda: True,
    )

    assert _ensure_windows_interop(progress=messages.append) is True
    assert any("attempting automatic recovery" in message for message in messages)
    assert any("recovered automatically" in message for message in messages)


def test_launch_windows_process_uses_explicit_powershell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.runtime.preflight import _launch_windows_process

    calls: list[list[str]] = []

    def fake_resolve(candidates: tuple[str, ...]) -> str | None:
        if any(candidate.endswith("powershell.exe") for candidate in candidates):
            return (
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/"
                "powershell.exe"
            )
        return None

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._resolve_windows_executable",
        fake_resolve,
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._run_quietly",
        lambda command: calls.append(command) or 0,
    )

    assert _launch_windows_process(
        "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe"
    ) is True
    assert calls[0][0].endswith("powershell.exe")
    assert "-EncodedCommand" in calls[0]


def test_wsl_recovery_reports_located_launcher_that_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.wsl_distribution_name",
        lambda: "Ubuntu-24.04",
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._ensure_windows_interop",
        lambda **_kwargs: True,
    )

    def fake_resolve(candidates: tuple[str, ...]) -> str | None:
        if any(candidate.endswith("wsl.exe") for candidate in candidates):
            return None
        if any(candidate.endswith("docker.exe") for candidate in candidates):
            return None
        return "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe"

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._resolve_windows_executable",
        fake_resolve,
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._discover_windows_executable",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._launch_windows_process",
        lambda _executable, **_kwargs: False,
    )

    assert _recover_wsl_docker_desktop(
        docker_cli_available=False,
        progress=messages.append,
    ) is False
    assert any("was located" in message for message in messages)
    assert not any("was not found" in message for message in messages)


def test_windows_path_conversion_supports_current_per_user_install() -> None:
    from egomodelkit.runtime.preflight import (
        _windows_path_to_wsl_path,
        _wsl_path_to_windows_path,
    )

    windows_path = (
        r"C:\Users\lab\AppData\Local\Programs\DockerDesktop"
        r"\Docker Desktop.exe"
    )
    wsl_path = (
        "/mnt/c/Users/lab/AppData/Local/Programs/DockerDesktop/"
        "Docker Desktop.exe"
    )

    assert _windows_path_to_wsl_path(windows_path) == wsl_path
    assert _wsl_path_to_windows_path(wsl_path) == windows_path


def test_discover_windows_executable_uses_where_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.runtime.preflight import _discover_windows_executable

    monkeypatch.setattr(
        "egomodelkit.runtime.preflight._resolve_windows_executable",
        lambda _candidates: "/mnt/c/Windows/System32/where.exe",
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.subprocess.run",
        lambda _command, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "C:\\Program Files\\Docker\\Docker\\resources"
                "\\bin\\docker.exe\r\n"
            ),
        ),
    )
    monkeypatch.setattr(
        "egomodelkit.runtime.preflight.Path.is_file",
        lambda path: str(path).endswith("/resources/bin/docker.exe"),
    )

    assert _discover_windows_executable("docker.exe") == (
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    )
