import pytest

from egomodelkit.runtime.preflight import (
    HostPrerequisiteError,
    ensure_host_runtime_ready,
)


def test_host_preflight_accepts_supported_python_and_docker() -> None:
    commands: list[list[str]] = []
    messages: list[str] = []
    
    def runner(command: list[str]) -> int:
        commands.append(command)
        return 0

    ensure_host_runtime_ready(
        docker_executable = "docker",
        command_runner = runner,
        executable_locator = lambda executable: "/usr/bin/docker",
        python_version = (3, 12, 2),
        progress = messages.append,
    )
    
    assert commands == [
        [
            "docker",
            "version",
            "--format",
            "{{.Server.Version}}",
        ]
    ]
    
    assert "Checking host runtime prerequisites." in messages
    assert "Python 3.12.2 detected." in messages
    assert "Docker executable found: /usr/bin/docker" in messages
    assert "Docker daemon is available." in messages
    
def test_host_preflight_rejects_unsupported_python() -> None:
    with pytest.raises(
        HostPrerequisiteError,
        match = "Python 3.10 or newer is required",
    ):
        ensure_host_runtime_ready(
            docker_executable = "docker",
            command_runner = lambda command: 0,
            executable_locator = lambda executable: "usr/bin/docker",
            python_version = (3, 9, 18),
        )

def test_host_preflight_rejects_missing_docker_executable() -> None:
    with pytest.raises(
        HostPrerequisiteError,
        match = "Docker executable 'docker' was not found",
    ):
        ensure_host_runtime_ready(
            docker_executable = "docker",
            command_runner = lambda command: 0,
            executable_locator = lambda executable: None,
            python_version = (3, 12, 2),
        )

def test_host_preflight_rejects_unavailable_docker_daemon() -> None:
    with pytest.raises(
        HostPrerequisiteError,
        match = "Docker is installed, but its daemon is not available",
    ):
        ensure_host_runtime_ready(
            docker_executable = "docker",
            command_runner = lambda command: 17,
            executable_locator = lambda executable: "usr/bin/docker",
            python_version = (3, 12, 2),
        )
