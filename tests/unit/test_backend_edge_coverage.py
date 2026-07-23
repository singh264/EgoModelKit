from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from fastapi.testclient import TestClient

import egomodelkit.bandini_metrics as bandini_metrics
import egomodelkit.cli as cli
import egomodelkit.gui_backend as gui_backend
import egomodelkit.output_contract as output_contract
import egomodelkit.runtime.docker_images as docker_images
import egomodelkit.runtime.host_platform as host_platform
import egomodelkit.runtime.preflight as preflight
from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_MODEL_ID,
    AdlRecognitionRequest,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_MODEL_ID,
    HandObjectContactRequest,
)
from egomodelkit.output_contract import build_run_output_layout
from egomodelkit.runtime.commands import CommandResult


def test_cli_progress_reporter_uses_typer_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(cli.typer, "echo", messages.append)

    cli._report_progress("Preparing video")

    assert messages == ["EgoModelKit: Preparing video"]


def test_cli_unique_run_id_covers_collision_and_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "build_run_id", lambda: "run-fixed")
    (tmp_path / "run-fixed").mkdir()

    assert cli._build_unique_cli_run_id(tmp_path) == "run-fixed-002"

    class ExistingLayout:
        run_dir = SimpleNamespace(exists=lambda: True)

    monkeypatch.setattr(cli, "build_run_output_layout", lambda *args, **kwargs: ExistingLayout())
    with pytest.raises(ValueError, match="Unable to create a unique run id"):
        cli._build_unique_cli_run_id(tmp_path)

def test_cli_payload_int_covers_all_external_value_types() -> None:
    assert cli._payload_int({"value": True}, "value") is None
    assert cli._payload_int({"value": 4}, "value") == 4
    assert cli._payload_int({"value": 4.9}, "value") == 4
    assert cli._payload_int({"value": "12"}, "value") == 12
    assert cli._payload_int({"value": "12.5"}, "value") is None
    assert cli._payload_int({}, "value") is None

def test_cli_progress_reporter_writes_structured_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = build_run_output_layout(tmp_path, run_id="run-progress")
    layout.logs_dir.mkdir(parents=True)
    echoed: list[str] = []
    monkeypatch.setattr(cli, "_report_progress", echoed.append)

    reporter = cli._cli_progress_reporter(layout)
    reporter(
        "EGOMODELKIT_PROGRESS "
        + json.dumps(
            {
                "kind": "model_step",
                "current": "2",
                "total": 3.8,
                "unit": "frames",
            }
        )
    )
    reporter(
        "EGOMODELKIT_PROGRESS "
        + json.dumps(
            {
                "kind": "second_step",
                "current": True,
                "total": "unknown",
                "unit": 4,
            }
        )
    )
    reporter("plain runtime message")

    lines = layout.progress_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["current"] == 2
    assert json.loads(lines[0])["total"] == 3
    assert json.loads(lines[0])["unit"] == "frames"
    assert json.loads(lines[1])["current"] is None
    assert json.loads(lines[1])["unit"] is None
    assert len(echoed) == 3
    assert "plain runtime message" in layout.runtime_log_path.read_text(encoding="utf-8")

@pytest.mark.parametrize(
    ("model_id", "request_type", "message"),
    [
        (
            HAND_OBJECT_CONTACT_MODEL_ID,
            AdlRecognitionRequest,
            "Unsupported model id",
        ),
        (
            ADL_RECOGNITION_MODEL_ID,
            HandObjectContactRequest,
            "ADL recognition requires",
        ),
        (
            "unsupported-model",
            HandObjectContactRequest,
            "Unsupported model id",
        ),
    ],
)
def test_cli_internal_model_dispatch_rejects_invalid_combinations(
    tmp_path: Path,
    model_id: str,
    request_type: type[HandObjectContactRequest] | type[AdlRecognitionRequest],
    message: str,
) -> None:
    input_path = tmp_path / "frame.jpg"
    input_path.write_bytes(b"image")
    output_dir = tmp_path / "out"
    request = request_type(input_path=input_path, output_dir=output_dir)

    with pytest.raises((TypeError, ValueError), match=message):
        cli._run_model_with_output_contract(model_id=model_id, request=request)

    assert not output_dir.exists()

def test_cli_gui_covers_linux_and_non_linux_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        cli,
        "ensure_host_runtime_ready",
        lambda **kwargs: calls.append(("ready", kwargs)),
    )
    monkeypatch.setattr(
        "egomodelkit.gui.launch_gui",
        lambda **kwargs: calls.append(("launch", kwargs)),
    )

    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    cli.gui(port=8001, no_browser=True)
    assert calls[0][0] == "ready"
    assert calls[1] == ("launch", {"server_port": 8001, "inbrowser": False})

    calls.clear()
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    cli.gui(port=8002, no_browser=False)
    assert calls == [("launch", {"server_port": 8002, "inbrowser": True})]

def test_open_output_folder_endpoint_reports_spawn_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run-restored"
    run_dir.mkdir()
    monkeypatch.setattr(gui_backend, "_open_output_folder", lambda _path: False)
    response = TestClient(gui_backend.create_app()).post(
        "/api/open-output-folder",
        json={"runId": run_dir.name, "outputFolder": str(run_dir)},
    )
    assert response.status_code == 500
    assert "could not start" in response.json()["detail"]

def test_gui_output_path_helpers_cover_wsl_and_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = gui_backend.OpenOutputFolderRequest(runId="run-missing")
    with pytest.raises(ValueError, match="no saved output path"):
        gui_backend._resolve_output_folder_to_open(request=request, state=None)

    windows_path_to_wsl_path = gui_backend._windows_path_to_wsl_path
    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: True)
    monkeypatch.setattr(gui_backend, "_windows_path_to_wsl_path", lambda _text: "/mnt/c/out")
    assert gui_backend._normalize_output_root(r"C:\out") == Path("/mnt/c/out")

    monkeypatch.setattr(gui_backend, "_windows_path_to_wsl_path", windows_path_to_wsl_path)
    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="/mnt/d/results\n"),
    )
    assert gui_backend._windows_path_to_wsl_path(r"D:\results") == "/mnt/d/results"

    def raise_oserror(*args, **kwargs):
        raise OSError("missing wslpath")

    monkeypatch.setattr(gui_backend.subprocess, "run", raise_oserror)
    assert gui_backend._windows_path_to_wsl_path(r"C:\Users\A\out") == "/mnt/c/Users/A/out"
    with pytest.raises(ValueError, match="Unsupported Windows output path"):
        gui_backend._windows_path_to_wsl_path(r"\\server\share")

def test_gui_wsl_to_windows_path_covers_all_subprocess_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="C:\\out\n"),
    )
    assert gui_backend._wsl_path_to_windows_path(tmp_path) == r"C:\out"

    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert gui_backend._wsl_path_to_windows_path(tmp_path) is None

    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="  "),
    )
    assert gui_backend._wsl_path_to_windows_path(tmp_path) is None

    def raise_oserror(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(gui_backend.subprocess, "run", raise_oserror)
    assert gui_backend._wsl_path_to_windows_path(tmp_path) is None

def test_windows_powershell_resolution_covers_wsl_native_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: True)
    monkeypatch.setattr(Path, "is_file", lambda self: str(self).endswith("powershell.exe"))
    monkeypatch.setattr(gui_backend.shutil, "which", lambda _name: None)
    assert gui_backend._resolve_windows_powershell_executable().endswith("powershell.exe")

    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(
        gui_backend.shutil,
        "which",
        lambda name: f"/bin/{name}" if name == "powershell" else None,
    )
    assert gui_backend._resolve_windows_powershell_executable() == "/bin/powershell"

    monkeypatch.setattr(gui_backend.shutil, "which", lambda _name: None)
    assert gui_backend._resolve_windows_powershell_executable() is None

def test_windows_folder_picker_wraps_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gui_backend, "_resolve_windows_powershell_executable", lambda: "powershell")
    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: False)

    def raise_oserror(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(gui_backend.subprocess, "run", raise_oserror)
    with pytest.raises(gui_backend.NativeOutputFolderPickerError, match="could not start"):
        gui_backend._select_output_folder_windows()

def test_linux_folder_picker_dispatch_and_command_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_folder_picker_command = gui_backend._run_folder_picker_command
    available = {"zenity": "/bin/zenity", "kdialog": "/bin/kdialog"}
    monkeypatch.setattr(gui_backend.shutil, "which", lambda name: available.get(name))
    results = iter([None, "/chosen"])
    monkeypatch.setattr(gui_backend, "_run_folder_picker_command", lambda _command: next(results))
    monkeypatch.setattr(gui_backend, "_select_output_folder_tkinter", lambda: "/tk")
    assert gui_backend._select_output_folder_linux() == "/chosen"

    monkeypatch.setattr(gui_backend.shutil, "which", lambda _name: None)
    assert gui_backend._select_output_folder_linux() == "/tk"

    monkeypatch.setattr(gui_backend, "_run_folder_picker_command", run_folder_picker_command)
    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="/selected\n"),
    )
    assert gui_backend._run_folder_picker_command(["picker"]) == "/selected"
    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="  "),
    )
    assert gui_backend._run_folder_picker_command(["picker"]) is None
    monkeypatch.setattr(
        gui_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout=""),
    )
    assert gui_backend._run_folder_picker_command(["picker"]) is None

def test_open_output_folder_covers_host_commands_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        gui_backend.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(),
    )

    monkeypatch.setattr(gui_backend.platform, "system", lambda: "Darwin")
    assert gui_backend._open_output_folder(tmp_path) is True
    assert commands[-1][0] == "open"

    monkeypatch.setattr(gui_backend.platform, "system", lambda: "Windows")
    assert gui_backend._open_output_folder(tmp_path) is True
    assert commands[-1][0] == "explorer.exe"

    monkeypatch.setattr(gui_backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: True)
    monkeypatch.setattr(gui_backend, "_wsl_path_to_windows_path", lambda _path: None)
    assert gui_backend._open_output_folder(tmp_path) is False

    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: False)
    monkeypatch.setattr(gui_backend, "_linux_file_manager_command", lambda _path: None)
    assert gui_backend._open_output_folder(tmp_path) is False

    monkeypatch.setattr(
        gui_backend,
        "_linux_file_manager_command",
        lambda path: ["xdg-open", str(path)],
    )

    def raise_oserror(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr(gui_backend.subprocess, "Popen", raise_oserror)
    assert gui_backend._open_output_folder(tmp_path) is False

def test_linux_file_manager_command_covers_search_and_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gui_backend.shutil,
        "which",
        lambda name: "/bin/thunar" if name == "thunar" else None,
    )
    assert gui_backend._linux_file_manager_command(tmp_path) == ["/bin/thunar", str(tmp_path)]
    monkeypatch.setattr(gui_backend.shutil, "which", lambda _name: None)
    assert gui_backend._linux_file_manager_command(tmp_path) is None

def test_output_contract_private_fallback_helpers(tmp_path: Path) -> None:
    layout = build_run_output_layout(tmp_path, run_id="run-output")
    assert layout.output_folder_path == layout.run_dir

    metric_inputs = output_contract._resolve_hand_interaction_metric_input_paths(layout)
    assert metric_inputs.shan_outputs_dir == layout.shan_outputs_dir
    assert metric_inputs.input_manifest_path == layout.hand_interaction_input_manifest_path

    assert output_contract._first_existing_file_path(
        [tmp_path / "one", tmp_path / "two"]
    ) == tmp_path / "two"
    assert output_contract._count_shan_prediction_files(tmp_path / "missing") == 0
    assert output_contract._csv_has_data_rows(tmp_path / "missing.csv") is False
    assert output_contract._count_csv_data_rows(tmp_path / "missing.csv") == 0

def test_stale_docker_image_removal_failure_is_reported() -> None:
    identity = docker_images.DockerImageIdentity(
        runtime_name="runtime",
        repository="egomodelkit-runtime",
        fingerprint="a" * 64,
    )
    messages: list[str] = []

    def capture(command: list[str]) -> CommandResult:
        if command[1:3] == ["image", "ls"]:
            return CommandResult(0, "egomodelkit-runtime:old\n", "")
        return CommandResult(1, "", "busy")

    assert docker_images.remove_stale_runtime_images(
        docker_executable="docker",
        current_image=identity,
        capture_runner=capture,
        progress=messages.append,
    ) == ()
    assert any("unable to remove" in message for message in messages)

def test_wsl_distribution_name_covers_environment_and_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    assert host_platform.wsl_distribution_name() == "this WSL distribution"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-Test")
    assert host_platform.wsl_distribution_name() == "Ubuntu-Test"


def test_preflight_wait_and_dispatch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    path, ready = preflight._wait_for_docker_ready(
        docker_executable="docker",
        command_runner=lambda _command: 1,
        executable_locator=lambda _name: "/bin/docker",
        sleep=sleeps.append,
        recovery_attempts=2,
    )
    assert (path, ready) == ("/bin/docker", False)
    assert sleeps == [preflight.DOCKER_RECOVERY_INTERVAL_SECONDS]

    monkeypatch.setattr(preflight, "_recover_wsl_docker_desktop", lambda **kwargs: True)
    assert preflight._recover_docker_runtime(
        "Linux", True, "docker", True, lambda _c: 0, lambda _m: None
    )
    monkeypatch.setattr(preflight, "_start_macos_docker_desktop", lambda **kwargs: True)
    assert preflight._recover_docker_runtime(
        "Darwin", False, "docker", True, lambda _c: 0, lambda _m: None
    )
    monkeypatch.setattr(preflight, "_start_linux_docker_runtime", lambda **kwargs: True)
    assert preflight._recover_docker_runtime(
        "Linux", False, "docker", True, lambda _c: 0, lambda _m: None
    )
    assert (
        preflight._recover_docker_runtime(
            "Windows", False, "docker", True, lambda _c: 0, lambda _m: None
        )
        is False
    )

def test_preflight_windows_executable_and_candidate_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/exists/tool.exe")
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: f"/path/{name}" if name == "tool.exe" else None,
    )
    assert preflight._resolve_windows_executable(("/exists/tool.exe",)) == "/exists/tool.exe"
    assert preflight._resolve_windows_executable(("missing.exe", "tool.exe")) == "/path/tool.exe"
    assert preflight._resolve_windows_executable(("missing.exe",)) is None

    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    assert preflight._windows_docker_cli_candidates() == (
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe",
    )
    assert preflight._windows_docker_desktop_candidates() == (
        "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe",
        "/mnt/c/Program Files/Docker/Docker/frontend/Docker Desktop.exe",
    )

    monkeypatch.setattr(Path, "is_dir", lambda self: str(self) == "/mnt/c/Users")

    def fake_glob(self: Path, pattern: str):
        return [Path("/mnt/c/Users/A") / pattern.replace("*", "A")]

    monkeypatch.setattr(Path, "glob", fake_glob)
    cli_candidates = preflight._windows_docker_cli_candidates()
    assert len(cli_candidates) == 3

    desktop_candidates = preflight._windows_docker_desktop_candidates(
        windows_docker_cli="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    )
    assert any(candidate.endswith("Docker Desktop.exe") for candidate in desktop_candidates)
    assert len(desktop_candidates) == len(set(desktop_candidates))

    short_candidates = preflight._windows_docker_desktop_candidates(windows_docker_cli="x")
    assert "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe" in short_candidates

def test_preflight_windows_path_conversion_and_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert preflight._windows_path_to_wsl_path(
        r'"C:\Program Files\Docker\docker.exe"'
    ) == "/mnt/c/Program Files/Docker/docker.exe"
    assert preflight._windows_path_to_wsl_path("relative/path") is None
    assert (
        preflight._wsl_path_to_windows_path("/mnt/c/Program Files/Docker")
        == r"C:\Program Files\Docker"
    )
    assert preflight._wsl_path_to_windows_path("/home/user") is None

    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: None)
    assert preflight._discover_windows_executable("docker.exe") is None

    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: "where.exe")

    def raise_oserror(*args, **kwargs):
        raise OSError("where unavailable")

    monkeypatch.setattr(preflight.subprocess, "run", raise_oserror)
    assert preflight._discover_windows_executable("docker.exe") is None

    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert preflight._discover_windows_executable("docker.exe") is None

    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="bad path\nC:\\Program Files\\Docker\\docker.exe\n",
        ),
    )
    monkeypatch.setattr(Path, "is_file", lambda self: str(self).endswith("docker.exe"))
    assert (
        preflight._discover_windows_executable("docker.exe")
        == "/mnt/c/Program Files/Docker/docker.exe"
    )

def test_preflight_quiet_run_and_interop_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7),
    )
    assert preflight._run_quietly(["command"]) == 7

    def raise_oserror(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(preflight.subprocess, "run", raise_oserror)
    assert preflight._run_quietly(["command"]) is None

    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: None)
    assert preflight._windows_interop_is_ready() is False
    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: "cmd.exe")
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: 0)
    assert preflight._windows_interop_is_ready() is True
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: 1)
    assert preflight._windows_interop_is_ready() is False

def test_preflight_repair_and_ensure_interop_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter([1, 0])
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: next(outcomes))
    assert preflight._repair_wsl_interop() is True

    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: 1)
    assert preflight._repair_wsl_interop() is False

    monkeypatch.setattr(preflight, "_windows_interop_is_ready", lambda: True)
    assert preflight._ensure_windows_interop(progress=lambda _message: None) is True

    messages: list[str] = []
    monkeypatch.setattr(preflight, "_windows_interop_is_ready", lambda: False)
    monkeypatch.setattr(preflight, "_repair_wsl_interop", lambda: False)
    assert preflight._ensure_windows_interop(progress=messages.append) is False
    assert any("could not be recovered" in message for message in messages)

def test_launch_windows_process_fallbacks_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []

    def resolve(candidates: tuple[str, ...]) -> str | None:
        if any("powershell" in candidate.lower() for candidate in candidates):
            return "powershell.exe"
        return "cmd.exe"

    monkeypatch.setattr(preflight, "_resolve_windows_executable", resolve)
    outcomes = iter([1, 0])
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: next(outcomes))
    assert preflight._launch_windows_process(
        "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe",
        progress=messages.append,
    ) is True
    assert any("PowerShell could not launch" in message for message in messages)

    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: None)
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace())
    assert preflight._launch_windows_process("relative.exe") is True

    def raise_oserror(*args, **kwargs):
        raise OSError("blocked")

    monkeypatch.setattr(preflight.subprocess, "Popen", raise_oserror)
    assert preflight._launch_windows_process("relative.exe", progress=messages.append) is False
    assert any("Direct Docker Desktop launch failed" in message for message in messages)

def test_wsl_docker_recovery_covers_failure_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "wsl_distribution_name", lambda: "Ubuntu")
    monkeypatch.setattr(preflight, "_ensure_windows_interop", lambda **kwargs: False)
    assert preflight._recover_wsl_docker_desktop(
        docker_cli_available=False,
        progress=lambda _message: None,
    ) is False

    monkeypatch.setattr(preflight, "_ensure_windows_interop", lambda **kwargs: True)

    def resolve(candidates: tuple[str, ...]) -> str | None:
        if any(candidate.endswith("wsl.exe") for candidate in candidates):
            return None
        if any(candidate.endswith("docker.exe") for candidate in candidates):
            return "docker.exe"
        return "Docker Desktop.exe"

    monkeypatch.setattr(preflight, "_resolve_windows_executable", resolve)
    monkeypatch.setattr(preflight, "_discover_windows_executable", lambda _name: None)
    outcomes = iter([1, 1, 0])
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: next(outcomes))
    assert preflight._recover_wsl_docker_desktop(
        docker_cli_available=True,
        progress=lambda _message: None,
    ) is True

    monkeypatch.setattr(
        preflight,
        "_resolve_windows_executable",
        lambda _candidates: "Docker Desktop.exe",
    )
    monkeypatch.setattr(preflight, "_discover_windows_executable", lambda _name: None)
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: 1)
    monkeypatch.setattr(preflight, "_launch_windows_process", lambda *args, **kwargs: False)
    assert preflight._recover_wsl_docker_desktop(
        docker_cli_available=False,
        progress=lambda _message: None,
    ) is False

    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: None)
    messages: list[str] = []
    assert preflight._recover_wsl_docker_desktop(
        docker_cli_available=False,
        progress=messages.append,
    ) is False
    assert any("was not found" in message for message in messages)

def test_platform_runtime_starters_cover_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_oserror(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(preflight.subprocess, "run", raise_oserror)
    assert preflight._start_macos_docker_desktop(progress=lambda _message: None) is False

    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    assert preflight._start_macos_docker_desktop(progress=lambda _message: None) is False

    assert preflight._start_linux_docker_runtime(
        command_runner=lambda _command: 1,
        progress=lambda _message: None,
    ) is False

def test_preflight_wsl_daemon_failure_uses_wsl_specific_error() -> None:
    with pytest.raises(preflight.HostPrerequisiteError, match="Docker Desktop/WSL2"):
        preflight.ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=lambda _command: 1,
            executable_locator=lambda _name: "/bin/docker",
            python_version=(3, 13, 0),
            wsl_detector=lambda: True,
            allow_runtime_recovery=False,
        )

def test_remaining_gui_picker_loop_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gui_backend, "_is_wsl", lambda: True)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(gui_backend.shutil, "which", lambda _name: None)
    assert gui_backend._resolve_windows_powershell_executable() is None

    monkeypatch.setattr(gui_backend.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(gui_backend, "_run_folder_picker_command", lambda _command: "/zenity")
    assert gui_backend._select_output_folder_linux() == "/zenity"

    results = iter([None, None])
    monkeypatch.setattr(gui_backend, "_run_folder_picker_command", lambda _command: next(results))
    monkeypatch.setattr(gui_backend, "_select_output_folder_tkinter", lambda: "/tk-fallback")
    assert gui_backend._select_output_folder_linux() == "/tk-fallback"

def test_remaining_preflight_windows_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)
    assert preflight._resolve_windows_executable(("/missing/tool.exe",)) is None

    monkeypatch.setattr(preflight, "_resolve_windows_executable", lambda _candidates: "where.exe")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="bad path\nC:\\missing\\docker.exe\n",
        ),
    )
    assert preflight._discover_windows_executable("docker.exe") is None

    messages: list[str] = []

    def resolve(candidates: tuple[str, ...]) -> str | None:
        if any("powershell" in candidate.lower() for candidate in candidates):
            return None
        return "cmd.exe"

    monkeypatch.setattr(preflight, "_resolve_windows_executable", resolve)
    monkeypatch.setattr(preflight, "_run_quietly", lambda _command: 1)
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace())
    assert preflight._launch_windows_process(
        "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe",
        progress=messages.append,
    ) is True
    assert any("Windows Command Prompt could not launch" in message for message in messages)


def test_remaining_simple_backend_branches(tmp_path: Path) -> None:
    assert bandini_metrics._as_hand_label(None, "right") == "right"
    assert bandini_metrics._as_hand_label(" LEFT ", "right") == "left"
    assert gui_backend._dominant_hand_from_text("right", model_id="hand-interaction") == "right"

    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"video")

    with pytest.raises(typer.Exit) as error:
        cli.run(
            input_path=input_path,
            output_dir=tmp_path / "output",
            model_id=ADL_RECOGNITION_MODEL_ID,
            dry_run=True,
            dominant_hand="left",
        )

    assert error.value.exit_code == cli.CLI_RUNTIME_ERROR_EXIT_CODE


def test_output_contract_child_directory_helpers_cover_all_paths(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    second_root = tmp_path / "second"
    source_dir = second_root / "wanted"
    source_dir.mkdir(parents=True)

    assert output_contract._first_existing_child_dir_path(
        search_roots=[tmp_path / "missing", first_root, second_root],
        dirnames=["absent", "wanted"],
    ) == source_dir
    assert output_contract._first_existing_child_dir_path(
        search_roots=[first_root],
        dirnames=["absent"],
    ) is None

    destination = tmp_path / "moved"
    output_contract._move_first_existing_child_dir(
        search_roots=[tmp_path / "missing", first_root, second_root],
        dirnames=["absent", "wanted"],
        destination=destination,
    )
    assert destination.is_dir()
    assert not source_dir.exists()

    output_contract._move_first_existing_child_dir(
        search_roots=[first_root],
        dirnames=["still-absent"],
        destination=tmp_path / "not-created",
    )
    assert not (tmp_path / "not-created").exists()


def test_finalize_bandini_metrics_covers_quiet_and_mismatch_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = build_run_output_layout(tmp_path, run_id="run-bandini")
    metric_inputs = output_contract._HandInteractionMetricInputPaths(
        extracted_frames_dir=tmp_path / "frames",
        shan_outputs_dir=tmp_path / "shan",
        input_manifest_path=tmp_path / "input.csv",
        subclip_manifest_path=tmp_path / "subclips.csv",
        metrics_config_path=tmp_path / "config.json",
    )
    removed_paths: list[Path] = []

    monkeypatch.setattr(output_contract, "_count_shan_prediction_files", lambda _path: 1)
    monkeypatch.setattr(output_contract, "_csv_has_data_rows", lambda _path: True)
    monkeypatch.setattr(
        output_contract,
        "_write_bandini_metric_input_diagnostics",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        output_contract,
        "read_video_processing_config",
        lambda _path: bandini_metrics.DEFAULT_VIDEO_PROCESSING_CONFIG,
    )
    monkeypatch.setattr(output_contract, "write_bandini_metric_files", lambda **_kwargs: None)
    monkeypatch.setattr(output_contract, "write_runtime_log_line", lambda *_args: None)
    monkeypatch.setattr(output_contract, "_remove_path", removed_paths.append)
    monkeypatch.setattr(output_contract, "_count_csv_data_rows", lambda _path: 1)

    output_contract._finalize_bandini_metric_outputs(
        layout=layout,
        metric_inputs=metric_inputs,
        pipeline_label="test pipeline",
        progress=lambda _message: None,
        emit_progress=False,
    )

    assert removed_paths == [layout.run_dir / output_contract.METRICS_CONFIG_FILENAME]

    monkeypatch.setattr(output_contract, "_count_csv_data_rows", lambda _path: 0)

    with pytest.raises(RuntimeError, match="did not consume every Shan JSON prediction"):
        output_contract._finalize_bandini_metric_outputs(
            layout=layout,
            metric_inputs=metric_inputs,
            pipeline_label="test pipeline",
            progress=lambda _message: None,
            emit_progress=False,
        )
