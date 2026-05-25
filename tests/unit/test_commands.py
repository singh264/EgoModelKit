from types import SimpleNamespace

import pytest

from egomodelkit.runtime.commands import subprocess_runner


def test_subprocess_runner_returns_completed_process_exit_code(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], bool]] = []
    
    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        calls.append((command, check))
        
        return SimpleNamespace(returncode = 23)

    monkeypatch.setattr(
        "egomodelkit.runtime.commands.subprocess.run",
        fake_run,
    )
    
    exit_code = subprocess_runner(["echo", "hello"])
    
    assert exit_code == 23
    
    assert calls == [
        (
            ["echo", "hello"],
            False,
        )
    ]
