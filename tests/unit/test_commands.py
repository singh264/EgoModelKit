import subprocess
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

def test_streaming_subprocess_runner_forwards_output_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.runtime.commands import streaming_subprocess_runner

    class FakeStdout:
        def __iter__(self):
            return iter(["first\n", "second\n"])

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self) -> int:
            return 7

    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append(command)
        
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"] is True
        
        return FakeProcess()

    monkeypatch.setattr(
        "egomodelkit.runtime.commands.subprocess.Popen",
        fake_popen,
    )

    lines: list[str] = []

    exit_code = streaming_subprocess_runner(["echo", "hello"], lines.append)

    assert exit_code == 7
    assert calls == [["echo", "hello"]]
    assert lines == ["first", "second"]

def test_streaming_subprocess_runner_handles_missing_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.runtime.commands import streaming_subprocess_runner

    class FakeProcess:
        stdout = None

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        "egomodelkit.runtime.commands.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )

    lines: list[str] = []

    assert streaming_subprocess_runner(["echo", "hello"], lines.append) == 0
    assert lines == []
