import subprocess
from types import SimpleNamespace

import pytest

from egomodelkit.runtime import commands
from egomodelkit.runtime.commands import (
    CommandCancelledError,
    ProcessCancellation,
    subprocess_runner,
)


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: object | None = None,
        returncode: int | None = None,
        pid: int | None = 1234,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.pid = pid
        self.terminated = False
        self.wait_called = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self) -> int:
        self.wait_called = True

        if self.returncode is None:
            return 0

        return self.returncode
    
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

def test_process_cancellation_terminates_bound_process() -> None:
    from egomodelkit.runtime.commands import ProcessCancellation

    class FakeProcess:
        pid = 1234
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

    process = FakeProcess()
    cancellation = ProcessCancellation()

    cancellation.bind_process(process, ["docker", "run", "egomodelkit-test"])
    messages = cancellation.cancel(operation_label = "run run-test")

    assert cancellation.is_cancelled()
    assert process.terminated is True
    
    assert messages == [
        (
            "Cancel requested for run run-test; sent terminate signal "
            "to subprocess pid=1234 command=['docker', 'run', 'egomodelkit-test']."
        )
    ]

def test_process_cancellation_rejects_new_process_after_cancel() -> None:
    from egomodelkit.runtime.commands import (
        CommandCancelledError,
        ProcessCancellation,
    )

    class FakeProcess:
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

    process = FakeProcess()
    cancellation = ProcessCancellation()
    cancellation.cancel()

    with pytest.raises(CommandCancelledError, match = "Run was cancelled"):
        cancellation.bind_process(process)

    assert process.terminated is True


def test_cancellable_streaming_subprocess_runner_raises_after_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.runtime.commands import (
        CommandCancelledError,
        ProcessCancellation,
        cancellable_streaming_subprocess_runner,
    )

    class FakeStdout:
        def __iter__(self):
            return iter(["first\n"])

    class FakeProcess:
        stdout = FakeStdout()
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self) -> int:
            return 0

    process = FakeProcess()

    monkeypatch.setattr(
        "egomodelkit.runtime.commands.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )

    cancellation = ProcessCancellation()
    lines: list[str] = []

    def record_line(line: str) -> None:
        lines.append(line)
        cancellation.cancel()

    with pytest.raises(CommandCancelledError, match = "Run was cancelled"):
        cancellable_streaming_subprocess_runner(
            ["docker", "run"],
            record_line,
            cancellation,
        )

    assert lines == ["first"]
    assert process.terminated is True

def test_process_cancellation_logs_when_no_process_is_running() -> None:
    from egomodelkit.runtime.commands import ProcessCancellation

    cancellation = ProcessCancellation()

    messages = cancellation.cancel(operation_label = "run run-test")

    assert messages == [
        (
            "Cancel requested for run run-test; no active subprocess was "
            "running at the time of cancellation."
        )
    ]

def test_bind_process_terminates_process_when_already_cancelled() -> None:
    cancellation = ProcessCancellation()
    cancellation.cancel(operation_label = "run cancelled-before-bind")

    process = _FakeProcess(returncode = None)

    with pytest.raises(CommandCancelledError, match = "Run was cancelled."):
        cancellation.bind_process(process, ["docker", "run"])

    assert process.terminated is True

def test_bind_process_does_not_terminate_finished_process_when_cancelled() -> None:
    cancellation = ProcessCancellation()
    cancellation.cancel(operation_label = "run cancelled-before-bind")

    process = _FakeProcess(returncode = 0)

    with pytest.raises(CommandCancelledError, match = "Run was cancelled."):
        cancellation.bind_process(process, ["docker", "run"])

    assert process.terminated is False

def test_release_process_ignores_untracked_process() -> None:
    cancellation = ProcessCancellation()
    tracked_process = _FakeProcess(returncode = None)
    other_process = _FakeProcess(returncode = None)

    cancellation.bind_process(tracked_process, ["tracked"])

    cancellation.release_process(other_process)

    assert cancellation.current_process is tracked_process
    assert cancellation.current_command == ["tracked"]
    
def test_cancel_logs_unknown_command_for_bound_process_without_command() -> None:
    cancellation = ProcessCancellation()
    process = _FakeProcess(returncode = None)

    cancellation.bind_process(process)

    messages = cancellation.cancel(operation_label = "run missing-command")

    assert process.terminated is True
    
    assert messages == [
        (
            "Cancel requested for run missing-command; sent terminate signal "
            "to subprocess pid=1234 command=unknown."
        )
    ]

def test_cancellable_subprocess_runner_delegates_to_streaming_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = ProcessCancellation()

    def fake_streaming_runner(
        command: list[str],
        output_line_callback: commands.OutputLineCallback,
        runner_cancellation: ProcessCancellation,
    ) -> int:
        output_line_callback("ignored")
        
        assert command == ["docker", "version"]
        assert runner_cancellation is cancellation

        return 17

    monkeypatch.setattr(
        commands,
        "cancellable_streaming_subprocess_runner",
        fake_streaming_runner,
    )

    assert (
        commands.cancellable_subprocess_runner(["docker", "version"], cancellation)
        == 17
    )

def test_cancellable_streaming_subprocess_runner_returns_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(stdout = None, returncode = 7)

    monkeypatch.setattr(
        commands.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    lines: list[str] = []

    exit_code = commands.cancellable_streaming_subprocess_runner(
        ["python", "--version"],
        lines.append,
        ProcessCancellation(),
    )

    assert exit_code == 7
    assert process.wait_called is True
    assert lines == []

def test_cancellable_streaming_subprocess_runner_cancels_during_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = ProcessCancellation()

    class CancellingStdout:
        def __iter__(self):
            cancellation.cancel(operation_label = "run streaming-cancel")
            yield "line after cancellation\n"

    process = _FakeProcess(stdout = CancellingStdout(), returncode = None)

    monkeypatch.setattr(
        commands.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    lines: list[str] = []

    with pytest.raises(CommandCancelledError, match = "Run was cancelled."):
        commands.cancellable_streaming_subprocess_runner(
            ["python", "slow.py"],
            lines.append,
            cancellation,
        )

    assert process.terminated is True
    assert lines == []
    assert cancellation.current_process is None
    assert cancellation.current_command is None
