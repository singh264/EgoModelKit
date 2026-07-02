""" Command execution helpers for model runtimes. """

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

OutputLineCallback = Callable[[str], None]
CommandRunner = Callable[[list[str]], int]

class CommandCancelledError(RuntimeError):
    """ Raised when a running command is cancelled by the GUI. """
    
@dataclass(slots = True)
class ProcessCancellation:
    """ Shared cancellation token for one GUI operation. """
    cancel_event: threading.Event = field(default_factory = threading.Event)
    lock: threading.Lock = field(default_factory = threading.Lock)
    current_process: subprocess.Popen[str] | None = None
    current_command: list[str] | None = None

    def cancel(self, *, operation_label: str = "GUI operation") -> list[str]:
        """ Cancel the operation and terminate the active subprocess if present. """
        self.cancel_event.set()

        with self.lock:
            process = self.current_process
            command = self.current_command

        if process is not None and process.poll() is None:
            process.terminate()

            return [
                (
                    f"Cancel requested for {operation_label}; sent terminate signal "
                    f"to subprocess pid={_process_pid_for_log(process)} "
                    f"command={_command_for_log(command)}."
                )
            ]

        return [
            (
                f"Cancel requested for {operation_label}; no active subprocess was "
                "running at the time of cancellation."
            )
        ]

    def is_cancelled(self) -> bool:
        """ Return whether cancellation has been requested. """
        return self.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        """ Raise if cancellation has been requested. """
        if self.is_cancelled():
            raise CommandCancelledError("Run was cancelled.")

    def bind_process(
        self,
        process: subprocess.Popen[str],
        command: list[str] | None = None,
    ) -> None:
        """ Track the active subprocess so a later cancel can terminate it. """
        with self.lock:
            if self.is_cancelled():
                if process.poll() is None:
                    process.terminate()
                    
                raise CommandCancelledError("Run was cancelled.")

            self.current_process = process
            self.current_command = command

    def release_process(self, process: subprocess.Popen[str]) -> None:
        """ Clear the active subprocess if it is still the tracked process. """
        with self.lock:
            if self.current_process is process:
                self.current_process = None
                self.current_command = None

def _process_pid_for_log(process: subprocess.Popen[str]) -> str:
    """ Return a stable subprocess pid label for cancellation logs. """
    pid = getattr(process, "pid", None)

    if pid is None:
        return "unknown"

    return str(pid)


def _command_for_log(command: list[str] | None) -> str:
    """ Return a simple command label for cancellation logs. """
    if not command:
        return "unknown"

    return repr(command)

def subprocess_runner(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)

    return completed.returncode

def streaming_subprocess_runner(
    command: list[str],
    output_line_callback: OutputLineCallback,
) -> int:
    """ Run a command while forwarding stdout/stderr lines to a callback. """
    process = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        bufsize = 1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            output_line_callback(line.rstrip())

    return process.wait()

def cancellable_subprocess_runner(
    command: list[str],
    cancellation: ProcessCancellation,
) -> int:
    """ Run a command that can be terminated through a cancellation token. """
    return cancellable_streaming_subprocess_runner(
        command,
        lambda _line: None,
        cancellation,
    )

def cancellable_streaming_subprocess_runner(
    command: list[str],
    output_line_callback: OutputLineCallback,
    cancellation: ProcessCancellation,
) -> int:
    """ Run a streaming command and terminate it when cancellation is requested. """
    cancellation.raise_if_cancelled()

    process = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        bufsize = 1,
    )

    cancellation.bind_process(process, command)

    try:
        if process.stdout is not None:
            for line in process.stdout:
                if cancellation.is_cancelled():
                    process.terminate()
                    
                    raise CommandCancelledError("Run was cancelled.")

                output_line_callback(line.rstrip())

        exit_code = process.wait()

        cancellation.raise_if_cancelled()

        return exit_code
    finally:
        cancellation.release_process(process)
