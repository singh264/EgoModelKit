""" Command execution helpers for model runtimes. """

from __future__ import annotations

import subprocess
from collections.abc import Callable

OutputLineCallback = Callable[[str], None]

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
