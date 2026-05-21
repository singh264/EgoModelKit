""" Hidden runtime execution for ADL recognition inference. """

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from egomodelkit.models.adl_recognition import (
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)
from egomodelkit.runtime.commands import subprocess_runner
from egomodelkit.runtime.preflight import (
    ProgressReporter,
    ensure_host_runtime_ready,
)

CommandRunner = Callable[[list[str]], int]

@dataclass(frozen = True, slots = True)
class AdlRecognitionRuntimeSpec:
    """ Build and execution settings for the hidden adl-recognition runtime. """

    docker_executable: str

DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC: Final[AdlRecognitionRuntimeSpec] = (
    AdlRecognitionRuntimeSpec(
        docker_executable = "docker",
    )
)

class AdlRecognitionRuntimeError(RuntimeError):
    """ Raised when ADL recognition runtime execution fails. """

def _ignore_progress(_: str) -> None:
    """ Default no-op progress reporter. """

def run_adl_recognition(
    request: AdlRecognitionRequest,
    *,
    runtime_spec: AdlRecognitionRuntimeSpec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> list[Path]:
    """ Run ADL recognition behind EgoModelKit's run command. """
    progress("Validating adl-recognition request.")
    validate_adl_recognition_request(request)
    
    ensure_host_runtime_ready(
        docker_executable = runtime_spec.docker_executable,
        command_runner = command_runner,
        progress = progress,
    )
    
    raise AdlRecognitionRuntimeError(
        "adl-recognition runtime is not available yet."
    )
