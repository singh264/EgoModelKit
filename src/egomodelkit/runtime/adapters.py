"""Runtime adapter registry for model-specific request and execution strategies."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, TypeAlias

from egomodelkit.bandini_metrics import DEFAULT_DOMINANT_HAND, HandLabel
from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_DRY_RUN_VALIDATION_MESSAGE,
    ADL_RECOGNITION_MODEL_ID,
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)
from egomodelkit.models.hand_interaction import (
    HAND_INTERACTION_DRY_RUN_VALIDATION_MESSAGE,
    HAND_INTERACTION_MODEL_ID,
    HandInteractionRequest,
    validate_hand_interaction_request,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_DRY_RUN_VALIDATION_MESSAGE,
    HAND_OBJECT_CONTACT_MODEL_ID,
    HandObjectContactRequest,
    validate_hand_object_contact_request,
)
from egomodelkit.runtime import adl_recognition as adl_runtime
from egomodelkit.runtime import hand_interaction as hand_interaction_runtime
from egomodelkit.runtime import hand_object_contact as hand_object_runtime
from egomodelkit.runtime.commands import CommandRunner

ModelRequest: TypeAlias = (
    HandObjectContactRequest | HandInteractionRequest | AdlRecognitionRequest
)
ProgressReporter: TypeAlias = Callable[[str], None]
StreamingCommandRunner: TypeAlias = Callable[[list[str], ProgressReporter], int]
RequestBuilder: TypeAlias = Callable[[Path, Path, HandLabel], ModelRequest]
RequestValidator: TypeAlias = Callable[[ModelRequest], None]
RuntimeRunner: TypeAlias = Callable[
    [ModelRequest, CommandRunner, StreamingCommandRunner, ProgressReporter],
    object,
]


@dataclass(frozen=True, slots=True)
class ModelRuntimeAdapter:
    """Normalize model-specific request validation and runtime execution."""

    model_id: str
    dry_run_validation_message: str
    docker_executable: str
    request_builder: RequestBuilder
    request_validator: RequestValidator
    runtime_runner: RuntimeRunner

    def build_request(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        dominant_hand: HandLabel = DEFAULT_DOMINANT_HAND,
    ) -> ModelRequest:
        """Build this model's request from shared interface inputs."""
        return self.request_builder(input_path, output_dir, dominant_hand)

    def validate(self, request: ModelRequest) -> None:
        """Validate a request through the model-specific validator."""
        self.request_validator(request)

    def with_output_dir(self, request: ModelRequest, output_dir: Path) -> ModelRequest:
        """Copy a frozen request while rebasing its runtime output directory."""
        return replace(request, output_dir=output_dir)

    def run(
        self,
        request: ModelRequest,
        *,
        command_runner: CommandRunner,
        streaming_command_runner: StreamingCommandRunner,
        progress: ProgressReporter,
    ) -> object:
        """Execute a model through a uniform runtime signature."""
        return self.runtime_runner(
            request,
            command_runner,
            streaming_command_runner,
            progress,
        )


def _build_hand_object_request(
    input_path: Path,
    output_dir: Path,
    _dominant_hand: HandLabel,
) -> ModelRequest:
    return HandObjectContactRequest(input_path=input_path, output_dir=output_dir)


def _build_hand_interaction_request(
    input_path: Path,
    output_dir: Path,
    dominant_hand: HandLabel,
) -> ModelRequest:
    return HandInteractionRequest(
        input_path=input_path,
        output_dir=output_dir,
        dominant_hand=dominant_hand,
    )


def _build_adl_request(
    input_path: Path,
    output_dir: Path,
    _dominant_hand: HandLabel,
) -> ModelRequest:
    return AdlRecognitionRequest(input_path=input_path, output_dir=output_dir)


def _validate_hand_object_request(request: ModelRequest) -> None:
    if not isinstance(request, HandObjectContactRequest):
        raise TypeError("Hand-object contact requires a HandObjectContactRequest.")
    validate_hand_object_contact_request(request)


def _validate_hand_interaction_request(request: ModelRequest) -> None:
    if not isinstance(request, HandInteractionRequest):
        raise TypeError("Hand interaction requires a HandInteractionRequest.")
    validate_hand_interaction_request(request)


def _validate_adl_request(request: ModelRequest) -> None:
    if not isinstance(request, AdlRecognitionRequest):
        raise TypeError("ADL recognition requires an AdlRecognitionRequest.")
    validate_adl_recognition_request(request)


def _run_hand_object_request(
    request: ModelRequest,
    command_runner: CommandRunner,
    streaming_command_runner: StreamingCommandRunner,
    progress: ProgressReporter,
) -> object:
    if not isinstance(request, HandObjectContactRequest):
        raise TypeError("Hand-object contact requires a HandObjectContactRequest.")
    return hand_object_runtime.run_hand_object_contact(
        request,
        command_runner=command_runner,
        streaming_command_runner=streaming_command_runner,
        progress=progress,
    )


def _run_hand_interaction_request(
    request: ModelRequest,
    command_runner: CommandRunner,
    streaming_command_runner: StreamingCommandRunner,
    progress: ProgressReporter,
) -> object:
    if not isinstance(request, HandInteractionRequest):
        raise TypeError("Hand interaction requires a HandInteractionRequest.")
    return hand_interaction_runtime.run_hand_interaction(
        request,
        command_runner=command_runner,
        streaming_command_runner=streaming_command_runner,
        progress=progress,
    )


def _run_adl_request(
    request: ModelRequest,
    command_runner: CommandRunner,
    streaming_command_runner: StreamingCommandRunner,
    progress: ProgressReporter,
) -> object:
    if not isinstance(request, AdlRecognitionRequest):
        raise TypeError("ADL recognition requires an AdlRecognitionRequest.")
    return adl_runtime.run_adl_recognition(
        request,
        command_runner=command_runner,
        streaming_command_runner=streaming_command_runner,
        progress=progress,
    )


MODEL_RUNTIME_ADAPTERS: Final[tuple[ModelRuntimeAdapter, ...]] = (
    ModelRuntimeAdapter(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        dry_run_validation_message=HAND_OBJECT_CONTACT_DRY_RUN_VALIDATION_MESSAGE,
        docker_executable=(
            hand_object_runtime.DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC.docker_executable
        ),
        request_builder=_build_hand_object_request,
        request_validator=_validate_hand_object_request,
        runtime_runner=_run_hand_object_request,
    ),
    ModelRuntimeAdapter(
        model_id=HAND_INTERACTION_MODEL_ID,
        dry_run_validation_message=HAND_INTERACTION_DRY_RUN_VALIDATION_MESSAGE,
        docker_executable=(
            hand_interaction_runtime.DEFAULT_HAND_INTERACTION_RUNTIME_SPEC.docker_executable
        ),
        request_builder=_build_hand_interaction_request,
        request_validator=_validate_hand_interaction_request,
        runtime_runner=_run_hand_interaction_request,
    ),
    ModelRuntimeAdapter(
        model_id=ADL_RECOGNITION_MODEL_ID,
        dry_run_validation_message=ADL_RECOGNITION_DRY_RUN_VALIDATION_MESSAGE,
        docker_executable=adl_runtime.DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC.docker_executable,
        request_builder=_build_adl_request,
        request_validator=_validate_adl_request,
        runtime_runner=_run_adl_request,
    ),
)

_RUNTIME_ADAPTERS_BY_ID: Final[dict[str, ModelRuntimeAdapter]] = {
    adapter.model_id: adapter for adapter in MODEL_RUNTIME_ADAPTERS
}


def get_runtime_adapter(model_id: str) -> ModelRuntimeAdapter:
    """Return the registered runtime adapter for one model id."""
    try:
        return _RUNTIME_ADAPTERS_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model id: {model_id}") from exc
