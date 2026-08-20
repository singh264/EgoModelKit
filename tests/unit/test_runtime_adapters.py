from pathlib import Path

import pytest

from egomodelkit.bandini_metrics import LEFT_HAND_LABEL
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID, AdlRecognitionRequest
from egomodelkit.models.hand_interaction import (
    HAND_INTERACTION_MODEL_ID,
    HandInteractionRequest,
)
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.runtime.adapters import (
    MODEL_RUNTIME_ADAPTERS,
    get_runtime_adapter,
)


def test_runtime_registry_contains_every_packaged_model() -> None:
    assert [adapter.model_id for adapter in MODEL_RUNTIME_ADAPTERS] == [
        HAND_OBJECT_CONTACT_MODEL_ID,
        HAND_INTERACTION_MODEL_ID,
        ADL_RECOGNITION_MODEL_ID,
    ]


def test_runtime_adapter_builds_model_specific_requests(tmp_path: Path) -> None:
    hand_request = get_runtime_adapter(HAND_INTERACTION_MODEL_ID).build_request(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "out",
        dominant_hand=LEFT_HAND_LABEL,
    )
    adl_request = get_runtime_adapter(ADL_RECOGNITION_MODEL_ID).build_request(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "out",
    )

    assert isinstance(hand_request, HandInteractionRequest)
    assert hand_request.dominant_hand == LEFT_HAND_LABEL
    assert isinstance(adl_request, AdlRecognitionRequest)


def test_runtime_adapter_rebases_frozen_request_output(tmp_path: Path) -> None:
    adapter = get_runtime_adapter(ADL_RECOGNITION_MODEL_ID)
    request = adapter.build_request(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "root",
    )

    rebased = adapter.with_output_dir(request, tmp_path / "root" / "run-1")

    assert rebased.output_dir == tmp_path / "root" / "run-1"
    assert request.output_dir == tmp_path / "root"


def test_runtime_adapters_reject_wrong_request_types(tmp_path: Path) -> None:
    hand_request = HandInteractionRequest(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "out",
    )
    adl_request = AdlRecognitionRequest(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "out",
    )

    invalid_requests = [
        (HAND_OBJECT_CONTACT_MODEL_ID, adl_request, "Hand-object contact requires"),
        (HAND_INTERACTION_MODEL_ID, adl_request, "Hand interaction requires"),
        (ADL_RECOGNITION_MODEL_ID, hand_request, "ADL recognition requires"),
    ]

    for model_id, request, message in invalid_requests:
        with pytest.raises(TypeError, match=message):
            get_runtime_adapter(model_id).validate(request)


def test_runtime_adapters_reject_wrong_request_types_before_execution(tmp_path: Path) -> None:
    hand_request = HandInteractionRequest(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "out",
    )
    adl_request = AdlRecognitionRequest(
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "out",
    )

    invalid_requests = [
        (HAND_OBJECT_CONTACT_MODEL_ID, adl_request, "Hand-object contact requires"),
        (HAND_INTERACTION_MODEL_ID, adl_request, "Hand interaction requires"),
        (ADL_RECOGNITION_MODEL_ID, hand_request, "ADL recognition requires"),
    ]

    def unexpected_command_runner(_command: list[str]) -> int:
        raise AssertionError("runtime command should not execute")

    def unexpected_streaming_command_runner(
        _command: list[str],
        _progress,
    ) -> int:
        raise AssertionError("runtime command should not execute")

    for model_id, request, message in invalid_requests:
        with pytest.raises(TypeError, match=message):
            get_runtime_adapter(model_id).run(
                request,
                command_runner=unexpected_command_runner,
                streaming_command_runner=unexpected_streaming_command_runner,
                progress=lambda _line: None,
            )


def test_runtime_adapter_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model id: unknown"):
        get_runtime_adapter("unknown")
