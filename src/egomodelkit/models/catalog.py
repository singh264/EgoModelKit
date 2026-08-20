"""Central catalog of model metadata exposed by EgoModelKit interfaces."""

from dataclasses import dataclass
from typing import Final

from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_MODEL_ID,
    ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES,
)
from egomodelkit.models.hand_interaction import (
    HAND_INTERACTION_MODEL_ID,
    HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_MODEL_ID,
    HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES,
)


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Stable user-facing metadata and capabilities for one model integration."""

    model_id: str
    display_name: str
    description: str
    supported_input_extensions: frozenset[str]
    accepted_input_label: str
    output_label: str
    recommended_output_path: str
    cli_enabled: bool = False
    gui_enabled: bool = False
    uses_dominant_hand: bool = False

    def gui_payload(self) -> dict[str, object]:
        """Return the frontend API representation for this model."""
        return {
            "id": self.model_id,
            "name": self.display_name,
            "description": self.description,
            "supportedInputExtensions": sorted(self.supported_input_extensions),
            "acceptedInputLabel": self.accepted_input_label,
            "outputLabel": self.output_label,
        }


MODEL_DEFINITIONS: Final[tuple[ModelDefinition, ...]] = (
    ModelDefinition(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        display_name="Hand-object contact",
        description="Detects hand-object contact in egocentric images.",
        supported_input_extensions=HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES,
        accepted_input_label="single image or multiple images",
        output_label="hand-object contact detections",
        recommended_output_path="visual_outputs/hand_object_contact/",
    ),
    ModelDefinition(
        model_id=HAND_INTERACTION_MODEL_ID,
        display_name="Hand interaction",
        description=(
            "Measures functional hand-object interactions in egocentric videos."
        ),
        supported_input_extensions=HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES,
        accepted_input_label="single MP4 video or multiple MP4 videos",
        output_label="interaction profiles and hand-use metrics",
        recommended_output_path="results/video_level_metrics.csv",
        cli_enabled=True,
        gui_enabled=True,
        uses_dominant_hand=True,
    ),
    ModelDefinition(
        model_id=ADL_RECOGNITION_MODEL_ID,
        display_name="Activity recognition (ADL)",
        description=(
            "Processes egocentric video clips for activity of daily living (ADL) "
            "recognition."
        ),
        supported_input_extensions=ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES,
        accepted_input_label="single MP4 video or multiple MP4 videos",
        output_label="segment predictions and video/session summaries",
        recommended_output_path="results/adl_segment_predictions.csv",
        cli_enabled=True,
        gui_enabled=True,
    ),
)

_MODEL_DEFINITIONS_BY_ID: Final[dict[str, ModelDefinition]] = {
    definition.model_id: definition for definition in MODEL_DEFINITIONS
}


def get_model_definition(model_id: str) -> ModelDefinition:
    """Return one registered model definition or fail with a stable error."""
    try:
        return _MODEL_DEFINITIONS_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model id: {model_id}") from exc


def cli_model_ids() -> tuple[str, ...]:
    """Return public model ids exposed by the CLI."""
    return tuple(
        definition.model_id
        for definition in MODEL_DEFINITIONS
        if definition.cli_enabled
    )


def gui_model_definitions() -> tuple[ModelDefinition, ...]:
    """Return model definitions exposed by the model-selection GUI."""
    return tuple(
        definition for definition in MODEL_DEFINITIONS if definition.gui_enabled
    )
