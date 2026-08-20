import pytest

from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.catalog import (
    cli_model_ids,
    get_model_definition,
    gui_model_definitions,
)
from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID


def test_catalog_exposes_only_public_cli_models() -> None:
    assert cli_model_ids() == (
        HAND_INTERACTION_MODEL_ID,
        ADL_RECOGNITION_MODEL_ID,
    )


def test_catalog_exposes_gui_metadata_from_one_definition() -> None:
    definitions = gui_model_definitions()

    assert [definition.model_id for definition in definitions] == [
        HAND_INTERACTION_MODEL_ID,
        ADL_RECOGNITION_MODEL_ID,
    ]
    assert definitions[0].uses_dominant_hand is True
    assert definitions[1].gui_payload()["name"] == "Activity recognition (ADL)"


def test_catalog_keeps_internal_model_registered_but_hidden() -> None:
    definition = get_model_definition(HAND_OBJECT_CONTACT_MODEL_ID)

    assert definition.display_name == "Hand-object contact"
    assert definition.cli_enabled is False
    assert definition.gui_enabled is False


def test_catalog_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model id: unknown"):
        get_model_definition("unknown")
