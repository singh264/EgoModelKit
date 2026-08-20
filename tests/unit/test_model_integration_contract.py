from egomodelkit.models.catalog import MODEL_DEFINITIONS
from egomodelkit.output_contract import get_model_output_strategy
from egomodelkit.runtime.adapters import get_runtime_adapter
from egomodelkit.runtime.disk_space import get_model_storage_strategy


def test_every_catalog_model_has_standard_integration_strategies() -> None:
    for definition in MODEL_DEFINITIONS:
        model_id = definition.model_id

        assert get_runtime_adapter(model_id).model_id == model_id
        assert get_model_output_strategy(model_id).model_id == model_id
        assert get_model_storage_strategy(model_id).model_id == model_id
