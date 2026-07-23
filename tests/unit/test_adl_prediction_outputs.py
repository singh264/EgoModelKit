import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_predict_module():
    polars = ModuleType("polars")
    polars.from_pandas = lambda value: value  # type: ignore[attr-defined]

    egoviz = ModuleType("egoviz")
    models = ModuleType("egoviz.models")
    models.inference = SimpleNamespace()
    models.processing = SimpleNamespace()

    sys.modules["polars"] = polars
    sys.modules["egoviz"] = egoviz
    sys.modules["egoviz.models"] = models

    script_path = (
        Path(__file__).parents[2]
        / "src"
        / "egomodelkit"
        / "resources"
        / "containers"
        / "adl_recognition"
        / "predict_adl.py"
    )
    spec = importlib.util.spec_from_file_location("egomodelkit_test_predict_adl", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adl_prediction_outputs_preserve_segment_metadata_and_summarize(tmp_path: Path) -> None:
    pd = pytest.importorskip(
        "pandas",
        reason="pandas is provided by the ADL runtime image, not the host package",
    )
    module = _load_predict_module()
    segment_manifest = tmp_path / "adl_segment_manifest.csv"
    segment_manifest.write_text(
        "session_id,source_video,staged_video_stem,segment_name,segment_index,"
        "start_time_seconds,end_time_seconds,valid_duration_seconds\n"
        "session001,a.mp4,video001,video001--1,1,0,60,60\n"
        "session001,a.mp4,video001,video001--2,2,60,75,15\n"
        "session001,b.mp4,video002,video002--1,1,0,20,20\n",
        encoding="utf-8",
    )
    input_manifest = tmp_path / "adl_input_manifest.csv"
    input_manifest.write_text(
        "input_name,source_duration_seconds\na.mp4,75\nb.mp4,20\n",
        encoding="utf-8",
    )
    results = pd.DataFrame(
        {
            "video": ["video001--1", "video001--2", "video002--1"],
            "predicted_label": ["\ufffdmeal\x00", "\ufffdmeal\x00", "\ufffdmobility\x00"],
            "predicted_class": [1, 1, 2],
            "prob_meal": [0.8, 0.7, 0.1],
            "prob_mobility": [0.2, 0.3, 0.9],
        }
    )

    segments = module.build_segment_predictions(
        results,
        segment_manifest_path=segment_manifest,
    )
    videos = module.build_video_summary(
        segments,
        input_manifest_path=input_manifest,
    )
    sessions = module.build_session_summary(segments)

    assert list(segments["source_video"]) == ["a.mp4", "a.mp4", "b.mp4"]
    assert list(segments["valid_duration_seconds"]) == [60, 15, 20]
    assert list(segments["predicted_adl"]) == ["meal", "meal", "mobility"]
    assert videos.loc[0, "source_duration_seconds"] == 75
    assert json.loads(videos.loc[0, "predicted_adl_counts"]) == {"meal": 2}
    assert json.loads(videos.loc[0, "predicted_adl_valid_duration_seconds"]) == {
        "meal": 75.0
    }
    assert sessions.loc[0, "input_video_count"] == 2
    assert sessions.loc[0, "segment_count"] == 3


def test_adl_prediction_labels_fall_back_safely_without_probability_columns() -> None:
    pd = pytest.importorskip(
        "pandas",
        reason="pandas is provided by the ADL runtime image and development extras",
    )
    module = _load_predict_module()
    results = pd.DataFrame(
        {
            "video": ["video001--1", "video002--1", "video003--1"],
            "predicted_label": [b"meal", "bad\x00label", "bad\ufffdlabel"],
            "predicted_class": [1, 2, 3],
        }
    )

    segments = module.build_segment_predictions(
        results,
        segment_manifest_path=None,
    )

    assert list(segments["predicted_adl"]) == ["meal", "", ""]
    assert list(segments["prediction_status"]) == ["predicted", "missing", "missing"]


def test_adl_classifier_wrapper_keeps_paper_feature_processing() -> None:
    script_path = (
        Path(__file__).parents[2]
        / "src"
        / "egomodelkit"
        / "resources"
        / "containers"
        / "adl_recognition"
        / "predict_adl.py"
    )
    source = script_path.read_text(encoding="utf-8")

    assert "processing.generate_binary_presence_df(frame_df)" in source
    assert "processing.row_wise_min_max_scaling(features)" in source
