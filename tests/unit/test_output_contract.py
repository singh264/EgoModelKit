from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from egomodelkit.bandini_metrics import VideoProcessingConfig
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.output_contract import (
    InputScenario,
    OutputPreviewContext,
    _input_names_for_preview,
    _preview_items,
    build_output_preview_context,
    build_output_preview_context_from_names,
    build_run_id,
    build_run_output_layout,
    create_output_scaffold,
    finalize_runtime_outputs,
    infer_input_scenario,
    infer_input_scenario_from_names,
    output_file_descriptions,
    output_folder_tree,
    output_preview_note,
    run_readme_text,
    write_run_summary,
)


def test_build_output_preview_context_from_names_for_hand_object_single_image() -> None:
    context = build_output_preview_context_from_names(
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_names = ("frame.jpg",),
        output_root = Path("Results"),
        run_id = "run-2026-06-08-010203"
    )
    
    assert context.scenario == "hand-object-single-image"
    assert "frame_det.png" in output_folder_tree(context)

def test_build_run_id_uses_explicit_datetime() -> None:
    assert build_run_id(datetime(2026, 6, 9, 1, 2, 3, tzinfo = timezone.utc)) == (
        "run-2026-06-09-010203"
    )    

def test_run_output_layout_exposes_all_backend_paths(tmp_path: Path) -> None:
    layout = build_run_output_layout(tmp_path, run_id="run-1")

    assert layout.results_dir == tmp_path / "run-1" / "results"
    assert layout.model_outputs_dir == tmp_path / "run-1" / "technical" / "model_outputs"
    assert layout.adl_input_manifest_path.name == "adl_input_manifest.csv"
    assert layout.adl_segment_manifest_path.name == "adl_segment_manifest.csv"
    assert layout.adl_processing_config_path.name == "adl_processing_config.json"
    assert layout.adl_segment_predictions_path.name == "adl_segment_predictions.csv"
    assert layout.adl_video_summary_path.name == "adl_video_summary.csv"
    assert layout.adl_session_summary_path.name == "adl_session_summary.csv"
    assert layout.metrics_config_path.name == "metrics_config.json"

def test_infer_input_scenario_covers_supported_and_unsupported_cases(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    
    all_preds = tmp_path / "all_preds.pkl"
    all_preds.write_bytes(b"fake-pickle")
    
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    
    assert (
        infer_input_scenario(model_id = HAND_OBJECT_CONTACT_MODEL_ID, input_path = image_dir) ==
        "hand-object-image-directory"
    )
    
    assert (
        infer_input_scenario(model_id = ADL_RECOGNITION_MODEL_ID, input_path = video_dir) ==
        "adl-video-directory"
    )

    assert (
        infer_input_scenario(model_id = ADL_RECOGNITION_MODEL_ID, input_path = all_preds) ==
        "adl-combined-predictions"
    )

    assert (
        infer_input_scenario(model_id = ADL_RECOGNITION_MODEL_ID, input_path = video) ==
        "adl-single-video"
    )

    with pytest.raises(ValueError, match = "Unsupported model id"):
        infer_input_scenario(model_id = "unknown", input_path = video)

def test_infer_input_scenario_from_names_covers_all_branches() -> None:
    assert infer_input_scenario_from_names(
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_names = ("one.jpg", "two.jpg")
    ) == "hand-object-image-directory"
    
    assert infer_input_scenario_from_names(
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_names = ("all_preds.pkl",)
    ) == "adl-combined-predictions"

    assert infer_input_scenario_from_names(
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_names = ("clip.mp4",)
    ) == "adl-single-video"

    assert infer_input_scenario_from_names(
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_names = ("one.mp4", "two.mp4")
    ) == "adl-video-directory"
    
    with pytest.raises(ValueError, match = "At least one input name"):
        infer_input_scenario_from_names(
            model_id = HAND_OBJECT_CONTACT_MODEL_ID, 
            input_names = ()
        )
    
    with pytest.raises(ValueError, match = "Unsupported model id"):
        infer_input_scenario_from_names(
            model_id = "unknown", 
            input_names = ("file.bin",)
        )

def test_build_output_preview_context_reads_input_names_from_directories(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    for name in ["a.jpg", "b.png", "notes.txt"]:
        (image_dir / name).write_bytes(b"fake")
    
    context = build_output_preview_context(
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = image_dir,
        output_root = tmp_path / "Results",
        run_id = "run-1",
    )
        
    assert context.input_names == ("a.jpg", "b.png")

    empty_video_dir = tmp_path / "empty-videos"
    empty_video_dir.mkdir()
    
    empty_context = build_output_preview_context(
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_path = empty_video_dir,
        output_root = Path("/"),
        run_id = "run-2",
    )
    
    assert empty_context.input_names == ("empty-videos",)
    assert empty_context.output_name == "/"
    
    with pytest.raises(ValueError, match = "Unsupported model id"):
        build_output_preview_context(
            model_id = "unknown",
            input_path = image_dir,
            output_root = tmp_path,
            run_id = "run-3",
        )

def test_create_output_scaffold_creates_hand_object_and_adl_contract_files(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-image")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")

    hand_layout = build_run_output_layout(tmp_path / "hand-results", run_id="run-hand")
    create_output_scaffold(
        layout=hand_layout,
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        input_path=image,
        scenario="hand-object-single-image",
        status="running",
    )
    assert (hand_layout.visual_outputs_dir / "hand_object_contact").is_dir()
    assert json.loads(hand_layout.run_summary_path.read_text())["status"] == "running"

    adl_layout = build_run_output_layout(tmp_path / "adl-results", run_id="run-adl")
    create_output_scaffold(
        layout=adl_layout,
        model_id=ADL_RECOGNITION_MODEL_ID,
        input_path=video,
        scenario="adl-single-video",
        video_processing_config=VideoProcessingConfig(dominant_hand="left"),
    )

    assert "prediction_status" in adl_layout.adl_segment_predictions_path.read_text()
    assert "summary_status" in adl_layout.adl_video_summary_path.read_text()
    assert adl_layout.adl_session_summary_path.exists()
    assert "results/adl_segment_predictions.csv" in adl_layout.readme_path.read_text()
    config = json.loads(adl_layout.adl_processing_config_path.read_text())
    assert config["segment_length_seconds"] == 60
    assert config["inference_frame_fps"] == 1
    assert config["subclip_encoding_fps"] == 10
    assert config["active_object_iou_threshold"] == 0.8
    assert config["frame_resize"] is None
    manifest = json.loads(adl_layout.run_manifest_path.read_text())
    assert "dominant_hand" not in manifest["model_configuration"]

def test_create_output_scaffold_records_multi_file_input_names_in_manifest(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "egomodelkit-gui-input-test"
    input_dir.mkdir()

    (input_dir / "part-02.jpg").write_bytes(b"fake-image-2")
    (input_dir / "part-01.jpg").write_bytes(b"fake-image-1")

    layout = build_run_output_layout(
        tmp_path / "results",
        run_id = "run-multi-input-manifest",
    )

    create_output_scaffold(
        layout = layout,
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = input_dir,
        scenario = "hand-object-image-directory",
    )

    manifest = json.loads(layout.run_manifest_path.read_text(encoding = "utf-8"))

    assert manifest["input_names"] == [
        "part-01.jpg",
        "part-02.jpg",
    ]

    assert "input_name" not in manifest 

def test_create_output_scaffold_rejects_unsupported_model(tmp_path: Path) -> None:
    input_file = tmp_path / "file.bin"
    input_file.write_bytes(b"fake")
    
    with pytest.raises(ValueError, match = "Unsupported model id"):
        create_output_scaffold(
            layout = build_run_output_layout(tmp_path, run_id = "run-bad"),
            model_id = "unknown",
            input_path = input_file,
            scenario = cast(InputScenario, "hand-object-single-image"),
        )

def test_output_folder_tree_covers_directory_and_adl_scenarios() -> None:
    hand_directory = OutputPreviewContext(
        scenario="hand-object-image-directory", run_id="run-hand-dir",
        input_names=("a.jpg", "b.jpg", "c.jpg", "d.jpg"), output_name="Results",
    )
    adl_single = OutputPreviewContext(
        scenario="adl-single-video", run_id="run-adl-one",
        input_names=("clip.mp4",), output_name="Results",
    )
    adl_directory = OutputPreviewContext(
        scenario="adl-video-directory", run_id="run-adl-dir",
        input_names=("a.mp4", "b.mp4", "c.mp4", "d.mp4"), output_name="Results",
    )
    adl_combined = OutputPreviewContext(
        scenario="adl-combined-predictions", run_id="run-adl-preds",
        input_names=("all_preds.pkl",), output_name="Results",
    )

    assert "a_det.png" in output_folder_tree(hand_directory)
    assert "adl_segment_predictions.csv" in output_folder_tree(adl_single)
    assert "adl_video_summary.csv" in output_folder_tree(adl_single)
    assert "adl_session_summary.csv" in output_folder_tree(adl_single)
    assert "adl_segment_manifest.csv" in output_folder_tree(adl_single)
    assert "adl_processing_config.json" in output_folder_tree(adl_single)
    assert "video_level_metrics.csv" not in output_folder_tree(adl_single)
    assert "video001--1/" in output_folder_tree(adl_single)
    assert "video001--1/" in output_folder_tree(adl_directory)
    assert "video003--1/" in output_folder_tree(adl_directory)
    assert "          ..." in output_folder_tree(adl_directory)
    assert "adl_segment_manifest.csv" not in output_folder_tree(adl_combined)

def test_output_folder_tree_rejects_unknown_scenario() -> None:
    context = OutputPreviewContext(
        scenario = cast(InputScenario, "unknown"),
        run_id = "run-bad",
        input_names = ("file.bin",),
        output_name = "Results",
    )
    
    with pytest.raises(ValueError, match = "Unsupported output scenario"):
        output_folder_tree(context)

def test_output_file_descriptions_and_notes_cover_remaining_scenarios() -> None:
    hand_directory = OutputPreviewContext(
        scenario="hand-object-image-directory", run_id="run-hand-dir",
        input_names=("a.jpg", "b.jpg"), output_name="Results",
    )
    adl_single = OutputPreviewContext(
        scenario="adl-single-video", run_id="run-adl-one",
        input_names=("clip.mp4",), output_name="Results",
    )

    names = {item.name for item in output_file_descriptions(adl_single)}
    assert "adl_segment_predictions.csv" in names
    assert "adl_video_summary.csv" in names
    assert "adl_session_summary.csv" in names
    assert "adl_segment_manifest.csv" in names
    assert "adl_processing_config.json" in names
    assert "video_level_metrics.csv" not in names
    assert "hand-use metrics" in output_preview_note("adl-single-video")
    assert "results/adl_segment_predictions.csv" in run_readme_text(
        model_id=ADL_RECOGNITION_MODEL_ID, context=adl_single
    )
    assert "visual_outputs/hand_object_contact" in run_readme_text(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID, context=hand_directory
    )

def test_finalize_runtime_outputs_moves_hand_object_visuals(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-image")
    layout = build_run_output_layout(tmp_path / "results", run_id = "run-hand")

    create_output_scaffold(
        layout = layout,
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = image,
        scenario = "hand-object-single-image",
    )

    existing_visual_destination = (
        layout.visual_outputs_dir / "hand_object_contact" / "frame_det.png"
    )
    
    existing_visual_destination.mkdir(parents = True)

    (layout.run_dir / "frame_det.png").write_bytes(b"visual")
    (layout.run_dir / "frame_shan.json").write_text("{}", encoding = "utf-8")
    (layout.run_dir / "frame_shan.pkl").write_bytes(b"pickle")
    (layout.run_dir / "runtime-scratch.tmp").write_text("remove me", encoding = "utf-8")

    finalize_runtime_outputs(
        layout = layout,
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = image,
        scenario = "hand-object-single-image",
    )

    assert (
        layout.visual_outputs_dir / "hand_object_contact" / "frame_det.png"
    ).read_bytes() == b"visual"

    assert not (layout.run_dir / "frame_det.png").exists()
    assert not (layout.run_dir / "frame_shan.json").exists()
    assert not (layout.run_dir / "frame_shan.pkl").exists()
    assert not (layout.run_dir / "runtime-scratch.tmp").exists()
    assert (layout.model_outputs_dir / "frame_shan.json").exists()
    assert (layout.model_outputs_dir / "frame_shan.pkl").exists()
    
def test_finalize_runtime_outputs_normalizes_adl_specific_outputs(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-adl")
    create_output_scaffold(
        layout=layout, model_id=ADL_RECOGNITION_MODEL_ID, input_path=video,
        scenario="adl-single-video",
    )

    (layout.run_dir / "adl_segment_predictions.csv").write_text(
        "session_id,source_video,segment_name,segment_index,prediction_status,predicted_adl\n"
        "session001,clip.mp4,video001--1,1,predicted,self-feeding\n"
    )
    (layout.run_dir / "adl_video_summary.csv").write_text(
        "session_id,source_video,segment_count\nsession001,clip.mp4,1\n"
    )
    (layout.run_dir / "adl_session_summary.csv").write_text(
        "session_id,input_video_count,segment_count\nsession001,1,1\n"
    )
    (layout.run_dir / "all_preds.pkl").write_bytes(b"pickle")
    (layout.run_dir / "adl_input_manifest.csv").write_text("input_name\nclip.mp4\n")
    (layout.run_dir / "adl_segment_manifest.csv").write_text(
        "source_video,segment_name,segment_index,valid_duration_seconds\n"
        "clip.mp4,video001--1,1,8\n"
    )
    (layout.run_dir / "adl_processing_config.json").write_text('{"segment_length_seconds": 60}')

    runtime_adl_dir = (
        layout.run_dir / "adl_recognition_work" / "egoviz_data" / "meal-preparation-cleanup"
    )
        
    (runtime_adl_dir / "subclips" / "video001--1").mkdir(parents=True)
    (runtime_adl_dir / "subclips" / "video001--1" / "frame_001.jpg").write_bytes(b"jpg")
    (runtime_adl_dir / "detic_raw" / "video001--1").mkdir(parents=True)
    (runtime_adl_dir / "detic_raw" / "video001--1" / "frame_001_detic.pkl").write_bytes(b"detic")
    (runtime_adl_dir / "subclips_shan" / "video001--1").mkdir(parents=True)
    (runtime_adl_dir / "subclips_shan" / "video001--1" / "frame_001_shan.pkl").write_bytes(b"shan")

    finalize_runtime_outputs(
        layout=layout, model_id=ADL_RECOGNITION_MODEL_ID, input_path=video,
        scenario="adl-single-video",
    )

    assert "self-feeding" in layout.adl_segment_predictions_path.read_text()
    assert layout.adl_video_summary_path.exists()
    assert layout.adl_session_summary_path.exists()
    assert layout.adl_segment_manifest_path.exists()
    assert layout.adl_processing_config_path.exists()
    assert (layout.model_outputs_dir / "all_preds.pkl").read_bytes() == b"pickle"
    assert (layout.adl_extracted_frames_dir / "video001--1" / "frame_001.jpg").exists()
    assert not layout.video_level_metrics_path.exists()
    assert not layout.frame_level_predictions_path.exists()
    assert not layout.interaction_segments_path.exists()

def test_finalize_runtime_outputs_tolerates_missing_adl_runtime_work(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-adl-missing")
    create_output_scaffold(
        layout=layout, model_id=ADL_RECOGNITION_MODEL_ID, input_path=video,
        scenario="adl-single-video",
    )
    (layout.run_dir / "adl_segment_predictions.csv").write_text("predicted_adl\nmeal\n")
    (layout.run_dir / "adl_video_summary.csv").write_text("segment_count\n1\n")
    (layout.run_dir / "adl_session_summary.csv").write_text("segment_count\n1\n")

    finalize_runtime_outputs(
        layout=layout, model_id=ADL_RECOGNITION_MODEL_ID, input_path=video,
        scenario="adl-single-video",
    )

    assert "meal" in layout.adl_segment_predictions_path.read_text()
    assert list(layout.adl_extracted_frames_dir.iterdir()) == []

def test_finalize_runtime_outputs_rejects_unsupported_model(tmp_path: Path) -> None:
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(b"fake")
    layout = build_run_output_layout(tmp_path, run_id = "run-bad")

    with pytest.raises(ValueError, match = "Unsupported model id"):
        finalize_runtime_outputs(
            layout = layout,
            model_id = "unknown",
            input_path = input_file,
            scenario = cast(InputScenario, "hand-object-single-image"),
        )

def test_write_run_summary_records_multi_file_input_names_without_temp_path(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "egomodelkit-gui-input-test"
    input_dir.mkdir()

    (input_dir / "part-02.mp4").write_bytes(b"fake-video-2")
    (input_dir / "part-01.mp4").write_bytes(b"fake-video-1")

    layout = build_run_output_layout(tmp_path, run_id = "run-summary")

    layout.run_dir.mkdir(parents = True)

    write_run_summary(
        layout = layout,
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_path = input_dir,
        scenario = "adl-video-directory",
        status = "completed",
    )

    summary = json.loads(layout.run_summary_path.read_text(encoding = "utf-8"))

    assert summary["status"] == "completed"
    assert summary["input_names"] == ["part-01.mp4", "part-02.mp4"]
    assert "input_name" not in summary
    assert "input_path" not in summary   

def test_private_preview_helpers_cover_small_lists_and_unsupported_model(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    assert _preview_items(["a", "b"]) == ["a", "b"]
    
    with pytest.raises(ValueError, match = "Unsupported model id"):
        _input_names_for_preview(model_id = "unknown", input_path = input_dir)

def test_finalize_runtime_outputs_requires_adl_prediction_files(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-adl-missing-output")
    create_output_scaffold(
        layout=layout, model_id=ADL_RECOGNITION_MODEL_ID, input_path=video,
        scenario="adl-single-video",
    )

    with pytest.raises(RuntimeError, match="ADL segment predictions"):
        finalize_runtime_outputs(
            layout=layout, model_id=ADL_RECOGNITION_MODEL_ID, input_path=video,
            scenario="adl-single-video",
        )
