from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

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
    layout = build_run_output_layout(tmp_path, run_id = "run-1")
    
    assert layout.results_dir == tmp_path / "run-1" / "results"
    assert layout.visual_outputs_dir == tmp_path / "run-1" / "visual_outputs"
    assert layout.technical_dir == tmp_path / "run-1" / "technical"
    assert layout.model_outputs_dir == tmp_path / "run-1" / "technical" / "model_outputs"
    
    assert layout.post_processing_intermediate_dir == (
        tmp_path / "run-1" / "technical" / "post_processing_intermediate"
    )
    
    assert layout.intermediate_files_dir == (
        tmp_path / "run-1" / "technical" / "intermediate_files"
    )

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
    
    hand_layout = build_run_output_layout(tmp_path / "hand-results", run_id = "run-hand")
    
    create_output_scaffold(
        layout = hand_layout,
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = image,
        scenario = "hand-object-single-image",
        status = "running",
    )
    
    assert (hand_layout.visual_outputs_dir / "hand_object_contact").is_dir()
    assert hand_layout.model_outputs_dir.is_dir()
    assert hand_layout.progress_log_path.exists()
    assert json.loads(hand_layout.run_summary_path.read_text())["status"] == "running"
    assert json.loads(hand_layout.run_manifest_path.read_text())["output_contract_version"] == 1
    
    adl_layout = build_run_output_layout(tmp_path / "adl-results", run_id = "run-adl")
    
    create_output_scaffold(
        layout = adl_layout,
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_path = video,
        scenario = "adl-single-video",
    )

    assert adl_layout.results_dir.is_dir()
    assert adl_layout.model_outputs_dir.is_dir()
    assert adl_layout.post_processing_intermediate_dir.is_dir()
    assert adl_layout.intermediate_files_dir.is_dir()
    assert "results/video_level_metrics.csv" in adl_layout.readme_path.read_text()
    
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
        scenario = "hand-object-image-directory",
        run_id = "run-hand-dir",
        input_names = ("a.jpg", "b.jpg", "c.jpg", "d.jpg"),
        output_name = "Results",
    )
    
    adl_single = OutputPreviewContext(
        scenario = "adl-single-video",
        run_id = "run-adl-one",
        input_names = ("clip.mp4",),
        output_name = "Results",
    )
    
    adl_directory = OutputPreviewContext(
        scenario = "adl-video-directory",
        run_id = "run-adl-dir",
        input_names = ("a.mp4", "b.mp4", "c.mp4", "d.mp4"),
        output_name = "Results",
    )
    
    adl_combined = OutputPreviewContext(
        scenario = "adl-combined-predictions",
        run_id = "run-adl-preds",
        input_names = ("all_preds.pkl",),
        output_name = "Results",
    )
    
    assert "a_det.png" in output_folder_tree(hand_directory)
    assert "        ..." in output_folder_tree(hand_directory)
    assert "clip/" in output_folder_tree(adl_single)
    assert "          ..." in output_folder_tree(adl_directory)
    assert "all_preds.pkl" in output_folder_tree(adl_combined)

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
        scenario = "hand-object-image-directory",
        run_id = "run-hand-dir",
        input_names = ("a.jpg", "b.jpg"),
        output_name = "Results",
    )
    
    adl_single = OutputPreviewContext(
        scenario = "adl-single-video",
        run_id = "run-adl-one",
        input_names = ("clip.mp4",),
        output_name = "Results",
    )
        
    assert any(
        description.name == "*_shan.json"
        for description in output_file_descriptions(hand_directory)
    )
    
    assert any(
        description.name == "video_level_metrics.csv"
        for description in output_file_descriptions(adl_single)
    )
    
    assert (
        "Frame-level metrics are not generated" in 
        output_preview_note("hand-object-image-directory")
    )
    
    assert "Most users should review" in output_preview_note("adl-single-video")
    
    assert (
        "visual_outputs/hand_object_contact" in 
        run_readme_text(model_id = HAND_OBJECT_CONTACT_MODEL_ID, context = hand_directory)
    )

def test_write_run_summary_updates_existing_summary(tmp_path: Path) -> None:
    input_file = tmp_path / "frame.jpg"
    input_file.write_bytes(b"fake-image")
    
    layout = build_run_output_layout(tmp_path, run_id = "run-summary")
    layout.run_dir.mkdir(parents = True)
    
    write_run_summary(
        layout = layout,
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = input_file,
        scenario = "hand-object-single-image",
        status = "completed",
    )
    
    assert json.loads(layout.run_summary_path.read_text())["status"] == "completed"    

def test_private_preview_helpers_cover_small_lists_and_unsupported_model(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    assert _preview_items(["a", "b"]) == ["a", "b"]
    
    with pytest.raises(ValueError, match = "Unsupported model id"):
        _input_names_for_preview(model_id = "unknown", input_path = input_dir)
