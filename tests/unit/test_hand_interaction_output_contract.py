from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID
from egomodelkit.output_contract import (
    OutputPreviewContext,
    build_output_preview_context,
    build_output_preview_context_from_names,
    build_run_output_layout,
    create_output_scaffold,
    finalize_runtime_outputs,
    infer_input_scenario,
    infer_input_scenario_from_names,
    output_file_descriptions,
    output_folder_tree,
    run_readme_text,
)
from egomodelkit.progress import parse_external_progress_line


def _write_runtime_inputs(layout, videos: list[tuple[str, str, int]]) -> None:
    work = layout.run_dir / "hand_interaction_work"
    extracted = work / "extracted_frames"
    shan = work / "shan_outputs"
    input_rows: list[str] = []
    subclip_rows: list[str] = []

    for index, (input_name, stem, contact_state) in enumerate(videos, start=1):
        input_rows.append(
            f"session001,{index},{input_name},{stem}.MP4,{stem},"
            f"2026-07-21T12:00:00+00:00,1,24,24"
        )
        subclip_rows.append(
            f"session001,{input_name},{stem},{stem}--1,1,0,1,1,30,1"
        )
        frame_dir = extracted / f"{stem}--1"
        shan_dir = shan / f"{stem}--1"
        frame_dir.mkdir(parents=True)
        shan_dir.mkdir(parents=True)
        for frame_index in range(30):
            (frame_dir / f"frame_{frame_index:06d}.jpg").write_bytes(b"image")
            (shan_dir / f"frame_{frame_index:06d}_shan.json").write_text(
                json.dumps(
                    {
                        "hands": [
                            [0, 0, 10, 10, 0.9, contact_state, 0, 0, 0, 1]
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (shan_dir / f"frame_{frame_index:06d}_shan.pkl").write_bytes(b"pickle")

    (layout.run_dir / "hand_interaction_input_manifest.csv").write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time,source_duration_seconds,"
        "source_fps,source_total_frames\n"
        + "\n".join(input_rows)
        + "\n",
        encoding="utf-8",
    )
    (layout.run_dir / "hand_interaction_subclip_manifest.csv").write_text(
        "session_id,input_name,staged_video_stem,subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,valid_duration_seconds,"
        "processing_fps,processing_subclip_duration_seconds\n"
        + "\n".join(subclip_rows)
        + "\n",
        encoding="utf-8",
    )
    (layout.run_dir / "metrics_config.json").write_text(
        json.dumps(
            {
                "subclip_length_seconds": 10,
                "subclip_fps": 30,
                "frame_fps": 30,
                "processing_fps": 30,
                "resize_width": 720,
                "resize_height": 405,
                "pooling_window_seconds": 1,
                "pooling_window_frames": 30,
                "interaction_contact_state_threshold": 3,
                "dominant_hand": "right",
                "non_dominant_hand": "left",
            }
        ),
        encoding="utf-8",
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_hand_interaction_scenario_inference_and_preview_filtering(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "b.MP4").write_bytes(b"video")
    (videos / "a.mp4").write_bytes(b"video")
    (videos / "image.jpg").write_bytes(b"image")

    assert infer_input_scenario(model_id=HAND_INTERACTION_MODEL_ID, input_path=video) == (
        "hand-interaction-single-video"
    )
    assert infer_input_scenario(model_id=HAND_INTERACTION_MODEL_ID, input_path=videos) == (
        "hand-interaction-video-directory"
    )
    assert infer_input_scenario_from_names(
        model_id=HAND_INTERACTION_MODEL_ID,
        input_names=("clip.mp4",),
    ) == "hand-interaction-single-video"
    assert infer_input_scenario_from_names(
        model_id=HAND_INTERACTION_MODEL_ID,
        input_names=("one.mp4", "two.mp4"),
    ) == "hand-interaction-video-directory"

    context = build_output_preview_context(
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=videos,
        output_root=tmp_path / "Results",
        run_id="run-test",
    )
    assert context.input_names == ("a.mp4", "b.MP4")
    assert context.scenario == "hand-interaction-video-directory"

    browser_context = build_output_preview_context_from_names(
        model_id=HAND_INTERACTION_MODEL_ID,
        input_names=("one.mp4", "two.mp4"),
        output_root=tmp_path / "Results",
        run_id="run-browser",
    )
    assert browser_context.scenario == "hand-interaction-video-directory"


def test_hand_interaction_layout_scaffold_tree_descriptions_and_readme(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"video")
    (videos / "two.mp4").write_bytes(b"video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-hi")

    create_output_scaffold(
        layout=layout,
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=videos,
        scenario="hand-interaction-video-directory",
    )

    assert layout.hand_interaction_input_manifest_path == (
        layout.model_outputs_dir / "hand_interaction_input_manifest.csv"
    )
    assert layout.hand_interaction_subclip_manifest_path == (
        layout.post_processing_dir / "hand_interaction_subclip_manifest.csv"
    )
    assert layout.extracted_frames_dir.is_dir()
    assert layout.shan_outputs_dir.is_dir()
    assert not layout.detic_outputs_dir.exists()
    assert layout.session_level_metrics_path.exists()
    manifest = json.loads(layout.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_configuration"] == {
        "dominant_hand": "right",
        "non_dominant_hand": "left",
    }

    context = OutputPreviewContext(
        scenario="hand-interaction-video-directory",
        run_id="run-hi",
        input_names=("one.mp4", "two.mp4"),
        output_name="Results",
    )
    tree = output_folder_tree(context)
    assert "hand_interaction_input_manifest.csv" in tree
    assert "hand_interaction_subclip_manifest.csv" in tree
    assert "detic_outputs" not in tree
    descriptions = {item.name: item.description for item in output_file_descriptions(context)}
    assert "Perc, Dur, and Num" in descriptions["video_level_metrics.csv"]
    assert "continuously active across adjacent input videos" in descriptions[
        "session_level_metrics.csv"
    ]
    assert "per-video metrics remain separate" in descriptions[
        "session_level_metrics.csv"
    ]
    assert "different start and end input names" in descriptions[
        "interaction_segments.csv"
    ]
    readme = run_readme_text(model_id=HAND_INTERACTION_MODEL_ID, context=context)
    single_video_readme = run_readme_text(
        model_id=HAND_INTERACTION_MODEL_ID,
        context=OutputPreviewContext(
            scenario="hand-interaction-single-video",
            run_id="run-hi-single",
            input_names=("one.mp4",),
            output_name="Results",
        ),
    )
    assert "results/video_level_metrics.csv" in readme
    assert "results/session_level_metrics.csv" in readme
    for generated_readme in (readme, single_video_readme):
        assert "continuously active across adjacent input videos" in generated_readme
        assert "different start and end input names" in generated_readme
        assert "without joining interactions across file boundaries" not in generated_readme
        assert "constructed independently inside each original video" not in generated_readme


def test_finalize_hand_interaction_outputs_computes_and_organizes_multi_video_session(
    tmp_path: Path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"video")
    (videos / "two.mp4").write_bytes(b"video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-hi")
    create_output_scaffold(
        layout=layout,
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=videos,
        scenario="hand-interaction-video-directory",
    )
    _write_runtime_inputs(layout, [("one.mp4", "video001", 3), ("two.mp4", "video002", 3)])
    progress: list[str] = []

    finalize_runtime_outputs(
        layout=layout,
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=videos,
        scenario="hand-interaction-video-directory",
        progress=progress.append,
    )

    assert layout.hand_interaction_input_manifest_path.exists()
    assert layout.hand_interaction_subclip_manifest_path.exists()
    assert layout.extracted_frames_dir.joinpath("video001--1").is_dir()
    assert layout.shan_outputs_dir.joinpath("video002--1", "frame_000000_shan.json").exists()
    assert not layout.run_dir.joinpath("hand_interaction_work").exists()
    assert not layout.run_dir.joinpath("hand_interaction_input_manifest.csv").exists()

    video_rows = _csv_rows(layout.video_level_metrics_path)
    session_rows = _csv_rows(layout.session_level_metrics_path)
    segment_rows = _csv_rows(layout.interaction_segments_path)
    frame_rows = _csv_rows(layout.frame_level_predictions_path)
    assert [row["input_name"] for row in video_rows] == ["one.mp4", "two.mp4"]
    assert [row["dur_dominant_hand_seconds"] for row in video_rows] == ["1.0", "1.0"]
    assert [row["num_dominant_hand_per_hour"] for row in video_rows] == ["3600.0", "3600.0"]
    assert len(session_rows) == 1
    assert session_rows[0]["input_video_count"] == "2"
    assert session_rows[0]["recording_time_seconds"] == "2"
    assert session_rows[0]["dur_dominant_hand_seconds"] == "2"
    assert session_rows[0]["num_dominant_hand_per_hour"] == "1800"
    assert len(segment_rows) == 1
    assert segment_rows[0]["start_input_name"] == "one.mp4"
    assert segment_rows[0]["end_input_name"] == "two.mp4"
    assert segment_rows[0]["duration_seconds"] == "2"
    assert len(frame_rows) == 60
    config = json.loads(layout.metrics_config_path.read_text(encoding="utf-8"))
    assert config["processing_fps"] == 30
    assert config["pooling_window_frames"] == 30
    assert config["non_dominant_hand"] == "left"
    kinds = [parse_external_progress_line(message).kind for message in progress]
    assert kinds == [
        "hand_interaction_profiles_calculating",
        "hand_interaction_metrics_calculating",
        "hand_interaction_metrics_calculated",
        "hand_interaction_outputs_organizing",
    ]


def test_finalize_hand_interaction_uses_stable_paths_and_rejects_partial_outputs(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-stable")
    create_output_scaffold(
        layout=layout,
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=video,
        scenario="hand-interaction-single-video",
    )
    layout.hand_interaction_input_manifest_path.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time\n"
        "session001,1,clip.mp4,video001.MP4,video001,now\n",
        encoding="utf-8",
    )
    layout.hand_interaction_subclip_manifest_path.write_text(
        "session_id,input_name,staged_video_stem,subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,valid_duration_seconds,"
        "processing_fps,processing_subclip_duration_seconds\n"
        "session001,clip.mp4,video001,video001--1,1,0,1,1,30,1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no Shan JSON frame predictions"):
        finalize_runtime_outputs(
            layout=layout,
            model_id=HAND_INTERACTION_MODEL_ID,
            input_path=video,
            scenario="hand-interaction-single-video",
        )
    assert "Shan JSON count=0" in layout.runtime_log_path.read_text(encoding="utf-8")


def test_finalize_hand_interaction_rejects_incomplete_shan_subclip(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-partial")
    create_output_scaffold(
        layout=layout,
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=video,
        scenario="hand-interaction-single-video",
    )
    _write_runtime_inputs(layout, [("clip.mp4", "video001", 3)])
    partial_prediction = (
        layout.run_dir
        / "hand_interaction_work"
        / "shan_outputs"
        / "video001--1"
        / "frame_000029_shan.json"
    )
    partial_prediction.unlink()

    with pytest.raises(
        RuntimeError,
        match=r"incomplete Shan JSON predictions.*expected 30, found 29",
    ):
        finalize_runtime_outputs(
            layout=layout,
            model_id=HAND_INTERACTION_MODEL_ID,
            input_path=video,
            scenario="hand-interaction-single-video",
        )


def test_finalize_hand_interaction_rejects_missing_extracted_frame_directory(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    layout = build_run_output_layout(tmp_path / "results", run_id="run-no-frames")
    create_output_scaffold(
        layout=layout,
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=video,
        scenario="hand-interaction-single-video",
    )
    layout.extracted_frames_dir.rmdir()

    with pytest.raises(RuntimeError, match="did not produce the extracted-frame directory"):
        finalize_runtime_outputs(
            layout=layout,
            model_id=HAND_INTERACTION_MODEL_ID,
            input_path=video,
            scenario="hand-interaction-single-video",
        )
