from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import egomodelkit.bandini_metrics as bandini_metrics
from egomodelkit.bandini_metrics import (
    DEFAULT_VIDEO_PROCESSING_CONFIG,
    InputVideoMapping,
    SubclipTimingMapping,
    VideoProcessingConfig,
    build_interaction_segments,
    build_session_level_metrics,
    build_video_level_metrics,
    load_frame_interaction_predictions,
    load_input_video_mappings,
    load_subclip_timing_mappings,
    read_video_processing_config,
    write_bandini_metric_files,
    write_video_processing_config,
)

SESSION_ID = "session001"
FIRST_VIDEO = "part1.mp4"
SECOND_VIDEO = "part2.mp4"
FIRST_STAGED_STEM = "video001"
SECOND_STAGED_STEM = "video002"

SYNTHETIC_HAND_BOX_LEFT = 0
SYNTHETIC_HAND_BOX_TOP = 0
SYNTHETIC_HAND_BOX_RIGHT = 10
SYNTHETIC_HAND_BOX_BOTTOM = 10

SYNTHETIC_OBJECT_BOX_LEFT = 0
SYNTHETIC_OBJECT_BOX_TOP = 0
SYNTHETIC_OBJECT_BOX_RIGHT = 0
SYNTHETIC_OBJECT_BOX_BOTTOM = 0

SYNTHETIC_HAND_SIDE = 1

NO_CONTACT_STATE = 0
SELF_CONTACT_STATE = 1
OTHER_PERSON_CONTACT_STATE = 2
NON_PORTABLE_OBJECT_CONTACT_STATE = 3
PORTABLE_OBJECT_CONTACT_STATE = 4

DEFAULT_FPS = 30
ONE_SECOND_FRAME_COUNT = 30
LOW_FPS = 6
LOW_FPS_POOLING_SECONDS = 0.5
LOW_FPS_POOLING_FRAMES = 3

HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.1
EXPECTED_ONE_SECOND_PERCENTAGE = 100.0
EXPECTED_ONE_SECOND_NUM_PER_HOUR = 3600.0

SYNTHETIC_OBJECT_OFFSET_MAGNITUDE = 0
SYNTHETIC_OBJECT_OFFSET_X = 0
SYNTHETIC_OBJECT_OFFSET_Y = 0

LEFT_HAND_VALUE = 0
RIGHT_HAND_VALUE = 1


def _write_shan_json_with_hands(
    path: Path,
    hands: list[list[object]],
) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps({"hands": hands}), encoding = "utf-8")

def _shan_hand_prediction(
    contact_state: int,
    *,
    side: int = RIGHT_HAND_VALUE,
    score: float = HIGH_CONFIDENCE,
) -> list[object]:
    return [
        SYNTHETIC_HAND_BOX_LEFT,
        SYNTHETIC_HAND_BOX_TOP,
        SYNTHETIC_HAND_BOX_RIGHT,
        SYNTHETIC_HAND_BOX_BOTTOM,
        score,
        contact_state,
        SYNTHETIC_OBJECT_OFFSET_MAGNITUDE,
        SYNTHETIC_OBJECT_OFFSET_X,
        SYNTHETIC_OBJECT_OFFSET_Y,
        side,
    ]

def _write_shan_json(
    path: Path,
    contact_state: int,
    *,
    side: int = RIGHT_HAND_VALUE,
    score: float = HIGH_CONFIDENCE,
) -> None:
    _write_shan_json_with_hands(
        path,
        [
            _shan_hand_prediction(
                contact_state,
                side=side,
                score=score,
            )
        ],
    )

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding = "utf-8", newline = "") as csv_file:
        return list(csv.DictReader(csv_file))

def _mapping(
    *,
    input_name: str = FIRST_VIDEO,
    staged_stem: str = FIRST_STAGED_STEM,
    session_sort_index: int = 1,
    input_modified_time: str = "2026-07-05T10:01:00",
) -> InputVideoMapping:
    return InputVideoMapping(
        session_id = SESSION_ID,
        session_sort_index = session_sort_index,
        input_name = input_name,
        staged_video_name = f"{staged_stem}.MP4",
        staged_video_stem = staged_stem,
        input_modified_time = input_modified_time,
    )

def _subclip_mapping(
    *,
    input_name: str = FIRST_VIDEO,
    staged_stem: str = FIRST_STAGED_STEM,
    subclip_index: int = 1,
    source_start_seconds: float = 0.0,
    valid_duration_seconds: float = 10.0,
    processing_fps: float = DEFAULT_FPS,
    processing_subclip_duration_seconds: float = 10.0,
) -> SubclipTimingMapping:
    return SubclipTimingMapping(
        session_id = SESSION_ID,
        input_name = input_name,
        staged_video_stem = staged_stem,
        subclip_name = f"{staged_stem}--{subclip_index}",
        subclip_index = subclip_index,
        source_start_seconds = source_start_seconds,
        source_end_seconds = source_start_seconds + valid_duration_seconds,
        valid_duration_seconds = valid_duration_seconds,
        processing_fps = processing_fps,
        processing_subclip_duration_seconds = processing_subclip_duration_seconds,
    )

def test_default_config_values_and_derived_pooling_window() -> None:
    config = DEFAULT_VIDEO_PROCESSING_CONFIG

    assert config.subclip_length_seconds == 10
    assert config.subclip_fps == DEFAULT_FPS
    assert config.frame_fps == DEFAULT_FPS
    assert config.resize_width == 720
    assert config.resize_height == 405
    assert config.pooling_window_seconds == 1.0
    assert config.pooling_window_frames == ONE_SECOND_FRAME_COUNT
    assert config.interaction_contact_state_threshold == NON_PORTABLE_OBJECT_CONTACT_STATE

def test_config_validation_rejects_invalid_values() -> None:
    invalid_kwargs = [
        {"subclip_length_seconds": 0},
        {"subclip_fps": 0},
        {"frame_fps": 0},
        {"resize_width": 0},
        {"resize_height": 0},
        {"pooling_window_seconds": 0},
        {"subclip_length_seconds": -1},
    ]

    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            VideoProcessingConfig(**kwargs)

def test_config_validation_rejects_frame_fps_greater_than_subclip_fps() -> None:
    with pytest.raises(ValueError, match = "frame_fps"):
        VideoProcessingConfig(subclip_fps = 15, frame_fps = DEFAULT_FPS)

def test_config_serialization_records_defaults_and_derived_values(tmp_path: Path) -> None:
    config_path = tmp_path / "metrics_config.json"

    write_video_processing_config(config_path, DEFAULT_VIDEO_PROCESSING_CONFIG)

    payload = json.loads(config_path.read_text(encoding = "utf-8"))
    
    assert payload["frame_fps"] == DEFAULT_FPS
    assert payload["pooling_window_frames"] == ONE_SECOND_FRAME_COUNT
    assert payload["interaction_contact_state_threshold"] == NON_PORTABLE_OBJECT_CONTACT_STATE
    assert read_video_processing_config(config_path) == DEFAULT_VIDEO_PROCESSING_CONFIG
    
    assert read_video_processing_config(tmp_path / "missing.json") == (
        DEFAULT_VIDEO_PROCESSING_CONFIG
    )

def test_manifest_is_required_for_metric_mapping(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match = "Input manifest is required for Bandini metric computation",
    ):
        load_input_video_mappings(
            input_manifest_path=tmp_path / "missing.csv",
        )

def test_manifest_preserves_session_order_names_and_modified_time(tmp_path: Path) -> None:
    manifest_path = tmp_path / "adl_input_manifest.csv"
    
    manifest_path.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time\n"
        "session001,2,part2.mp4,video002.MP4,video002,2026-07-05T10:21:00\n"
        "session001,1,part1.mp4,video001.MP4,video001,2026-07-05T10:01:00\n",
        encoding="utf-8",
    )

    mappings = load_input_video_mappings(
        input_manifest_path = manifest_path,
    )

    assert [mapping.input_name for mapping in mappings] == [FIRST_VIDEO, SECOND_VIDEO]
    assert [mapping.session_sort_index for mapping in mappings] == [1, 2]
    assert mappings[0].input_modified_time == "2026-07-05T10:01:00"

def test_contact_state_threshold_comes_from_config(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"
    
    contact_states = [
        NO_CONTACT_STATE,
        SELF_CONTACT_STATE,
        OTHER_PERSON_CONTACT_STATE,
        NON_PORTABLE_OBJECT_CONTACT_STATE,
        PORTABLE_OBJECT_CONTACT_STATE,
    ]

    for index, contact_state in enumerate(contact_states, start = 1):
        _write_shan_json(
            shan_dir / "video001.--1" / f"frame_{index:03d}_shan.json",
            contact_state,
        )

    config = VideoProcessingConfig(
        subclip_fps = len(contact_states),
        frame_fps = len(contact_states),
        pooling_window_seconds = 1 / len(contact_states),
    )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = config,
    )

    assert [frame.raw_any_interaction for frame in frames] == [0, 0, 0, 1, 1]
    assert [frame.any_interaction for frame in frames] == [0, 0, 0, 1, 1]

def test_contact_state_threshold_can_be_overridden(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"
    
    _write_shan_json(
        shan_dir / "video001.--1" / "frame_001_shan.json",
        NON_PORTABLE_OBJECT_CONTACT_STATE,
    )
    
    _write_shan_json(
        shan_dir / "video001.--1" / "frame_002_shan.json",
        PORTABLE_OBJECT_CONTACT_STATE,
    )

    config = VideoProcessingConfig(
        subclip_fps = 2,
        frame_fps = 2,
        pooling_window_seconds = 0.5,
        interaction_contact_state_threshold = PORTABLE_OBJECT_CONTACT_STATE,
    )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = config,
    )

    assert [frame.raw_any_interaction for frame in frames] == [0, 1]

def test_low_confidence_hands_still_affect_bandini_interaction_labels(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"

    _write_shan_json(
        shan_dir / "video001.--1" / "frame_001_shan.json",
        PORTABLE_OBJECT_CONTACT_STATE,
        score = LOW_CONFIDENCE,
    )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = VideoProcessingConfig(subclip_fps=1, frame_fps=1),
    )

    assert frames[0].detected_hand_count == 1
    assert frames[0].max_contact_state == PORTABLE_OBJECT_CONTACT_STATE
    assert frames[0].raw_any_interaction == 1

def test_statepool_default_one_second_window_uses_30_frames(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for index in range(1, ONE_SECOND_FRAME_COUNT + 1):
        contact_state = (
            NON_PORTABLE_OBJECT_CONTACT_STATE
            if index <= 16
            else NO_CONTACT_STATE
        )
        
        _write_shan_json(
            shan_dir / "video001.--1" / f"frame_{index:03d}_shan.json",
            contact_state,
        )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    assert DEFAULT_VIDEO_PROCESSING_CONFIG.pooling_window_frames == ONE_SECOND_FRAME_COUNT
    assert {frame.any_interaction for frame in frames} == {1}

def test_statepool_window_adapts_to_lower_configured_fps(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"
    
    contact_states = [
        NON_PORTABLE_OBJECT_CONTACT_STATE,
        NON_PORTABLE_OBJECT_CONTACT_STATE,
        NO_CONTACT_STATE,
        NO_CONTACT_STATE,
        NO_CONTACT_STATE,
        NON_PORTABLE_OBJECT_CONTACT_STATE,
    ]

    for index, contact_state in enumerate(contact_states, start = 1):
        _write_shan_json(
            shan_dir / "video001.--1" / f"frame_{index:03d}_shan.json",
            contact_state,
        )

    config = VideoProcessingConfig(
        subclip_fps = LOW_FPS,
        frame_fps = LOW_FPS,
        pooling_window_seconds = LOW_FPS_POOLING_SECONDS,
    )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = config,
    )

    assert config.pooling_window_frames == LOW_FPS_POOLING_FRAMES
    assert [frame.any_interaction for frame in frames] == [1, 1, 1, 0, 0, 0]

def test_video_frame_index_continues_across_egoviz_subclips(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for subclip_index in [1, 2]:
        for frame_index in [1, 2]:
            _write_shan_json(
                shan_dir
                / f"video001.--{subclip_index}"
                / f"frame_{frame_index:03d}_shan.json",
                NON_PORTABLE_OBJECT_CONTACT_STATE,
            )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = VideoProcessingConfig(subclip_fps = LOW_FPS, frame_fps = LOW_FPS),
    )

    assert [frame.video_frame_index for frame in frames] == [1, 2, 3, 4]

def test_session_frame_index_continues_across_default_session_videos(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for staged_stem in [FIRST_STAGED_STEM, SECOND_STAGED_STEM]:
        for frame_index in [1, 2]:
            _write_shan_json(
                shan_dir / f"{staged_stem}.--1" / f"frame_{frame_index:03d}_shan.json",
                NON_PORTABLE_OBJECT_CONTACT_STATE,
            )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [
            _mapping(
                input_name = FIRST_VIDEO, 
                staged_stem = FIRST_STAGED_STEM, 
                session_sort_index = 1
            ),
            _mapping(
                input_name = SECOND_VIDEO, 
                staged_stem= SECOND_STAGED_STEM, 
                session_sort_index = 2),
        ],
        config = VideoProcessingConfig(
            subclip_fps = LOW_FPS, 
            frame_fps = LOW_FPS),
    )

    assert [frame.session_frame_index for frame in frames] == [1, 2, 3, 4]
    
    assert [frame.input_name for frame in frames] == [
        FIRST_VIDEO,
        FIRST_VIDEO,
        SECOND_VIDEO,
        SECOND_VIDEO,
    ]

def test_video_and_session_metrics_use_configured_fps(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for index in range(1, ONE_SECOND_FRAME_COUNT + 1):
        _write_shan_json(
            shan_dir / "video001.--1" / f"frame_{index:03d}_shan.json",
            NON_PORTABLE_OBJECT_CONTACT_STATE,
        )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )
    
    segments = build_interaction_segments(frames, config = DEFAULT_VIDEO_PROCESSING_CONFIG)
    
    video_metrics = build_video_level_metrics(
        frames,
        mappings = [_mapping()],
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )
    
    session_metrics = build_session_level_metrics(
        frames,
        segments,
        mappings = [_mapping()],
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    assert len(segments) == 1
    assert segments[0].duration_seconds == 1.0
    assert video_metrics[0].recording_time_seconds == 1.0
    assert video_metrics[0].perc_dominant_hand == EXPECTED_ONE_SECOND_PERCENTAGE
    assert video_metrics[0].perc_non_dominant_hand == 0.0
    assert video_metrics[0].perc_bilateral == EXPECTED_ONE_SECOND_PERCENTAGE / 2
    assert video_metrics[0].num_dominant_hand_per_hour == EXPECTED_ONE_SECOND_NUM_PER_HOUR
    assert video_metrics[0].num_non_dominant_hand_per_hour == 0.0
    assert video_metrics[0].num_bilateral_per_hour == EXPECTED_ONE_SECOND_NUM_PER_HOUR

    assert session_metrics[0].recording_time_seconds == 1.0
    assert session_metrics[0].num_dominant_hand_per_hour == EXPECTED_ONE_SECOND_NUM_PER_HOUR
    assert session_metrics[0].num_non_dominant_hand_per_hour == 0.0
    assert session_metrics[0].num_bilateral_per_hour == EXPECTED_ONE_SECOND_NUM_PER_HOUR

def test_session_metrics_combine_multiple_camera_chunks(tmp_path: Path) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for staged_stem in [FIRST_STAGED_STEM, SECOND_STAGED_STEM]:
        for frame_index in range(1, ONE_SECOND_FRAME_COUNT + 1):
            _write_shan_json(
                shan_dir / f"{staged_stem}.--1" / f"frame_{frame_index:03d}_shan.json",
                NON_PORTABLE_OBJECT_CONTACT_STATE,
            )

    mappings = [
        _mapping(
            input_name = FIRST_VIDEO, 
            staged_stem = FIRST_STAGED_STEM, 
            session_sort_index = 1
        ),
        _mapping(
            input_name = SECOND_VIDEO, 
            staged_stem = SECOND_STAGED_STEM, 
            session_sort_index= 2 
        ),
    ]
    
    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = mappings,
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )
    
    segments = build_interaction_segments(frames, config = DEFAULT_VIDEO_PROCESSING_CONFIG)
    
    video_metrics = build_video_level_metrics(
        frames,
        mappings = mappings,
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )
    
    session_metrics = build_session_level_metrics(
        frames,
        segments,
        mappings = mappings,
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    assert [metric.input_name for metric in video_metrics] == [FIRST_VIDEO, SECOND_VIDEO]
    assert [metric.recording_time_seconds for metric in video_metrics] == [1.0, 1.0]
    assert len(session_metrics) == 1
    assert session_metrics[0].session_id == SESSION_ID
    assert session_metrics[0].input_video_count == 2
    assert session_metrics[0].recording_time_seconds == 2.0

def test_session_interaction_continues_across_input_video_boundary(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for staged_stem in [FIRST_STAGED_STEM, SECOND_STAGED_STEM]:
        for frame_index in range(1, ONE_SECOND_FRAME_COUNT + 1):
            _write_shan_json(
                shan_dir / f"{staged_stem}.--1" / f"frame_{frame_index:03d}_shan.json",
                NON_PORTABLE_OBJECT_CONTACT_STATE,
            )

    mappings = [
        _mapping(
            input_name = FIRST_VIDEO,
            staged_stem = FIRST_STAGED_STEM,
            session_sort_index = 1,
        ),
        _mapping(
            input_name = SECOND_VIDEO,
            staged_stem = SECOND_STAGED_STEM,
            session_sort_index = 2,
        ),
    ]

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = mappings,
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    segments = build_interaction_segments(frames, config = DEFAULT_VIDEO_PROCESSING_CONFIG)

    session_metrics = build_session_level_metrics(
        frames,
        segments,
        mappings = mappings,
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    assert len(segments) == 1
    assert segments[0].start_input_name == FIRST_VIDEO
    assert segments[0].end_input_name == SECOND_VIDEO
    assert segments[0].start_session_frame_index == 1
    assert segments[0].end_session_frame_index == 2 * ONE_SECOND_FRAME_COUNT
    assert segments[0].duration_seconds == 2.0
    assert session_metrics[0].dur_dominant_hand_seconds == 2.0
    assert session_metrics[0].num_dominant_hand_per_hour == 1800.0
    assert session_metrics[0].num_non_dominant_hand_per_hour == 0.0
    assert session_metrics[0].num_bilateral_per_hour == 1800.0

def test_missing_shan_predictions_produce_no_prediction_statuses(tmp_path: Path) -> None:
    video_csv = tmp_path / "video_level_metrics.csv"
    session_csv = tmp_path / "session_level_metrics.csv"
    manifest_path = tmp_path / "adl_input_manifest.csv"
    
    manifest_path.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time\n"
        "session001,1,part1.mp4,video001.MP4,video001,2026-07-05T10:01:00+00:00\n",
        encoding="utf-8",
    )
    
    subclip_manifest_path = tmp_path / "adl_subclip_manifest.csv"
    
    subclip_manifest_path.write_text(
        "session_id,input_name,staged_video_stem,subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,valid_duration_seconds,"
        "processing_fps,processing_subclip_duration_seconds\n",
        encoding = "utf-8",
    )

    write_bandini_metric_files(
        shan_outputs_dir = tmp_path / "missing_shan_outputs",
        input_manifest_path = manifest_path,
        subclip_manifest_path = subclip_manifest_path,
        frame_level_predictions_path = tmp_path / "frame_level_predictions.csv",
        interaction_segments_path = tmp_path / "interaction_segments.csv",
        video_level_metrics_path = video_csv,
        session_level_metrics_path = session_csv,
        video_level_metrics_summary_path = tmp_path / "video_level_metrics_summary.csv",
        metrics_config_path = tmp_path / "metrics_config.json",
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    assert _read_csv(video_csv)[0]["metric_status"] == "no_frame_predictions"
    assert _read_csv(session_csv)[0]["metric_status"] == "no_session_predictions"

def test_write_metric_files_preserves_manifest_session_and_original_names(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"

    for index in range(1, ONE_SECOND_FRAME_COUNT + 1):
        _write_shan_json(
            shan_dir / "video001.--1" / f"frame_{index:03d}_shan.json",
            NON_PORTABLE_OBJECT_CONTACT_STATE,
        )

    manifest_path = tmp_path / "adl_input_manifest.csv"
    
    manifest_path.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time\n"
        "session001,1,making-tea.mp4,video001.MP4,video001,2026-07-05T10:01:00\n",
        encoding = "utf-8",
    )
    
    subclip_manifest_path = tmp_path / "adl_subclip_manifest.csv"
    
    subclip_manifest_path.write_text(
        "session_id,input_name,staged_video_stem,subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,valid_duration_seconds,"
        "processing_fps,processing_subclip_duration_seconds\n"
        "session001,making-tea.mp4,video001,video001.--1,1,0,10,10,30,10\n",
        encoding = "utf-8",
    )

    frame_csv = tmp_path / "frame_level_predictions.csv"
    segment_csv = tmp_path / "interaction_segments.csv"
    video_csv = tmp_path / "video_level_metrics.csv"
    session_csv = tmp_path / "session_level_metrics.csv"
    summary_csv = tmp_path / "video_level_metrics_summary.csv"
    config_json = tmp_path / "metrics_config.json"

    write_bandini_metric_files(
        shan_outputs_dir = shan_dir,
        input_manifest_path = manifest_path,
        subclip_manifest_path = subclip_manifest_path,
        frame_level_predictions_path = frame_csv,
        interaction_segments_path = segment_csv,
        video_level_metrics_path = video_csv,
        session_level_metrics_path = session_csv,
        video_level_metrics_summary_path = summary_csv,
        metrics_config_path = config_json,
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    )

    assert _read_csv(frame_csv)[0]["session_id"] == SESSION_ID
    assert _read_csv(frame_csv)[0]["input_name"] == "making-tea.mp4"
    assert _read_csv(segment_csv)[0]["session_id"] == SESSION_ID
    assert _read_csv(video_csv)[0]["input_name"] == "making-tea.mp4"
    assert _read_csv(session_csv)[0]["session_id"] == SESSION_ID

    payload = json.loads(config_json.read_text(encoding = "utf-8"))
    
    assert payload["frame_fps"] == DEFAULT_FPS
    assert payload["pooling_window_frames"] == ONE_SECOND_FRAME_COUNT
    assert any(row["metric"] == "computed_video_count" for row in _read_csv(summary_csv))

def test_bandini_metrics_are_hand_specific_not_any_hand_union(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"

    frames_per_second = 4
    frame_duration_seconds = 1.0 / frames_per_second
    pooling_window_seconds = frame_duration_seconds

    total_frame_count = 4
    interacting_frame_count_per_hand = 2
    expected_recording_time_seconds = total_frame_count / frames_per_second
    
    expected_interaction_time_seconds = (
        interacting_frame_count_per_hand * frame_duration_seconds
    )
    
    expected_interaction_percentage = (
        expected_interaction_time_seconds / expected_recording_time_seconds * 100.0
    )
    
    expected_segments_per_hand = 1
    expected_hand_specific_segment_count = 2
    
    expected_interactions_per_hour_per_hand = (
        expected_segments_per_hand / (expected_recording_time_seconds / 3600.0)
    )

    config = VideoProcessingConfig(
        subclip_fps = frames_per_second,
        frame_fps = frames_per_second,
        pooling_window_seconds = pooling_window_seconds,
        dominant_hand = "right",
    )
    
    interacting_frames = range(1, interacting_frame_count_per_hand + 1)
    
    non_interacting_frames = range(interacting_frame_count_per_hand + 1,
        total_frame_count + 1,
    )

    for frame_index in interacting_frames:
        _write_shan_json_with_hands(
            shan_dir / "video001.--1" / f"frame_{frame_index:03d}_shan.json",
            [
                _shan_hand_prediction(
                    NON_PORTABLE_OBJECT_CONTACT_STATE,
                    side = RIGHT_HAND_VALUE,
                ),
                _shan_hand_prediction(
                    NON_PORTABLE_OBJECT_CONTACT_STATE,
                    side = LEFT_HAND_VALUE,
                ),
            ],
        )

    for frame_index in non_interacting_frames:
        _write_shan_json_with_hands(
            shan_dir / "video001.--1" / f"frame_{frame_index:03d}_shan.json",
            [
                _shan_hand_prediction(NO_CONTACT_STATE, side = RIGHT_HAND_VALUE),
                _shan_hand_prediction(NO_CONTACT_STATE, side = LEFT_HAND_VALUE),
            ],
        )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        config = config,
    )

    segments = build_interaction_segments(frames, config = config)

    metrics = build_session_level_metrics(
        frames,
        segments,
        mappings = [_mapping()],
        config = config,
    )

    assert len(frames) == total_frame_count
    assert len(segments) == expected_hand_specific_segment_count

    assert {segment.hand_role for segment in segments} == {
        "dominant",
        "non_dominant",
    }

    assert metrics[0].recording_time_seconds == expected_recording_time_seconds

    assert metrics[0].perc_dominant_hand == expected_interaction_percentage
    assert metrics[0].perc_non_dominant_hand == expected_interaction_percentage
    assert metrics[0].perc_bilateral == expected_interaction_percentage

    assert metrics[0].dur_dominant_hand_seconds == expected_interaction_time_seconds
    assert metrics[0].dur_non_dominant_hand_seconds == expected_interaction_time_seconds
    assert metrics[0].dur_bilateral_seconds == expected_interaction_time_seconds * 2

    assert (
        metrics[0].num_dominant_hand_per_hour
        == expected_interactions_per_hour_per_hand
    )
    
    assert (
        metrics[0].num_non_dominant_hand_per_hour
        == expected_interactions_per_hour_per_hand
    )
    
    assert metrics[0].num_bilateral_per_hour == (
        expected_interactions_per_hour_per_hand * 2
    )

def test_write_metric_files_loads_real_egoviz_dash_subclip_dirs(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"
    frame_fps = LOW_FPS
    subclip_length_seconds = 1
    subclip_count = 2

    for subclip_index in range(1, subclip_count + 1):
        for frame_index in range(frame_fps * subclip_length_seconds):
            _write_shan_json(
                shan_dir
                / f"video001--{subclip_index}"
                / f"frame_{frame_index}_shan.json",
                NON_PORTABLE_OBJECT_CONTACT_STATE,
            )

    manifest_path = tmp_path / "adl_input_manifest.csv"

    manifest_path.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time\n"
        "session001,1,cooking.MP4,video001.MP4,video001,"
        "2026-07-07T00:01:17+00:00\n",
        encoding="utf-8",
    )
    
    subclip_manifest_path = tmp_path / "adl_subclip_manifest.csv"
    
    subclip_manifest_path.write_text(
        "session_id,input_name,staged_video_stem,"
        "subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,"
        "valid_duration_seconds,"
        "processing_fps,"
        "processing_subclip_duration_seconds\n"
        f"session001,cooking.MP4,video001,"
        f"video001--1,1,0,1,1,{frame_fps},1\n"
        f"session001,cooking.MP4,video001,"
        f"video001--2,2,1,2,1,{frame_fps},1\n",
        encoding = "utf-8",
    )

    frame_csv = tmp_path / "frame_level_predictions.csv"
    segment_csv = tmp_path / "interaction_segments.csv"
    video_csv = tmp_path / "video_level_metrics.csv"
    session_csv = tmp_path / "session_level_metrics.csv"
    summary_csv = tmp_path / "video_level_metrics_summary.csv"
    config_json = tmp_path / "metrics_config.json"
    
    config = VideoProcessingConfig(
        subclip_length_seconds = subclip_length_seconds,
        subclip_fps = frame_fps,
        frame_fps = frame_fps,
        pooling_window_seconds = 1 / frame_fps,
    )

    write_bandini_metric_files(
        shan_outputs_dir = shan_dir,
        input_manifest_path = manifest_path,
        subclip_manifest_path = subclip_manifest_path,
        frame_level_predictions_path = frame_csv,
        interaction_segments_path = segment_csv,
        video_level_metrics_path = video_csv,
        session_level_metrics_path = session_csv,
        video_level_metrics_summary_path = summary_csv,
        metrics_config_path = config_json,
        config = config,
    )

    frame_rows = _read_csv(frame_csv)
    video_rows = _read_csv(video_csv)
    session_rows = _read_csv(session_csv)

    assert len(frame_rows) == frame_fps * subclip_length_seconds * subclip_count
    
    assert [row["video_frame_index"] for row in frame_rows] == [
        str(index) for index in range(1, 13)
    ]
    
    assert frame_rows[0]["frame_path"] == "video001--1/frame_0_shan.json"
    assert frame_rows[6]["frame_path"] == "video001--2/frame_0_shan.json"
    assert frame_rows[-1]["video_timestamp_seconds"] == "1.833333"

    assert video_rows[0]["metric_status"] == "computed"
    assert video_rows[0]["analyzed_frame_count"] == "12"
    assert video_rows[0]["recording_time_seconds"] == "2"

    assert session_rows[0]["metric_status"] == "computed"
    assert session_rows[0]["analyzed_frame_count"] == "12"
    assert session_rows[0]["recording_time_seconds"] == "2"

def test_padded_tail_frames_are_kept_but_excluded_from_metrics(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"
    processing_fps = 2
    processed_frame_count = 20
    valid_duration_seconds = 3.0

    for frame_index in range(processed_frame_count):
        _write_shan_json(
            shan_dir / "video001--1" / f"frame_{frame_index}_shan.json",
            NON_PORTABLE_OBJECT_CONTACT_STATE,
        )

    input_manifest_path = tmp_path / "adl_input_manifest.csv"
    
    input_manifest_path.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time,source_duration_seconds,"
        "source_fps,source_total_frames\n"
        "session001,1,carrot.mp4,video001.MP4,video001,"
        "2026-07-11T00:00:00+00:00,3.0,15.0,45\n",
        encoding = "utf-8",
    )

    subclip_manifest_path = tmp_path / "adl_subclip_manifest.csv"
    
    subclip_manifest_path.write_text(
        "session_id,input_name,staged_video_stem,"
        "subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,"
        "valid_duration_seconds,"
        "processing_fps,"
        "processing_subclip_duration_seconds\n"
        f"session001,carrot.mp4,video001,"
        f"video001--1,1,0,3,"
        f"{valid_duration_seconds},"
        f"{processing_fps},10\n",
        encoding = "utf-8",
    )

    frame_csv = tmp_path / "frame_level_predictions.csv"
    segment_csv = tmp_path / "interaction_segments.csv"
    video_csv = tmp_path / "video_level_metrics.csv"
    session_csv = tmp_path / "session_level_metrics.csv"

    config = VideoProcessingConfig(
        subclip_length_seconds = 10,
        subclip_fps = processing_fps,
        frame_fps = processing_fps,
        pooling_window_seconds = 0.5,
    )

    write_bandini_metric_files(
        shan_outputs_dir = shan_dir,
        input_manifest_path = input_manifest_path,
        subclip_manifest_path = subclip_manifest_path,
        frame_level_predictions_path = frame_csv,
        interaction_segments_path = segment_csv,
        video_level_metrics_path = video_csv,
        session_level_metrics_path = session_csv,
        video_level_metrics_summary_path = tmp_path / "summary.csv",
        metrics_config_path = tmp_path / "metrics_config.json",
        config = config,
    )

    frame_rows = _read_csv(frame_csv)
    video_row = _read_csv(video_csv)[0]
    session_row = _read_csv(session_csv)[0]
    segment_row = _read_csv(segment_csv)[0]

    assert len(frame_rows) == processed_frame_count

    assert [row["is_valid_source_frame"] for row in frame_rows[:6]] == (
        ["true"] * 6
    )

    assert [row["is_valid_source_frame"] for row in frame_rows[6:]] == (
        ["false"] * 14
    )

    assert frame_rows[5]["subclip_timestamp_seconds"] == "2.5"
    assert frame_rows[6]["subclip_timestamp_seconds"] == "3"
    assert frame_rows[6]["source_timestamp_seconds"] == "3"
    assert frame_rows[6]["valid_source_duration_seconds"] == "3"

    assert video_row["analyzed_frame_count"] == "6"
    assert video_row["recording_time_seconds"] == "3"

    assert session_row["analyzed_frame_count"] == "6"
    assert session_row["recording_time_seconds"] == "3"
    assert session_row["perc_dominant_hand"] == "100"
    assert session_row["dur_dominant_hand_seconds"] == "3"
    assert session_row["num_dominant_hand_per_hour"] == "1200"

    assert segment_row["duration_seconds"] == "3"
    
def test_statepool_ignores_invalid_padded_tail_frames(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"

    config = VideoProcessingConfig(
        subclip_length_seconds = 1,
        subclip_fps = 4,
        frame_fps = 4,
        pooling_window_seconds = 1.0,
    )

    for frame_index, contact_state in enumerate(
        [
            NON_PORTABLE_OBJECT_CONTACT_STATE,
            NON_PORTABLE_OBJECT_CONTACT_STATE,
            NO_CONTACT_STATE,
            NO_CONTACT_STATE,
        ]
    ):
        _write_shan_json(
            shan_dir / "video001--1" / f"frame_{frame_index}_shan.json",
            contact_state,
        )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
        subclip_mappings = [
            _subclip_mapping(
                valid_duration_seconds = 0.5,
                processing_fps = 4,
                processing_subclip_duration_seconds = 1,
            )
        ],
        config = config,
    )

    assert [frame.is_valid_source_frame for frame in frames] == [
        True,
        True,
        False,
        False,
    ]

    assert [frame.right_interaction for frame in frames] == [
        1,
        1,
        0,
        0,
    ]

def test_multi_file_session_concatenates_only_valid_source_frames(
    tmp_path: Path,
) -> None:
    shan_dir = tmp_path / "shan_outputs"

    config = VideoProcessingConfig(
        subclip_length_seconds = 10,
        subclip_fps = 1,
        frame_fps = 1,
        pooling_window_seconds = 1.0,
    )

    mappings = [
        _mapping(
            input_name = "part1.mp4",
            staged_stem = "video001",
            session_sort_index = 1,
        ),
        _mapping(
            input_name = "part2.mp4",
            staged_stem = "video002",
            session_sort_index = 2,
        ),
        _mapping(
            input_name = "part3.mp4",
            staged_stem = "video003",
            session_sort_index = 3,
        ),
    ]

    valid_durations = [3.0, 3.0, 2.0]
    subclip_mappings: list[SubclipTimingMapping] = []

    for mapping, valid_duration in zip(
        mappings,
        valid_durations,
        strict = True,
    ):
        for frame_index in range(10):
            _write_shan_json(
                shan_dir
                / f"{mapping.staged_video_stem}--1"
                / f"frame_{frame_index}_shan.json",
                NON_PORTABLE_OBJECT_CONTACT_STATE,
            )

        subclip_mappings.append(
            _subclip_mapping(
                input_name = mapping.input_name,
                staged_stem = mapping.staged_video_stem,
                valid_duration_seconds = valid_duration,
                processing_fps = 1,
            )
        )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = mappings,
        subclip_mappings = subclip_mappings,
        config = config,
    )

    segments = build_interaction_segments(
        frames,
        config = config,
    )

    video_metrics = build_video_level_metrics(
        frames,
        mappings = mappings,
        config = config,
    )

    session_metrics = build_session_level_metrics(
        frames,
        segments,
        mappings = mappings,
        config = config,
    )

    assert len(frames) == 30
    assert sum(frame.is_valid_source_frame for frame in frames) == 8

    assert [
        metric.recording_time_seconds
        for metric in video_metrics
    ] == [3.0, 3.0, 2.0]

    assert len(segments) == 1
    assert segments[0].start_input_name == "part1.mp4"
    assert segments[0].end_input_name == "part3.mp4"
    assert segments[0].start_session_time_seconds == 0.0
    assert segments[0].end_session_time_seconds == 8.0
    assert segments[0].duration_seconds == 8.0

    assert session_metrics[0].analyzed_frame_count == 8
    assert session_metrics[0].recording_time_seconds == 8.0
    assert session_metrics[0].dur_dominant_hand_seconds == 8.0
    assert session_metrics[0].num_dominant_hand_per_hour == 450.0

def test_defensive_config_manifest_and_subclip_validation_branches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match = "dominant_hand"):
        VideoProcessingConfig(
            dominant_hand = "ambidextrous",  # type: ignore[arg-type]
        )

    non_mapping_config = tmp_path / "non-mapping-config.json"
    
    non_mapping_config.write_text("[]", encoding = "utf-8")

    assert read_video_processing_config(non_mapping_config) == (
        DEFAULT_VIDEO_PROCESSING_CONFIG
    )

    empty_manifest = tmp_path / "empty-input-manifest.csv"

    empty_manifest.write_text(
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time\n",
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "does not contain any valid input mappings"):
        load_input_video_mappings(input_manifest_path = empty_manifest)

    missing_subclip_manifest = tmp_path / "missing-subclip-manifest.csv"

    with pytest.raises(
        FileNotFoundError,
        match = "Subclip manifest is required for Bandini metric computation",
    ):
        load_subclip_timing_mappings(subclip_manifest_path = missing_subclip_manifest)

    invalid_subclip_manifest = tmp_path / "invalid-subclip-manifest.csv"

    invalid_subclip_manifest.write_text(
        "session_id,input_name,staged_video_stem,subclip_name,subclip_index,"
        "source_start_seconds,source_end_seconds,valid_duration_seconds,"
        "processing_fps,processing_subclip_duration_seconds\n"
        "session001,part1.mp4,video001,video001--1,1,0,1,1,0,10\n",
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "non-positive processing_fps"):
        load_subclip_timing_mappings(subclip_manifest_path = invalid_subclip_manifest)

    shan_dir = tmp_path / "shan_outputs"

    _write_shan_json(
        shan_dir
        / "video001--1"
        / "frame_0_shan.json",
        NON_PORTABLE_OBJECT_CONTACT_STATE,
    )

    with pytest.raises(ValueError, match = "does not contain timing metadata"):
        load_frame_interaction_predictions(
            shan_outputs_dir = shan_dir,
            mappings = [_mapping()],
            subclip_mappings = [],
        )

def test_private_parsing_and_segment_defensive_branches(tmp_path: Path) -> None:
    assert bandini_metrics._subclip_sort_key("video001") == ("video001", 0)

    assert bandini_metrics._frame_sort_key(Path("frame_shan.json")) == ("frame_shan", 0)

    non_mapping_prediction = tmp_path / "prediction.json"

    non_mapping_prediction.write_text("[]", encoding = "utf-8")

    assert bandini_metrics._load_shan_prediction_json(non_mapping_prediction) == {}

    assert bandini_metrics._detected_hands_from_payload({"hands": "invalid"}) == []

    assert bandini_metrics._is_shan_hand_prediction({"contact_state": 3}) is True

    assert bandini_metrics._contact_state({"contactstate": 4}) == 4
    assert bandini_metrics._contact_state("invalid") == 0

    assert bandini_metrics._hand_label({"hand_side": LEFT_HAND_VALUE}) == "left"
    assert bandini_metrics._hand_label({"hand_side": RIGHT_HAND_VALUE}) == "right"

    assert bandini_metrics._hand_label({"hand_side": 99}) is None
    assert bandini_metrics._hand_label([0] * 9 + [99]) is None
    assert bandini_metrics._as_hand_label("unknown", "right") == "right"

    assert bandini_metrics._segments_from_frames(
        [],
        config = DEFAULT_VIDEO_PROCESSING_CONFIG,
    ) == []

    shan_dir = tmp_path / "shan_outputs"

    _write_shan_json(
        shan_dir
        / "video001--1"
        / "frame_0_shan.json",
        NON_PORTABLE_OBJECT_CONTACT_STATE,
    )

    frames = load_frame_interaction_predictions(
        shan_outputs_dir = shan_dir,
        mappings = [_mapping()],
    )

    mixed_session_frames = [
        frames[0],
        bandini_metrics.replace(
            frames[0],
            session_id = "session002",
        ),
    ]

    with pytest.raises(ValueError, match = "single session"):
        bandini_metrics._segments_from_frames(
            mixed_session_frames,
            config = DEFAULT_VIDEO_PROCESSING_CONFIG,
        )
