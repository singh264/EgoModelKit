""" Configurable Bandini-style hand-use metrics for ADL post-processing. """

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

DEFAULT_SESSION_ID: Final[str] = "session001"
DEFAULT_SUBCLIP_LENGTH_SECONDS: Final[int] = 10
DEFAULT_SUBCLIP_FPS: Final[int] = 30
DEFAULT_FRAME_FPS: Final[int] = 30
DEFAULT_RESIZE_WIDTH: Final[int] = 720
DEFAULT_RESIZE_HEIGHT: Final[int] = 405
DEFAULT_POOLING_WINDOW_SECONDS: Final[float] = 1.0
DEFAULT_INTERACTION_CONTACT_STATE_THRESHOLD: Final[int] = 3

SHAN_HAND_CONTACT_STATE_INDEX: Final[int] = 5
SHAN_HAND_SIDE_INDEX: Final[int] = 9

LEFT_HAND_VALUE: Final[int] = 0
RIGHT_HAND_VALUE: Final[int] = 1

LEFT_HAND_LABEL: Final[str] = "left"
RIGHT_HAND_LABEL: Final[str] = "right"
DOMINANT_HAND_ROLE: Final[str] = "dominant"
NON_DOMINANT_HAND_ROLE: Final[str] = "non_dominant"

DEFAULT_DOMINANT_HAND: Final[str] = RIGHT_HAND_LABEL

COMPUTED_STATUS: Final[str] = "computed"
NO_FRAME_PREDICTIONS_STATUS: Final[str] = "no_frame_predictions"
NO_SESSION_PREDICTIONS_STATUS: Final[str] = "no_session_predictions"

HandLabel = Literal["left", "right"]
HandRole = Literal["dominant", "non_dominant"]


@dataclass(frozen=True, slots=True)
class VideoProcessingConfig:
    """ Shared video-processing and metric assumptions for one ADL run. """
    subclip_length_seconds: int = DEFAULT_SUBCLIP_LENGTH_SECONDS
    subclip_fps: int = DEFAULT_SUBCLIP_FPS
    frame_fps: int = DEFAULT_FRAME_FPS
    resize_width: int = DEFAULT_RESIZE_WIDTH
    resize_height: int = DEFAULT_RESIZE_HEIGHT
    pooling_window_seconds: float = DEFAULT_POOLING_WINDOW_SECONDS
    interaction_contact_state_threshold: int = DEFAULT_INTERACTION_CONTACT_STATE_THRESHOLD
    dominant_hand: HandLabel = DEFAULT_DOMINANT_HAND

    def __post_init__(self) -> None:
        if self.subclip_length_seconds <= 0:
            raise ValueError("subclip_length_seconds must be greater than zero.")

        if self.subclip_fps <= 0:
            raise ValueError("subclip_fps must be greater than zero.")

        if self.frame_fps <= 0:
            raise ValueError("frame_fps must be greater than zero.")

        if self.frame_fps > self.subclip_fps:
            raise ValueError("frame_fps must not be greater than subclip_fps.")

        if self.resize_width <= 0:
            raise ValueError("resize_width must be greater than zero.")

        if self.resize_height <= 0:
            raise ValueError("resize_height must be greater than zero.")

        if self.pooling_window_seconds <= 0:
            raise ValueError("pooling_window_seconds must be greater than zero.")
        
        if self.dominant_hand not in {LEFT_HAND_LABEL, RIGHT_HAND_LABEL}:
            raise ValueError("dominant_hand must be 'left' or 'right'.")

    @property
    def pooling_window_frames(self) -> int:
        """ Derived Statepool window length in analyzed frames. """
        return max(1, round(self.frame_fps * self.pooling_window_seconds))

    def to_dict(self) -> dict[str, int | float | str]:
        """ Return a JSON-serializable config including derived values. """
        return {
            "subclip_length_seconds": self.subclip_length_seconds,
            "subclip_fps": self.subclip_fps,
            "frame_fps": self.frame_fps,
            "resize_width": self.resize_width,
            "resize_height": self.resize_height,
            "pooling_window_seconds": self.pooling_window_seconds,
            "pooling_window_frames": self.pooling_window_frames,
            "interaction_contact_state_threshold": self.interaction_contact_state_threshold,
            "dominant_hand": self.dominant_hand,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> VideoProcessingConfig:
        """ Build a config from a partially populated JSON payload. """
        return cls(
            subclip_length_seconds = _as_int(
                payload.get("subclip_length_seconds"),
                DEFAULT_SUBCLIP_LENGTH_SECONDS,
            ),
            subclip_fps = _as_int(payload.get("subclip_fps"), DEFAULT_SUBCLIP_FPS),
            frame_fps = _as_int(payload.get("frame_fps"), DEFAULT_FRAME_FPS),
            resize_width = _as_int(payload.get("resize_width"), DEFAULT_RESIZE_WIDTH),
            resize_height = _as_int(payload.get("resize_height"), DEFAULT_RESIZE_HEIGHT),
            pooling_window_seconds = _as_float(
                payload.get("pooling_window_seconds"),
                DEFAULT_POOLING_WINDOW_SECONDS,
            ),
            interaction_contact_state_threshold = _as_int(
                payload.get("interaction_contact_state_threshold"),
                DEFAULT_INTERACTION_CONTACT_STATE_THRESHOLD,
            ),
            dominant_hand = _as_hand_label(
                payload.get("dominant_hand"),
                DEFAULT_DOMINANT_HAND,
            ),
        )

DEFAULT_VIDEO_PROCESSING_CONFIG: Final[VideoProcessingConfig] = VideoProcessingConfig()

@dataclass(frozen = True, slots = True)
class InputVideoMapping:
    """ Mapping from EgoVizML staged names back to original user inputs. """
    session_id: str
    session_sort_index: int
    input_name: str
    staged_video_name: str
    staged_video_stem: str
    input_modified_time: str = ""
    source_duration_seconds: float = 0.0
    source_fps: float = 0.0
    source_total_frames: int = 0

@dataclass(frozen = True, slots = True)
class SubclipTimingMapping:
    """ Source-time validity metadata for one processed EgoVizML subclip. """
    session_id: str
    input_name: str
    staged_video_stem: str
    subclip_name: str
    subclip_index: int
    source_start_seconds: float
    source_end_seconds: float
    valid_duration_seconds: float
    processing_fps: float
    processing_subclip_duration_seconds: float

@dataclass(frozen = True, slots = True)
class FrameInteractionPrediction:
    """ One Shan-derived frame-level hand-use prediction. """
    session_id: str
    input_name: str
    staged_video_stem: str
    subclip_name: str
    frame_path: str
    subclip_timestamp_seconds: float
    source_timestamp_seconds: float
    valid_source_duration_seconds: float
    is_valid_source_frame: bool
    video_frame_index: int
    session_frame_index: int
    video_timestamp_seconds: float
    session_timestamp_seconds: float
    detected_hand_count: int
    left_detected_hand_count: int
    right_detected_hand_count: int
    max_contact_state: int
    max_left_contact_state: int
    max_right_contact_state: int
    raw_left_interaction: int
    raw_right_interaction: int
    raw_dominant_interaction: int
    raw_non_dominant_interaction: int
    raw_any_interaction: int
    left_interaction: int
    right_interaction: int
    dominant_interaction: int
    non_dominant_interaction: int
    any_interaction: int

@dataclass(frozen = True, slots = True)
class InteractionSegment:
    """ One continuous pooled hand-specific interaction segment. """
    session_id: str
    hand_label: HandLabel
    hand_role: HandRole
    segment_index: int
    start_session_time_seconds: float
    end_session_time_seconds: float
    duration_seconds: float
    start_input_name: str
    end_input_name: str
    start_session_frame_index: int
    end_session_frame_index: int

@dataclass(frozen = True, slots = True)
class VideoLevelMetric:
    """ One row of per-input-video Bandini hand-use metrics. """
    session_id: str
    input_name: str
    staged_video_stem: str
    metric_status: str
    dominant_hand: HandLabel
    non_dominant_hand: HandLabel
    analyzed_frame_count: int
    recording_time_seconds: float
    perc_dominant_hand: float
    dur_dominant_hand_seconds: float
    num_dominant_hand_per_hour: float
    perc_non_dominant_hand: float
    dur_non_dominant_hand_seconds: float
    num_non_dominant_hand_per_hour: float
    perc_bilateral: float
    dur_bilateral_seconds: float
    num_bilateral_per_hour: float
    notes: str

@dataclass(frozen = True, slots = True)
class SessionLevelMetric:
    """ One row of session-level Bandini hand-use metrics. """
    session_id: str
    metric_status: str
    dominant_hand: HandLabel
    non_dominant_hand: HandLabel
    input_video_count: int
    analyzed_frame_count: int
    recording_time_seconds: float
    perc_dominant_hand: float
    dur_dominant_hand_seconds: float
    num_dominant_hand_per_hour: float
    perc_non_dominant_hand: float
    dur_non_dominant_hand_seconds: float
    num_non_dominant_hand_per_hour: float
    perc_bilateral: float
    dur_bilateral_seconds: float
    num_bilateral_per_hour: float
    notes: str

@dataclass(frozen = True, slots = True)
class _HandMetricValues:
    perc: float
    dur_seconds: float
    num_per_hour: float

def _hand_metric_values(
    *,
    segments: list[InteractionSegment],
    recording_time_seconds: float,
) -> _HandMetricValues:
    interaction_time_seconds = sum(segment.duration_seconds for segment in segments)
    interaction_count = len(segments)

    return _HandMetricValues(
        perc = _safe_divide(interaction_time_seconds, recording_time_seconds) * 100.0,
        dur_seconds = _safe_divide(interaction_time_seconds, interaction_count),
        num_per_hour = _safe_divide(interaction_count, recording_time_seconds / 3600.0),
    )

def _metric_values_by_role(
    *,
    segments: list[InteractionSegment],
    recording_time_seconds: float,
) -> dict[HandRole, _HandMetricValues]:
    return {
        DOMINANT_HAND_ROLE: _hand_metric_values(
            segments = [
                segment for segment in segments
                if segment.hand_role == DOMINANT_HAND_ROLE
            ],
            recording_time_seconds = recording_time_seconds,
        ),
        NON_DOMINANT_HAND_ROLE: _hand_metric_values(
            segments = [
                segment for segment in segments
                if segment.hand_role == NON_DOMINANT_HAND_ROLE
            ],
            recording_time_seconds = recording_time_seconds,
        ),
    }

def _bilateral_perc(
    dominant: _HandMetricValues,
    non_dominant: _HandMetricValues,
) -> float:
    return (dominant.perc + non_dominant.perc) / 2.0

def _bilateral_dur(
    dominant: _HandMetricValues,
    non_dominant: _HandMetricValues,
) -> float:
    return dominant.dur_seconds + non_dominant.dur_seconds

def _bilateral_num(
    dominant: _HandMetricValues,
    non_dominant: _HandMetricValues,
) -> float:
    return dominant.num_per_hour + non_dominant.num_per_hour

def write_video_processing_config(path: Path, config: VideoProcessingConfig) -> None:
    """ Write the actual video-processing config used for a run. """
    path.parent.mkdir(parents = True, exist_ok = True)
    
    path.write_text(
        json.dumps(config.to_dict(), indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8",
    )

def read_video_processing_config(path: Path) -> VideoProcessingConfig:
    """ Read a run config, falling back to defaults when it is missing. """
    if not path.is_file():
        return DEFAULT_VIDEO_PROCESSING_CONFIG

    payload = json.loads(path.read_text(encoding = "utf-8"))

    if not isinstance(payload, dict):
        return DEFAULT_VIDEO_PROCESSING_CONFIG

    return VideoProcessingConfig.from_dict(payload)

def write_bandini_metric_files(
    *,
    shan_outputs_dir: Path,
    input_manifest_path: Path,
    subclip_manifest_path: Path,
    frame_level_predictions_path: Path,
    interaction_segments_path: Path,
    video_level_metrics_path: Path,
    session_level_metrics_path: Path,
    video_level_metrics_summary_path: Path,
    metrics_config_path: Path,
    config: VideoProcessingConfig = DEFAULT_VIDEO_PROCESSING_CONFIG,
    diagnostic_log: Callable[[str], None] | None = None,
) -> None:
    """ Write frame, segment, video, session, summary, and config files. """
    mappings = load_input_video_mappings(input_manifest_path = input_manifest_path)

    subclip_mappings = load_subclip_timing_mappings(
        subclip_manifest_path = subclip_manifest_path,
    )

    frame_predictions = load_frame_interaction_predictions(
        shan_outputs_dir = shan_outputs_dir,
        mappings = mappings,
        subclip_mappings = subclip_mappings,
        config = config,
        diagnostic_log = diagnostic_log,
    )
    
    segments = build_interaction_segments(frame_predictions, config = config)
    
    video_metrics = build_video_level_metrics(
        frame_predictions,
        mappings = mappings,
        config = config,
    )
    
    session_metrics = build_session_level_metrics(
        frame_predictions,
        segments,
        mappings = mappings,
        config = config,
    )

    _write_frame_level_predictions(frame_level_predictions_path, frame_predictions)
    _write_interaction_segments(interaction_segments_path, segments)
    _write_video_level_metrics(video_level_metrics_path, video_metrics)
    _write_session_level_metrics(session_level_metrics_path, session_metrics)
    
    _write_video_level_metrics_summary(
        video_level_metrics_summary_path,
        video_metrics,
        session_metrics,
    )
    
    write_video_processing_config(metrics_config_path, config)

def load_input_video_mappings(
    *,
    input_manifest_path: Path,
) -> list[InputVideoMapping]:
    """ Load staged-name mappings from the required ADL input manifest. """
    if not input_manifest_path.is_file():
        raise FileNotFoundError(
            "ADL input manifest is required for Bandini metric computation: "
            f"{input_manifest_path}"
        )

    with input_manifest_path.open("r", encoding= "utf-8", newline= "") as csv_file:
        rows = list(csv.DictReader(csv_file))

    mappings = [
        InputVideoMapping(
            session_id = row.get("session_id", "") or DEFAULT_SESSION_ID,
            session_sort_index = _as_int(row.get("session_sort_index"), index),
            input_name = row.get("input_name", ""),
            staged_video_name = row.get("staged_video_name", ""),
            staged_video_stem = (
                row.get("staged_video_stem", "")
                or Path(row.get("staged_video_name", "")).stem
            ),
            input_modified_time = row.get("input_modified_time", ""),
            source_duration_seconds = _as_float(row.get("source_duration_seconds"), 0.0),
            source_fps = _as_float(row.get("source_fps"), 0.0),
            source_total_frames = _as_int(row.get("source_total_frames"), 0),
        )
        for index, row in enumerate(rows, start=1)
        if row.get("input_name")
    ]

    if not mappings:
        raise ValueError(
            "ADL input manifest does not contain any valid input mappings: "
            f"{input_manifest_path}"
        )

    return sorted(
        mappings,
        key = lambda mapping: (mapping.session_id, mapping.session_sort_index),
    )

def load_subclip_timing_mappings(
    *,
    subclip_manifest_path: Path,
) -> list[SubclipTimingMapping]:
    """ Load source-time validity metadata for processed EgoVizML subclips. """
    if not subclip_manifest_path.is_file():
        raise FileNotFoundError(
            "ADL subclip manifest is required for Bandini metric computation: "
            f"{subclip_manifest_path}"
        )

    with subclip_manifest_path.open(
        "r",
        encoding = "utf-8",
        newline = "",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    mappings = [
        SubclipTimingMapping(
            session_id = row.get("session_id", "") or DEFAULT_SESSION_ID,
            input_name = row.get("input_name", ""),
            staged_video_stem = row.get( "staged_video_stem", ""),
            subclip_name = row.get("subclip_name", ""),
            subclip_index = _as_int(row.get("subclip_index"), 0),
            source_start_seconds = _as_float(row.get("source_start_seconds"), 0.0),
            source_end_seconds = _as_float(row.get("source_end_seconds"), 0.0),
            valid_duration_seconds = _as_float(row.get("valid_duration_seconds"), 0.0),
            processing_fps = _as_float(row.get("processing_fps"), 0.0),
            processing_subclip_duration_seconds = _as_float(
                row.get("processing_subclip_duration_seconds"),
                0.0,
            ),
        )
        for row in rows
        if row.get("subclip_name")
    ]

    invalid_processing_fps = [
        mapping.subclip_name
        for mapping in mappings
        if mapping.processing_fps <= 0
    ]

    if invalid_processing_fps:
        raise ValueError(
            "ADL subclip manifest contains non-positive "
            "processing_fps for: "
            + ", ".join(invalid_processing_fps[:5])
        )

    return sorted(
        mappings,
        key = lambda mapping: (
            mapping.session_id,
            mapping.staged_video_stem,
            mapping.subclip_index,
        ),
    )

def load_frame_interaction_predictions(
    *,
    shan_outputs_dir: Path,
    mappings: list[InputVideoMapping],
    subclip_mappings: list[SubclipTimingMapping] | None = None,
    config: VideoProcessingConfig = DEFAULT_VIDEO_PROCESSING_CONFIG,
    diagnostic_log: Callable[[str], None] | None = None,
) -> list[FrameInteractionPrediction]:
    """ Convert Shan contact states into raw and pooled interaction predictions. """
    if not shan_outputs_dir.is_dir():
        _write_diagnostic(
            diagnostic_log,
            f"Shan directory does not exist: {shan_outputs_dir}",
        )

        return []

    raw_predictions: list[FrameInteractionPrediction] = []
    video_frame_offsets: dict[str, int] = {}
    session_frame_offsets: dict[str, int] = {}
    paths_by_clip = _prediction_paths_by_clip(shan_outputs_dir)
    require_subclip_mapping = subclip_mappings is not None

    clip_counts = {
        name: len(paths)
        for name, paths in paths_by_clip.items()
    }

    staged_stems = [
        mapping.staged_video_stem
        for mapping in mappings
    ]

    subclip_names = [
        mapping.subclip_name
        for mapping in (subclip_mappings or [])
    ]

    _write_diagnostic(
        diagnostic_log,
        f"discovered Shan clip groups={clip_counts}",
    )

    _write_diagnostic(
        diagnostic_log,
        f"input manifest staged stems={staged_stems}",
    )

    _write_diagnostic(
        diagnostic_log,
        f"subclip manifest names={subclip_names}",
    )

    subclip_timings = {
        (
            mapping.session_id,
            mapping.staged_video_stem,
            mapping.subclip_name,
        ): mapping
        for mapping in (subclip_mappings or [])
    }

    sorted_mappings = sorted(
        mappings,
        key = lambda item: (item.session_id, item.session_sort_index),
    )

    for mapping in sorted_mappings:
        clip_groups = [
            (clip_name, prediction_paths)
            for clip_name, prediction_paths in paths_by_clip.items()
            if _clip_belongs_to_mapping(clip_name, mapping)
        ]
        
        clip_groups.sort(key = lambda item: _subclip_sort_key(item[0]))
        
        matched_clip_names = [
            clip_name
            for clip_name, _prediction_paths in clip_groups
        ]

        _write_diagnostic(
            diagnostic_log,
            (
                f"staged stem {mapping.staged_video_stem!r} "
                f"matched Shan folders={matched_clip_names}"
            ),
        )

        for clip_name, prediction_paths in clip_groups:
            subclip_timing = subclip_timings.get(
                (
                    mapping.session_id,
                    mapping.staged_video_stem,
                    clip_name,
                )
            )

            if require_subclip_mapping and subclip_timing is None:
                raise ValueError(
                    "ADL subclip manifest does not contain timing metadata for "
                    f"Shan output folder: {clip_name}"
                )

            subclip_index = _subclip_sort_key(clip_name)[1] or 1
            source_start_seconds = (subclip_index - 1) * config.subclip_length_seconds
            valid_duration_seconds = float(config.subclip_length_seconds)
            processing_fps = float(config.frame_fps)

            if subclip_timing is not None:
                subclip_index = subclip_timing.subclip_index
                source_start_seconds = subclip_timing.source_start_seconds
                valid_duration_seconds = subclip_timing.valid_duration_seconds
                processing_fps = subclip_timing.processing_fps

            for local_frame_index, prediction_path in enumerate(prediction_paths):
                subclip_timestamp_seconds = local_frame_index / processing_fps
                source_timestamp_seconds = source_start_seconds + subclip_timestamp_seconds
                is_valid_source_frame = subclip_timestamp_seconds < valid_duration_seconds
                
                video_frame_index = video_frame_offsets.get(mapping.staged_video_stem, 0) + 1
                session_frame_index = session_frame_offsets.get(mapping.session_id, 0) + 1
                video_frame_offsets[mapping.staged_video_stem] = video_frame_index
                session_frame_offsets[mapping.session_id] = session_frame_index

                payload = _load_shan_prediction_json(prediction_path)
                hands = _detected_hands_from_payload(payload)
                
                left_hands = [
                    hand for hand in hands
                    if _hand_label(hand) == LEFT_HAND_LABEL
                ]
                
                right_hands = [
                    hand for hand in hands
                    if _hand_label(hand) == RIGHT_HAND_LABEL
                ]

                contact_states = [_contact_state(hand) for hand in hands]
                left_contact_states = [_contact_state(hand) for hand in left_hands]
                right_contact_states = [_contact_state(hand) for hand in right_hands]

                max_contact_state = max(contact_states, default = 0)
                max_left_contact_state = max(left_contact_states, default = 0)
                max_right_contact_state = max(right_contact_states, default = 0)

                raw_left_interaction = int(
                    max_left_contact_state >= config.interaction_contact_state_threshold
                )
                
                raw_right_interaction = int(
                    max_right_contact_state >= config.interaction_contact_state_threshold
                )

                raw_dominant_interaction = _interaction_for_hand(
                    left_value = raw_left_interaction,
                    right_value = raw_right_interaction,
                    hand = config.dominant_hand,
                )
                
                raw_non_dominant_interaction = _interaction_for_hand(
                    left_value = raw_left_interaction,
                    right_value = raw_right_interaction,
                    hand = _non_dominant_hand(config.dominant_hand),
                )

                raw_any_interaction = int(
                    raw_left_interaction or raw_right_interaction
                )

                raw_predictions.append(
                    FrameInteractionPrediction(
                        session_id = mapping.session_id,
                        input_name = mapping.input_name,
                        staged_video_stem = mapping.staged_video_stem,
                        subclip_name = clip_name,
                        frame_path = str(prediction_path.relative_to(shan_outputs_dir)),
                        subclip_timestamp_seconds = subclip_timestamp_seconds,
                        source_timestamp_seconds = source_timestamp_seconds,
                        valid_source_duration_seconds = valid_duration_seconds,
                        is_valid_source_frame = is_valid_source_frame,
                        video_frame_index = video_frame_index,
                        session_frame_index = session_frame_index,
                        video_timestamp_seconds = (video_frame_index - 1) / config.frame_fps,
                        session_timestamp_seconds= (session_frame_index - 1) / config.frame_fps,
                        detected_hand_count = len(hands),
                        left_detected_hand_count = len(left_hands),
                        right_detected_hand_count = len(right_hands),
                        max_contact_state = max_contact_state,
                        max_left_contact_state = max_left_contact_state,
                        max_right_contact_state = max_right_contact_state,
                        raw_left_interaction = raw_left_interaction,
                        raw_right_interaction = raw_right_interaction,
                        raw_dominant_interaction = raw_dominant_interaction,
                        raw_non_dominant_interaction = raw_non_dominant_interaction,
                        raw_any_interaction = raw_any_interaction,
                        left_interaction = raw_left_interaction,
                        right_interaction = raw_right_interaction,
                        dominant_interaction = raw_dominant_interaction,
                        non_dominant_interaction = raw_non_dominant_interaction,
                        any_interaction = raw_any_interaction,
                    )
                )

    pooled_predictions = apply_statepool(
        raw_predictions,
        config = config,
    )

    _write_diagnostic(
        diagnostic_log,
        f"frame predictions produced={len(pooled_predictions)}",
    )

    return pooled_predictions

def apply_statepool(
    frame_predictions: list[FrameInteractionPrediction],
    *,
    config: VideoProcessingConfig,
) -> list[FrameInteractionPrediction]:
    """ Apply Statepool only across source-valid session frames. """
    pooled_by_key: dict[tuple[str, str], FrameInteractionPrediction] = {}
    window_size = config.pooling_window_frames

    for _session_id, session_frames in _group_frames_by_session(frame_predictions).items():
        valid_frames = sorted(
            [
                frame
                for frame in session_frames
                if frame.is_valid_source_frame
            ],
            key = lambda item: item.session_frame_index,
        )

        for start in range(0, len(valid_frames), window_size):
            window = valid_frames[start : start + window_size]

            left_pooled = int(
                sum(frame.raw_left_interaction for frame in window) > len(window) / 2
            )

            right_pooled = int(
                sum(frame.raw_right_interaction for frame in window) > len(window) / 2
            )

            dominant_pooled = _interaction_for_hand(
                left_value = left_pooled,
                right_value = right_pooled,
                hand = config.dominant_hand,
            )

            non_dominant_pooled = _interaction_for_hand(
                left_value = left_pooled,
                right_value = right_pooled,
                hand = _non_dominant_hand(config.dominant_hand),
            )

            any_pooled = int(left_pooled or right_pooled)

            for frame in window:
                pooled_by_key[
                    (
                        frame.session_id,
                        frame.frame_path,
                    )
                ] = replace(
                    frame,
                    left_interaction = left_pooled,
                    right_interaction = right_pooled,
                    dominant_interaction = dominant_pooled,
                    non_dominant_interaction = non_dominant_pooled,
                    any_interaction = any_pooled,
                )

    pooled_frames: list[FrameInteractionPrediction] = []

    for frame in frame_predictions:
        key = (frame.session_id, frame.frame_path,)

        if key in pooled_by_key:
            pooled_frames.append(pooled_by_key[key])

            continue

        pooled_frames.append(
            replace(
                frame,
                left_interaction = 0,
                right_interaction = 0,
                dominant_interaction = 0,
                non_dominant_interaction = 0,
                any_interaction = 0,
            )
        )

    return sorted(
        pooled_frames,
        key = lambda frame: (frame.session_id, frame.session_frame_index),
    )

def build_interaction_segments(
    frame_predictions: list[FrameInteractionPrediction],
    *,
    config: VideoProcessingConfig = DEFAULT_VIDEO_PROCESSING_CONFIG,
) -> list[InteractionSegment]:
    """ Build continuous hand-specific interaction segments from pooled frames. """
    segments: list[InteractionSegment] = []

    for session_id, session_frames in _group_frames_by_session(frame_predictions).items():
        sorted_frames = sorted(
            [frame for frame in session_frames if frame.is_valid_source_frame],
            key = lambda item: item.session_frame_index,
        )

        for hand_label in [LEFT_HAND_LABEL, RIGHT_HAND_LABEL]:
            hand_role = _hand_role(
                hand_label = hand_label,
                dominant_hand = config.dominant_hand,
            )
            
            segments.extend(
                _segments_from_frames_for_hand(
                    sorted_frames,
                    session_id = session_id,
                    hand_label = hand_label,
                    hand_role = hand_role,
                    config = config,
                )
            )

    return sorted(
        segments,
        key = lambda segment: (
            segment.session_id,
            segment.hand_role,
            segment.start_session_frame_index,
        ),
    )

def build_video_level_metrics(
    frame_predictions: list[FrameInteractionPrediction],
    *,
    mappings: list[InputVideoMapping],
    config: VideoProcessingConfig = DEFAULT_VIDEO_PROCESSING_CONFIG,
) -> list[VideoLevelMetric]:
    """ Calculate per-input-video hand-use metrics. """
    frame_groups = _group_frames_by_video(frame_predictions)
    metrics: list[VideoLevelMetric] = []

    for mapping in mappings:
        key = (mapping.session_id, mapping.staged_video_stem)
       
        video_frames = sorted(
            [frame for frame in frame_groups.get(key, []) if frame.is_valid_source_frame],
            key = lambda frame: frame.video_frame_index,
        )
        
        analyzed_frame_count = len(video_frames)

        if analyzed_frame_count == 0:
            metrics.append(
                VideoLevelMetric(
                    session_id = mapping.session_id,
                    input_name = mapping.input_name,
                    staged_video_stem = mapping.staged_video_stem,
                    metric_status = NO_FRAME_PREDICTIONS_STATUS,
                    dominant_hand = config.dominant_hand,
                    non_dominant_hand = _non_dominant_hand(config.dominant_hand),
                    analyzed_frame_count = 0,
                    recording_time_seconds = 0.0,
                    perc_dominant_hand = 0.0,
                    dur_dominant_hand_seconds = 0.0,
                    num_dominant_hand_per_hour = 0.0,
                    perc_non_dominant_hand = 0.0,
                    dur_non_dominant_hand_seconds = 0.0,
                    num_non_dominant_hand_per_hour = 0.0,
                    perc_bilateral = 0.0,
                    dur_bilateral_seconds = 0.0,
                    num_bilateral_per_hour = 0.0,
                    notes = _metric_notes(
                        "No Shan frame-level predictions were available for this input."
                    ),
                )
            )
            
            continue

        video_segments = _segments_from_video_frames(video_frames, config = config)
        
        metrics.append(
            _video_metric_from_frames_and_segments(
                mapping = mapping,
                frames = video_frames,
                segments = video_segments,
                config = config,
            )
        )

    return metrics

def build_session_level_metrics(
    frame_predictions: list[FrameInteractionPrediction],
    segments: list[InteractionSegment],
    *,
    mappings: list[InputVideoMapping],
    config: VideoProcessingConfig = DEFAULT_VIDEO_PROCESSING_CONFIG,
) -> list[SessionLevelMetric]:
    """ Calculate session-level metrics from session-continuous segments. """
    frames_by_session = _group_frames_by_session(frame_predictions)
    segments_by_session = _group_segments_by_session(segments)
    mappings_by_session = _group_mappings_by_session(mappings)
    metrics: list[SessionLevelMetric] = []

    for session_id, session_mappings in mappings_by_session.items():
        session_frames = [
            frame
            for frame in frames_by_session.get(session_id, [])
            if frame.is_valid_source_frame
        ]
        
        session_segments = segments_by_session.get(session_id, [])
        analyzed_frame_count = len(session_frames)

        if analyzed_frame_count == 0:
            metrics.append(
                SessionLevelMetric(
                    session_id = session_id,
                    metric_status = NO_SESSION_PREDICTIONS_STATUS,
                    dominant_hand = config.dominant_hand,
                    non_dominant_hand = _non_dominant_hand(config.dominant_hand),
                    input_video_count = len(session_mappings),
                    analyzed_frame_count = 0,
                    recording_time_seconds = 0.0,
                    perc_dominant_hand = 0.0,
                    dur_dominant_hand_seconds = 0.0,
                    num_dominant_hand_per_hour = 0.0,
                    perc_non_dominant_hand = 0.0,
                    dur_non_dominant_hand_seconds = 0.0,
                    num_non_dominant_hand_per_hour = 0.0,
                    perc_bilateral = 0.0,
                    dur_bilateral_seconds = 0.0,
                    num_bilateral_per_hour = 0.0,
                    notes = _metric_notes(
                        "No Shan frame-level predictions were available for this session."
                    ),
                )
            )

            continue

        recording_time_seconds = analyzed_frame_count / config.frame_fps
        
        role_metrics = _metric_values_by_role(
            segments = session_segments,
            recording_time_seconds = recording_time_seconds,
        )
        
        dominant = role_metrics[DOMINANT_HAND_ROLE]
        non_dominant = role_metrics[NON_DOMINANT_HAND_ROLE]

        metrics.append(
            SessionLevelMetric(
                session_id = session_id,
                metric_status = COMPUTED_STATUS,
                dominant_hand = config.dominant_hand,
                non_dominant_hand = _non_dominant_hand(config.dominant_hand),
                input_video_count = len(session_mappings),
                analyzed_frame_count = analyzed_frame_count,
                recording_time_seconds = recording_time_seconds,
                perc_dominant_hand = dominant.perc,
                dur_dominant_hand_seconds = dominant.dur_seconds,
                num_dominant_hand_per_hour = dominant.num_per_hour,
                perc_non_dominant_hand = non_dominant.perc,
                dur_non_dominant_hand_seconds = non_dominant.dur_seconds,
                num_non_dominant_hand_per_hour = non_dominant.num_per_hour,
                perc_bilateral = _bilateral_perc(dominant, non_dominant),
                dur_bilateral_seconds = _bilateral_dur(dominant, non_dominant),
                num_bilateral_per_hour = _bilateral_num(dominant, non_dominant),
                notes = _metric_notes(
                    "Session metrics combine all input videos assigned to this session."
                ),
            )
        )

    return metrics

def _segment_from_frames(
    *,
    session_id: str,
    hand_label: HandLabel,
    hand_role: HandRole,
    segment_index: int,
    start_frame: FrameInteractionPrediction,
    end_frame: FrameInteractionPrediction,
    start_metric_frame_position: int,
    end_metric_frame_position: int,
    frame_duration: float,
) -> InteractionSegment:
    start_time_seconds = (start_metric_frame_position - 1) * frame_duration
    end_time_seconds = end_metric_frame_position * frame_duration

    return InteractionSegment(
        session_id = session_id,
        hand_label = hand_label,
        hand_role = hand_role,
        segment_index = segment_index,
        start_session_time_seconds = start_time_seconds,
        end_session_time_seconds = end_time_seconds,
        duration_seconds = end_time_seconds - start_time_seconds,
        start_input_name = start_frame.input_name,
        end_input_name = end_frame.input_name,
        start_session_frame_index = start_frame.session_frame_index,
        end_session_frame_index = end_frame.session_frame_index,
    )

def _segments_from_video_frames(
    video_frames: list[FrameInteractionPrediction],
    *,
    config: VideoProcessingConfig,
) -> list[InteractionSegment]:
    if not video_frames:
        return []

    session_id = video_frames[0].session_id

    if any(frame.session_id != session_id for frame in video_frames):
        raise ValueError("video_frames must belong to a single session.")

    segments: list[InteractionSegment] = []

    for hand_label in [LEFT_HAND_LABEL, RIGHT_HAND_LABEL]:
        segments.extend(
            _segments_from_frames_for_hand(
                video_frames,
                session_id = session_id,
                hand_label = hand_label,
                hand_role = _hand_role(
                    hand_label = hand_label,
                    dominant_hand = config.dominant_hand,
                ),
                config = config,
            )
        )

    return segments

def _segments_from_frames_for_hand(
    frames: list[FrameInteractionPrediction],
    *,
    session_id: str,
    hand_label: HandLabel,
    hand_role: HandRole,
    config: VideoProcessingConfig,
) -> list[InteractionSegment]:
    frame_duration = 1.0 / config.frame_fps
    segments: list[InteractionSegment] = []
    start_interaction_frame: FrameInteractionPrediction | None = None
    last_interaction_frame: FrameInteractionPrediction | None = None
    start_metric_frame_position: int | None = None
    last_metric_frame_position: int | None = None

    for metric_frame_position, frame in enumerate(frames, start = 1):
        is_interacting = _pooled_interaction_for_hand(frame, hand_label)

        if is_interacting:
            if start_interaction_frame is None:
                start_interaction_frame = frame
                start_metric_frame_position = metric_frame_position

            last_interaction_frame = frame
            last_metric_frame_position = metric_frame_position

            continue

        if start_interaction_frame is None:
            continue

        assert last_interaction_frame is not None
        assert start_metric_frame_position is not None
        assert last_metric_frame_position is not None

        segments.append(
            _segment_from_frames(
                session_id = session_id,
                hand_label = hand_label,
                hand_role = hand_role,
                segment_index = len(segments) + 1,
                start_frame = start_interaction_frame,
                end_frame = last_interaction_frame,
                start_metric_frame_position = start_metric_frame_position,
                end_metric_frame_position = last_metric_frame_position,
                frame_duration = frame_duration,
            )
        )

        start_interaction_frame = None
        last_interaction_frame = None
        start_metric_frame_position = None
        last_metric_frame_position = None

    if start_interaction_frame is not None:
        assert last_interaction_frame is not None
        assert start_metric_frame_position is not None
        assert last_metric_frame_position is not None

        segments.append(
            _segment_from_frames(
                session_id = session_id,
                hand_label = hand_label,
                hand_role = hand_role,
                segment_index = len(segments) + 1,
                start_frame = start_interaction_frame,
                end_frame = last_interaction_frame,
                start_metric_frame_position = start_metric_frame_position,
                end_metric_frame_position = last_metric_frame_position,
                frame_duration = frame_duration,
            )
        )

    return segments

def _video_metric_from_frames_and_segments(
    *,
    mapping: InputVideoMapping,
    frames: list[FrameInteractionPrediction],
    segments: list[InteractionSegment],
    config: VideoProcessingConfig,
) -> VideoLevelMetric:
    recording_time_seconds = len(frames) / config.frame_fps
    
    role_metrics = _metric_values_by_role(
        segments = segments,
        recording_time_seconds = recording_time_seconds,
    )
    
    dominant = role_metrics[DOMINANT_HAND_ROLE]
    non_dominant = role_metrics[NON_DOMINANT_HAND_ROLE]

    return VideoLevelMetric(
        session_id = mapping.session_id,
        input_name = mapping.input_name,
        staged_video_stem = mapping.staged_video_stem,
        metric_status = COMPUTED_STATUS,
        dominant_hand = config.dominant_hand,
        non_dominant_hand = _non_dominant_hand(config.dominant_hand),
        analyzed_frame_count = len(frames),
        recording_time_seconds = recording_time_seconds,
        perc_dominant_hand = dominant.perc,
        dur_dominant_hand_seconds = dominant.dur_seconds,
        num_dominant_hand_per_hour = dominant.num_per_hour,
        perc_non_dominant_hand = non_dominant.perc,
        dur_non_dominant_hand_seconds = non_dominant.dur_seconds,
        num_non_dominant_hand_per_hour = non_dominant.num_per_hour,
        perc_bilateral = _bilateral_perc(dominant, non_dominant),
        dur_bilateral_seconds = _bilateral_dur(dominant, non_dominant),
        num_bilateral_per_hour = _bilateral_num(dominant, non_dominant),
        notes = _metric_notes("Video metrics are computed per original input video."),
    )

def _metric_notes(prefix: str) -> str:
    return (
        f"{prefix} Metrics assume detected egocentric hands belong to the camera "
        "wearer; hand ownership is not explicitly verified, and other people's "
        "hands visible in the frame may affect contact-state metrics. These are "
        "hand-use interaction metrics, not clinical scores."
    )

def _write_diagnostic(
    diagnostic_log: Callable[[str], None] | None,
    message: str,
) -> None:
    if diagnostic_log is not None:
        diagnostic_log(message)

def _prediction_paths_by_clip(shan_outputs_dir: Path) -> dict[str, list[Path]]:
    """ Group Shan JSON frame predictions by subclip folder.

    Expected input structure:

        shan_outputs/
          video001.--1/
            frame_5_shan.json
            frame_10_shan.json
          video001.--2/
            frame_5_shan.json

    Returned structure:

        {
            "video001.--1": [
                Path("shan_outputs/video001.--1/frame_5_shan.json"),
                Path("shan_outputs/video001.--1/frame_10_shan.json"),
            ],
            "video001.--2": [
                Path("shan_outputs/video001.--2/frame_5_shan.json"),
            ],
        }

    Each list is sorted by frame number using ``_frame_sort_key``.
    """
    grouped_paths: dict[str, list[Path]] = {}

    for prediction_path in sorted(shan_outputs_dir.rglob("*_shan.json")):
        clip_name = prediction_path.parent.name
        grouped_paths.setdefault(clip_name, []).append(prediction_path)

    return {
        clip_name: sorted(paths, key = _frame_sort_key)
        for clip_name, paths in grouped_paths.items()
    }

def _clip_belongs_to_mapping(clip_name: str, mapping: InputVideoMapping) -> bool:
    """ Return whether a Shan subclip folder belongs to a staged input video.

    Expected input examples:

        clip_name = "video001--1"
        clip_name = "video001.--1"
        mapping.staged_video_stem = "video001"

    Returned result:

        True when the clip name is exactly the staged stem, starts with the
        staged stem followed by ".", or starts with the staged stem followed
        by "--".

    This maps real EgoVizML/Shan subclip folders such as "video001--1" and
    older test fixtures such as "video001.--1" back to the staged video
    recorded in adl_input_manifest.csv.
    """
    staged_stem = mapping.staged_video_stem

    return (
        clip_name == staged_stem
        or clip_name.startswith(f"{staged_stem}.")
        or clip_name.startswith(f"{staged_stem}--")
    )


def _subclip_sort_key(clip_name: str) -> tuple[str, int]:
    """ Return a natural sort key for an EgoVizML/Shan subclip folder name.

    Expected input examples:

        "video001.--1"
        "video001.--2"
        "video001.--10"

    Returned sort keys:

        ("video001", 1)
        ("video001", 2)
        ("video001", 10)

    This makes subclip folders sort by numeric subclip order instead of
    plain alphabetical order.
    """
    match = re.search(r"[._-]+(\d+)$", clip_name)

    if match is None:
        return (clip_name, 0)

    return (clip_name[: match.start()], int(match.group(1)))

def _frame_sort_key(path: Path) -> tuple[str, int]:
    """ Return a natural sort key for a Shan frame prediction path.

    Expected input examples:

        Path("shan_outputs/video001.--1/frame_5_shan.json")
        Path("shan_outputs/video001.--1/frame_10_shan.json")

    Returned sort keys:

        ("frame_", 5)
        ("frame_", 10)

    This makes frame paths sort by numeric frame order instead of plain
    alphabetical order.
    """
    match = re.search(r"(\d+)(?=_shan$|$)", path.stem)

    if match is None:
        return (path.stem, 0)

    return (path.stem[: match.start()], int(match.group(1)))

def _load_shan_prediction_json(prediction_path: Path) -> dict[str, object]:
    payload = json.loads(prediction_path.read_text(encoding = "utf-8"))

    if isinstance(payload, dict):
        return payload

    return {}

def _detected_hands_from_payload(payload: dict[str, object]) -> list[object]:
    hands = payload.get("hands", [])

    if not isinstance(hands, list):
        return []

    return [hand for hand in hands if _is_shan_hand_prediction(hand)]

def _is_shan_hand_prediction(hand: object) -> bool:
    if isinstance(hand, dict):
        return True

    return isinstance(hand, list) and len(hand) > SHAN_HAND_SIDE_INDEX

def _contact_state(hand: object) -> int:
    if isinstance(hand, dict):
        value = hand.get("contact_state", hand.get("contactstate", 0))
    elif isinstance(hand, list) and len(hand) > SHAN_HAND_CONTACT_STATE_INDEX:
        value = hand[SHAN_HAND_CONTACT_STATE_INDEX]
    else:
        value = 0

    return _as_int(value, 0)

def _hand_label(hand: object) -> HandLabel | None:
    if isinstance(hand, list) and len(hand) > SHAN_HAND_SIDE_INDEX:
        side_value = _as_int(hand[SHAN_HAND_SIDE_INDEX], -1)

        if side_value == LEFT_HAND_VALUE:
            return LEFT_HAND_LABEL

        if side_value == RIGHT_HAND_VALUE:
            return RIGHT_HAND_LABEL

    if isinstance(hand, dict) and "hand_side" in hand:
        side_value = _as_int(hand["hand_side"], -1)

        if side_value == LEFT_HAND_VALUE:
            return LEFT_HAND_LABEL

        if side_value == RIGHT_HAND_VALUE:
            return RIGHT_HAND_LABEL

    return None

def _non_dominant_hand(dominant_hand: HandLabel) -> HandLabel:
    if dominant_hand == LEFT_HAND_LABEL:
        return RIGHT_HAND_LABEL

    return LEFT_HAND_LABEL

def _hand_role(
    *,
    hand_label: HandLabel,
    dominant_hand: HandLabel,
) -> HandRole:
    if hand_label == dominant_hand:
        return DOMINANT_HAND_ROLE

    return NON_DOMINANT_HAND_ROLE

def _interaction_for_hand(
    *,
    left_value: int,
    right_value: int,
    hand: HandLabel,
) -> int:
    if hand == LEFT_HAND_LABEL:
        return left_value

    return right_value

def _pooled_interaction_for_hand(
    frame: FrameInteractionPrediction,
    hand_label: HandLabel,
) -> int:
    if hand_label == LEFT_HAND_LABEL:
        return frame.left_interaction

    return frame.right_interaction

def _group_frames_by_session(
    frame_predictions: list[FrameInteractionPrediction],
) -> dict[str, list[FrameInteractionPrediction]]:
    """ Group frame-level predictions by session id.

    Expected input examples:

        [
            FrameInteractionPrediction(session_id = "session001", input_name = "part1.mp4", ...),
            FrameInteractionPrediction(session_id = "session001", input_name = "part2.mp4", ...),
            FrameInteractionPrediction(session_id = "session002", input_name = "part1.mp4", ...),
            FrameInteractionPrediction(session_id = "session002", input_name = "part2.mp4", ...),
        ]

    Returned structure:

        {
            "session001": [
                FrameInteractionPrediction(... session001 / part1.mp4 ...),
                FrameInteractionPrediction(... session001 / part2.mp4 ...),
            ],
            "session002": [
                FrameInteractionPrediction(... session002 / part1.mp4 ...),
                FrameInteractionPrediction(... session002 / part2.mp4 ...),
            ],
        }

    This keeps all frames from the same session together so session-level
    interaction segments and metrics can be computed across the input videos
    that belong to that session.
    """
    groups: dict[str, list[FrameInteractionPrediction]] = {}

    for frame in frame_predictions:
        groups.setdefault(frame.session_id, []).append(frame)

    return groups

def _group_frames_by_video(
    frame_predictions: list[FrameInteractionPrediction],
) -> dict[tuple[str, str], list[FrameInteractionPrediction]]:
    """ Group frame-level predictions by session and staged input video.

    Expected input examples:

        [
            FrameInteractionPrediction(
                session_id = "session001",
                input_name = "part1.mp4",
                staged_video_stem = "video001",
                ...
            ),
            FrameInteractionPrediction(
                session_id = "session001",
                input_name = "part2.mp4",
                staged_video_stem = "video002",
                ...
            ),
            FrameInteractionPrediction(
                session_id = "session002",
                input_name = "part1.mp4",
                staged_video_stem = "video001",
                ...
            ),
        ]

    Returned structure:

        {
            ("session001", "video001"): [
                FrameInteractionPrediction(... session001 / part1.mp4 / video001 ...),
            ],
            ("session001", "video002"): [
                FrameInteractionPrediction(... session001 / part2.mp4 / video002 ...),
            ],
            ("session002", "video001"): [
                FrameInteractionPrediction(... session002 / part1.mp4 / video001 ...),
            ],
        }

    The grouped key uses ``staged_video_stem`` instead of ``input_name``
    because original filenames may repeat across folders or sessions. The
    user-facing ``input_name`` can still be read from the frames inside each
    group when writing video-level metric rows.
    """
    groups: dict[tuple[str, str], list[FrameInteractionPrediction]] = {}

    for frame in frame_predictions:
        groups.setdefault(
            (frame.session_id, frame.staged_video_stem),
            [],
        ).append(frame)

    return groups

def _group_segments_by_session(
    segments: list[InteractionSegment],
) -> dict[str, list[InteractionSegment]]:
    """ Group interaction segments by session id.

    Expected input examples:

        [
            InteractionSegment(session_id = "session001", segment_index = 1, ...),
            InteractionSegment(session_id = "session001", segment_index = 2, ...),
            InteractionSegment(session_id = "session002", segment_index = 1, ...),
        ]

    Returned structure:

        {
            "session001": [
                InteractionSegment(... session001 / segment 1 ...),
                InteractionSegment(... session001 / segment 2 ...),
            ],
            "session002": [
                InteractionSegment(... session002 / segment 1 ...),
            ],
        }

    This keeps interaction segments from the same session together so
    session-level metrics can compute interaction count, total interaction
    time, and average interaction duration per session.
    """
    groups: dict[str, list[InteractionSegment]] = {}

    for segment in segments:
        groups.setdefault(segment.session_id, []).append(segment)

    return groups

def _group_mappings_by_session(
    mappings: list[InputVideoMapping],
) -> dict[str, list[InputVideoMapping]]:
    """ Group input-video mappings by session id.

    Expected input examples:

        [
            InputVideoMapping(session_id = "session001", input_name = "part1.mp4", ...),
            InputVideoMapping(session_id = "session001", input_name = "part2.mp4", ...),
            InputVideoMapping(session_id = "session002", input_name = "part1.mp4", ...),
        ]

    Returned structure:

        {
            "session001": [
                InputVideoMapping(... session001 / part1.mp4 ...),
                InputVideoMapping(... session001 / part2.mp4 ...),
            ],
            "session002": [
                InputVideoMapping(... session002 / part1.mp4 ...),
            ],
        }

    This keeps the input videos for each session together so session-level
    metrics can report how many videos contributed to each session.
    """
    groups: dict[str, list[InputVideoMapping]] = {}

    for mapping in mappings:
        groups.setdefault(mapping.session_id, []).append(mapping)

    return groups


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator

def _as_int(value: object, default: int) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def _as_hand_label(value: object, default: HandLabel) -> HandLabel:
    if not isinstance(value, str):
        return default

    normalized = value.strip().lower()

    if normalized == LEFT_HAND_LABEL:
        return LEFT_HAND_LABEL

    if normalized == RIGHT_HAND_LABEL:
        return RIGHT_HAND_LABEL

    return default

def _format_number(value: float) -> str:
    """ Format a metric number for compact CSV output.

    Expected input examples:

        1.0
        1.25
        0.333333333
        12.000000

    Returned values:

        "1"
        "1.25"
        "0.333333"
        "12"

    Values are rounded to 6 decimal places, then trailing zeros and a
    trailing decimal point are removed.
    """
    return f"{value:.6f}".rstrip("0").rstrip(".")

def _write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_frame_level_predictions(
    path: Path,
    rows: list[FrameInteractionPrediction],
) -> None:
    _write_csv(
        path,
        fieldnames=[
            "session_id",
            "input_name",
            "staged_video_stem",
            "subclip_name",
            "frame_path",
            "subclip_timestamp_seconds",
            "source_timestamp_seconds",
            "valid_source_duration_seconds",
            "is_valid_source_frame",
            "video_frame_index",
            "session_frame_index",
            "video_timestamp_seconds",
            "session_timestamp_seconds",
            "detected_hand_count",
            "left_detected_hand_count",
            "right_detected_hand_count",
            "max_contact_state",
            "max_left_contact_state",
            "max_right_contact_state",
            "raw_left_interaction",
            "raw_right_interaction",
            "raw_dominant_interaction",
            "raw_non_dominant_interaction",
            "raw_any_interaction",
            "left_interaction",
            "right_interaction",
            "dominant_interaction",
            "non_dominant_interaction",
            "any_interaction",
        ],
        rows=[
            {
                "session_id": row.session_id,
                "input_name": row.input_name,
                "staged_video_stem": row.staged_video_stem,
                "subclip_name": row.subclip_name,
                "frame_path": row.frame_path,
                "subclip_timestamp_seconds": _format_number(row.subclip_timestamp_seconds),
                "source_timestamp_seconds": _format_number(row.source_timestamp_seconds),
                "valid_source_duration_seconds": _format_number(
                    row.valid_source_duration_seconds
                ),
                "is_valid_source_frame": "true" if row.is_valid_source_frame else "false",
                "video_frame_index": row.video_frame_index,
                "session_frame_index": row.session_frame_index,
                "video_timestamp_seconds": _format_number(row.video_timestamp_seconds),
                "session_timestamp_seconds": _format_number(row.session_timestamp_seconds),
                "detected_hand_count": row.detected_hand_count,
                "left_detected_hand_count": row.left_detected_hand_count,
                "right_detected_hand_count": row.right_detected_hand_count,
                "max_contact_state": row.max_contact_state,
                "max_left_contact_state": row.max_left_contact_state,
                "max_right_contact_state": row.max_right_contact_state,
                "raw_left_interaction": row.raw_left_interaction,
                "raw_right_interaction": row.raw_right_interaction,
                "raw_dominant_interaction": row.raw_dominant_interaction,
                "raw_non_dominant_interaction": row.raw_non_dominant_interaction,
                "raw_any_interaction": row.raw_any_interaction,
                "left_interaction": row.left_interaction,
                "right_interaction": row.right_interaction,
                "dominant_interaction": row.dominant_interaction,
                "non_dominant_interaction": row.non_dominant_interaction,
                "any_interaction": row.any_interaction,
            }
            for row in rows
        ],
    )

def _write_interaction_segments(path: Path, rows: list[InteractionSegment]) -> None:
    _write_csv(
        path,
        fieldnames=[
            "session_id",
            "segment_index",
            "start_session_time_seconds",
            "end_session_time_seconds",
            "duration_seconds",
            "start_input_name",
            "end_input_name",
            "start_session_frame_index",
            "end_session_frame_index",
            "hand_label",
            "hand_role",
        ],
        rows=[
            {
                "session_id": row.session_id,
                "segment_index": row.segment_index,
                "start_session_time_seconds": _format_number(row.start_session_time_seconds),
                "end_session_time_seconds": _format_number(row.end_session_time_seconds),
                "duration_seconds": _format_number(row.duration_seconds),
                "start_input_name": row.start_input_name,
                "end_input_name": row.end_input_name,
                "start_session_frame_index": row.start_session_frame_index,
                "end_session_frame_index": row.end_session_frame_index,
                "hand_label": row.hand_label,
                "hand_role": row.hand_role,
            }
            for row in rows
        ],
    )

def _write_video_level_metrics(path: Path, rows: list[VideoLevelMetric]) -> None:
    _write_csv(
        path,
        fieldnames=[
            "session_id",
            "input_name",
            "staged_video_stem",
            "metric_status",
            "dominant_hand",
            "non_dominant_hand",
            "analyzed_frame_count",
            "recording_time_seconds",
            "perc_dominant_hand",
            "dur_dominant_hand_seconds",
            "num_dominant_hand_per_hour",
            "perc_non_dominant_hand",
            "dur_non_dominant_hand_seconds",
            "num_non_dominant_hand_per_hour",
            "perc_bilateral",
            "dur_bilateral_seconds",
            "num_bilateral_per_hour",
            "notes",
        ],
        rows=[
            {
                "session_id": row.session_id,
                "input_name": row.input_name,
                "staged_video_stem": row.staged_video_stem,
                "metric_status": row.metric_status,
                "dominant_hand": row.dominant_hand,
                "non_dominant_hand": row.non_dominant_hand,
                "analyzed_frame_count": row.analyzed_frame_count,
                "recording_time_seconds": _format_number(row.recording_time_seconds),
                "perc_dominant_hand": row.perc_dominant_hand,
                "dur_dominant_hand_seconds": row.dur_dominant_hand_seconds,
                "num_dominant_hand_per_hour": row.num_dominant_hand_per_hour,
                "perc_non_dominant_hand": row.perc_non_dominant_hand,
                "dur_non_dominant_hand_seconds": row.dur_non_dominant_hand_seconds,
                "num_non_dominant_hand_per_hour": row.num_non_dominant_hand_per_hour,
                "perc_bilateral": row.perc_bilateral,
                "dur_bilateral_seconds": row.dur_bilateral_seconds,
                "num_bilateral_per_hour": row.num_bilateral_per_hour,
                "notes": row.notes,
            }
            for row in rows
        ],
    )

def _write_session_level_metrics(path: Path, rows: list[SessionLevelMetric]) -> None:
    _write_csv(
        path,
        fieldnames=[
            "session_id",
            "metric_status",
            "dominant_hand",
            "non_dominant_hand",
            "input_video_count",
            "analyzed_frame_count",
            "recording_time_seconds",
            "perc_dominant_hand",
            "dur_dominant_hand_seconds",
            "num_dominant_hand_per_hour",
            "perc_non_dominant_hand",
            "dur_non_dominant_hand_seconds",
            "num_non_dominant_hand_per_hour",
            "perc_bilateral",
            "dur_bilateral_seconds",
            "num_bilateral_per_hour",
            "notes",
        ],
        rows=[
            {
                "session_id": row.session_id,
                "metric_status": row.metric_status,
                "dominant_hand": row.dominant_hand,
                "non_dominant_hand": row.non_dominant_hand,
                "input_video_count": row.input_video_count,
                "analyzed_frame_count": row.analyzed_frame_count,
                "recording_time_seconds": _format_number(row.recording_time_seconds),
                "perc_dominant_hand": _format_number(row.perc_dominant_hand),
                "dur_dominant_hand_seconds": _format_number(
                    row.dur_dominant_hand_seconds
                ),
                "num_dominant_hand_per_hour": _format_number(
                    row.num_dominant_hand_per_hour
                ),
                "perc_non_dominant_hand": _format_number(row.perc_non_dominant_hand),
                "dur_non_dominant_hand_seconds": _format_number(
                    row.dur_non_dominant_hand_seconds
                ),
                "num_non_dominant_hand_per_hour": _format_number(
                    row.num_non_dominant_hand_per_hour
                ),
                "perc_bilateral": _format_number(row.perc_bilateral),
                "dur_bilateral_seconds": _format_number(row.dur_bilateral_seconds),
                "num_bilateral_per_hour": _format_number(row.num_bilateral_per_hour),
                "notes": row.notes,
            }
            for row in rows
        ],
    )

def _write_video_level_metrics_summary(
    path: Path,
    video_rows: list[VideoLevelMetric],
    session_rows: list[SessionLevelMetric],
) -> None:
    computed_video_rows = [
        row for row in video_rows if row.metric_status == COMPUTED_STATUS
    ]
    
    computed_session_rows = [
        row for row in session_rows if row.metric_status == COMPUTED_STATUS
    ]

    _write_csv(
        path,
        fieldnames = ["metric", "value", "notes"],
        rows=[
            {
                "metric": "video_count",
                "value": len(video_rows),
                "notes": "Number of input videos included in the metric file.",
            },
            {
                "metric": "computed_video_count",
                "value": len(computed_video_rows),
                "notes": "Number of videos with at least one Shan frame prediction.",
            },
            {
                "metric": "session_count",
                "value": len(session_rows),
                "notes": "Number of sessions included in the session-level metric file.",
            },
            {
                "metric": "computed_session_count",
                "value": len(computed_session_rows),
                "notes": "Number of sessions with at least one Shan frame prediction.",
            },
            {
                "metric": "bandini_metric_set",
                "value": "PercDH,DurDH,NumDH,PercNH,DurNH,NumNH,PercBi,DurBi,NumBi",
                "notes": (
                    "Bandini hand-use metric set computed from hand-specific "
                    "Statepool interaction profiles."
                ),
            },
        ],
    )
