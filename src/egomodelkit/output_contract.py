""" Output-folder contracts and dynamic previews for EgoModelKit runs. """

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_MODEL_ID,
    ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES,
    COMBINED_PREDS_FILENAME,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_MODEL_ID,
    HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES,
)

InputScenario = Literal[
    "hand-object-single-image",
    "hand-object-image-directory",
    "adl-single-video",
    "adl-video-directory",
    "adl-combined-predictions",
]

RUN_README_FILENAME: Final[str] = "README.txt"
RUN_SUMMARY_FILENAME: Final[str] = "run_summary.json"
RUN_MANIFEST_FILENAME: Final[str] = "run_manifest.json"

PROGRESS_LOG_FILENAME: Final[str] = "progress.jsonl"
RUNTIME_LOG_FILENAME: Final[str] = "runtime.log"

VIDEO_LEVEL_METRICS_FILENAME: Final[str] = "video_level_metrics.csv"
VIDEO_LEVEL_METRICS_SUMMARY_FILENAME: Final[str] = "video_level_metrics_summary.csv"

HAND_OBJECT_VISUAL_OUTPUT_DIRNAME: Final[str] = "hand_object_contact"
MODEL_OUTPUTS_DIRNAME: Final[str] = "model_outputs"
POST_PROCESSING_DIRNAME: Final[str] = "post_processing"
INTERMEDIATE_FILES_DIRNAME: Final[str] = "intermediate_files"

MAX_TREE_EXAMPLE_FILES: Final[int] = 3

HAND_OBJECT_DETECTION_VISUAL_SUFFIX: Final[str] = ".png"

@dataclass(frozen = True, slots = True)
class OutputFileDescription:
    """ Plain-language description for one output file or directory. """
    name: str
    description: str

@dataclass(frozen = True, slots = True)
class OutputPreviewContext:
    """ Dynamic context for an output-folder preview. """
    scenario: InputScenario
    run_id: str
    input_names: tuple[str, ...]
    output_name: str

@dataclass(frozen = True, slots = True)
class RunOutputLayout:
    """ Stable output-folder paths for one EgoModelKit run. """
    run_dir: Path
    
    @property
    def readme_path(self) -> Path:
        return self.run_dir / RUN_README_FILENAME
    
    @property
    def run_summary_path(self) -> Path:
        return self.run_dir / RUN_SUMMARY_FILENAME

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / RUN_MANIFEST_FILENAME
    
    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"
    
    @property
    def visual_outputs_dir(self) -> Path:
        return self.run_dir / "visual_outputs"
    
    @property
    def technical_dir(self) -> Path:
        return self.run_dir / "technical"
    
    @property
    def model_outputs_dir(self) -> Path:
        return self.technical_dir / MODEL_OUTPUTS_DIRNAME
    
    @property
    def post_processing_dir(self) -> Path:
        return self.technical_dir / POST_PROCESSING_DIRNAME
    
    @property
    def intermediate_files_dir(self) -> Path:
        return self.technical_dir / INTERMEDIATE_FILES_DIRNAME

    @property
    def adl_extracted_frames_dir(self) -> Path:
        return self.intermediate_files_dir / "extracted_frames"

    @property
    def adl_detic_outputs_dir(self) -> Path:
        return self.intermediate_files_dir / "detic_outputs"

    @property
    def adl_shan_outputs_dir(self) -> Path:
        return self.intermediate_files_dir / "shan_outputs"

    @property
    def frame_level_predictions_path(self) -> Path:
        return self.post_processing_dir / "frame_level_predictions.csv"

    @property
    def interaction_segments_path(self) -> Path:
        return self.post_processing_dir / "interaction_segments.csv"

    @property
    def video_level_metrics_path(self) -> Path:
        return self.results_dir / VIDEO_LEVEL_METRICS_FILENAME

    @property
    def video_level_metrics_summary_path(self) -> Path:
        return self.results_dir / VIDEO_LEVEL_METRICS_SUMMARY_FILENAME

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"
    
    @property
    def progress_log_path(self) -> Path:
        return self.logs_dir / PROGRESS_LOG_FILENAME
    
    @property
    def runtime_log_path(self) -> Path:
        return self.logs_dir / RUNTIME_LOG_FILENAME
    
    @property
    def output_folder_path(self) -> Path:
        return self.run_dir.parent
    
def build_run_id(now: datetime | None = None) -> str:
    """ Return a neutral run id safe for privacy-sensitive workflows. """
    timestamp = now if now is not None else datetime.now().astimezone()
    
    return f"run-{timestamp:%Y-%m-%d-%H%M%S}"

def build_run_output_layout(output_root: Path, *, run_id: str) -> RunOutputLayout:
    """ Return the output layout for one run under a user-selected output root. """
    return RunOutputLayout(run_dir = output_root / run_id)

def infer_input_scenario(*, model_id: str, input_path: Path) -> InputScenario:
    """ Infer the GUI/output-preview scenario from a model id and input path. """
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        if input_path.is_dir():
            return "hand-object-image-directory"
        
        return "hand-object-single-image"
    
    if model_id == ADL_RECOGNITION_MODEL_ID:
        if input_path.is_dir():
            return "adl-video-directory"
        
        if input_path.name == COMBINED_PREDS_FILENAME:
            return "adl-combined-predictions"
        
        return "adl-single-video"
    
    raise ValueError(f"Unsupported model id: {model_id}")

def infer_input_scenario_from_names(
    *,
    model_id: str,
    input_names: tuple[str, ...],
) -> InputScenario:
    """ Infer an output-preview scenario before browser uploads are staged. """
    if not input_names:
        raise ValueError("At least one input name is required for output preview.")
    
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        if len(input_names) == 1:
            return "hand-object-single-image"
        
        return "hand-object-image-directory"
    
    if model_id == ADL_RECOGNITION_MODEL_ID:
        if len(input_names) == 1 and input_names[0] == COMBINED_PREDS_FILENAME:
            return "adl-combined-predictions"
        
        if len(input_names) == 1:
            return "adl-single-video"
        
        return "adl-video-directory"
    
    raise ValueError(f"Unsupported model id: {model_id}")

def build_output_preview_context_from_names(
    *,
    model_id: str,
    input_names: tuple[str, ...],
    output_root: Path,
    run_id: str,
) -> OutputPreviewContext:
    """ Build output preview context from browser-selected input names. """
    scenario = infer_input_scenario_from_names(
        model_id = model_id,
        input_names = input_names,
    )
        
    return OutputPreviewContext(
        scenario = scenario,
        run_id = run_id,
        input_names = input_names,
        output_name = _display_output_root_name(output_root)
    )

def build_output_preview_context(
    *,
    model_id: str,
    input_path: Path,
    output_root: Path,
    run_id: str
) -> OutputPreviewContext:
    """ Build a dynamic output preview context from the selected input and output. """
    scenario = infer_input_scenario(model_id = model_id, input_path = input_path)
    input_names = tuple(_input_names_for_preview(model_id = model_id, input_path = input_path))
    
    return OutputPreviewContext(
        scenario = scenario,
        run_id = run_id,
        input_names = input_names,
        output_name = _display_output_root_name(output_root),
    )

def create_output_scaffold(
    *,
    layout: RunOutputLayout,
    model_id: str,
    input_path: Path,
    scenario: InputScenario,
    status: str = "created",
) -> None:
    """ Create run-folder metadata files and stable top-level directories. """
    layout.run_dir.mkdir(parents = True, exist_ok = True)
    layout.logs_dir.mkdir(parents = True, exist_ok = True)
    layout.technical_dir.mkdir(parents = True, exist_ok = True)
    
    if model_id == ADL_RECOGNITION_MODEL_ID:
        layout.results_dir.mkdir(parents = True, exist_ok = True)
        layout.model_outputs_dir.mkdir(parents = True, exist_ok = True)
        layout.post_processing_dir.mkdir(parents = True, exist_ok = True)
        layout.intermediate_files_dir.mkdir(parents = True, exist_ok = True)
        layout.adl_extracted_frames_dir.mkdir(parents = True, exist_ok = True)
        layout.adl_detic_outputs_dir.mkdir(parents = True, exist_ok = True)
        layout.adl_shan_outputs_dir.mkdir(parents = True, exist_ok = True)
    elif model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        (layout.visual_outputs_dir / HAND_OBJECT_VISUAL_OUTPUT_DIRNAME).mkdir(
            parents = True,
            exist_ok = True,
        )

        layout.model_outputs_dir.mkdir(parents = True, exist_ok = True)
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    
    preview_context = build_output_preview_context(
        model_id = model_id,
        input_path = input_path,
        output_root = layout.run_dir.parent,
        run_id = layout.run_dir.name,
    )
    
    layout.readme_path.write_text(
        run_readme_text(model_id = model_id, context = preview_context),
        encoding = "utf-8",
    )
    
    if model_id == ADL_RECOGNITION_MODEL_ID:
        write_stub_video_level_metric_files(
            layout = layout,
            context = preview_context,
            status = "pending",
        )
    
    _write_json(
        layout.run_summary_path,
        {
            "model_id": model_id,
            "input_name": input_path.name,
            "input_path": str(input_path),
            "output_folder": str(layout.run_dir),
            "scenario": scenario,
            "status": status,
        },
    )
    
    _write_json(
        layout.run_manifest_path,
        {
            "model_id": model_id,
            "input_name": input_path.name,
            "scenario": scenario,
            "output_contract_version": 1,
            "notes": (
                "Runtime image IDs and exact code/model pins may be populated "
                "by runtime execution."
            ),
        },
    )
    
    layout.progress_log_path.touch(exist_ok = True)
    layout.runtime_log_path.touch(exist_ok = True)
    
def output_folder_tree(context: OutputPreviewContext) -> str:
    """ Return a dynamic plain-text folder tree for the GUI output preview. """
    if context.scenario == "hand-object-single-image":
        stem = _stem(context.input_names[0])
        
        lines = [
            f"{context.output_name}/",
            f"  {context.run_id}/",
            "    README.txt",
            "    run_summary.json",
            "    run_manifest.json",
            "    visual_outputs/",
            "      hand_object_contact/",
            f"        {stem}_det{HAND_OBJECT_DETECTION_VISUAL_SUFFIX}",
            "    technical/",
            "      model_outputs/",
            f"        {stem}_shan.json",
            f"        {stem}_shan.pkl",
            "    logs/",
            "      progress.jsonl",
            "      runtime.log",
        ]
    elif context.scenario == "hand-object-image-directory":
        stems = [_stem(name) for name in context.input_names]
        
        visual_lines = [
            f"        {stem}_det{HAND_OBJECT_DETECTION_VISUAL_SUFFIX}"
            for stem in _preview_items(stems)]
        
        model_lines: list[str] = []
        
        for stem in _preview_items(stems):
            if stem == "...":
                model_lines.append("        ...")
            else:
                model_lines.append(f"        {stem}_shan.json")
                model_lines.append(f"        {stem}_shan.pkl")

        lines = [
            f"{context.output_name}/",
            f"  {context.run_id}/",
            "    README.txt",
            "    run_summary.json",
            "    run_manifest.json",
            "    visual_outputs/",
            "      hand_object_contact/",
            *visual_lines,
            "    technical/",
            "      model_outputs/",
            *model_lines,
            "    logs/",
            "      progress.jsonl",
            "      runtime.log",
        ]
    elif context.scenario == "adl-single-video":
        lines = [
            f"{context.output_name}/",
            f"  {context.run_id}/",
            "    README.txt",
            "    run_summary.json",
            "    run_manifest.json",
            "    results/",
            "      video_level_metrics.csv",
            "      video_level_metrics_summary.csv",
            "    technical/",
            "      model_outputs/",
            "        predictions.csv",
            "        predictions_summary.csv",
            "        all_preds.pkl",
            "      post_processing/",
            "        frame_level_predictions.csv",
            "        interaction_segments.csv",
            "      intermediate_files/",
            "        extracted_frames/",
            "        detic_outputs/",
            "        shan_outputs/",
            "    logs/",
            "      progress.jsonl",
            "      runtime.log",
        ]
    elif context.scenario == "adl-video-directory":
        video_stems = [_stem(name) for name in context.input_names]
        
        session_lines = [
            "          ..." if stem == "..." else f"          {stem}/" 
            for stem in _preview_items(video_stems)
        ]
        
        lines = [
            f"{context.output_name}/",
            f"  {context.run_id}/",
            "    README.txt",
            "    run_summary.json",
            "    run_manifest.json",
            "    results/",
            "      video_level_metrics.csv",
            "      video_level_metrics_summary.csv",
            "    technical/",
            "      model_outputs/",
            "        predictions.csv",
            "        predictions_summary.csv",
            "        all_preds.pkl",
            "      post_processing/",
            "        frame_level_predictions.csv",
            "        interaction_segments.csv",
            "      intermediate_files/",
            "        extracted_frames/",
            *session_lines,
            "        detic_outputs/",
            "        shan_outputs/",
            "    logs/",
            "      progress.jsonl",
            "      runtime.log",
        ]
    elif context.scenario == "adl-combined-predictions":
        lines = [
            f"{context.output_name}/",
            f"  {context.run_id}/",
            "    README.txt",
            "    run_summary.json",
            "    run_manifest.json",
            "    results/",
            "      video_level_metrics.csv",
            "      video_level_metrics_summary.csv",
            "    technical/",
            "      model_outputs/",
            "        predictions.csv",
            "        predictions_summary.csv",
            f"        {context.input_names[0]}",
            "      post_processing/",
            "        frame_level_predictions.csv",
            "        interaction_segments.csv",
            "    logs/",
            "      progress.jsonl",
            "      runtime.log"
        ]
    else:
        raise ValueError(f"Unsupported output scenario: {context.scenario}")  
    
    return "\n".join(lines)

def output_file_descriptions(context: OutputPreviewContext) -> list[OutputFileDescription]:
    """ Return plain-language file descriptions for an output scenario. """
    common = [
        OutputFileDescription(
            "README.txt",
            (
                "Plain-language guide explaining the output folder contents "
                "and which files to review first."   
            ),
        ),
        OutputFileDescription(
            "run_summary.json",
            (
                "Summary of the run, including model name, input name, "
                "output folder, status, and completion time."
            )
        ),
        OutputFileDescription(
            "run_manifest.json",
            (
                "Reproducibility record, including model ID, runtime details, "
                "container/image information, code/model version pins, and run settings."
            )
        ),
    ]
    
    logs = [
        OutputFileDescription("progress.jsonl", "Progress events written during the run."),
        OutputFileDescription("runtime.log", "Technical runtime log for troubleshooting."),
    ]
    
    if context.scenario == "hand-object-single-image":
        stem = _stem(context.input_names[0])
        
        return [
            *common,
            OutputFileDescription(
                f"{stem}_det{HAND_OBJECT_DETECTION_VISUAL_SUFFIX}",
                (
                    "User-facing visualization image showing detected hands, "
                    "objects, and contact results."
                ),
            ),
            OutputFileDescription(
                f"{stem}_shan.json",
                "Structured hand-object contact model output in a readable data format.",
            ),
            OutputFileDescription(
                f"{stem}_shan.pkl",
                "Raw model output used for downstream processing or debugging.",
            ),
            *logs,
        ]
    
    if context.scenario == "hand-object-image-directory":
        return [
            *common,
            OutputFileDescription(
                f"*_det{HAND_OBJECT_DETECTION_VISUAL_SUFFIX}",
                "One user-facing visualization image for each processed input image.",
            ),
            OutputFileDescription(
                "*_shan.json",
                "One structured hand-object contact output file for each processed image.",
            ),
            OutputFileDescription(
                "*_shan.pkl",
                "One raw model output file for each processed image."
            ),
            *logs,
        ]
    
    return [
        *common,
        OutputFileDescription(
            "video_level_metrics.csv",
            (
                "Main user-facing post-processing output with video-level summary metrics "
                "such as interaction time, interaction percentage, interaction count, "
                "interactions per hour, and average interaction duration."
            ),
        ),
        OutputFileDescription(
            "video_level_metrics_summary.csv",
            "Compact summary of the video-level metrics for quick review."
        ),
        OutputFileDescription(
            "predictions.csv",
            "Technical activity recognition prediction output before final post-processing.",
        ),
        OutputFileDescription(
            "predictions_summary.csv",
            "Technical summary of activity recognition predictions before final post-processing.",
        ),
        OutputFileDescription(
            "all_preds.pkl",
            (
                "Combined raw prediction file used internally "
                "by the Activity recognition (ADL) pipeline."
            ),
        ),
        OutputFileDescription(
            "frame_level_predictions.csv",
            (
                "Technical intermediate table with one row per analyzed frame or "
                "frame-level observations. Used to calculate video-level metrics."
            ),
        ),
        OutputFileDescription(
            "interaction_segments.csv",
            (
                "Technical intermediate table describing continuous interaction periods "
                "with start time, end time, and duration."
            )
        ),
        OutputFileDescription(
            "extracted_frames/",
            "Technical intermediate frames extracted from input video(s).",
        ),
        OutputFileDescription(
            "detic_outputs/",
            "Technical object detection outputs used by the Activity recognition (ADL) pipeline.",
        ),
        OutputFileDescription(
            "shan_outputs/",
            "Technical hand-object contact outputs generated on extracted frames.",
        ),
        *logs,
    ]

def output_preview_note(scenario: InputScenario) -> str:
    """ Return the scenario-specific note displayed below the output preview. """
    if scenario == "hand-object-single-image":
        return "Frame-level metrics are not generated for a single image."
    
    if scenario == "hand-object-image-directory":
        return (
            "Frame-level metrics are not generated for a general image folder "
            "unless the images are treated as an ordered frame sequence with "
            "known timestamps or FPS."
        )
    
    return (
        "Most users should review video_level_metrics.csv first. Frame-level outputs "
        "and raw model files are stored separately as technical intermediate files."
    )

def run_readme_text(*, model_id: str, context: OutputPreviewContext) -> str:
    """ Return a plain-language README for a run output folder. """
    main_files = (
        "visual_outputs/hand_object_contact/"
        if model_id == HAND_OBJECT_CONTACT_MODEL_ID
        else "results/video_level_metrics.csv"
    )
    
    descriptions = "\n".join(
        f"- {description.name}: {description.description}"
        for description in output_file_descriptions(context)
    )
    
    return (
        f"EgoModelKit Run Output\n"
        "\n"
        "This folder contains the files generated by one EgoModelKit run.\n"
        "\n"
        f"Recommended file or folder to review first: {main_files}\n"
        "\n"
        "File descriptions:\n"
        f"{descriptions}\n"
        "\n"
        "Please review these outputs according to your research team's approved workflow.\n"
    )

def finalize_runtime_outputs(
    *,
    layout: RunOutputLayout,
    model_id: str,
    input_path: Path,
    scenario: InputScenario,
) -> None:
    """ Move runtime outputs into the stable wireframe-aligned output contract. """
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        _organize_hand_object_runtime_outputs(layout)
        
        return

    if model_id == ADL_RECOGNITION_MODEL_ID:
        context = build_output_preview_context(
            model_id = model_id,
            input_path = input_path,
            output_root = layout.run_dir.parent,
            run_id = layout.run_dir.name,
        )

        _organize_adl_runtime_outputs(layout)
        
        write_stub_video_level_metric_files(
            layout = layout,
            context = context,
            status = "stub",
        )
        
        return

    raise ValueError(f"Unsupported model id: {model_id}")

def write_stub_video_level_metric_files(
    *,
    layout: RunOutputLayout,
    context: OutputPreviewContext,
    status: str,
) -> None:
    """ Write placeholder ADL video-level metrics until Bandini metrics are implemented. """
    layout.results_dir.mkdir(parents = True, exist_ok = True)
    layout.post_processing_dir.mkdir(parents = True, exist_ok = True)

    metric_rows = [
        {
            "input_name": input_name,
            "metric_status": status,
            "interaction_time_seconds": "",
            "interaction_percentage": "",
            "interaction_count": "",
            "interactions_per_hour": "",
            "average_interaction_duration_seconds": "",
            "notes": (
                "Stub metric row reserved for Bandini-style video-level "
                "hand-function metrics in a later commit."
            ),
        }
        for input_name in context.input_names
    ]

    _write_csv(
        layout.video_level_metrics_path,
        fieldnames = [
            "input_name",
            "metric_status",
            "interaction_time_seconds",
            "interaction_percentage",
            "interaction_count",
            "interactions_per_hour",
            "average_interaction_duration_seconds",
            "notes",
        ],
        rows = metric_rows,
    )

    _write_csv(
        layout.video_level_metrics_summary_path,
        fieldnames = ["metric", "metric_status", "notes"],
        rows = [
            {
                "metric": "video_level_hand_function_metrics",
                "metric_status": status,
                "notes": (
                    "Placeholder summary only; final Bandini-style metric "
                    "definitions will be implemented separately."
                ),
            }
        ],
    )

    _write_csv(
        layout.frame_level_predictions_path,
        fieldnames = ["input_name", "frame_index", "timestamp_seconds", "prediction_status"],
        rows = [
            {
                "input_name": input_name,
                "frame_index": "",
                "timestamp_seconds": "",
                "prediction_status": status,
            }
            for input_name in context.input_names
        ],
    )

    _write_csv(
        layout.interaction_segments_path,
        fieldnames = [
            "input_name",
            "segment_index",
            "start_time_seconds",
            "end_time_seconds",
            "duration_seconds",
            "segment_status",
        ],
        rows = [
            {
                "input_name": input_name,
                "segment_index": "",
                "start_time_seconds": "",
                "end_time_seconds": "",
                "duration_seconds": "",
                "segment_status": status,
            }
            for input_name in context.input_names
        ],
    )

def _organize_hand_object_runtime_outputs(layout: RunOutputLayout) -> None:
    visual_dir = layout.visual_outputs_dir / HAND_OBJECT_VISUAL_OUTPUT_DIRNAME
    visual_dir.mkdir(parents = True, exist_ok = True)
    layout.model_outputs_dir.mkdir(parents = True, exist_ok = True)

    for output_path in sorted(layout.run_dir.iterdir()):
        if not output_path.is_file():
            continue

        if output_path.stem.endswith("_det"):
            _move_path(output_path, visual_dir / output_path.name)
        elif _is_hand_object_model_output(output_path):
            _move_path(output_path, layout.model_outputs_dir / output_path.name)

    _remove_non_contract_root_outputs(layout)

def _organize_adl_runtime_outputs(layout: RunOutputLayout) -> None:
    layout.results_dir.mkdir(parents = True, exist_ok = True)
    layout.model_outputs_dir.mkdir(parents = True, exist_ok = True)
    layout.post_processing_dir.mkdir(parents = True, exist_ok = True)
    layout.intermediate_files_dir.mkdir(parents = True, exist_ok = True)

    runtime_work_dir = layout.run_dir / "adl_recognition_work"
    egoviz_data_dir = runtime_work_dir / "egoviz_data"

    _move_first_existing_file(
        sources = [
            layout.run_dir / "adl_predictions.csv",
        ],
        destination = layout.model_outputs_dir / "predictions.csv",
    )

    _move_first_existing_file(
        sources = [
            layout.run_dir / "adl_predictions_summary.csv",
        ],
        destination = layout.model_outputs_dir / "predictions_summary.csv",
    )

    _move_first_existing_file(
        sources = [
            layout.run_dir / "all_preds.pkl",
            egoviz_data_dir / "all_preds.pkl",
        ],
        destination = layout.model_outputs_dir / "all_preds.pkl",
    )

    staged_adl_dirs = _adl_runtime_staging_dirs(runtime_work_dir)

    _move_first_existing_child_dir(
        search_roots = staged_adl_dirs,
        dirnames = ["subclips"],
        destination = layout.adl_extracted_frames_dir,
    )

    _move_first_existing_child_dir(
        search_roots = staged_adl_dirs,
        dirnames = ["detic", "detic_raw"],
        destination = layout.adl_detic_outputs_dir,
    )

    _move_first_existing_child_dir(
        search_roots = staged_adl_dirs,
        dirnames = ["shan", "subclips_shan"],
        destination = layout.adl_shan_outputs_dir,
    )

    _remove_path(runtime_work_dir)
    _remove_non_contract_root_outputs(layout)

def _adl_runtime_staging_dirs(runtime_work_dir: Path) -> list[Path]:
    egoviz_data_dir = runtime_work_dir / "egoviz_data"
    known_demo_dir = egoviz_data_dir / "meal-preparation-cleanup"

    staging_dirs: list[Path] = []

    if known_demo_dir.is_dir():
        staging_dirs.append(known_demo_dir)

    if egoviz_data_dir.is_dir():
        staging_dirs.extend(
            child
            for child in sorted(egoviz_data_dir.iterdir())
            if child.is_dir() and child != known_demo_dir
        )

    staging_dirs.append(runtime_work_dir)

    return staging_dirs

def _move_first_existing_child_dir(
    *,
    search_roots: list[Path],
    dirnames: list[str],
    destination: Path,
) -> None:
    for search_root in search_roots:
        if not search_root.is_dir():
            continue

        for dirname in dirnames:
            candidate = search_root / dirname

            if candidate.is_dir():
                _replace_dir(candidate, destination)
                return

def _move_first_existing_file(*, sources: list[Path], destination: Path) -> None:
    for source in sources:
        if source.is_file():
            _move_path(source, destination)
            return

def _replace_dir(source: Path, destination: Path) -> None:
    _remove_path(destination)
    destination.parent.mkdir(parents = True, exist_ok = True)
    shutil.move(str(source), str(destination))

def _is_hand_object_model_output(output_path: Path) -> bool:
    protected_root_files = {
        RUN_SUMMARY_FILENAME,
        RUN_MANIFEST_FILENAME,
    }

    return (
        output_path.name not in protected_root_files
        and output_path.suffix.lower() in {".json", ".pkl", ".pickle"}
    )

def _move_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    _remove_path(destination)
    shutil.move(str(source), str(destination))

def _remove_path(path: Path) -> None:
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

def _remove_non_contract_root_outputs(layout: RunOutputLayout) -> None:
    contract_root_names = {
        RUN_README_FILENAME,
        RUN_SUMMARY_FILENAME,
        RUN_MANIFEST_FILENAME,
        "results",
        "visual_outputs",
        "technical",
        "logs",
    }

    for child in sorted(layout.run_dir.iterdir()):
        if child.name not in contract_root_names:
            _remove_path(child)

def _input_names_for_preview(*, model_id: str, input_path: Path) -> list[str]:
    if input_path.is_file():
        return [input_path.name]
    
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        suffixes = HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES
    elif model_id == ADL_RECOGNITION_MODEL_ID:
        suffixes = ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    
    names = sorted(
        child.name
        for child in input_path.iterdir()
        if child.is_file() and child.suffix.lower() in suffixes
    )
    
    if not names:
        return [input_path.name]
    
    return names

def _preview_items(items: list[str]) -> list[str]:
    if len(items) <= MAX_TREE_EXAMPLE_FILES:
        return items
    
    return [*items[:MAX_TREE_EXAMPLE_FILES], "..."]

def _stem(filename: str) -> str:
    return Path(filename).stem

def _display_output_root_name(output_root: Path) -> str:
    return output_root.name or str(output_root)

def _write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("w", encoding = "utf-8", newline = "") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent = 2, sort_keys = True) + "\n", encoding = "utf-8")

def write_run_summary(
    *,
    layout: RunOutputLayout,
    model_id: str,
    input_path: Path,
    scenario: InputScenario,
    status: str,
) -> None:
    """ Write the current run summary without recreating the scaffold. """
    _write_json(
        layout.run_summary_path,
        {
            "model_id": model_id,
            "input_name": input_path.name,
            "input_path": str(input_path),
            "output_folder": str(layout.run_dir),
            "scenario": scenario,
            "status": status,
        },
    )
