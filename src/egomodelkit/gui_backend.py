""" Local FastAPI backend used by the React EgoModelKit GUI. """

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import tempfile
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Final, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_MODEL_ID,
    ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES,
    AdlRecognitionInputError,
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_MODEL_ID,
    HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES,
    HandObjectContactInputError,
    HandObjectContactRequest,
    validate_hand_object_contact_request,
)
from egomodelkit.output_contract import (
    InputScenario,
    RunOutputLayout,
    build_output_preview_context,
    build_output_preview_context_from_names,
    build_run_id,
    build_run_output_layout,
    create_output_scaffold,
    finalize_runtime_outputs,
    output_file_descriptions,
    output_folder_tree,
    output_preview_note,
    write_run_summary,
)
from egomodelkit.progress import (
    ExternalProgressUpdate,
    ProgressEvent,
    parse_external_progress_line,
    write_progress_event,
    write_runtime_log_line,
)
from egomodelkit.runtime.adl_recognition import (
    DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    AdlRecognitionRuntimeError,
    run_adl_recognition,
)
from egomodelkit.runtime.commands import (
    streaming_subprocess_runner,
    subprocess_runner,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    HandObjectContactRuntimeError,
    run_hand_object_contact,
)
from egomodelkit.runtime.preflight import (
    HostPrerequisiteError,
    ensure_host_runtime_ready,
)

GUI_LOCAL_SERVER_NAME: Final[str] = "127.0.0.1"
GUI_DEFAULT_SERVER_PORT: Final[int] = 7860
GUI_UPLOAD_CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024

GuiRunStatus = Literal["ready", "running", "completed", "failed"]
ProgressCallback = Callable[[str], None]
ModelRunner = Callable[[Path, Path, ProgressCallback], None]
RuntimeReadyChecker = Callable[[str, ProgressCallback], None]

GUI_REQUEST_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
    ValueError,
    HandObjectContactInputError,
    AdlRecognitionInputError,
    HostPrerequisiteError,
    HandObjectContactRuntimeError,
    AdlRecognitionRuntimeError
)

class OutputPreviewRequest(BaseModel):
    """ Request body for a browser-side output preview. """
    model_id: str = Field(alias = "modelId")
    input_names: list[str] = Field(alias = "inputNames", min_length = 1)
    output_root: str = Field(alias = "outputRoot")

class OpenOutputFolderRequest(BaseModel):
    """ Request body for opening a completed run folder. """
    run_id: str = Field(alias = "runId")

class SelectOutputFolderResponse(BaseModel):
    """ Response body for native output-folder selection. """
    output_root: str = Field(alias = "outputRoot")

@dataclass(frozen = True, slots = True)
class StagedInput:
    """ Temporary local copy of uploaded browser files. """
    root_dir: Path
    input_path: Path
    input_names: tuple[str, ...]

@dataclass(frozen = True, slots = True)
class RuntimeStatus:
    """ Temporary red-line Docker build status shown under Running model. """
    model_name: str
    current_step: int | None = None
    total_steps: int | None = None

@dataclass(frozen = True, slots = True)
class RuntimeBuildStage:
    """ One Docker image build counted as one progress-bar stage. """
    stage_id: str
    model_name: str
    current: int
    total: int

@dataclass(slots = True)
class GuiRunState:
    """ In-memory state for one local GUI run. """
    run_id: str
    model_id: str
    status: GuiRunStatus
    layout: RunOutputLayout
    scenario: InputScenario
    input_name: str
    input_path: Path
    staged_root: Path
    output_preview: dict[str, object]
    error_message: str | None = None
    runtime_status: RuntimeStatus | None = None
    active_runtime_stage_id: str | None = None
    runtime_build_stages: dict[str, RuntimeBuildStage] = field(default_factory=dict)
    progress_events: list[ProgressEvent] = field(default_factory=list)
    lock: threading.Lock = field(default_factory = threading.Lock)

def create_app(
    *,
    static_dir: Path | None = None,
    hand_object_runner: ModelRunner | None = None,
    adl_runner: ModelRunner | None = None,
    runtime_checker: RuntimeReadyChecker | None = None,
) -> FastAPI:
    """ Create the local FastAPI app used by the React GUI. 
    
    The optional runner arguments make endpoint tests fast and GPU-free.
    Production uses the existing EgoModelKit runtime functions.
    """
    app = FastAPI(title = "EgoModelKit Local GUI API")
    runs: dict[str, GuiRunState] = {}
    runtime_ready_checker = runtime_checker or _check_runtime_ready_for_gui
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins = [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials = False,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )
    
    @app.get("/api/models")
    def models() -> dict[str, object]:
        """ Return GUI-displayable model choices. """
        return {
            "models": [
                {
                    "id": HAND_OBJECT_CONTACT_MODEL_ID,
                    "name": "Hand-object contact",
                    "description": "Detects hands, objects, and hand-object contact in images.",
                    "acceptedInputLabel": "single image or multiple images",
                    "supportedInputExtensions": sorted(
                        HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES,
                    ),
                    "outputLabel": "detection visualizations and structured results",
                },
                {
                    "id": ADL_RECOGNITION_MODEL_ID,
                    "name": "Activity recognition (ADL)",
                    "description": (
                        "Processes egocentric video clips for "
                        "activity of daily living (ADL) recognition."
                    ),
                    "supportedInputExtensions": sorted(
                        ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES,
                    ),
                    "acceptedInputLabel": "single MP4 video or multiple MP4 videos",
                    "outputLabel": "predictions and processed frame-level files"
                },
            ]
        }
    
    @app.post("/api/output-preview")
    def output_preview(request: OutputPreviewRequest) -> dict[str, object]:
        """ Return a dynamic folder preview before files are uploaded. """
        try:
            output_root = _normalize_output_root(request.output_root)
            
            input_names = tuple(
                _safe_upload_filename(input_name)
                for input_name in request.input_names
            )
            
            run_id = _build_unique_run_id(output_root, runs)
            
            context = build_output_preview_context_from_names(
                model_id = request.model_id,
                input_names = input_names,
                output_root = output_root,
                run_id = run_id,
            )
            
            return _output_preview_response(context)
        except GUI_REQUEST_EXCEPTIONS as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from exc
    
    @app.post("/api/dry-run")
    async def dry_run(
        model_id: Annotated[str, Form(alias = "modelId")],
        output_root_text: Annotated[str, Form(alias = "outputRoot")],
        files: Annotated[list[UploadFile], File()],
    ) -> dict[str, object]:
        """ Validate uploaded files and output folder without running a model. """
        staged = await _stage_uploaded_files(files)
        
        try:
            output_root = _normalize_output_root(output_root_text)
            
            _validate_gui_request(
                model_id = model_id,
                input_path = staged.input_path,
                output_root = output_root,
            )
            
            runtime_ready_checker(model_id, _ignore_progress)
            
            run_id = _build_unique_run_id(output_root, runs)
            layout = build_run_output_layout(output_root, run_id = run_id)
            
            context = build_output_preview_context(
                model_id = model_id,
                input_path = staged.input_path,
                output_root = output_root,
                run_id = run_id,
            )

            return {
                "runId": run_id,
                "status": "ready",
                "scenario": context.scenario,
                "summary": {
                    "modelId": model_id,
                    "model": _model_display_name(model_id),
                    "input": _input_label(staged.input_names),
                    "outputFolder": str(layout.run_dir),
                    "status": "Ready",
                },
                "outputPreview": _output_preview_response(context),
            }
        except GUI_REQUEST_EXCEPTIONS as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from exc
        finally:
            shutil.rmtree(staged.root_dir, ignore_errors = True)
    
    @app.post("/api/runs")
    async def start_run(
        model_id: Annotated[str, Form(alias = "modelId")],
        output_root_text: Annotated[str, Form(alias = "outputRoot")],
        files: Annotated[list[UploadFile], File()],
    ) -> dict[str, object]:
        """ Start a model run and return immediately with a run id. """
        staged = await _stage_uploaded_files(files)
        
        try:
            output_root = _normalize_output_root(output_root_text)
            
            _validate_gui_request(
                model_id = model_id,
                input_path = staged.input_path,
                output_root = output_root,
            )
            
            run_id = _build_unique_run_id(output_root, runs)
            layout = build_run_output_layout(output_root, run_id = run_id)
            
            context = build_output_preview_context(
                model_id = model_id,
                input_path = staged.input_path,
                output_root = output_root,
                run_id = run_id,
            )
            
            create_output_scaffold(
                layout = layout,
                model_id = model_id,
                input_path = staged.input_path,
                scenario = context.scenario,
                status = "running",
            )
            
            state = GuiRunState(
                run_id = run_id,
                model_id = model_id,
                status = "running",
                layout = layout,
                scenario = context.scenario,
                input_name = _input_label(staged.input_names),
                input_path = staged.input_path,
                staged_root = staged.root_dir,
                output_preview = _output_preview_response(context),
            )
            
            runs[run_id] = state
            
            _initialize_wireframe_progress(state)
            
            thread = threading.Thread(
                target = _execute_run,
                kwargs = {
                    "state": state,
                    "hand_object_runner": hand_object_runner,
                    "adl_runner": adl_runner,
                },
                daemon = True
            )
            
            thread.start()
            
            return {
                "runId": run_id,
                "status": "running",
                "scenario": context.scenario,
                "summary": {
                    "modelId": model_id,
                    "model": _model_display_name(model_id),
                    "input": state.input_name,
                    "outputFolder": str(layout.run_dir),
                    "status": "Running",
                },
                "outputPreview": state.output_preview,
            }
        except GUI_REQUEST_EXCEPTIONS as exc:
            shutil.rmtree(staged.root_dir, ignore_errors = True)
            raise HTTPException(status_code = 400, detail = str(exc)) from exc
    
    @app.get("/api/runs/{run_id}/progress")
    def run_progress(run_id: str) -> dict[str, object]:
        """ Return current progress for one GUI run. """
        state = runs.get(run_id)
        
        if state is None:
            raise HTTPException(status_code = 404, detail = "Run was not found.")
        
        return _progress_response(state)
    
    @app.post("/api/open-output-folder")
    def open_output_folder(request: OpenOutputFolderRequest) -> dict[str, object]:
        """ Open the output folder for a run created by this GUI session. """
        state = runs.get(request.run_id)
        
        if state is None:
            raise HTTPException(status_code = 404, detail = "Run was not found.")
        
        if not state.layout.run_dir.exists():
            raise HTTPException(
                status_code = 404,
                detail = "Output folder does not exist yet.",
            )
        
        webbrowser.open(state.layout.output_folder_path.resolve().as_uri())
        
        return {
            "opened": True,
            "runId": state.run_id,
            "outputFolder": str(state.layout.output_folder_path),
        }
    
    @app.post("/api/select-output-folder")
    def select_output_folder() -> dict[str, str]:
        """Open a local native folder picker when the host platform supports it."""
        output_root = _select_output_folder()

        if output_root is None:
            raise HTTPException(
                status_code = 404,
                detail = (
                    "Native output folder picker is not available on this host. "
                    "Use the manual output path fallback."
                ),
            )

        return {"outputRoot": output_root}
    
    if static_dir is not None and static_dir.exists():
        app.mount(
            "/",
            StaticFiles(directory = str(static_dir), html = True),
            name = "egomodelkit-gui",
        )
    
    return app

def _execute_run(
    *,
    state: GuiRunState,
    hand_object_runner: ModelRunner | None,
    adl_runner: ModelRunner | None,
) -> None:
    """ Execute one run in a background thread. """
    try:
        def progress(message: str) -> None:
            update = parse_external_progress_line(message)

            if update is not None:
                _record_external_progress_update(state, update)
                return

            _record_runtime_output(state, message)
        
        if state.model_id == HAND_OBJECT_CONTACT_MODEL_ID:
            runner = (
                hand_object_runner
                if hand_object_runner is not None
                else _run_hand_object_contact_for_gui
            )
        elif state.model_id == ADL_RECOGNITION_MODEL_ID:
            runner = (
                adl_runner
                if adl_runner is not None
                else _run_adl_recognition_for_gui
            )
        else:
            raise ValueError(f"Unsupported model id: {state.model_id}")

        runner(state.input_path, state.layout.run_dir, progress)

        finalize_runtime_outputs(
            layout = state.layout,
            model_id = state.model_id,
            input_path = state.input_path,
            scenario = state.scenario,
        )
        
        _record_final_output_progress(state)
        
        _finish_runtime_build_stage(state)
        
        write_run_summary(
            layout = state.layout,
            model_id = state.model_id,
            input_path = state.input_path,
            scenario = state.scenario,
            status = "completed",
        )
        
        with state.lock:
            state.status = "completed"
    except GUI_REQUEST_EXCEPTIONS as exc:
        write_run_summary(
            layout = state.layout,
            model_id = state.model_id,
            input_path = state.input_path,
            scenario = state.scenario,
            status = "failed",
        )
        
        _record_progress(
            state,
            ProgressEvent(stage = "failed", message = str(exc))
        )
        
        with state.lock:
            state.runtime_status = None
            state.status = "failed"
            state.error_message = str(exc)
    finally:
        shutil.rmtree(state.staged_root, ignore_errors = True)
        
def _run_hand_object_contact_for_gui(
    input_path: Path,
    output_dir: Path,
    progress: Callable[[str], None],
) -> None:
    """ Run the existing hand-object-contact runtime. """
    run_hand_object_contact(
        HandObjectContactRequest(input_path = input_path, output_dir = output_dir),
        command_runner = subprocess_runner,
        streaming_command_runner = streaming_subprocess_runner,
        progress = progress,
    )

def _run_adl_recognition_for_gui(
    input_path: Path,
    output_dir: Path,
    progress: Callable[[str], None],
) -> None:
    """ Run the existing ADL-recognition runtime. """
    run_adl_recognition(
        AdlRecognitionRequest(input_path = input_path, output_dir = output_dir),
        command_runner = subprocess_runner,
        streaming_command_runner = streaming_subprocess_runner,
        progress = progress,
    )

def _ignore_progress(_: str) -> None:
    """ Default no-op progress reporter for GUI runtime checks. """

def _check_runtime_ready_for_gui(
    model_id: str,
    progress: ProgressCallback = _ignore_progress,
) -> None:
    """ Validate that the host can run packaged GPU model containers. """
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        docker_executable = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC.docker_executable
    elif model_id == ADL_RECOGNITION_MODEL_ID:
        docker_executable = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC.docker_executable
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

    ensure_host_runtime_ready(
        docker_executable = docker_executable,
        command_runner = subprocess_runner,
        require_linux_nvidia_gpu = True,
        progress = progress,
    )

def _validate_gui_request(
    *,
    model_id: str,
    input_path: Path,
    output_root: Path,
) -> None:
    """ Validate GUI input through the existing model validators. """
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        validate_hand_object_contact_request(
            HandObjectContactRequest(
                input_path = input_path,
                output_dir = output_root,
            ),
        )
        
        return

    if model_id == ADL_RECOGNITION_MODEL_ID:
        validate_adl_recognition_request(
            AdlRecognitionRequest(
                input_path = input_path,
                output_dir = output_root,
            ),
        )

        return

    raise ValueError(f"Unsupported model id: {model_id}")

async def _stage_uploaded_files(files: list[UploadFile]) -> StagedInput:
    """ Copy uploaded browser files into a temporary local staging folder. """
    if not files:
        raise ValueError("Choose an input file or folder before continuing.")
    
    root_dir = Path(tempfile.mkdtemp(prefix = "egomodelkit-gui-input-"))
    saved_paths: list[Path] = []
    
    try:
        for uploaded_file in files:
            filename = _safe_upload_filename(uploaded_file.filename)
            destination = _unique_destination_path(root_dir, filename)
            
            with destination.open("wb") as stream:
                while True:
                    chunk = await uploaded_file.read(GUI_UPLOAD_CHUNK_SIZE_BYTES)
                    
                    if not chunk:
                        break
                    
                    stream.write(chunk)
            
            await uploaded_file.close()
            saved_paths.append(destination)
        
        if not saved_paths:
            raise ValueError("Choose an input file or folder before continuing.")
        
        input_path = saved_paths[0] if len(saved_paths) == 1 else root_dir
        
        return StagedInput(
            root_dir = root_dir,
            input_path = input_path,
            input_names = tuple(path.name for path in saved_paths)
        )
    except Exception:
        shutil.rmtree(root_dir, ignore_errors = True)
        raise

def _record_progress(state: GuiRunState, event: ProgressEvent) -> None:
    """ Add one progress event to memory and disk. """
    with state.lock:
        state.progress_events.append(event)
    
    write_progress_event(state.layout.progress_log_path, event)

def _initialize_wireframe_progress(state: GuiRunState) -> None:
    """ Create the exact visible running-card rows from the approved wireframes. """
    for event in _initial_wireframe_events(state):
        _upsert_progress_event(state, event)

def _initial_wireframe_events(state: GuiRunState) -> list[ProgressEvent]:
    if state.scenario == "hand-object-single-image":
        return [
            ProgressEvent(stage = "prepare_input", message = "Preparing image input..."),
            ProgressEvent(stage = "check_input", message = "Checking selected image..."),
            ProgressEvent(
                stage="run_hand_object",
                message="Running hand-object contact model on the image: waiting",
            ),
            ProgressEvent(
                stage = "save_outputs", 
                message = "Saving detection outputs: waiting",
            ),
        ]

    if state.scenario == "hand-object-image-directory":
        return [
            ProgressEvent(stage = "prepare_input", message = "Preparing image inputs..."),
            ProgressEvent(stage = "check_input", message = "Checking images: waiting"),
            ProgressEvent(
                stage="run_hand_object", 
                message = "Running hand-object contact model on the images: waiting",
            ),
            ProgressEvent(
                stage = "save_outputs", 
                message = "Saving detection outputs: waiting",
            ),
        ]

    if state.scenario == "adl-single-video":
        return [
            ProgressEvent(stage = "prepare_input", message = "Preparing video input..."),
            ProgressEvent(stage = "extract_frames", message = "Extracting frames: waiting"),
            ProgressEvent(
                stage = "run_detic",
                message="Running object detection model on extracted frames: waiting",
            ),
            ProgressEvent(
                stage="run_hand_object", 
                message="Running hand-object contact on extracted frames: waiting",
            ),
            ProgressEvent(
                stage = "combine_predictions", 
                message = "Combining predictions: waiting"),
            ProgressEvent(
                stage = "calculate_metrics",
                message = "Calculating video-level summary metrics: waiting",
            ),
            ProgressEvent(stage = "save_outputs", message = "Saving outputs: waiting"),
        ]

    if state.scenario == "adl-video-directory":
        return [
            ProgressEvent(stage = "prepare_input", message = "Preparing video inputs..."),
            ProgressEvent(stage = "check_input", message = "Checking video: waiting"),
            ProgressEvent(
                stage="extract_frames", 
                message="Extracting frames across all videos: waiting"
            ),
            ProgressEvent(
                stage = "run_detic", 
                message = "Running object detection model: waiting"
            ),
            ProgressEvent(
                stage = "run_hand_object",
                message = "Running hand-object contact on extracted frames: waiting",
            ),
            ProgressEvent(
                stage = "combine_predictions", 
                message = "Combining predictions: waiting"),
            ProgressEvent(
                stage = "calculate_metrics",
                message = "Calculating video-level summary metrics: waiting",
            ),
            ProgressEvent(stage = "save_outputs", message = "Saving outputs: waiting"),
        ]

    if state.scenario == "adl-combined-predictions":
        return [
            ProgressEvent(
                stage = "prepare_input", 
                message = "Preparing combined predictions input..."),
            ProgressEvent(
                stage = "combine_predictions", 
                message = "Combining predictions: waiting"),
            ProgressEvent(
                stage = "calculate_metrics",
                message = "Calculating video-level summary metrics: waiting",
            ),
            ProgressEvent(stage = "save_outputs", message = "Saving outputs: waiting"),
        ]

    raise ValueError(f"Unsupported progress scenario: {state.scenario}")

def _record_external_progress_update(
    state: GuiRunState,
    update: ExternalProgressUpdate,
) -> None:
    """ Map in-container progress updates to exact wireframe visible rows. """
    payload = update.payload

    if update.kind in {"hand_object_images_discovered", "hand_object_images_checked"}:
        total = _payload_int(payload, "total")

        if state.scenario == "hand-object-single-image":
            _upsert_progress_event(
                state,
                ProgressEvent(stage = "check_input", message = "Checking selected image..."),
            )
        elif state.scenario == "hand-object-image-directory":
            _upsert_progress_event(
                state,
                ProgressEvent(
                    stage = "check_input",
                    message = "Checking images",
                    current = total,
                    total = total,
                    unit = "valid images",
                ),
            )

        return

    if update.kind == "hand_object_image_processed":
        current = _payload_int(payload, "current")
        total = _payload_int(payload, "total")

        if state.model_id == HAND_OBJECT_CONTACT_MODEL_ID:
            if state.scenario == "hand-object-single-image":
                _upsert_progress_event(
                    state,
                    ProgressEvent(
                        stage = "run_hand_object",
                        message = "Running hand-object contact model on the image",
                        current = current,
                        total = total,
                        unit = "image processed",
                    ),
                )
            else:
                _upsert_progress_event(
                    state,
                    ProgressEvent(
                        stage = "run_hand_object",
                        message = "Running hand-object contact model on the images",
                        current = current,
                        total = total,
                        unit = "images processed",
                    ),
                )
        else:
            _upsert_progress_event(
                state,
                ProgressEvent(
                    stage = "run_hand_object",
                    message = "Running hand-object contact on extracted frames",
                    current = current,
                    total = total,
                    unit = "frames",
                ),
            )

        return

    if update.kind == "hand_object_output_saved":
        current = _payload_int(payload, "current")
        total = _payload_int(payload, "total")

        if state.scenario == "hand-object-image-directory":
            _upsert_progress_event(
                state,
                ProgressEvent(
                    stage = "save_outputs",
                    message = "Saving detection outputs",
                    current = current,
                    total = total,
                    unit = "images",
                ),
            )

        return

    if update.kind == "adl_video_checked":
        current = _payload_int(payload, "current")
        total = _payload_int(payload, "total")

        if state.scenario == "adl-video-directory":
            _upsert_progress_event(
                state,
                ProgressEvent(
                    stage = "check_input",
                    message = "Checking videos",
                    current = current,
                    total = total,
                    unit = "valid videos",
                ),
            )

        return

    if update.kind == "adl_frame_extracted":
        current = _payload_int(payload, "current")
        total = _payload_int(payload, "total")

        message = (
            "Extracting frames across all videos"
            if state.scenario == "adl-video-directory"
            else "Extracting frames"
        )

        _upsert_progress_event(
            state,
            ProgressEvent(
                stage = "extract_frames",
                message = message,
                current = current,
                total = total,
                unit = "frames",
            ),
        )

        return

    if update.kind == "detic_frame_processed":
        current = _payload_int(payload, "current")
        total = _payload_int(payload, "total")

        message = (
            "Running object detection model"
            if state.scenario == "adl-video-directory"
            else "Running object detection model on extracted frames"
        )

        _upsert_progress_event(
            state,
            ProgressEvent(
                stage = "run_detic",
                message = message,
                current = current,
                total = total,
                unit = "frames",
            ),
        )

        return

    if update.kind == "adl_prediction_frames_discovered":
        _upsert_progress_event(
            state,
            ProgressEvent(
                stage = "combine_predictions",
                message = "Combining predictions: waiting",
            ),
        )

        return

    if update.kind == "adl_prediction_frame_processed":
        current = _payload_int(payload, "current")
        total = _payload_int(payload, "total")

        _upsert_progress_event(
            state,
            ProgressEvent(
                stage = "combine_predictions",
                message = "Combining predictions",
                current = current,
                total = total,
                unit = "frames",
            ),
        )

        return

    if update.kind == "adl_predictions_combining":
        _upsert_progress_event(
            state,
            ProgressEvent(
                stage = "combine_predictions",
                message = "Combining predictions: waiting",
            ),
        )

        return

    if update.kind == "adl_predictions_combined":
        return

def _display_video_name_from_payload(payload: dict[str, object]) -> str:
    """ Return a clean user-facing video label. """
    display_video = payload.get("displayVideo")

    if isinstance(display_video, str) and display_video:
        return display_video

    video = str(payload.get("video", "unknown"))

    path = Path(video)
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name

    return f"{stem.rstrip('.')}{suffix}"

def _stage_has_numeric_progress(state: GuiRunState, stage: str) -> bool:
    """ Return true when a visible stage already has real current/total progress. """
    with state.lock:
        return any(
            event.stage == stage
            and event.current is not None
            and event.total is not None
            and event.total > 0
            for event in state.progress_events
        )
        
def _record_final_output_progress(state: GuiRunState) -> None:
    """ Mark final wireframe rows complete once output finalization is reached. """
    if state.model_id == ADL_RECOGNITION_MODEL_ID:
        return

    if not _stage_has_numeric_progress(state, "save_outputs"):
        _upsert_progress_event(
            state,
            ProgressEvent(
                stage="save_outputs",
                message="Saving detection outputs",
                current=1,
                total=1,
            ),
        )

def _record_runtime_output(state: GuiRunState, message: str) -> None:
    """ Write raw runtime output to runtime.log and update Docker build state. """
    write_runtime_log_line(state.layout.runtime_log_path, message)
    _update_docker_build_progress(state, message)

def _update_docker_build_progress(state: GuiRunState, message: str) -> None:
    """ Track Docker image builds as simple equal-weight progress stages. """
    model_name = _docker_model_name_from_message(message)
    normalized = message.lower()

    if model_name is not None and "missing; preparing it now" in normalized:
        _start_runtime_build_stage(state, model_name)
        
        return

    if model_name is not None and "runtime image is already available" in normalized:
        with state.lock:
            state.runtime_status = None
            state.active_runtime_stage_id = None
            
        return

    if model_name is not None and "runtime image is ready" in normalized:
        _finish_runtime_build_stage(state)
        
        return

    docker_step = _docker_build_step_counts(message)

    if docker_step is not None:
        current, total = docker_step
        
        _update_active_runtime_build_stage(
            state,
            current=current,
            total=total,
        )

def _start_runtime_build_stage(state: GuiRunState, model_name: str) -> None:
    """ Start counting one Docker image build as one progress stage. """
    stage_id = f"docker:{model_name}"

    with state.lock:
        state.active_runtime_stage_id = stage_id
        
        state.runtime_status = RuntimeStatus(
            model_name = model_name,
            current_step = None,
            total_steps = None,
        )
        
        state.runtime_build_stages[stage_id] = RuntimeBuildStage(
            stage_id = stage_id,
            model_name = model_name,
            current = 0,
            total = 1,
        )

def _update_active_runtime_build_stage(
    state: GuiRunState,
    *,
    current: int,
    total: int,
) -> None:
    """ Update the active Docker build without allowing progress to go backward. """
    with state.lock:
        stage_id = state.active_runtime_stage_id

        if stage_id is None:
            return

        existing = state.runtime_build_stages[stage_id]
        next_total = max(existing.total, total)
        next_current = max(existing.current, min(current, next_total))

        state.runtime_build_stages[stage_id] = RuntimeBuildStage(
            stage_id = stage_id,
            model_name = existing.model_name,
            current = next_current,
            total = next_total,
        )
        
        state.runtime_status = RuntimeStatus(
            model_name = existing.model_name,
            current_step = next_current,
            total_steps = next_total,
        )

def _finish_runtime_build_stage(state: GuiRunState) -> None:
    """ Mark the active Docker build stage complete and clear the red line. """
    with state.lock:
        stage_id = state.active_runtime_stage_id

        if stage_id is not None:
            existing = state.runtime_build_stages[stage_id]
            
            state.runtime_build_stages[stage_id] = RuntimeBuildStage(
                stage_id = stage_id,
                model_name = existing.model_name,
                current = existing.total,
                total = existing.total,
            )

        state.runtime_status = None
        state.active_runtime_stage_id = None

def _docker_model_name_from_message(message: str) -> str | None:
    """ Extract the Docker image's readable model name from EgoModelKit logs. """
    normalized = message.lower()

    if "hand-object-contact runtime image" in normalized:
        return "hand-object-detector"

    if "adl-recognition core runtime image" in normalized:
        return "EgoVizML"

    if "adl-recognition detic runtime image" in normalized:
        return "Detic"

    return None

def _docker_build_step_counts(message: str) -> tuple[int, int] | None:
    """ Extract Dockerfile step counts from BuildKit or classic Docker output. """
    buildkit_match = re.search(r"\[\s*(\d+)/(\d+)\]", message)

    if buildkit_match is not None:
        return int(buildkit_match.group(1)), int(buildkit_match.group(2))

    classic_match = re.search(r"Step\s+(\d+)/(\d+)", message)

    if classic_match is not None:
        return int(classic_match.group(1)), int(classic_match.group(2))

    return None

def _stage_from_external_progress_kind(kind: str) -> str | None:
    """ Return the visible stage updated by one external progress kind. """
    if kind in {"hand_object_images_discovered", "hand_object_images_checked"}:
        return "check_input"

    if kind == "hand_object_image_processed":
        return "run_hand_object"

    if kind == "hand_object_output_saved":
        return "save_outputs"

    if kind == "adl_video_checked":
        return "check_input"

    if kind == "adl_frame_extracted":
        return "extract_frames"

    if kind == "detic_frame_processed":
        return "run_detic"

    if kind in {
        "adl_prediction_frames_discovered",
        "adl_prediction_frame_processed",
        "adl_predictions_combining",
        "adl_predictions_combined",
    }:
        return "combine_predictions"

    return None

def _upsert_progress_event(state: GuiRunState, event: ProgressEvent) -> None:
    """ Insert or replace one visible progress row by stage without going backward. """
    with state.lock:
        for index, existing_event in enumerate(state.progress_events):
            if existing_event.stage == event.stage:
                state.progress_events[index] = _merge_progress_event(
                    existing_event,
                    event,
                )
                
                break
        else:
            state.progress_events.append(event)

    write_progress_event(state.layout.progress_log_path, event)

def _merge_progress_event(
    existing_event: ProgressEvent,
    next_event: ProgressEvent,
) -> ProgressEvent:
    """ Keep the newest message, but never reduce numeric progress. """
    if (
        existing_event.current is None
        or existing_event.total is None
        or next_event.current is None
        or next_event.total is None
    ):
        return next_event

    if existing_event.total != next_event.total:
        return next_event

    return ProgressEvent(
        stage = next_event.stage,
        message = next_event.message,
        current = max(existing_event.current, next_event.current),
        total = next_event.total,
        unit = next_event.unit,
    )

def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str) and value.isdigit():
        return int(value)

    return 0

def _progress_response(state: GuiRunState) -> dict[str, object]:
    """ Convert run state into a JSON-safe progress response. """
    with state.lock:
        status = state.status
        error_message = state.error_message
        events = list(state.progress_events)
        runtime_status = state.runtime_status
        runtime_build_stages = list(state.runtime_build_stages.values())
    
    return {
        "runId": state.run_id,
        "status": status,
        "errorMessage": error_message,
        "outputFolder": str(state.layout.output_folder_path),
        "events": [
            {
                "stage": event.stage,
                "message": event.message,
                "current": event.current,
                "total": event.total,
                "unit": event.unit,
                "displayText": event.display_text,
            }
            for event in events
        ],
        "runtimeStatus": (
            None
            if runtime_status is None
            else {
                "modelName": runtime_status.model_name,
                "currentStep": runtime_status.current_step,
                "totalSteps": runtime_status.total_steps,
            }
        ),
        "runtimeBuildStages": [
            {
                "stageId": stage.stage_id,
                "modelName": stage.model_name,
                "current": stage.current,
                "total": stage.total,
            }
            for stage in runtime_build_stages
        ],
        "outputPreview": state.output_preview,
    }

def _output_preview_response(context) -> dict[str, object]:
    """ Convert an output preview context into a JSON-safe response. """
    return {
        "runId": context.run_id,
        "scenario": context.scenario,
        "folderTree": output_folder_tree(context),
        "note": output_preview_note(context.scenario),
        "files": [
            {
                "name": item.name,
                "description": item.description,
            }
            for item in output_file_descriptions(context)
        ],
    }

def _normalize_output_root(output_root_text: str) -> Path:
    """ Return a validated output root path. """
    if output_root_text is None or not output_root_text.strip():
        raise ValueError("Choose an output folder before continuing.")
    
    return Path(output_root_text).expanduser()

def _build_unique_run_id(output_root: Path, runs: dict[str, GuiRunState]) -> str:
    """ Build a neutral run id that does not collide in memory or on disk. """
    base_run_id = build_run_id()
    
    for index in range(1000):
        run_id = base_run_id if index == 0 else f"{base_run_id}-{index + 1:03d}"
        layout = build_run_output_layout(output_root, run_id = run_id)
        
        if run_id not in runs and not layout.run_dir.exists():
            return run_id
    
    raise ValueError("Unable to create a unique run id.")

def _safe_upload_filename(filename: str | None) -> str:
    """ Return a safe basename for an uploaded browser file. """
    name = Path(filename or "").name
    
    if not name or name in {".", ".."}:
        raise ValueError("Uploaded file is missing a valid name.")
    
    return name

def _unique_destination_path(directory: Path, filename: str) -> Path:
    """ Return a non-conflicting staged file path. """
    destination = directory / filename
    
    if not destination.exists():
        return destination
    
    stem = destination.stem
    suffix = destination.suffix
    index = 2
    
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        
        if not candidate.exists():
            return candidate
        
        index += 1

def _input_label(input_names: tuple[str, ...]) -> str:
    """ Return a concise label for uploaded input files. """
    if not input_names:
        return "No input selected"
    
    if len(input_names) == 1:
        return input_names[0]
    
    return f"{len(input_names)} files"

def _model_display_name(model_id: str) -> str:
    """ Return the GUI display name for a supported model. """
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        return "Hand-object contact"
    
    if model_id == ADL_RECOGNITION_MODEL_ID:
        return "Activity recognition (ADL)"
    
    raise ValueError(f"Unsupported model id: {model_id}")

def _select_output_folder() -> str | None:
    """ Return a user-selected output folder using a local native picker when possible. """
    system_name = platform.system()

    if system_name == "Darwin":
        return _select_output_folder_macos()

    if system_name == "Windows":
        return _select_output_folder_windows()

    return _select_output_folder_tkinter()


def _select_output_folder_macos() -> str | None:
    """ Return a folder selected through macOS Finder. """
    script = (
        'POSIX path of (choose folder '
        'with prompt "Choose an EgoModelKit output folder")'
    )

    completed = subprocess.run(
        ["osascript", "-e", script],
        check = False,
        capture_output = True,
        text = True,
    )

    if completed.returncode != 0:
        return None

    selected_path = completed.stdout.strip()

    return selected_path or None


def _select_output_folder_windows() -> str | None:
    """ Return a folder selected through the Windows folder picker. """
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "Choose an EgoModelKit output folder"
$dialog.ShowNewFolderButton = $true
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
"""

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check = False,
        capture_output = True,
        text = True,
    )

    if completed.returncode != 0:
        return None

    selected_path = completed.stdout.strip()

    return selected_path or None


def _select_output_folder_tkinter() -> str | None:
    """ Return a folder selected through Tkinter on Linux/other desktop hosts. """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        selected_path = filedialog.askdirectory(
            title = "Choose an EgoModelKit output folder",
            mustexist = False,
        )

        root.destroy()

        return selected_path or None
    except Exception:
        return None
