""" Local FastAPI backend used by the React EgoModelKit GUI. """

from __future__ import annotations

import platform
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
    AdlRecognitionInputError,
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_MODEL_ID,
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
    output_file_descriptions,
    output_folder_tree,
    output_preview_note,
    write_run_summary,
)
from egomodelkit.progress import ProgressEvent, write_progress_event
from egomodelkit.runtime.adl_recognition import (
    AdlRecognitionRuntimeError,
    run_adl_recognition,
)
from egomodelkit.runtime.commands import subprocess_runner
from egomodelkit.runtime.hand_object_contact import (
    HandObjectContactRuntimeError,
    run_hand_object_contact,
)
from egomodelkit.runtime.preflight import HostPrerequisiteError

GUI_LOCAL_SERVER_NAME: Final[str] = "127.0.0.1"
GUI_DEFAULT_SERVER_PORT: Final[int] = 7860
GUI_UPLOAD_CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024

GuiRunStatus = Literal["ready", "running", "completed", "failed"]
ProgressCallback = Callable[[str], None]
ModelRunner = Callable[[Path, Path, ProgressCallback], None]

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
    progress_events: list[ProgressEvent] = field(default_factory = list)
    lock: threading.Lock = field(default_factory = threading.Lock)

def create_app(
    *,
    static_dir: Path | None = None,
    hand_object_runner: ModelRunner | None = None,
    adl_runner: ModelRunner | None = None,
) -> FastAPI:
    """ Create the local FastAPI app used by the React GUI. 
    
    The optional runner arguments make endpoint tests fast and GPU-free.
    Production uses the existing EgoModelKit runtime functions.
    """
    app = FastAPI(title = "EgoModelKit Local GUI API")
    runs: dict[str, GuiRunState] = {}
    
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
                    "description": "Detect hands, objects, and hand-object contact in images.",
                    "acceptedInputLabel": "single image or folder of images",
                    "outputLabel": "detection visualizations and structured results",
                },
                {
                    "id": ADL_RECOGNITION_MODEL_ID,
                    "name": "Activity recognition (ADL)",
                    "description": (
                        "Process egocentric video clips for "
                        "activity of daily living (ADL) recognition"
                    ),
                    "acceptedInputLabel": "single MP4 video or folder of MP4 videos",
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
            
            _record_progress(
                state,
                ProgressEvent(stage = "setup", message = "Preparing input"),
            )
            
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
        
        webbrowser.open(state.layout.run_dir.resolve().as_uri())
        
        return {
            "opened": True,
            "runId": state.run_id,
            "outputFolder": str(state.layout.run_dir),
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
            _record_progress(
                state,
                ProgressEvent(stage = "runtime", message = message),
            )
        
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
        
        write_run_summary(
            layout = state.layout,
            model_id = state.model_id,
            input_path = state.input_path,
            scenario = state.scenario,
            status = "completed",
        )
        
        _record_progress(
            state,
            ProgressEvent(stage = "completed", message = "Run completed"),
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

def _progress_response(state: GuiRunState) -> dict[str, object]:
    """ Convert run state into a JSON-safe progress response. """
    with state.lock:
        status = state.status
        error_message = state.error_message
        events = list(state.progress_events)
    
    return {
        "runId": state.run_id,
        "status": status,
        "errorMessage": error_message,
        "outputFolder": str(state.layout.run_dir),
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
