from __future__ import annotations

import asyncio
import builtins
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

import egomodelkit.gui_backend as gui_backend
from egomodelkit.gui_backend import (
    CancelableGuiOperation,
    GuiRunState,
    ProgressCallback,
    _build_unique_run_id,
    _cancel_operation_for_request,
    _check_runtime_ready_for_gui,
    _execute_run,
    _input_label,
    _model_display_name,
    _normalize_output_root,
    _open_output_folder,
    _register_operation,
    _run_adl_recognition_for_gui,
    _run_hand_object_contact_for_gui,
    _safe_upload_filename,
    _select_output_folder,
    _select_output_folder_macos,
    _select_output_folder_tkinter,
    _select_output_folder_windows,
    _should_run_start_preflight,
    _stage_uploaded_files,
    _unique_destination_path,
    _validate_existing_output_root,
    _validate_gui_request,
    create_app,
)
from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_MODEL_ID,
)
from egomodelkit.models.hand_interaction import (
    HAND_INTERACTION_MODEL_ID,
    HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES,
)
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.output_contract import build_run_output_layout, create_output_scaffold
from egomodelkit.progress import ExternalProgressUpdate, ProgressEvent
from egomodelkit.runtime.commands import CommandCancelledError, ProcessCancellation
from egomodelkit.runtime.hand_object_contact import HandObjectContactRuntimeError


def _progress_line(kind: str, **payload: object) -> str:
    import json

    return "EGOMODELKIT_PROGRESS " + json.dumps(
        {
            "kind": kind,
            **payload,
        },
        sort_keys = True,
    )

def _ready_runtime_checker(
    model_id: str,
    progress: ProgressCallback,
    command_runner,
) -> None:
    progress(f"Runtime ready for {model_id}.")

def _failing_runtime_checker(
    model_id: str,
    progress: ProgressCallback,
    command_runner,
) -> None:
    progress(f"Runtime unavailable for {model_id}.")
    
    raise gui_backend.HostPrerequisiteError(
        "EgoModelKit model runs require a Linux host with an NVIDIA GPU; "
        "detected Darwin."
    )

def _write_adl_input_manifest(
    output_dir: Path,
    input_names: tuple[str, ...] = ("clip.mp4",),
) -> None:
    rows = [
        "session_id,session_sort_index,input_name,staged_video_name,"
        "staged_video_stem,input_modified_time",
    ]

    for index, input_name in enumerate(input_names, start = 1):
        rows.append(
            f"session001,{index},{input_name},video{index:03d}.MP4,"
            f"video{index:03d},2026-07-05T10:{index:02d}:00+00:00"
        )

    (output_dir / "adl_input_manifest.csv").write_text(
        "\n".join(rows) + "\n",
        encoding = "utf-8",
    )
    
    (output_dir / "adl_segment_manifest.csv").write_text(
        "session_id,source_video,staged_video_stem,segment_name,segment_index,"
        "start_time_seconds,end_time_seconds,valid_duration_seconds,"
        "segment_length_seconds,inference_frame_fps,subclip_encoding_fps\n"
        "session001,clip.mp4,video001,video001_001,1,0.0,60.0,60.0,60,1,10\n",
        encoding = "utf-8",
    )


def _write_adl_result_files(output_dir: Path) -> None:
    (output_dir / "adl_segment_predictions.csv").write_text(
        "session_id,source_video,segment_name,segment_index,start_time_seconds,"
        "end_time_seconds,valid_duration_seconds,predicted_adl\n"
        "session001,clip.mp4,video001_001,1,0.0,60.0,60.0,"
        "Meal Preparation and Cleanup\n",
        encoding = "utf-8",
    )
    (output_dir / "adl_video_summary.csv").write_text(
        "session_id,source_video,segment_count,total_valid_duration_seconds,"
        "predicted_adl_segment_counts\n"
        "session001,clip.mp4,1,60.0,"
        "{\"Meal Preparation and Cleanup\": 1}\n",
        encoding = "utf-8",
    )
    (output_dir / "adl_session_summary.csv").write_text(
        "session_id,source_video_count,segment_count,total_valid_duration_seconds,"
        "predicted_adl_segment_counts\n"
        "session001,1,1,60.0,"
        "{\"Meal Preparation and Cleanup\": 1}\n",
        encoding = "utf-8",
    )

def _existing_output_root(tmp_path: Path) -> Path:
    """ Create and return an output root for GUI tests that should pass validation. """
    output_root = tmp_path / "results"
    output_root.mkdir(exist_ok = True)

    return output_root

def _test_run_state(
    tmp_path: Path,
    *,
    model_id: str = HAND_OBJECT_CONTACT_MODEL_ID,
    scenario: str = "hand-object-single-image",
) -> GuiRunState:
    return GuiRunState(
        run_id="run-test",
        model_id=model_id,
        status="running",
        layout=build_run_output_layout(tmp_path / "results", run_id="run-test"),
        scenario=scenario,  # type: ignore[arg-type]
        input_name="input",
        input_path=tmp_path / "input",
        staged_root=tmp_path / "staged",
        operation_id="operation-test",
        cancellation=ProcessCancellation(),
        output_preview={},
    )

def test_models_endpoint_returns_only_public_video_models() -> None:
    client = TestClient(create_app())

    response = client.get("/api/models")

    assert response.status_code == 200

    body = response.json()
    assert [model["id"] for model in body["models"]] == [
        HAND_INTERACTION_MODEL_ID,
        ADL_RECOGNITION_MODEL_ID,
    ]
    assert all(model["supportedInputExtensions"] == [".mp4"] for model in body["models"])

def test_output_preview_endpoint_returns_dynamic_tree(tmp_path: Path) -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/output-preview",
        json = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "inputNames": ["frame.jpg"],
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
    )
    
    assert response.status_code == 200
    
    body = response.json()
    
    assert body["scenario"] == "hand-object-single-image"
    assert "frame_det.png" in body["folderTree"]
    assert body["files"]

def test_dry_run_validates_uploaded_file(tmp_path: Path) -> None:
    client = TestClient(create_app(runtime_checker = _ready_runtime_checker))
    
    response = client.post(
        "/api/dry-run",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [
            ("files", ("frame.jpg", b"fake-image", "image/jpeg")),  
        ],
    )
    
    assert response.status_code == 200
    
    body = response.json()
    
    assert body["status"] == "ready"
    assert body["scenario"] == "hand-object-single-image"
    assert body["summary"]["model"] == "Hand-object contact"

def test_register_operation_uses_provided_operation_id() -> None:
    operations = {}

    operation = _register_operation(operations, "operation-1")

    assert operation.operation_id == "operation-1"
    assert operations["operation-1"] is operation

def test_cancel_operation_for_request_cancels_operation_id() -> None:
    operations = {}
    runs = {}
    operation = _register_operation(operations, "operation-1")

    response = _cancel_operation_for_request(
        run_id = None,
        operation_id = "operation-1",
        operations = operations,
        runs = runs,
    )

    assert response["cancelled"] is True
    assert operation.cancellation.is_cancelled()

def test_cancel_operation_for_request_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match = "No active run or operation was found"):
        _cancel_operation_for_request(
            run_id = None,
            operation_id = "missing-operation",
            operations = {},
            runs = {},
        )

def test_run_endpoint_uses_injected_runner_without_docker(tmp_path: Path) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        assert input_path.exists()
        assert output_dir.exists()
        
        progress("Fake model step")
        (output_dir / "frame_det.png").write_bytes(b"visual")
        (output_dir / "frame_shan.json").write_text("{}", encoding = "utf-8")
        (output_dir / "frame_shan.pkl").write_bytes(b"pickle")

    client = TestClient(
        create_app(hand_object_runner = fake_runner),
    )
    
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path))
        },
        files = [
            ("files", ("frame.jpg", b"fake-image", "image/jpeg")),
        ],
    )
    
    assert start_response.status_code == 200
    
    run_id = start_response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)
    
    assert progress_body["status"] == "completed"
    
    visible_lines = [event["displayText"] for event in progress_body["events"]]
    assert "Fake model step" not in visible_lines

    runtime_log = (
        Path(progress_body["outputFolder"])
        / "logs"
        / "runtime.log"
    )

    assert "Fake model step" in runtime_log.read_text(encoding="utf-8")
    
    run_dir = Path(progress_body["outputFolder"])

    assert (
        run_dir / "visual_outputs" / "hand_object_contact" / "frame_det.png"
    ).read_bytes() == b"visual"

    assert (
        run_dir / "technical" / "model_outputs" / "frame_shan.json"
    ).read_text(encoding = "utf-8") == "{}"

    assert (
        run_dir / "technical" / "model_outputs" / "frame_shan.pkl"
    ).read_bytes() == b"pickle"

    assert not (run_dir / "frame_det.png").exists()
    assert not (run_dir / "frame_shan.json").exists()
    assert not (run_dir / "frame_shan.pkl").exists()

def test_run_endpoint_organizes_hand_object_runtime_outputs(tmp_path: Path) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        assert output_dir.is_dir()
        
        progress("Fake hand-object model step")
        
        (output_dir / "frame_det.png").write_bytes(b"visual")
        (output_dir / "frame_shan.json").write_text("{}", encoding = "utf-8")
        (output_dir / "frame_shan.pkl").write_bytes(b"pickle")
        (output_dir / "frame_extra.json").write_text("{}", encoding = "utf-8")

    client = TestClient(create_app(hand_object_runner = fake_runner))

    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    assert start_response.status_code == 200

    run_id = start_response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert progress_body["status"] == "completed"

    run_dir = Path(progress_body["outputFolder"])

    assert (
        run_dir / "visual_outputs" / "hand_object_contact" / "frame_det.png"
    ).read_bytes() == b"visual"

    assert (
        run_dir / "technical" / "model_outputs" / "frame_shan.json"
    ).read_text(encoding = "utf-8") == "{}"

    assert (
        run_dir / "technical" / "model_outputs" / "frame_shan.pkl"
    ).read_bytes() == b"pickle"
    
    assert (
        run_dir / "technical" / "model_outputs" / "frame_extra.json"
    ).exists()

    assert not (run_dir / "frame_det.png").exists()
    assert not (run_dir / "frame_shan.json").exists()
    assert not (run_dir / "frame_shan.pkl").exists()
    assert not (run_dir / "frame_extra.json").exists()

def test_run_endpoint_writes_adl_predictions_and_normalized_outputs(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        assert output_dir.name.startswith("run-")
        
        progress("Fake ADL model step")
        
        _write_adl_result_files(output_dir)

        (output_dir / "all_preds.pkl").write_bytes(b"pickle")
        
        runtime_adl_dir = (
            output_dir
            / "adl_recognition_work"
            / "egoviz_data"
            / "meal-preparation-cleanup"
        )
        
        (runtime_adl_dir / "subclips" / "clip_001").mkdir(parents = True)
        (runtime_adl_dir / "subclips" / "clip_001" / "frame_001.jpg").write_bytes(b"jpg")

        (runtime_adl_dir / "detic").mkdir(parents = True)
        (runtime_adl_dir / "detic" / "clip_001_frame_001_detic.pkl").write_bytes(b"detic")

        (runtime_adl_dir / "shan").mkdir(parents = True)
        (runtime_adl_dir / "shan" / "clip_001_frame_001_shan.pkl").write_bytes(b"shan")
        
        _write_adl_input_manifest(output_dir)

    client = TestClient(
        create_app(
            adl_runner = fake_runner,
            runtime_checker = _ready_runtime_checker,
        )
    )

    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("clip.mp4", b"fake-video", "video/mp4"))],
    )

    assert start_response.status_code == 200

    run_id = start_response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert progress_body["status"] == "completed"

    run_dir = Path(progress_body["outputFolder"])

    adl_config = json.loads(
        (
            run_dir
            / "technical"
            / "post_processing"
            / "adl_processing_config.json"
        ).read_text(encoding = "utf-8")
    )
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding = "utf-8"))

    assert adl_config["segment_length_seconds"] == 60
    assert adl_config["inference_frame_fps"] == 1
    assert adl_config["subclip_encoding_fps"] == 10
    assert adl_config["frame_resize"] is None
    assert "dominant_hand" not in run_manifest["model_configuration"]

    assert (run_dir / "results" / "adl_segment_predictions.csv").exists()
    assert (run_dir / "results" / "adl_video_summary.csv").exists()
    assert (run_dir / "results" / "adl_session_summary.csv").exists()
    assert (run_dir / "technical" / "model_outputs" / "all_preds.pkl").exists()
    assert (
        run_dir / "technical" / "model_outputs" / "adl_input_manifest.csv"
    ).exists()
    assert (
        run_dir / "technical" / "post_processing" / "adl_segment_manifest.csv"
    ).exists()

    assert (
        run_dir
        / "technical"
        / "intermediate_files"
        / "extracted_frames"
        / "clip_001"
        / "frame_001.jpg"
    ).exists()

    assert (
        run_dir
        / "technical"
        / "intermediate_files"
        / "detic_outputs"
        / "clip_001_frame_001_detic.pkl"
    ).exists()

    assert (
        run_dir
        / "technical"
        / "intermediate_files"
        / "shan_outputs"
        / "clip_001_frame_001_shan.pkl"
    ).exists()

    assert not (
        run_dir / "technical" / "post_processing" / "frame_level_predictions.csv"
    ).exists()
    assert not (
        run_dir / "technical" / "post_processing" / "interaction_segments.csv"
    ).exists()
    assert not (run_dir / "results" / "video_level_metrics.csv").exists()
    assert not (run_dir / "results" / "session_level_metrics.csv").exists()
    assert not (run_dir / "technical" / "post_processing" / "metrics_config.json").exists()

    assert not (run_dir / "adl_segment_predictions.csv").exists()
    assert not (run_dir / "adl_video_summary.csv").exists()
    assert not (run_dir / "adl_session_summary.csv").exists()
    assert not (run_dir / "all_preds.pkl").exists()
    assert not (run_dir / "adl_recognition_work").exists()

def test_cancel_run_endpoint_cancels_running_state(tmp_path: Path) -> None:
    block_runner = threading.Event()

    def slow_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress("Fake model started.")
        block_runner.wait(timeout = 2)

    client = TestClient(create_app(hand_object_runner = slow_runner))

    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
            "operationId": "operation-run-1",
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    assert start_response.status_code == 200
    
    run_id = start_response.json()["runId"]

    cancel_response = client.post(
        "/api/cancel-run",
        json = {"runId": run_id, "operationId": "operation-run-1"},
    )

    assert cancel_response.status_code == 200
    assert cancel_response.json()["cancelled"] is True

    progress_body = client.get(f"/api/runs/{run_id}/progress").json()

    assert progress_body["status"] == "cancelled"
    assert progress_body["errorMessage"] == "Run was cancelled."

    run_dir = tmp_path / "results" / run_id
    runtime_log = run_dir / "logs" / "runtime.log"
    progress_log = run_dir / "logs" / "progress.jsonl"

    runtime_text = runtime_log.read_text(encoding = "utf-8")
    progress_text = progress_log.read_text(encoding = "utf-8")

    assert f"Run {run_id} was cancelled by the user." in runtime_text
    assert "Backend worker thread signalled for cancellation:" in runtime_text
    assert "no active subprocess was running at the time of cancellation" in runtime_text
    assert '"stage": "cancelled"' in progress_text
    assert '"message": "Run cancelled by user."' in progress_text

    block_runner.set()

def test_open_output_folder_uses_tracked_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    opened_folders: list[Path] = []

    def fake_open_output_folder(output_folder: Path) -> bool:
        opened_folders.append(output_folder)

        return True
    
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress,
    ) -> None:
        progress("Fake model step")
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend._open_output_folder",
        fake_open_output_folder,
    )
        
    client = TestClient(
        create_app(hand_object_runner = fake_runner),
    )
    
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [
            ("files", ("frame.jpg", b"fake-image", "image/jpeg")),
        ],
    )
    
    assert start_response.status_code == 200
    
    run_id = start_response.json()["runId"]
    _wait_for_run_completion(client, run_id)
    
    open_response = client.post(
        "/api/open-output-folder",
        json = {"runId": run_id},
    )
    
    assert open_response.status_code == 200
    assert open_response.json()["opened"] is True

    expected_run_dir = tmp_path / "results" / run_id

    assert opened_folders == [expected_run_dir]
    assert open_response.json()["outputFolder"] == str(expected_run_dir)
    
def test_open_output_folder_rejects_unknown_run() -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/open-output-folder",
        json = {"runId": "run-does-not-exist"}
    )
    
    assert response.status_code == 404

def _wait_for_run_completion(
    client: TestClient,
    run_id: str
) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/runs/{run_id}/progress")
        
        assert response.status_code == 200
        
        body = response.json()
        
        if body["status"] in {"completed", "failed"}:
            return body
        
        time.sleep(0.01)

    raise AssertionError("Run did not complete.")

def test_select_output_folder_returns_empty_path_when_picker_is_cancelled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "egomodelkit.gui_backend._select_output_folder",
        lambda: None,
    )

    client = TestClient(create_app())

    response = client.post("/api/select-output-folder")

    assert response.status_code == 200
    assert response.json() == {"outputRoot": ""}


def test_select_output_folder_returns_503_when_picker_cannot_start(monkeypatch) -> None:
    def unavailable_picker() -> None:
        raise gui_backend.NativeOutputFolderPickerError("picker failed")

    monkeypatch.setattr(
        "egomodelkit.gui_backend._select_output_folder",
        unavailable_picker,
    )

    client = TestClient(create_app())

    response = client.post("/api/select-output-folder")

    assert response.status_code == 503
    assert response.json() == {"detail": "picker failed"}

def test_select_output_folder_returns_selected_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "egomodelkit.gui_backend._select_output_folder",
        lambda: "/tmp/EgoModelKit Results",
    )

    client = TestClient(create_app())

    response = client.post("/api/select-output-folder")

    assert response.status_code == 200
    assert response.json() == {"outputRoot": "/tmp/EgoModelKit Results"}

def test_output_preview_endpoint_reports_bad_request(tmp_path: Path) -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/output-preview",
        json = {
            "modelId": "unknown-model",
            "inputNames": ["frame.jpg"],
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
    )
    
    assert response.status_code == 400
    assert "Unsupported model id" in response.json()["detail"]

def test_dry_run_endpoint_checks_local_runtime_before_reporting_ready(
    tmp_path: Path,
) -> None:
    checked_models: list[str] = []

    def runtime_checker(model_id: str, progress: ProgressCallback, command_runner) -> None:
        checked_models.append(model_id)
        progress("Runtime check finished.")

    client = TestClient(create_app(runtime_checker = runtime_checker))

    response = client.post(
        "/api/dry-run",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    assert response.status_code == 200
    assert checked_models == [HAND_OBJECT_CONTACT_MODEL_ID]
    
def test_dry_run_endpoint_fails_when_runtime_check_fails(tmp_path: Path) -> None:
    client = TestClient(create_app(runtime_checker = _failing_runtime_checker))

    response = client.post(
        "/api/dry-run",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    assert response.status_code == 400
    assert "Linux host with an NVIDIA GPU" in response.json()["detail"]

def test_default_gui_runtime_check_requires_linux_nvidia_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_ready(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(gui_backend, "ensure_host_runtime_ready", fake_ready)

    _check_runtime_ready_for_gui(HAND_OBJECT_CONTACT_MODEL_ID, lambda _message: None)
    _check_runtime_ready_for_gui(ADL_RECOGNITION_MODEL_ID, lambda _message: None)

    assert [call["require_linux_nvidia_gpu"] for call in calls] == [True, True]
    assert [call["docker_executable"] for call in calls] == ["docker", "docker"]

def test_default_gui_runtime_check_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match = "Unsupported model id"):
        _check_runtime_ready_for_gui("unknown", lambda _message: None)

def test_run_endpoint_reports_runtime_check_failure_gracefully(tmp_path: Path) -> None:
    def failing_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress("Checking host runtime prerequisites.")
        
        raise gui_backend.HostPrerequisiteError(
            "EgoModelKit model runs require a Linux host with an NVIDIA GPU; "
            "detected Darwin."
        )

    client = TestClient(create_app(hand_object_runner = failing_runner))

    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    assert start_response.status_code == 200

    run_id = start_response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert progress_body["status"] == "failed"
    assert "Linux host with an NVIDIA GPU" in progress_body["errorMessage"]
    assert progress_body["runtimeStatus"] is None

def test_dry_run_endpoint_reports_validation_error(tmp_path: Path) -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/dry-run",
        data = {
              "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
              "outputRoot": "   ",
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )
    
    assert response.status_code == 400
    assert "Choose an output folder" in response.json()["detail"]

def test_run_endpoint_reports_validation_error_and_cleans_staging(tmp_path: Path) -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/runs",
        data = {
            "modelId": "unknown-model",
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )
    
    assert response.status_code == 400
    assert "Unsupported model id" in response.json()["detail"]

def test_progress_endpoint_rejects_unknown_run() -> None:
    client = TestClient(create_app())
    
    response = client.get("/api/runs/run-does-not-exist/progress")
    
    assert response.status_code == 404

def test_open_output_folder_rejects_missing_output_folder(tmp_path: Path) -> None:
    def fake_runner(input_path: Path, output_dir: Path, progress) -> None:
        progress("done")
    
    client = TestClient(create_app(hand_object_runner = fake_runner))
    
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )
    
    run_id = start_response.json()["runId"]
    
    progress_body = _wait_for_run_completion(client, run_id)

    output_folder = Path(progress_body["outputFolder"])
    
    import shutil
    
    shutil.rmtree(output_folder)
    
    response = client.post("/api/open-output-folder", json = {"runId": run_id})
    
    assert response.status_code == 404
    assert "Output folder does not exist" in response.json()["detail"]

def test_create_app_mounts_static_directory(tmp_path: Path) -> None:
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>GUI</html>", encoding = "utf-8")
    
    client = TestClient(create_app(static_dir = static_dir))
    
    response = client.get("/")
    
    assert response.status_code == 200
    assert "GUI" in response.text

def test_run_endpoint_uses_injected_adl_runner(tmp_path: Path) -> None:
    def fake_runner(input_path: Path, output_dir: Path, progress) -> None:
        assert input_path.suffix == ".mp4"
        assert output_dir.exists()
        
        progress("ADL step")

        _write_adl_result_files(output_dir)
        _write_adl_input_manifest(output_dir)
    
    client = TestClient(
        create_app(
            adl_runner = fake_runner,
            runtime_checker = _ready_runtime_checker,
        )
    )

    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("clip.mp4", b"fake-video", "video/mp4"))],
    )
    
    assert start_response.status_code == 200
    
    run_id = start_response.json()["runId"]
    
    for _ in range(100):
        body = client.get(f"/api/runs/{run_id}/progress").json()
        
        if body["status"] == "completed":
            break
    else:
        raise AssertionError("ADL run did not complete")
    
    visible_lines = [event["displayText"] for event in body["events"]]
    
    assert "ADL step" not in visible_lines

    runtime_log = (
        Path(body["outputFolder"])
        / "logs"
        / "runtime.log"
    )

    progress_log = (
        Path(body["outputFolder"])
        / "logs"
        / "progress.jsonl"
    )

    assert "ADL step" in runtime_log.read_text(encoding="utf-8")
    assert "ADL step" not in progress_log.read_text(encoding="utf-8")

def test_execute_run_records_failure_and_cleans_staged_root(tmp_path: Path) -> None:
    input_path = tmp_path / "frame.jpg"
    input_path.write_bytes(b"fake-image")
    
    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    
    layout = build_run_output_layout(tmp_path / "results", run_id = "run-failed")
    
    create_output_scaffold(
        layout = layout,
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        input_path = input_path,
        scenario = "hand-object-single-image",
    )
    
    state = GuiRunState(
        run_id = "run-failed",
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        status = "running",
        layout = layout,
        scenario = "hand-object-single-image",
        input_name = "frame.jpg",
        input_path = input_path,
        staged_root = staged_root,
        operation_id="operation-test",
        cancellation=ProcessCancellation(),
        output_preview = {}
    )
    
    def failing_runner(input_path: Path, output_dir: Path, progress) -> None:
        raise HandObjectContactRuntimeError("simulated failure")
    
    _execute_run(
        state=state,
        hand_object_runner=failing_runner,
        adl_runner=None,
        operations={},
    )

    assert state.status == "failed"
    assert state.error_message == "simulated failure"
    assert not staged_root.exists()
    assert any(event.stage == "failed" for event in state.progress_events)

def test_execute_run_rejects_unsupported_model_id(tmp_path: Path) -> None:
    input_path = tmp_path / "frame.jpg"
    input_path.write_bytes(b"fake-image")
    
    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    
    layout = build_run_output_layout(tmp_path / "results", run_id = "run-unsupported")
    layout.run_dir.mkdir(parents = True)
    
    state = GuiRunState(
        run_id = "run-supported",
        model_id = "unknown-model",
        status = "running",
        layout = layout,
        scenario = "hand-object-single-image",
        input_name = "frame.jpg",
        input_path = input_path,
        staged_root = staged_root,
        operation_id="operation-test",
        cancellation=ProcessCancellation(),
        output_preview = {},
    )
    
    def failing_runner(input_path: Path, output_dir: Path, progress) -> None:
        raise HandObjectContactRuntimeError("Unsupported model id: unknown-model")
    
    _execute_run(
        state = state,
        hand_object_runner = failing_runner,
        adl_runner = None,
        operations = {},
    )

    assert state.status == "failed"
    assert state.error_message == "Unsupported model id: unknown-model"
    
def test_runtime_wrappers_delegate_to_existing_runners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    
    def fake_hand_object_runner(
       request,
        *,
        command_runner,
        streaming_command_runner,
        progress,
    ) -> None:
        captured["hand_request"] = request
        captured["hand_command_runner"] = command_runner
        captured["hand_streaming_command_runner"] = streaming_command_runner

        progress("hand progress")
    
    def fake_adl_runner(
        request,
        *,
        command_runner,
        streaming_command_runner,
        progress,
    ) -> None:
        captured["adl_request"] = request
        captured["adl_command_runner"] = command_runner
        captured["adl_streaming_command_runner"] = streaming_command_runner

        progress("adl progress")
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend.run_hand_object_contact",
        fake_hand_object_runner,
    )

    monkeypatch.setattr(
        "egomodelkit.gui_backend.run_adl_recognition",
        fake_adl_runner,
    )
    
    messages: list[str] = []
    
    _run_hand_object_contact_for_gui(
        tmp_path / "frame.jpg",
        tmp_path / "out",
        messages.append,
        ProcessCancellation(),
    )
    
    _run_adl_recognition_for_gui(
        tmp_path / "clip.mp4",
        tmp_path / "out",
        messages.append,
        ProcessCancellation(),
    )
    
    assert captured["hand_request"].input_path == tmp_path / "frame.jpg"
    assert captured["adl_request"].input_path == tmp_path / "clip.mp4"
    assert messages == ["hand progress", "adl progress"]
    assert not hasattr(captured["adl_request"], "dominant_hand")

def test_validate_gui_request_accepts_adl(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    
    _validate_gui_request(
        model_id = ADL_RECOGNITION_MODEL_ID,
        input_path = video_path,
        output_root = tmp_path / "results"
    )

def test_stage_uploaded_files_handles_duplicate_names() -> None:
    async def run() -> None:
        files = [
            UploadFile(filename = "frame.jpg", file = __import__("io").BytesIO(b"one")),
            UploadFile(filename = "frame.jpg", file = __import__("io").BytesIO(b"two")),
        ]
        
        staged = await _stage_uploaded_files(files)
        
        try:
            assert staged.input_path == staged.root_dir
            assert staged.input_names == ("frame.jpg", "frame_2.jpg")
            assert (staged.root_dir / "frame.jpg").read_bytes() == b"one"
            assert (staged.root_dir / "frame_2.jpg").read_bytes() == b"two"
        finally:
            import shutil
            
            shutil.rmtree(staged.root_dir, ignore_errors = True)
    
    asyncio.run(run())

def test_stage_uploaded_files_rejects_empty_file_list() -> None:
    with pytest.raises(ValueError, match = "Choose an input file"):
        asyncio.run(_stage_uploaded_files([]))

def test_stage_uploaded_files_cleans_up_after_bad_filename(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    
    with pytest.raises(ValueError, match = "valid name"):
        asyncio.run(
            _stage_uploaded_files(
                [
                    UploadFile(filename = "", file = __import__("io").BytesIO())
                ]
            )
        )
        
    assert set(tmp_path.iterdir()) == before
    
def test_normalize_output_root_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match = "Choose an output folder"):
        _normalize_output_root(" \n\t ")

def test_validate_existing_output_root_rejects_missing_folder(tmp_path: Path) -> None:
    missing_output_root = tmp_path / "missing-results"

    with pytest.raises(ValueError, match = "Output folder does not exist"):
        _validate_existing_output_root(missing_output_root)

def test_validate_existing_output_root_rejects_file(tmp_path: Path) -> None:
    output_file = tmp_path / "results.txt"
    output_file.write_text("not a folder", encoding = "utf-8")

    with pytest.raises(ValueError, match = "Output path exists but is not a folder"):
        _validate_existing_output_root(output_file)

def test_validate_existing_output_root_accepts_existing_folder(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir()

    _validate_existing_output_root(output_root)

def test_build_unique_run_id_uses_suffix_for_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("egomodelkit.gui_backend.build_run_id", lambda: "run-fixed")
    
    (tmp_path / "run-fixed").mkdir()
    
    assert _build_unique_run_id(tmp_path, {}) == "run-fixed-002"

def test_build_unique_run_id_reports_exhausted_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExistingRunDir:
        def exists(self) -> bool:
            return True

    class ExistingLayout:
        run_dir = ExistingRunDir()
        
    monkeypatch.setattr("egomodelkit.gui_backend.build_run_id", lambda: "run-fixed")
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend.build_run_output_layout",
        lambda output_root, *, run_id: ExistingLayout(),
    )
    
    with pytest.raises(ValueError, match = "Unable to create a unique run id"):
        _build_unique_run_id(tmp_path, {})

def test_safe_upload_filename_rejects_missing_name() -> None:
    with pytest.raises(ValueError, match = "valid name"):
        _safe_upload_filename(None)

def test_unique_destination_path_skips_existing_numbered_names(tmp_path: Path) -> None:
    (tmp_path / "frame.jpg").write_bytes(b"one")
    (tmp_path / "frame_2.jpg").write_bytes(b"two")
    
    assert _unique_destination_path(tmp_path, "frame.jpg") == tmp_path / "frame_3.jpg"

def test_input_label_handles_empty_and_multiple_names() -> None:
    assert _input_label(()) == "No input selected"
    assert _input_label(("one.jpg", "two.jpg")) == "2 files"

def test_model_display_name_accepts_adl_and_rejects_unknown() -> None:
    assert _model_display_name(ADL_RECOGNITION_MODEL_ID) == "Activity recognition (ADL)"
    
    with pytest.raises(ValueError, match = "Unsupported model id"):
        _model_display_name("unknown")

def test_select_output_folder_dispatches_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("egomodelkit.gui_backend._is_wsl", lambda: False)
    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("egomodelkit.gui_backend._select_output_folder_macos", lambda: "/mac")
    assert _select_output_folder() == "/mac"
    
    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Windows")
    monkeypatch.setattr("egomodelkit.gui_backend._select_output_folder_windows", lambda: "C:/out")
    assert _select_output_folder() == "C:/out"
    
    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Linux")
    monkeypatch.setattr("egomodelkit.gui_backend._select_output_folder_linux", lambda: "/linux")
    assert _select_output_folder() == "/linux"

    monkeypatch.setattr("egomodelkit.gui_backend._is_wsl", lambda: True)
    assert _select_output_folder() == "C:/out"

def test_select_output_folder_macos_handles_success_cancel_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 0, stdout = "/tmp/out\n"),
    )
    
    assert _select_output_folder_macos() == "/tmp/out"

    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 0, stdout = "\n"),
    )
    
    assert _select_output_folder_macos() is None
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 1, stdout = ""),
    )
    
    assert _select_output_folder_macos() is None

def test_select_output_folder_windows_handles_success_cancel_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(
        "egomodelkit.gui_backend._resolve_windows_powershell_executable",
        lambda: "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    )
    monkeypatch.setattr("egomodelkit.gui_backend._is_wsl", lambda: True)
    monkeypatch.setattr(
        "egomodelkit.gui_backend.Path.is_dir",
        lambda self: str(self) == "/mnt/c/Windows/System32",
    )

    def successful_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode = 0, stdout = "C:/out\n", stderr = "")

    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        successful_run,
    )

    assert _select_output_folder_windows() == "C:/out"
    assert commands[0][0][0] == (
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    assert "-STA" in commands[0][0]
    assert "-EncodedCommand" in commands[0][0]
    assert commands[0][1]["cwd"] == "/mnt/c/Windows/System32"

    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 2, stdout = "", stderr = ""),
    )

    assert _select_output_folder_windows() is None

    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 1, stdout = "", stderr = ""),
    )

    with pytest.raises(
        gui_backend.NativeOutputFolderPickerError,
        match = "could not be opened",
    ):
        _select_output_folder_windows()


def test_select_output_folder_windows_reports_missing_powershell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "egomodelkit.gui_backend._resolve_windows_powershell_executable",
        lambda: None,
    )

    with pytest.raises(
        gui_backend.NativeOutputFolderPickerError,
        match = "could not locate Windows PowerShell",
    ):
        _select_output_folder_windows()

def test_select_output_folder_tkinter_handles_missing_tkinter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    
    def fake_import(name, *args, **kwargs):
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError("missing tkinter")
        
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    
    assert _select_output_folder_tkinter() is None

def test_select_output_folder_tkinter_handles_success_cancel_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tkinter_module = ModuleType("tkinter")
    filedialog_module = ModuleType("tkinter.filedialog")
    
    class FakeRoot:
        def withdraw(self) -> None:
            pass
    
        def attributes(self, name: str, value: bool) -> None:
            assert name == "-topmost"
            assert value is True
        
        def destroy(self) -> None:
            pass
    
    tkinter_module.Tk = FakeRoot
    filedialog_module.askdirectory = lambda **kwargs: "/tmp/out"
    tkinter_module.filedialog = filedialog_module
    
    monkeypatch.setitem(sys.modules, "tkinter", tkinter_module)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog_module)
    
    assert _select_output_folder_tkinter() == "/tmp/out"
    
    filedialog_module.askdirectory = lambda **kwargs: ""
    
    assert _select_output_folder_tkinter() is None
    
    class ExplodingRoot:
        def __init__(self) -> None:
            raise RuntimeError("no display")
    
    tkinter_module.Tk = ExplodingRoot
    
    assert _select_output_folder_tkinter() is None

def test_stage_uploaded_files_rejects_truthy_empty_file_iterable() -> None:
    class TruthyEmptyUploads(list):
        def __bool__(self) -> bool:
            return True
    
    with pytest.raises(ValueError, match="Choose an input file"):
        asyncio.run(_stage_uploaded_files(TruthyEmptyUploads()))

def test_run_endpoint_reports_wireframe_hand_object_single_image_progress(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress(_progress_line("hand_object_images_discovered", current = 1, total = 1))
        progress(_progress_line("hand_object_image_processed", current = 1, total = 1))
        progress(_progress_line("hand_object_output_saved", current = 1, total = 1))
        (output_dir / "frame_det.png").write_bytes(b"visual")

    client = TestClient(create_app(hand_object_runner = fake_runner))

    response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    run_id = response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert [event["displayText"] for event in progress_body["events"]] == [
        "Preparing image input...",
        "Checking selected image...",
        "Running hand-object contact model on the image: 1 / 1 image processed",
        "Saving detection outputs: 1 / 1",
    ]

def test_run_endpoint_reports_wireframe_hand_object_image_set_progress(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress(_progress_line("hand_object_images_discovered", current = 25, total = 25))
        progress(_progress_line("hand_object_image_processed", current = 25, total = 25))
        progress(_progress_line("hand_object_output_saved", current = 21, total = 25))
        (output_dir / "frame_det.png").write_bytes(b"visual")

    client = TestClient(create_app(hand_object_runner = fake_runner))

    response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [
            ("files", (f"frame-{index:03d}.jpg", b"fake-image", "image/jpeg"))
            for index in range(25)
        ],
    )

    run_id = response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert [event["displayText"] for event in progress_body["events"]] == [
        "Preparing image inputs...",
        "Checking images: 25 / 25 valid images",
        "Running hand-object contact model on the images: 25 / 25 images processed",
        "Saving detection outputs: 21 / 25 images",
    ]

def test_run_endpoint_reports_wireframe_adl_single_video_progress(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress(_progress_line("adl_frame_extracted", current = 1200, total = 1200))
        progress(_progress_line("detic_frame_processed", current = 1200, total = 1200))
        progress(_progress_line("hand_object_image_processed", current = 1200, total = 1200))
        progress(_progress_line("adl_prediction_frames_discovered", current = 1200, total = 1200))
        progress(_progress_line("adl_prediction_frame_processed", current = 1200, total = 1200))
        progress(_progress_line("adl_predictions_combined"))
        
        _write_adl_result_files(output_dir)
        _write_adl_input_manifest(output_dir, ("participant-session-01.mp4",))

    client = TestClient(
        create_app(
            adl_runner = fake_runner,
            runtime_checker = _ready_runtime_checker,
        )
    )

    response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("participant-session-01.mp4", b"fake-video", "video/mp4"))],
    )

    run_id = response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert [event["displayText"] for event in progress_body["events"]] == [
        "Preparing video input...",
        "Extracting frames: 1,200 / 1,200 frames",
        "Running object detection model on extracted frames: 1,200 / 1,200 frames",
        "Running hand-object contact on extracted frames: 1,200 / 1,200 frames",
        "Combining predictions: 1,200 / 1,200 frames",
        "Building ADL video and session summaries: waiting",
        "Saving outputs: waiting",
    ]

def test_run_endpoint_reports_wireframe_adl_video_directory_progress(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress(
            _progress_line(
                "adl_video_checked",
                current = 5,
                total = 5,
                video = "participant-session-03.mp4",
            )
        )
        progress(_progress_line("adl_frame_extracted", current = 3600, total = 6000))
        progress(_progress_line("detic_frame_processed", current = 3050, total = 6000))
        
        _write_adl_result_files(output_dir)
        _write_adl_input_manifest(
            output_dir,
            tuple(f"participant-session-{index:02d}.mp4" for index in range(1, 6)),
        )

    client = TestClient(
        create_app(
            adl_runner = fake_runner,
            runtime_checker = _ready_runtime_checker,
        )
    )

    response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [
            ("files", (f"participant-session-{index:02d}.mp4", b"fake-video", "video/mp4"))
            for index in range(1, 6)
        ],
    )

    run_id = response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    assert [event["displayText"] for event in progress_body["events"]] == [
        "Preparing video inputs...",
        "Checking videos: 5 / 5 valid videos",
        "Extracting frames across all videos: 3,600 / 6,000 frames",
        "Running object detection model: 3,050 / 6,000 frames",
        "Running hand-object contact on extracted frames: waiting",
        "Combining predictions: waiting",
        "Building ADL video and session summaries: waiting",
        "Saving outputs: waiting",
    ]

def test_run_endpoint_writes_technical_progress_without_showing_it(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress("Checking packaged hand-object-contact runtime image.")
        progress("Packaged hand-object-contact runtime image is already available.")
        progress(_progress_line("hand_object_images_discovered", current = 1, total = 1))
        progress(_progress_line("hand_object_image_processed", current = 1, total = 1))
        (output_dir / "frame_det.png").write_bytes(b"visual")

    client = TestClient(create_app(hand_object_runner=fake_runner))

    response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    run_id = response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)
    visible_lines = [event["displayText"] for event in progress_body["events"]]

    assert "Checking packaged hand-object-contact runtime image." not in visible_lines

    output_folder = Path(progress_body["outputFolder"])
    runtime_log = output_folder / "logs" / "runtime.log"

    assert "Checking packaged hand-object-contact runtime image." in runtime_log.read_text(
        encoding="utf-8",
    )
    
    progress_log = output_folder / "logs" / "progress.jsonl"

    assert "Checking packaged hand-object-contact runtime image." not in progress_log.read_text(
        encoding="utf-8",
    )

def test_run_endpoint_reports_runtime_status_separately_from_progress_rows(
    tmp_path: Path,
) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        progress("Checking packaged hand-object-contact runtime image.")
        
        progress(
            "Packaged hand-object-contact runtime image is missing; preparing it now. "
            "The first run may take longer."
        )
        
        progress(_progress_line("hand_object_images_discovered", current = 1, total = 1))
        progress(_progress_line("hand_object_image_processed", current = 1, total = 1))
        
        (output_dir / "frame_det.png").write_bytes(b"visual")

    client = TestClient(create_app(hand_object_runner = fake_runner))

    response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    run_id = response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)

    visible_lines = [event["displayText"] for event in progress_body["events"]]

    assert "Building Docker image for hand-object detector" not in visible_lines
    assert progress_body["runtimeStatus"] is None

def test_internal_progress(tmp_path: Path) -> None:
    state = _test_run_state(
        tmp_path,
        model_id = ADL_RECOGNITION_MODEL_ID,
        scenario = "adl-combined-predictions",
    )

    assert [event.display_text for event in gui_backend._initial_wireframe_events(state)] == [
        "Preparing combined predictions input...",
        "Combining predictions: waiting",
        "Building ADL video and session summaries: waiting",
        "Saving outputs: waiting",
    ]

    unknown_state = _test_run_state(tmp_path, scenario = "unknown-scenario")

    with pytest.raises(ValueError, match = "Unsupported progress scenario"):
        gui_backend._initial_wireframe_events(unknown_state)

    gui_backend._record_external_progress_update(
        state,
        ExternalProgressUpdate(kind = "adl_predictions_combining", payload = {}),
    )

    assert any(
        event.display_text == "Combining predictions: waiting"
        for event in state.progress_events
    )

    before_count = len(state.progress_events)

    gui_backend._record_external_progress_update(
        state,
        ExternalProgressUpdate(kind = "adl_predictions_combined", payload = {}),
    )

    assert len(state.progress_events) == before_count
    
def test_hand_object_discovery_update_ignores_non_hand_object_scenarios(
    tmp_path: Path,
) -> None:
    state = _test_run_state(
        tmp_path,
        model_id = ADL_RECOGNITION_MODEL_ID,
        scenario = "adl-single-video",
    )

    gui_backend._record_external_progress_update(
        state,
        ExternalProgressUpdate(
            kind = "hand_object_images_discovered",
            payload = {"total": 3},
        ),
    )

    assert state.progress_events == []

def test_external_progress_update_ignores_mismatched_or_unknown_scenarios(
    tmp_path: Path,
) -> None:
    single_image_state = _test_run_state(tmp_path, scenario = "hand-object-single-image")
    
    adl_single_state = _test_run_state(
        tmp_path,
        model_id = ADL_RECOGNITION_MODEL_ID,
        scenario = "adl-single-video",
    )
   
    before_single_count = len(single_image_state.progress_events)
    before_adl_count = len(adl_single_state.progress_events)

    gui_backend._record_external_progress_update(
        single_image_state,
        ExternalProgressUpdate(
            kind = "hand_object_images_discovered",
            payload = {"total": 3},
        ),
    )
    
    gui_backend._record_external_progress_update(
        adl_single_state,
        ExternalProgressUpdate(
            kind = "adl_video_checked",
            payload = {"current": 1, "total": 1},
        ),
    )
    
    gui_backend._record_external_progress_update(
        adl_single_state,
        ExternalProgressUpdate(kind = "unknown_kind", payload = {}),
    )

    assert len(single_image_state.progress_events) == before_single_count + 1
    assert len(adl_single_state.progress_events) == before_adl_count

def test_display_video_name_from_payload_cleans_fallback_names() -> None:
    assert (
        gui_backend._display_video_name_from_payload({"displayVideo": "video1"}) 
        == "video1"
    )
    
    assert (
        gui_backend._display_video_name_from_payload({"video": "folder/video002..MP4"})
        == "video002.MP4"
    )
    
    assert gui_backend._display_video_name_from_payload({}) == "unknown"

def test_final_output_progress_preserves_existing_numeric_hand_object_progress(
    tmp_path: Path,
) -> None:
    state = _test_run_state(tmp_path)
    
    gui_backend._upsert_progress_event(
        state,
        ProgressEvent(
            stage = "save_outputs",
            message = "Saving detection outputs",
            current = 2,
            total = 3,
            unit = "images",
        ),
    )

    gui_backend._record_final_output_progress(state)

    assert (
        state.progress_events[-1].display_text 
        == "Saving detection outputs: 2 / 3 images"
    )

def test_docker_progress_helpers_cover_runtime_branches(tmp_path: Path) -> None:
    state = _test_run_state(tmp_path)

    gui_backend._update_docker_build_progress(
        state,
        "Packaged adl-recognition core runtime image is missing; preparing it now.",
    )

    assert state.runtime_status is not None
    assert state.runtime_status.model_name == "EgoVizML"

    gui_backend._update_docker_build_progress(state, "Step 2/6 : RUN pip install")

    assert state.runtime_status is not None
    assert state.runtime_status.current_step == 2
    assert state.runtime_status.total_steps == 6

    gui_backend._update_active_runtime_build_stage(state, current = 1, total = 4)

    assert state.runtime_status.current_step == 2
    assert state.runtime_status.total_steps == 6

    gui_backend._update_docker_build_progress(
        state,
        "Packaged adl-recognition core runtime image is ready.",
    )

    assert state.runtime_status is None
    assert state.active_runtime_stage_id is None
    assert state.runtime_build_stages["docker:EgoVizML"].current == 6

    gui_backend._update_docker_build_progress(
        state,
        "Packaged adl-recognition Detic runtime image is already available.",
    )

    assert (
        gui_backend._docker_model_name_from_message(
            "Packaged adl-recognition Detic runtime image is missing; preparing it now."
        )
        == "Detic"
    )
    
    assert gui_backend._docker_model_name_from_message("unrelated output") is None
    assert gui_backend._docker_build_step_counts("#7 [ 3/12] RUN apt-get update") == (
        3,
        12,
    )
    
    assert gui_backend._docker_build_step_counts("Step 4/9 : COPY . .") == (4, 9)
    assert gui_backend._docker_build_step_counts("no docker step here") is None

def test_stage_and_payload_helpers() -> None:
    assert gui_backend._stage_from_external_progress_kind(
        "hand_object_images_checked"
    ) == "check_input"
    
    assert gui_backend._stage_from_external_progress_kind(
        "hand_object_image_processed"
    ) == "run_hand_object"
    
    assert gui_backend._stage_from_external_progress_kind(
        "hand_object_output_saved"
    ) == "save_outputs"
    
    assert (
        gui_backend._stage_from_external_progress_kind("adl_video_checked") 
        == "check_input"
    )
    
    assert gui_backend._stage_from_external_progress_kind(
        "adl_frame_extracted"
    ) == "extract_frames"
    
    assert gui_backend._stage_from_external_progress_kind(
        "detic_frame_processed"
    ) == "run_detic"
    
    assert gui_backend._stage_from_external_progress_kind(
        "adl_prediction_frame_processed"
    ) == "combine_predictions"
    
    assert gui_backend._stage_from_external_progress_kind("unknown") is None

    payload = {
        "integer": 3,
        "floating": 4.9,
        "digits": "5",
        "letters": "five",
    }

    assert gui_backend._payload_int(payload, "integer") == 3
    assert gui_backend._payload_int(payload, "floating") == 4
    assert gui_backend._payload_int(payload, "digits") == 5
    assert gui_backend._payload_int(payload, "letters") == 0
    assert gui_backend._payload_int(payload, "missing") == 0

def test_progress_merge_helpers_cover_remaining_branches() -> None:
    existing = ProgressEvent(
        stage = "run_detic",
        message = "Running object detection model",
        current = 10,
        total = 20,
        unit = "frames",
    )

    lower_next = ProgressEvent(
        stage = "run_detic",
        message = "Running object detection model",
        current = 5,
        total = 20,
        unit = "frames",
    )

    assert gui_backend._merge_progress_event(existing, lower_next).current == 10

    different_total = ProgressEvent(
        stage = "run_detic",
        message = "Running object detection model",
        current = 5,
        total = 40,
        unit = "frames",
    )

    assert gui_backend._merge_progress_event(existing, different_total) == different_total
    
    assert gui_backend._merge_progress_event(
        ProgressEvent(stage = "run_detic", message = "waiting"),
        lower_next,
    ) == lower_next

def test_active_runtime_build_stage_ignores_missing_active_stage(tmp_path: Path) -> None:
    state = _test_run_state(tmp_path)

    gui_backend._update_active_runtime_build_stage(
        state,
        current = 1,
        total = 2,
    )

    assert state.runtime_status is None
    assert state.active_runtime_stage_id is None
    assert state.runtime_build_stages == {}

def test_dry_run_rejects_missing_output_folder(tmp_path: Path) -> None:
    app = create_app(runtime_checker = _ready_runtime_checker)
    client = TestClient(app)

    input_file = tmp_path / "frame.jpg"
    input_file.write_bytes(b"fake image")

    missing_output_root = tmp_path / "missing-results"

    with input_file.open("rb") as stream:
        response = client.post(
            "/api/dry-run",
            data = {
                "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
                "outputRoot": str(missing_output_root),
            },
            files = {
                "files": ("frame.jpg", stream, "image/jpeg"),
            },
        )

    assert response.status_code == 400
    
    assert response.json()["detail"] == (
        "Output folder does not exist. Choose an existing folder before continuing."
    )

def test_start_run_rejects_missing_output_folder(tmp_path: Path) -> None:
    app = create_app(runtime_checker = _ready_runtime_checker)
    client = TestClient(app)

    input_file = tmp_path / "frame.jpg"
    input_file.write_bytes(b"fake image")

    missing_output_root = tmp_path / "missing-results"

    with input_file.open("rb") as stream:
        response = client.post(
            "/api/runs",
            data = {
                "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
                "outputRoot": str(missing_output_root),
            },
            files = {
                "files": ("frame.jpg", stream, "image/jpeg"),
            },
        )

    assert response.status_code == 400
    
    assert response.json()["detail"] == (
        "Output folder does not exist. Choose an existing folder before continuing."
    )

def test_dry_run_returns_499_when_runtime_check_is_cancelled(tmp_path: Path) -> None:
    def cancelling_runtime_checker(
        _model_id: str,
        _progress: ProgressCallback,
        _command_runner: Callable[[list[str]], int],
    ) -> None:
        raise CommandCancelledError("Run was cancelled.")

    app = create_app(runtime_checker = cancelling_runtime_checker)
    client = TestClient(app)

    input_file = tmp_path / "frame.jpg"
    input_file.write_bytes(b"fake image")

    with input_file.open("rb") as stream:
        response = client.post(
            "/api/dry-run",
            data = {
                "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
                "outputRoot": str(_existing_output_root(tmp_path)),
                "operationId": "operation-cancelled-dry-run",
            },
            files = {
                "files": ("frame.jpg", stream, "image/jpeg"),
            },
        )

    assert response.status_code == 499
    assert response.json()["detail"] == "Run was cancelled."

def test_cancel_run_endpoint_returns_404_when_operation_is_missing() -> None:
    client = TestClient(create_app(runtime_checker = _ready_runtime_checker))

    response = client.post(
        "/api/cancel-run",
        json = {
            "runId": "missing-run",
            "operationId": "missing-operation",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "No active run or operation was found to cancel."

def test_execute_run_uses_default_hand_object_gui_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _test_run_state(tmp_path)
    calls: list[tuple[Path, Path, ProcessCancellation]] = []

    def fake_hand_object_gui_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
        cancellation: ProcessCancellation,
    ) -> None:
        calls.append((input_path, output_dir, cancellation))
        progress("Hand-object runtime output")

    monkeypatch.setattr(
        "egomodelkit.gui_backend._run_hand_object_contact_for_gui",
        fake_hand_object_gui_runner,
    )

    monkeypatch.setattr(
        "egomodelkit.gui_backend.finalize_runtime_outputs",
        lambda **_kwargs: None,
    )

    monkeypatch.setattr(
        "egomodelkit.gui_backend.write_run_summary",
        lambda **_kwargs: None,
    )

    operations = {
        state.operation_id: CancelableGuiOperation(
            operation_id = state.operation_id,
            cancellation = state.cancellation,
        )
    }

    _execute_run(
        state = state,
        hand_object_runner = None,
        adl_runner = None,
        operations = operations,
    )

    assert calls == [
        (
            state.input_path,
            state.layout.run_dir,
            state.cancellation,
        )
    ]

    assert state.status == "completed"
    assert operations == {}

def test_execute_run_uses_default_adl_gui_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _test_run_state(tmp_path)
    state.model_id = ADL_RECOGNITION_MODEL_ID
    state.scenario = "adl-single-video"

    calls: list[tuple[Path, Path, ProcessCancellation]] = []

    def fake_adl_gui_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
        cancellation: ProcessCancellation,
    ) -> None:
        calls.append((input_path, output_dir, cancellation))
        progress("ADL runtime output")

    monkeypatch.setattr(
        "egomodelkit.gui_backend._run_adl_recognition_for_gui",
        fake_adl_gui_runner,
    )

    monkeypatch.setattr(
        "egomodelkit.gui_backend.finalize_runtime_outputs",
        lambda **_kwargs: None,
    )

    monkeypatch.setattr(
        "egomodelkit.gui_backend.write_run_summary",
        lambda **_kwargs: None,
    )

    operations = {
        state.operation_id: CancelableGuiOperation(
            operation_id = state.operation_id,
            cancellation = state.cancellation,
        )
    }

    _execute_run(
        state = state,
        hand_object_runner = None,
        adl_runner = None,
        operations = operations,
    )

    assert calls == [
        (
            state.input_path,
            state.layout.run_dir,
            state.cancellation,
        )
    ]

    assert state.status == "completed"
    assert operations == {}

def test_dry_run_ignores_legacy_adl_dominant_hand_field(tmp_path: Path) -> None:
    client = TestClient(create_app(runtime_checker = _ready_runtime_checker))

    response = client.post(
        "/api/dry-run",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
            "dominantHand": "middle",
        },
        files = [("files", ("clip.mp4", b"fake-video", "video/mp4"))],
    )

    assert response.status_code == 200
    assert response.json()["summary"]["modelId"] == ADL_RECOGNITION_MODEL_ID

def test_run_endpoint_returns_same_preflight_error_as_dry_run(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(runtime_checker = _failing_runtime_checker))
    output_root = _existing_output_root(tmp_path)

    dry_run_response = client.post(
        "/api/dry-run",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(output_root),
            "dominantHand": "right",
        },
        files = [("files", ("clip.mp4", b"fake-video", "video/mp4"))],
    )

    run_response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(output_root),
            "dominantHand": "right",
        },
        files = [("files", ("clip.mp4", b"fake-video", "video/mp4"))],
    )

    assert dry_run_response.status_code == 400
    assert run_response.status_code == 400
    assert run_response.json()["detail"] == dry_run_response.json()["detail"]
    assert "Linux host with an NVIDIA GPU" in run_response.json()["detail"]

def test_run_endpoint_preflight_failure_does_not_create_run_folder(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(runtime_checker = _failing_runtime_checker))
    output_root = _existing_output_root(tmp_path)

    response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(output_root),
            "dominantHand": "right",
        },
        files = [("files", ("clip.mp4", b"fake-video", "video/mp4"))],
    )

    assert response.status_code == 400
    assert not any(path.name.startswith("run-") for path in output_root.iterdir())

def test_start_run_returns_499_when_runtime_check_is_cancelled(
    tmp_path: Path,
) -> None:
    def cancelling_runtime_checker(
        _model_id: str,
        _progress: ProgressCallback,
        _command_runner: Callable[[list[str]], int],
    ) -> None:
        raise CommandCancelledError("Run was cancelled before start.")

    client = TestClient(create_app(runtime_checker = cancelling_runtime_checker))

    output_root = _existing_output_root(tmp_path)

    response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(output_root),
            "operationId": "operation-cancelled-start-run",
        },
        files = {
            "files": (
                "frame.jpg",
                b"fake image",
                "image/jpeg",
            ),
        },
    )

    assert response.status_code == 499
    assert response.json()["detail"] == "Run was cancelled before start."
    assert list(output_root.iterdir()) == []

def test_run_start_preflight_policy_covers_adl_and_unknown_models() -> None:
    assert _should_run_start_preflight(
        model_id = ADL_RECOGNITION_MODEL_ID,
        hand_object_runner = None,
        adl_runner = None,
        runtime_checker_was_injected = False,
    ) is True

    assert _should_run_start_preflight(
        model_id = "unsupported-model",
        hand_object_runner = None,
        adl_runner = None,
        runtime_checker_was_injected = False,
    ) is True


def test_windows_display_path_is_preserved_in_run_responses_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _existing_output_root(tmp_path)
    windows_output_root = r"C:\Users\researcher\EgoModelKit Results"

    monkeypatch.setattr(
        "egomodelkit.gui_backend._normalize_output_root",
        lambda output_root_text: output_root,
    )

    def fake_runner(input_path: Path, output_dir: Path, progress) -> None:
        progress("done")

    client = TestClient(create_app(hand_object_runner = fake_runner))
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": windows_output_root,
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )

    assert start_response.status_code == 200
    run_id = start_response.json()["runId"]
    expected_display_path = windows_output_root + "\\" + run_id

    assert start_response.json()["summary"]["outputFolder"] == expected_display_path

    progress_response = _wait_for_run_completion(client, run_id)
    assert progress_response["outputFolder"] == expected_display_path

    summary_payload = json.loads(
        (output_root / run_id / "run_summary.json").read_text(encoding = "utf-8")
    )
    assert summary_payload["output_folder"] == expected_display_path


def test_open_output_folder_accepts_restored_saved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-restored"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    opened_folders: list[Path] = []

    monkeypatch.setattr(
        "egomodelkit.gui_backend._open_output_folder",
        lambda output_folder: opened_folders.append(output_folder) or True,
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/open-output-folder",
        json = {
            "runId": run_id,
            "outputFolder": str(run_dir),
        },
    )

    assert response.status_code == 200
    assert opened_folders == [run_dir]
    assert response.json()["outputFolder"] == str(run_dir)


def test_open_output_folder_rejects_mismatched_restored_saved_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "different-run"
    run_dir.mkdir()
    client = TestClient(create_app())

    response = client.post(
        "/api/open-output-folder",
        json = {
            "runId": "run-restored",
            "outputFolder": str(run_dir),
        },
    )

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_open_output_folder_returns_success_after_file_manager_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_commands: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs):
        del kwargs
        started_commands.append(command)
        return SimpleNamespace()

    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Linux")
    monkeypatch.setattr("egomodelkit.gui_backend._is_wsl", lambda: True)
    monkeypatch.setattr(
        "egomodelkit.gui_backend._wsl_path_to_windows_path",
        lambda path: r"C:\Users\researcher\results\run-1",
    )
    monkeypatch.setattr("egomodelkit.gui_backend.subprocess.Popen", fake_popen)

    assert _open_output_folder(tmp_path) is True
    assert started_commands == [
        ["explorer.exe", r"C:\Users\researcher\results\run-1"]
    ]


def test_models_endpoint_includes_hand_interaction_in_preferred_order() -> None:
    body = TestClient(create_app()).get("/api/models").json()
    model = body["models"][0]
    assert model["name"] == "Hand interaction"
    assert model["supportedInputExtensions"] == sorted(
        HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES
    )
    assert "functional hand-object interactions" in model["description"]


def test_hand_interaction_preview_and_dry_run_endpoints(tmp_path: Path) -> None:
    from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID

    client = TestClient(create_app(runtime_checker=_ready_runtime_checker))
    output_root = _existing_output_root(tmp_path)
    preview = client.post(
        "/api/output-preview",
        json={
            "modelId": HAND_INTERACTION_MODEL_ID,
            "inputNames": ["one.mp4", "two.mp4"],
            "outputRoot": str(output_root),
        },
    )
    assert preview.status_code == 200
    assert preview.json()["scenario"] == "hand-interaction-video-directory"
    assert "hand_interaction_input_manifest.csv" in preview.json()["folderTree"]
    assert "detic_outputs" not in preview.json()["folderTree"]

    dry_run = client.post(
        "/api/dry-run",
        data={
            "modelId": HAND_INTERACTION_MODEL_ID,
            "outputRoot": str(output_root),
            "dominantHand": "left",
        },
        files=[("files", ("clip.mp4", b"video", "video/mp4"))],
    )
    assert dry_run.status_code == 200
    body = dry_run.json()
    assert body["scenario"] == "hand-interaction-single-video"
    assert body["summary"]["model"] == "Hand interaction"

    invalid = client.post(
        "/api/dry-run",
        data={
            "modelId": HAND_INTERACTION_MODEL_ID,
            "outputRoot": str(output_root),
            "dominantHand": "middle",
        },
        files=[("files", ("clip.mp4", b"video", "video/mp4"))],
    )
    assert invalid.status_code == 400
    assert "Dominant hand" in invalid.json()["detail"]


def test_hand_interaction_run_endpoint_persists_dominant_hand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID

    runner_calls: list[tuple[Path, Path]] = []

    def fake_runner(input_path: Path, output_dir: Path, progress: ProgressCallback) -> None:
        runner_calls.append((input_path, output_dir))
        progress(_progress_line("hand_interaction_frame_extracted", current=30, total=30))

    monkeypatch.setattr(gui_backend, "finalize_runtime_outputs", lambda **_kwargs: None)
    client = TestClient(
        create_app(
            hand_interaction_runner=fake_runner,
            runtime_checker=_ready_runtime_checker,
        )
    )
    response = client.post(
        "/api/runs",
        data={
            "modelId": HAND_INTERACTION_MODEL_ID,
            "outputRoot": str(_existing_output_root(tmp_path)),
            "dominantHand": "left",
        },
        files=[("files", ("clip.mp4", b"video", "video/mp4"))],
    )
    assert response.status_code == 200
    run_id = response.json()["runId"]
    body = _wait_for_run_completion(client, run_id)
    assert body["status"] == "completed"
    assert response.json()["summary"]["model"] == "Hand interaction"
    assert runner_calls and runner_calls[0][0].suffix == ".mp4"
    run_dir = Path(body["outputFolder"])
    config = json.loads(
        (run_dir / "technical" / "post_processing" / "metrics_config.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert config["dominant_hand"] == "left"
    assert manifest["model_configuration"] == {
        "dominant_hand": "left",
        "non_dominant_hand": "right",
    }


def test_default_hand_interaction_gui_runner_uses_cancellable_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egomodelkit.gui_backend import _run_hand_interaction_for_gui

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    cancellation = ProcessCancellation()
    captured: dict[str, object] = {}

    def fake_runtime(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        assert kwargs["command_runner"](["true"]) == 0
        assert kwargs["streaming_command_runner"](
            ["true"],
            lambda _message: None,
        ) == 0
        return []

    monkeypatch.setattr(gui_backend, "run_hand_interaction", fake_runtime)
    monkeypatch.setattr(gui_backend, "cancellable_subprocess_runner", lambda command, token: 0)
    monkeypatch.setattr(
        gui_backend,
        "cancellable_streaming_subprocess_runner",
        lambda command, progress, token: 0,
    )
    _run_hand_interaction_for_gui(
        video,
        tmp_path / "results",
        lambda _message: None,
        cancellation,
        dominant_hand="left",
    )
    assert captured["request"].dominant_hand == "left"
    assert captured["request"].input_path == video


def test_hand_interaction_backend_helpers_and_progress(tmp_path: Path, monkeypatch) -> None:
    from egomodelkit.gui_backend import (
        _initialize_wireframe_progress,
        _record_external_progress_update,
    )
    from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID

    checked: list[dict[str, object]] = []
    monkeypatch.setattr(
        gui_backend,
        "ensure_host_runtime_ready",
        lambda **kwargs: checked.append(kwargs),
    )
    _check_runtime_ready_for_gui(HAND_INTERACTION_MODEL_ID)
    assert checked[0]["docker_executable"] == "docker"

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    _validate_gui_request(
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=video,
        output_root=tmp_path / "results",
        dominant_hand="left",
    )
    assert _model_display_name(HAND_INTERACTION_MODEL_ID) == "Hand interaction"
    assert _should_run_start_preflight(
        model_id=HAND_INTERACTION_MODEL_ID,
        hand_object_runner=None,
        hand_interaction_runner=None,
        adl_runner=None,
        runtime_checker_was_injected=False,
    )
    assert not _should_run_start_preflight(
        model_id=HAND_INTERACTION_MODEL_ID,
        hand_object_runner=None,
        hand_interaction_runner=lambda *_args: None,
        adl_runner=None,
        runtime_checker_was_injected=False,
    )

    state = _test_run_state(
        tmp_path,
        model_id=HAND_INTERACTION_MODEL_ID,
        scenario="hand-interaction-video-directory",
    )
    _initialize_wireframe_progress(state)
    assert [event.stage for event in state.progress_events] == [
        "prepare_input",
        "check_input",
        "extract_frames",
        "run_hand_object",
        "calculate_profiles",
        "calculate_metrics",
        "save_outputs",
    ]
    updates = [
        ExternalProgressUpdate("hand_interaction_video_checked", {"current": 2, "total": 2}),
        ExternalProgressUpdate("hand_interaction_frame_extracted", {"current": 60, "total": 60}),
        ExternalProgressUpdate(
            "hand_interaction_hoc_frame_processed",
            {"current": 60, "total": 60},
        ),
        ExternalProgressUpdate("hand_interaction_profiles_calculating", {}),
        ExternalProgressUpdate("hand_interaction_metrics_calculating", {}),
        ExternalProgressUpdate(
            "hand_interaction_metrics_calculated",
            {"current": 1, "total": 1},
        ),
        ExternalProgressUpdate("hand_interaction_outputs_organizing", {}),
    ]
    for update in updates:
        _record_external_progress_update(state, update)
    events = {event.stage: event for event in state.progress_events}
    assert events["check_input"].current == 2
    assert events["extract_frames"].current == 60
    assert events["run_hand_object"].current == 60
    assert events["calculate_profiles"].message == "Calculating interaction profiles..."
    assert events["calculate_metrics"].current == 1
    assert events["save_outputs"].current == 1

    single_state = _test_run_state(
        tmp_path / "single",
        model_id=HAND_INTERACTION_MODEL_ID,
        scenario="hand-interaction-single-video",
    )
    _initialize_wireframe_progress(single_state)
    _record_external_progress_update(
        single_state,
        ExternalProgressUpdate("hand_interaction_video_checked", {"current": 1, "total": 1}),
    )
    _record_external_progress_update(
        single_state,
        ExternalProgressUpdate("hand_interaction_frame_extracted", {"current": 30, "total": 30}),
    )
    assert single_state.progress_events[1].message == "Extracting frames"


def test_execute_run_uses_default_hand_interaction_gui_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _test_run_state(
        tmp_path,
        model_id="hand-interaction",
        scenario="hand-interaction-single-video",
    )
    state.dominant_hand = "left"
    calls: list[tuple[Path, Path, ProcessCancellation, str]] = []

    def fake_hand_interaction_gui_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
        cancellation: ProcessCancellation,
        *,
        dominant_hand: str,
    ) -> None:
        calls.append((input_path, output_dir, cancellation, dominant_hand))
        progress("Hand-interaction runtime output")

    monkeypatch.setattr(
        "egomodelkit.gui_backend._run_hand_interaction_for_gui",
        fake_hand_interaction_gui_runner,
    )
    monkeypatch.setattr(
        "egomodelkit.gui_backend.finalize_runtime_outputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "egomodelkit.gui_backend.write_run_summary",
        lambda **_kwargs: None,
    )

    operations = {
        state.operation_id: CancelableGuiOperation(
            operation_id=state.operation_id,
            cancellation=state.cancellation,
        )
    }
    _execute_run(
        state=state,
        hand_object_runner=None,
        hand_interaction_runner=None,
        adl_runner=None,
        operations=operations,
    )

    assert calls == [
        (
            state.input_path,
            state.layout.run_dir,
            state.cancellation,
            "left",
        )
    ]
    assert state.status == "completed"
    assert operations == {}
