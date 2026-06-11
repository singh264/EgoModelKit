from __future__ import annotations

import asyncio
import builtins
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from egomodelkit.gui_backend import (
    GuiRunState,
    ProgressCallback,
    _build_unique_run_id,
    _execute_run,
    _input_label,
    _model_display_name,
    _normalize_output_root,
    _run_adl_recognition_for_gui,
    _run_hand_object_contact_for_gui,
    _safe_upload_filename,
    _select_output_folder,
    _select_output_folder_macos,
    _select_output_folder_tkinter,
    _select_output_folder_windows,
    _stage_uploaded_files,
    _unique_destination_path,
    _validate_gui_request,
    create_app,
)
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.output_contract import build_run_output_layout, create_output_scaffold
from egomodelkit.runtime.hand_object_contact import HandObjectContactRuntimeError


def test_models_endpoint_return_supported_models() -> None:
    client = TestClient(create_app())
    
    response = client.get("/api/models")
    
    assert response.status_code == 200
    
    body = response.json()
    model_ids = {model["id"] for model in body["models"]}
    
    assert HAND_OBJECT_CONTACT_MODEL_ID in model_ids

def test_output_preview_endpoint_returns_dynamic_tree(tmp_path: Path) -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/output-preview",
        json = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "inputNames": ["frame.jpg"],
            "outputRoot": str(tmp_path / "results"), 
        },
    )
    
    assert response.status_code == 200
    
    body = response.json()
    
    assert body["scenario"] == "hand-object-single-image"
    assert "frame_det.png" in body["folderTree"]
    assert body["files"]

def test_dry_run_validates_uploaded_file(tmp_path: Path) -> None:
    client = TestClient(create_app())
    
    response = client.post(
        "/api/dry-run",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(tmp_path / "results"),
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

def test_run_endpoint_uses_injected_runner_without_docker(tmp_path: Path) -> None:
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> None:
        assert input_path.exists()
        assert output_dir.exists()
        
        progress("Fake model step")
        (output_dir / "fake-output.txt").write_text("done", encoding = "utf-8")

    client = TestClient(
        create_app(hand_object_runner = fake_runner),
    )
    
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(tmp_path / "results")
        },
        files = [
            ("files", ("frame.jpg", b"fake-image", "image/jpeg")),
        ],
    )
    
    assert start_response.status_code == 200
    
    run_id = start_response.json()["runId"]
    progress_body = _wait_for_run_completion(client, run_id)
    
    assert progress_body["status"] == "completed"
    
    assert any(
        event["displayText"] == "Fake model step"
        for event in progress_body["events"]
    )
    
    output_folder = Path(progress_body["outputFolder"])
    
    assert (output_folder / "fake-output.txt").read_text(encoding = "utf-8") == "done"

def test_open_output_folder_uses_tracked_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    opened_urls: list[str] = []
    
    def fake_open(url: str) -> bool:
        opened_urls.append(url)
        
        return True
    
    def fake_runner(
        input_path: Path,
        output_dir: Path,
        progress,
    ) -> None:
        progress("Fake model step")
    
    monkeypatch.setattr("egomodelkit.gui_backend.webbrowser.open", fake_open)
    
    client = TestClient(
        create_app(hand_object_runner = fake_runner),
    )
    
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": HAND_OBJECT_CONTACT_MODEL_ID,
            "outputRoot": str(tmp_path / "results"),
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
    assert opened_urls
    assert opened_urls[0].startswith("file://")
    
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

def test_select_output_folder_returns_404_when_picker_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "egomodelkit.gui_backend._select_output_folder",
        lambda: None,
    )

    client = TestClient(create_app())

    response = client.post("/api/select-output-folder")

    assert response.status_code == 404

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
            "outputRoot": str(tmp_path / "results"),
        },
    )
    
    assert response.status_code == 400
    assert "Unsupported model id" in response.json()["detail"]

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
            "outputRoot": str(tmp_path / "results"),
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
            "outputRoot": str(tmp_path / "results"),
        },
        files = [("files", ("frame.jpg", b"fake-image", "image/jpeg"))],
    )
    
    run_id = start_response.json()["runId"]
    
    progress_response = client.get(f"/api/runs/{run_id}/progress")
    
    output_folder = Path(progress_response.json()["outputFolder"])
    
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
    
    client = TestClient(create_app(adl_runner = fake_runner))
    
    start_response = client.post(
        "/api/runs",
        data = {
            "modelId": ADL_RECOGNITION_MODEL_ID,
            "outputRoot": str(tmp_path / "results"),
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
    
    assert any(event["displayText"] == "ADL step" for event in body["events"])

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
        output_preview = {}
    )
    
    def failing_runner(input_path: Path, output_dir: Path, progress) -> None:
        raise HandObjectContactRuntimeError("simulated failure")
    
    _execute_run(state = state, hand_object_runner = failing_runner, adl_runner = None)

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
        output_preview = {},
    )
    
    _execute_run(state = state, hand_object_runner = None, adl_runner = None)

    assert state.status == "failed"
    assert state.error_message == "Unsupported model id: unknown-model"
    
def test_runtime_wrappers_delegate_to_existing_runners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    
    def fake_hand_object_runner(request, *, command_runner, progress) -> None:
        captured["hand_request"] = request
        captured["hand_command_runner"] = command_runner
        
        progress("hand progress")
    
    def fake_adl_runner(request, *, command_runner, progress) -> None:
        captured["adl_request"] = request
        captured["adl_command_runner"] = command_runner
        
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
    
    _run_hand_object_contact_for_gui(tmp_path / "frame.jpg", tmp_path / "out", messages.append)
    _run_adl_recognition_for_gui(tmp_path / "clip.mp4", tmp_path / "out", messages.append)
    
    assert captured["hand_request"].input_path == tmp_path / "frame.jpg"
    assert captured["adl_request"].input_path == tmp_path / "clip.mp4"
    assert messages == ["hand progress", "adl progress"]

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
    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("egomodelkit.gui_backend._select_output_folder_macos", lambda: "/mac")
    assert _select_output_folder() == "/mac"
    
    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Windows")
    monkeypatch.setattr("egomodelkit.gui_backend._select_output_folder_windows", lambda: "C:/out")
    assert _select_output_folder() == "C:/out"
    
    monkeypatch.setattr("egomodelkit.gui_backend.platform.system", lambda: "Linux")
    monkeypatch.setattr("egomodelkit.gui_backend._select_output_folder_tkinter", lambda: "/linux")
    assert _select_output_folder() == "/linux"

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
    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 0, stdout = "C:/out\n"),
    )
    
    assert _select_output_folder_windows() == "C:/out"
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 0, stdout = "\n"),
    )
    
    assert _select_output_folder_windows() is None
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode = 1, stdout = ""),
    )
    
    assert _select_output_folder_windows() is None

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
