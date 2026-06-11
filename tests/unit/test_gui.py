from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from egomodelkit import gui as gui_module


def test_launch_gui_runs_uvicorn_without_opening_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    
    fake_uvicorn = SimpleNamespace(
        run = lambda app, **kwargs: captured.update({"app": app, "kwargs": kwargs}),
    )
    
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(gui_module, "_create_gui_app", lambda static_dir: "fake-app")
    
    gui_module.launch_gui(server_port = 1234, inbrowser = False)
    
    assert captured == {
        "app": "fake-app",
        "kwargs": {
            "host": gui_module.GUI_LOCAL_SERVER_NAME,
            "port": 1234,
            "log_level": "info",
        }
    }

def test_launch_gui_opens_browser_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_urls: list[str] = []
    timer_callbacks: list[object] = []
    
    class ImmediateTimer:
        def __init__(self, interval, function):
            assert interval == 1.0

            timer_callbacks.append(function)
            self.function = function
        
        def start(self) -> None:
            self.function()
    
    fake_uvicorn = SimpleNamespace(run = lambda *args, **kwargs: None)
    
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(gui_module, "_create_gui_app", lambda static_dir: "fake-app")    
    monkeypatch.setattr(gui_module.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(gui_module.webbrowser, "open", lambda url: opened_urls.append(url))

    gui_module.launch_gui(server_port = 4321, inbrowser = True)
    
    assert timer_callbacks
    assert opened_urls == ["http://127.0.0.1:4321"]  
       
def test_launch_gui_reports_missing_gui_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("missing uvicorn")
        
        return original_import(name, *args, **kwargs)
    
    monkeypatch.delitem(sys.modules, "uvicorn", raising = False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    
    with pytest.raises(RuntimeError, match = "GUI dependencies are not installed"):
        gui_module.launch_gui()

def test_create_gui_app_delegates_to_backend_create_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    expected_app = object()
    
    def fake_create_app(*, static_dir: Path) -> object:
        captured["static_dir"] = static_dir
        
        return expected_app

    static_dir = tmp_path / "dist"
    
    monkeypatch.setattr(
        "egomodelkit.gui_backend.create_app",
        fake_create_app,
    )
    
    static_dir = tmp_path / "dist"
    
    result = gui_module._create_gui_app(static_dir)
    
    assert result is expected_app
    assert captured == {"static_dir": static_dir}
