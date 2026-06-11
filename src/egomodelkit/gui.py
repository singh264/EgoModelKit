""" Launcher for the local React/FastAPI EgoModelKit GUI. """

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Final

GUI_LOCAL_SERVER_NAME: Final[str] = "127.0.0.1"
GUI_DEFAULT_SERVER_PORT: Final[int] = 7860
STATIC_GUI_DIR: Final[Path] = Path(__file__).with_name("web") / "dist"

def _create_gui_app(static_dir: Path) -> object:
    """ Create the GUI backend app after optional GUI dependencies are available. """
    from egomodelkit.gui_backend import create_app
    
    return create_app(static_dir = static_dir)

def launch_gui(
    *,
    server_port: int = GUI_DEFAULT_SERVER_PORT,
    inbrowser: bool = True,
) -> None:
    """ Launch the local browser GUI on 127.0.0.1. """
    try:
        import uvicorn
        
        app = _create_gui_app(STATIC_GUI_DIR)
    except ImportError as exc:
        raise RuntimeError(
            "The GUI dependencies are not installed. Install them with: "
            'python -m pip install -e ".[gui]"'
        ) from exc
    
    url = f"http://{GUI_LOCAL_SERVER_NAME}:{server_port}"
    
    if inbrowser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    uvicorn.run(
        app,
        host = GUI_LOCAL_SERVER_NAME,
        port = server_port,
        log_level = "info",
    )
    