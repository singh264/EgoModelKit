import os
from pathlib import Path

import pytest


@pytest.fixture(autouse = True)
def fake_docker_executable_on_path(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ Keep runtime unit tests independent from host Docker installation. """
    if request.node.get_closest_marker("e2e") is not None:
        return

    fake_bin_dir = tmp_path / "fake-bin"
    fake_bin_dir.mkdir()
    
    fake_docker = fake_bin_dir / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    
    existing_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{fake_bin_dir}{os.pathsep}{existing_path}")
