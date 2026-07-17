""" Host-platform detection shared by CLI, GUI, and runtime preflight. """

from __future__ import annotations

import os
import platform
from collections.abc import Mapping


def is_wsl(
    *,
    system_name: str | None = None,
    release_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """ Return whether the current process is running inside WSL. """
    detected_system = system_name if system_name is not None else platform.system()
    detected_release = release_name if release_name is not None else platform.release()
    detected_environment = environment if environment is not None else os.environ

    return (
        detected_system == "Linux"
        and (
            "microsoft" in detected_release.lower()
            or "WSL_DISTRO_NAME" in detected_environment
        )
    )


def wsl_distribution_name() -> str:
    """ Return the current WSL distribution name for actionable messages. """
    return os.environ.get("WSL_DISTRO_NAME", "this WSL distribution")
