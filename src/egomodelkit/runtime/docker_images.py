"""Deterministic identities and automatic cleanup for managed Docker images."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from egomodelkit.runtime.commands import CommandResult, capturing_subprocess_runner

CaptureRunner = Callable[[list[str]], CommandResult]
ProgressReporter = Callable[[str], None]

RUNTIME_LABEL_PREFIX: Final[str] = "org.egomodelkit.runtime"
MANAGED_RUNTIME_LABEL: Final[str] = f"{RUNTIME_LABEL_PREFIX}.managed"
RUNTIME_NAME_LABEL: Final[str] = f"{RUNTIME_LABEL_PREFIX}.name"
RUNTIME_FINGERPRINT_LABEL: Final[str] = f"{RUNTIME_LABEL_PREFIX}.fingerprint"
RUNTIME_TAG_FINGERPRINT_LENGTH: Final[int] = 12
FINGERPRINT_FORMAT: Final[int] = 1

_IGNORED_CONTEXT_PARTS: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache"}
)
_IGNORED_CONTEXT_SUFFIXES: Final[frozenset[str]] = frozenset({".pyc", ".pyo"})


@dataclass(frozen=True, slots=True)
class DockerImageIdentity:
    """Immutable tag and labels for one packaged runtime image."""

    runtime_name: str
    repository: str
    fingerprint: str

    @property
    def tag(self) -> str:
        """Return a content-addressed image tag safe for local Docker use."""
        short_fingerprint = self.fingerprint[:RUNTIME_TAG_FINGERPRINT_LENGTH]
        return f"{self.repository}:sha-{short_fingerprint}"

    @property
    def label_arguments(self) -> list[str]:
        """Return Docker build labels identifying this managed image."""
        return [
            "--label",
            f"{MANAGED_RUNTIME_LABEL}=true",
            "--label",
            f"{RUNTIME_NAME_LABEL}={self.runtime_name}",
            "--label",
            f"{RUNTIME_FINGERPRINT_LABEL}={self.fingerprint}",
        ]


def build_runtime_image_identity(
    *,
    runtime_name: str,
    repository: str,
    context_dir: Path,
    build_arguments: list[str],
) -> DockerImageIdentity:
    """Build an image identity from its Docker context and effective inputs."""
    hasher = hashlib.sha256()
    metadata = {
        "fingerprint_format": FINGERPRINT_FORMAT,
        "runtime_name": runtime_name,
        "repository": repository,
        "build_arguments": build_arguments,
    }
    hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))

    for path in _context_files(context_dir):
        relative_path = path.relative_to(context_dir).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")

    return DockerImageIdentity(
        runtime_name=runtime_name,
        repository=repository,
        fingerprint=hasher.hexdigest(),
    )


def remove_stale_runtime_images(
    *,
    docker_executable: str,
    current_image: DockerImageIdentity,
    capture_runner: CaptureRunner = capturing_subprocess_runner,
    progress: ProgressReporter | None = None,
) -> tuple[str, ...]:
    """Best-effort removal of older tags in one EgoModelKit image repository."""
    report = progress or (lambda _message: None)
    listing = capture_runner(
        [
            docker_executable,
            "image",
            "ls",
            current_image.repository,
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ]
    )

    if listing.returncode != 0:
        report(
            f"Warning: unable to inspect older {current_image.runtime_name} images; "
            "the current image remains usable."
        )
        return ()

    discovered = {
        line.strip()
        for line in listing.stdout.splitlines()
        if line.strip()
        and line.strip() != "<none>:<none>"
        and line.strip().startswith(f"{current_image.repository}:")
    }
    stale_tags = tuple(sorted(discovered - {current_image.tag}))
    removed: list[str] = []

    for stale_tag in stale_tags:
        result = capture_runner(
            [docker_executable, "image", "rm", stale_tag]
        )
        if result.returncode == 0:
            removed.append(stale_tag)
        else:
            report(
                f"Warning: unable to remove older Docker image {stale_tag}; "
                "it will not be selected by EgoModelKit."
            )

    if removed:
        report(
            "Removed older EgoModelKit Docker image tags: " + ", ".join(removed)
        )

    return tuple(removed)


def _context_files(context_dir: Path) -> list[Path]:
    """Return deterministic runtime-context files used for image fingerprinting."""
    return sorted(
        path
        for path in context_dir.rglob("*")
        if path.is_file()
        and not any(part in _IGNORED_CONTEXT_PARTS for part in path.parts)
        and path.suffix not in _IGNORED_CONTEXT_SUFFIXES
    )
