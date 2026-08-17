"""Disk-capacity estimates and preflight checks for packaged model runs."""

from __future__ import annotations

import math
import os
import platform
import shutil
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from egomodelkit.models.adl_recognition import (
    ADL_INFERENCE_FRAME_FPS,
    ADL_RECOGNITION_MODEL_ID,
    ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES,
)
from egomodelkit.models.hand_interaction import (
    HAND_INTERACTION_MODEL_ID,
    HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_MODEL_ID,
    HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES,
)
from egomodelkit.runtime.commands import CommandResult, capturing_subprocess_runner
from egomodelkit.runtime.docker_images import DockerImageIdentity, remove_stale_runtime_images
from egomodelkit.runtime.host_platform import is_wsl

CaptureRunner = Callable[[list[str]], CommandResult]
ProgressReporter = Callable[[str], None]
PlatformDetector = Callable[[], str]
WslDetector = Callable[[], bool]

GIB: Final[int] = 1024**3
MIB: Final[int] = 1024**2
MINIMUM_FREE_SPACE_RESERVE_BYTES: Final[int] = 5 * GIB
MINIMUM_FREE_INODE_RESERVE: Final[int] = 10_000
ESTIMATE_MARGIN: Final[float] = 1.20
PEAK_MARGIN: Final[float] = 1.10
DOCKER_BUILD_SCRATCH_RATIO: Final[float] = 0.50
DOCKER_BUILD_MINIMUM_SCRATCH_BYTES: Final[int] = 4 * GIB
DOCKER_RUNTIME_TRANSIENT_BYTES: Final[int] = 1 * GIB
HAND_INTERACTION_BYTES_PER_PIXEL_PER_FRAME: Final[float] = 2.0
ADL_BYTES_PER_PIXEL_PER_FRAME: Final[float] = 2.25
ADL_SUBCLIP_BYTES_PER_SOURCE_BYTE: Final[float] = 1.50
HOC_OUTPUT_BYTES_PER_INPUT_BYTE: Final[float] = 6.0
FIXED_OUTPUT_OVERHEAD_BYTES: Final[int] = 64 * MIB

DEFAULT_IMAGE_SIZE_BYTES: Final[dict[str, int]] = {
    "hand-interaction": 1 * GIB,
    "hand-object-contact": 24 * GIB,
    "adl-recognition-core": 5 * GIB,
    "adl-recognition-detic": 24 * GIB,
}


class DiskSpacePreflightError(RuntimeError):
    """Raised when a model run cannot be given sufficient storage headroom."""


@dataclass(frozen=True, slots=True)
class PipelineStorageEstimate:
    """Conservative output and file-count estimate for one model request."""

    estimated_output_bytes: int
    peak_output_bytes: int
    estimated_file_count: int
    input_bytes: int


@dataclass(frozen=True, slots=True)
class DockerImageBuildPlan:
    """One runtime image that is missing and must be rebuilt."""

    identity: DockerImageIdentity
    estimated_final_size_bytes: int
    removed_stale_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiskSpaceReport:
    """Storage preflight result returned to CLI and GUI dry runs."""

    estimated_output_bytes: int
    peak_output_bytes: int
    output_free_bytes: int
    docker_incremental_bytes: int
    docker_free_bytes: int
    images_to_build: tuple[str, ...]
    removed_images: tuple[str, ...]
    estimated_file_count: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly dry-run details."""
        return {
            "estimatedOutputBytes": self.estimated_output_bytes,
            "peakOutputBytes": self.peak_output_bytes,
            "outputFreeBytes": self.output_free_bytes,
            "dockerIncrementalBytes": self.docker_incremental_bytes,
            "dockerFreeBytes": self.docker_free_bytes,
            "imagesToBuild": list(self.images_to_build),
            "removedImages": list(self.removed_images),
            "estimatedFileCount": self.estimated_file_count,
        }


@dataclass(slots=True)
class _FilesystemRequirement:
    path: Path
    free_bytes: int
    growth_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    size_bytes: int


def ensure_sufficient_disk_space(
    *,
    model_id: str,
    input_path: Path,
    output_dir: Path,
    docker_executable: str = "docker",
    capture_runner: CaptureRunner = capturing_subprocess_runner,
    progress: ProgressReporter | None = None,
    cleanup_stale_images: bool = True,
    platform_detector: PlatformDetector = platform.system,
    wsl_detector: WslDetector = is_wsl,
) -> DiskSpaceReport:
    """Estimate run storage and enforce free space before model execution.

    ``cleanup_stale_images`` is enabled for user-facing preflight checks so
    obsolete EgoModelKit image tags can free space before a replacement build.
    Both dry runs and immediate pre-run checks may opt into the same cleanup
    behavior for consistent capacity assessment; runtime image builders still
    clean stale managed tags when a replacement image is actually required.
    """
    report_progress = progress or (lambda _message: None)
    estimate = estimate_pipeline_storage(model_id=model_id, input_path=input_path)

    report_progress(
        "Estimated final pipeline output: "
        f"{format_bytes(estimate.estimated_output_bytes)}."
    )
    report_progress(
        "Estimated peak output working space: "
        f"{format_bytes(estimate.peak_output_bytes)}."
    )

    build_plans = _plan_runtime_image_builds(
        model_id=model_id,
        docker_executable=docker_executable,
        capture_runner=capture_runner,
        progress=report_progress,
        cleanup_stale_images=cleanup_stale_images,
    )
    docker_incremental_bytes = _docker_incremental_bytes(build_plans)

    output_anchor = _storage_anchor(
        output_dir,
        platform_detector=platform_detector,
        wsl_detector=wsl_detector,
    )
    docker_anchor = _docker_storage_anchor(
        docker_executable=docker_executable,
        capture_runner=capture_runner,
        platform_detector=platform_detector,
        wsl_detector=wsl_detector,
    )

    requirements: dict[tuple[int, int], _FilesystemRequirement] = {}
    output_requirement = _add_filesystem_requirement(
        requirements,
        output_anchor,
        estimate.peak_output_bytes,
    )
    docker_requirement = _add_filesystem_requirement(
        requirements,
        docker_anchor,
        docker_incremental_bytes,
    )

    for requirement in requirements.values():
        required = requirement.growth_bytes + MINIMUM_FREE_SPACE_RESERVE_BYTES
        if requirement.free_bytes < required:
            raise DiskSpacePreflightError(
                "Insufficient disk space on "
                f"{requirement.path}: requires about {format_bytes(required)} free "
                "for this run, including safety headroom, but only "
                f"{format_bytes(requirement.free_bytes)} is available."
            )

    _ensure_output_inodes(
        output_anchor,
        estimated_file_count=estimate.estimated_file_count,
    )

    image_names = tuple(plan.identity.runtime_name for plan in build_plans)
    removed_images = tuple(
        tag for plan in build_plans for tag in plan.removed_stale_tags
    )
    if image_names:
        report_progress(
            "Docker images to build: " + ", ".join(image_names) + "."
        )
        report_progress(
            "Estimated additional Docker build/runtime space: "
            f"{format_bytes(docker_incremental_bytes)}."
        )
    else:
        report_progress("All required EgoModelKit Docker images are already available.")

    report_progress("Disk-space preflight passed.")
    return DiskSpaceReport(
        estimated_output_bytes=estimate.estimated_output_bytes,
        peak_output_bytes=estimate.peak_output_bytes,
        output_free_bytes=output_requirement.free_bytes,
        docker_incremental_bytes=docker_incremental_bytes,
        docker_free_bytes=docker_requirement.free_bytes,
        images_to_build=image_names,
        removed_images=removed_images,
        estimated_file_count=estimate.estimated_file_count,
    )


def estimate_pipeline_storage(*, model_id: str, input_path: Path) -> PipelineStorageEstimate:
    """Return a conservative model-specific output estimate before inference."""
    if model_id == HAND_INTERACTION_MODEL_ID:
        videos = _supported_files(input_path, HAND_INTERACTION_SUPPORTED_VIDEO_SUFFIXES)
        metadata = [_probe_mp4(path, require_dimensions=False) for path in videos]
        frame_count = round(sum(video.duration_seconds for video in metadata) * 30)
        frame_bytes = frame_count * 720 * 405 * HAND_INTERACTION_BYTES_PER_PIXEL_PER_FRAME
        estimated_output = _with_margin(frame_bytes + FIXED_OUTPUT_OVERHEAD_BYTES)
        return PipelineStorageEstimate(
            estimated_output_bytes=estimated_output,
            peak_output_bytes=_with_margin(estimated_output, PEAK_MARGIN),
            estimated_file_count=max(1, frame_count * 4 + len(videos) * 10),
            input_bytes=sum(video.size_bytes for video in metadata),
        )

    if model_id == ADL_RECOGNITION_MODEL_ID:
        if input_path.is_file() and input_path.name == "all_preds.pkl":
            input_bytes = input_path.stat().st_size
            estimated_output = _with_margin(
                input_bytes + FIXED_OUTPUT_OVERHEAD_BYTES
            )
            return PipelineStorageEstimate(
                estimated_output_bytes=estimated_output,
                peak_output_bytes=_with_margin(estimated_output, PEAK_MARGIN),
                estimated_file_count=100,
                input_bytes=input_bytes,
            )

        videos = _supported_files(input_path, ADL_RECOGNITION_SUPPORTED_VIDEO_SUFFIXES)
        metadata = [_probe_mp4(path, require_dimensions=True) for path in videos]
        frame_artifact_bytes = sum(
            round(video.duration_seconds * ADL_INFERENCE_FRAME_FPS)
            * video.width
            * video.height
            * ADL_BYTES_PER_PIXEL_PER_FRAME
            for video in metadata
        )
        input_bytes = sum(video.size_bytes for video in metadata)
        subclip_bytes = input_bytes * ADL_SUBCLIP_BYTES_PER_SOURCE_BYTE
        estimated_output = _with_margin(
            frame_artifact_bytes + subclip_bytes + FIXED_OUTPUT_OVERHEAD_BYTES
        )
        peak_output = _with_margin(
            estimated_output + input_bytes,
            PEAK_MARGIN,
        )
        frame_count = round(
            sum(video.duration_seconds for video in metadata) * ADL_INFERENCE_FRAME_FPS
        )
        segment_count = sum(math.ceil(video.duration_seconds / 60) for video in metadata)
        return PipelineStorageEstimate(
            estimated_output_bytes=estimated_output,
            peak_output_bytes=peak_output,
            estimated_file_count=max(
                1,
                frame_count * 5 + segment_count * 3 + len(videos) * 10,
            ),
            input_bytes=input_bytes,
        )

    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        images = _supported_files(input_path, HAND_OBJECT_CONTACT_SUPPORTED_IMAGE_SUFFIXES)
        input_bytes = sum(path.stat().st_size for path in images)
        estimated_output = _with_margin(
            input_bytes * HOC_OUTPUT_BYTES_PER_INPUT_BYTE + FIXED_OUTPUT_OVERHEAD_BYTES
        )
        return PipelineStorageEstimate(
            estimated_output_bytes=estimated_output,
            peak_output_bytes=_with_margin(estimated_output, PEAK_MARGIN),
            estimated_file_count=max(1, len(images) * 3 + 20),
            input_bytes=input_bytes,
        )

    raise ValueError(f"Unsupported model id: {model_id}")


def format_bytes(value: int) -> str:
    """Format a byte count for concise user-facing dry-run output."""
    if value >= GIB:
        return f"{value / GIB:.1f} GiB"
    return f"{value / MIB:.1f} MiB"


def _with_margin(value: float | int, margin: float = ESTIMATE_MARGIN) -> int:
    return math.ceil(float(value) * margin)


def _supported_files(input_path: Path, suffixes: frozenset[str]) -> list[Path]:
    candidates = [input_path] if input_path.is_file() else sorted(input_path.iterdir())
    files = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not files:
        raise DiskSpacePreflightError(
            "Disk-space estimation found no supported input files."
        )
    return files


def _probe_mp4(path: Path, *, require_dimensions: bool) -> _VideoMetadata:
    duration = 0.0
    width = 0
    height = 0
    file_size = path.stat().st_size

    with path.open("rb") as stream:
        for box_type, payload_start, box_end in _iter_boxes(stream, 0, file_size):
            if box_type != b"moov":
                continue
            for child_type, child_start, child_end in _iter_boxes(
                stream,
                payload_start,
                box_end,
            ):
                if child_type == b"mvhd":
                    duration = _mvhd_duration(stream, child_start, child_end)
                elif child_type == b"trak":
                    track_width, track_height = _trak_dimensions(
                        stream,
                        child_start,
                        child_end,
                    )
                    if track_width > 0 and track_height > 0:
                        width = max(width, track_width)
                        height = max(height, track_height)
            break

    if duration <= 0:
        raise DiskSpacePreflightError(
            f"Could not determine MP4 duration for disk-space estimation: {path.name}."
        )
    if require_dimensions and (width <= 0 or height <= 0):
        raise DiskSpacePreflightError(
            f"Could not determine MP4 frame dimensions for disk-space estimation: {path.name}."
        )

    return _VideoMetadata(
        duration_seconds=duration,
        width=width,
        height=height,
        size_bytes=file_size,
    )


def _iter_boxes(stream, start: int, end: int):
    position = start
    while position + 8 <= end:
        stream.seek(position)
        header = stream.read(8)
        if len(header) != 8:
            return
        size32, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size32 == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                return
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size32 == 0:
            size = end - position
        else:
            size = size32
        if size < header_size or position + size > end:
            return
        yield box_type, position + header_size, position + size
        position += size


def _mvhd_duration(stream, start: int, end: int) -> float:
    stream.seek(start)
    payload = stream.read(min(end - start, 32))
    if len(payload) < 20:
        return 0.0
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return 0.0
        timescale = struct.unpack_from(">I", payload, 20)[0]
        duration = struct.unpack_from(">Q", payload, 24)[0]
    else:
        timescale = struct.unpack_from(">I", payload, 12)[0]
        duration = struct.unpack_from(">I", payload, 16)[0]
    return duration / timescale if timescale else 0.0


def _trak_dimensions(stream, start: int, end: int) -> tuple[int, int]:
    for box_type, payload_start, box_end in _iter_boxes(stream, start, end):
        if box_type != b"tkhd":
            continue
        stream.seek(payload_start)
        payload = stream.read(min(box_end - payload_start, 96))
        if len(payload) < 84:
            return 0, 0
        version = payload[0]
        offset = 88 if version == 1 else 76
        if len(payload) < offset + 8:
            return 0, 0
        width_fixed, height_fixed = struct.unpack_from(">II", payload, offset)
        return width_fixed >> 16, height_fixed >> 16
    return 0, 0


def _runtime_image_identities(model_id: str) -> tuple[DockerImageIdentity, ...]:
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        from egomodelkit.runtime.hand_object_contact import hand_object_contact_image_identity

        return (hand_object_contact_image_identity(),)
    if model_id == HAND_INTERACTION_MODEL_ID:
        from egomodelkit.runtime.hand_interaction import hand_interaction_image_identity
        from egomodelkit.runtime.hand_object_contact import hand_object_contact_image_identity

        return (hand_interaction_image_identity(), hand_object_contact_image_identity())
    if model_id == ADL_RECOGNITION_MODEL_ID:
        from egomodelkit.runtime.adl_recognition import (
            adl_core_image_identity,
            adl_detic_image_identity,
        )
        from egomodelkit.runtime.hand_object_contact import hand_object_contact_image_identity

        return (
            adl_core_image_identity(),
            adl_detic_image_identity(),
            hand_object_contact_image_identity(),
        )
    raise ValueError(f"Unsupported model id: {model_id}")


def _plan_runtime_image_builds(
    *,
    model_id: str,
    docker_executable: str,
    capture_runner: CaptureRunner,
    progress: ProgressReporter,
    cleanup_stale_images: bool,
) -> tuple[DockerImageBuildPlan, ...]:
    plans: list[DockerImageBuildPlan] = []
    for identity in _runtime_image_identities(model_id):
        if _docker_image_size_bytes(
            docker_executable=docker_executable,
            image_tag=identity.tag,
            capture_runner=capture_runner,
        ) is not None:
            continue

        stale_tags = _repository_image_tags(
            docker_executable=docker_executable,
            repository=identity.repository,
            capture_runner=capture_runner,
        )
        stale_sizes = [
            size
            for tag in stale_tags
            if tag != identity.tag
            if (
                size := _docker_image_size_bytes(
                    docker_executable=docker_executable,
                    image_tag=tag,
                    capture_runner=capture_runner,
                )
            )
            is not None
        ]
        estimated_size = max(
            [DEFAULT_IMAGE_SIZE_BYTES[identity.runtime_name], *stale_sizes]
        )
        removed = (
            remove_stale_runtime_images(
                docker_executable=docker_executable,
                current_image=identity,
                capture_runner=capture_runner,
                progress=progress,
            )
            if cleanup_stale_images
            else ()
        )
        plans.append(
            DockerImageBuildPlan(
                identity=identity,
                estimated_final_size_bytes=estimated_size,
                removed_stale_tags=removed,
            )
        )
    return tuple(plans)


def _docker_image_size_bytes(
    *,
    docker_executable: str,
    image_tag: str,
    capture_runner: CaptureRunner,
) -> int | None:
    result = capture_runner(
        [docker_executable, "image", "inspect", image_tag, "--format", "{{.Size}}"]
    )
    if result.returncode != 0:
        return None
    try:
        value = int(result.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _repository_image_tags(
    *,
    docker_executable: str,
    repository: str,
    capture_runner: CaptureRunner,
) -> tuple[str, ...]:
    result = capture_runner(
        [
            docker_executable,
            "image",
            "ls",
            repository,
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ]
    )
    if result.returncode != 0:
        return ()
    return tuple(
        sorted(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and line.strip() != "<none>:<none>"
        )
    )


def _docker_incremental_bytes(plans: tuple[DockerImageBuildPlan, ...]) -> int:
    if not plans:
        return DOCKER_RUNTIME_TRANSIENT_BYTES
    final_bytes = sum(plan.estimated_final_size_bytes for plan in plans)
    largest_image = max(plan.estimated_final_size_bytes for plan in plans)
    scratch_bytes = max(
        DOCKER_BUILD_MINIMUM_SCRATCH_BYTES,
        math.ceil(largest_image * DOCKER_BUILD_SCRATCH_RATIO),
    )
    return final_bytes + scratch_bytes + DOCKER_RUNTIME_TRANSIENT_BYTES


def _docker_storage_anchor(
    *,
    docker_executable: str,
    capture_runner: CaptureRunner,
    platform_detector: PlatformDetector,
    wsl_detector: WslDetector,
) -> Path:
    if wsl_detector() and Path("/mnt/c").exists():
        return Path("/mnt/c")

    info = capture_runner(
        [docker_executable, "info", "--format", "{{.DockerRootDir}}"]
    )
    docker_root = Path(info.stdout.strip()) if info.returncode == 0 else Path()
    if info.returncode == 0 and info.stdout.strip() and docker_root.exists():
        return docker_root

    if platform_detector() == "Darwin":
        return Path.home()
    return _nearest_existing_path(Path.cwd())


def _storage_anchor(
    path: Path,
    *,
    platform_detector: PlatformDetector,
    wsl_detector: WslDetector,
) -> Path:
    del platform_detector
    existing = _nearest_existing_path(path)
    if not wsl_detector():
        return existing

    parts = existing.resolve().parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive_root = Path("/mnt") / parts[2]
        if drive_root.exists():
            return drive_root
    if Path("/mnt/c").exists():
        return Path("/mnt/c")
    return existing


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _filesystem_key(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino if path.is_file() else 0


def _add_filesystem_requirement(
    requirements: dict[tuple[int, int], _FilesystemRequirement],
    path: Path,
    growth_bytes: int,
) -> _FilesystemRequirement:
    key = (_filesystem_key(path)[0], 0)
    requirement = requirements.get(key)
    if requirement is None:
        requirement = _FilesystemRequirement(
            path=path,
            free_bytes=shutil.disk_usage(path).free,
        )
        requirements[key] = requirement
    requirement.growth_bytes += growth_bytes
    return requirement


def _ensure_output_inodes(path: Path, *, estimated_file_count: int) -> None:
    if not hasattr(os, "statvfs"):
        return
    stats = os.statvfs(path)
    available = stats.f_favail
    if available <= 0:
        return
    required = estimated_file_count + MINIMUM_FREE_INODE_RESERVE
    if available < required:
        raise DiskSpacePreflightError(
            "Insufficient filesystem inodes for the estimated per-frame outputs on "
            f"{path}: requires about {required:,} available file entries, but only "
            f"{available:,} are available."
        )
