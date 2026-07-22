"""Hidden runtime execution for standalone hand-interaction inference."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Final

from egomodelkit.bandini_metrics import DEFAULT_VIDEO_PROCESSING_CONFIG, VideoProcessingConfig
from egomodelkit.models.hand_interaction import (
    HandInteractionRequest,
    validate_hand_interaction_request,
)
from egomodelkit.models.hand_object_contact import HandObjectContactRequest
from egomodelkit.progress import external_progress_line, parse_external_progress_line
from egomodelkit.runtime.commands import (
    CommandResult,
    capturing_subprocess_runner,
    subprocess_runner,
)
from egomodelkit.runtime.docker_images import (
    DockerImageIdentity,
    build_runtime_image_identity,
    remove_stale_runtime_images,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    HandObjectContactRuntimeSpec,
    run_hand_object_contact,
)
from egomodelkit.runtime.preflight import (
    ExecutableLocator,
    PlatformDetector,
    ProgressReporter,
    ensure_host_runtime_ready,
)

CommandRunner = Callable[[list[str]], int]
CaptureRunner = Callable[[list[str]], CommandResult]
StreamingCommandRunner = Callable[[list[str], ProgressReporter], int]

HAND_INTERACTION_INPUT_MANIFEST_FILENAME: Final[str] = (
    "hand_interaction_input_manifest.csv"
)
HAND_INTERACTION_SUBCLIP_MANIFEST_FILENAME: Final[str] = (
    "hand_interaction_subclip_manifest.csv"
)
METRICS_CONFIG_FILENAME: Final[str] = "metrics_config.json"


@dataclass(frozen=True, slots=True)
class HandInteractionRuntimeSpec:
    """Build and execution settings for standalone hand interaction."""

    docker_executable: str
    image_repository: str
    container_input_dir: PurePosixPath
    container_output_dir: PurePosixPath
    work_dir_name: str
    video_processing_config: VideoProcessingConfig
    hand_object_contact_runtime_spec: HandObjectContactRuntimeSpec
    host_uid: int
    host_gid: int

    @property
    def image_tag(self) -> str:
        """Return the content-addressed frame-extraction image tag."""
        return hand_interaction_image_identity(self).tag


DEFAULT_HAND_INTERACTION_RUNTIME_SPEC: Final[HandInteractionRuntimeSpec] = (
    HandInteractionRuntimeSpec(
        docker_executable="docker",
        image_repository="egomodelkit-hand-interaction",
        container_input_dir=PurePosixPath("/workspace/input"),
        container_output_dir=PurePosixPath("/workspace/output"),
        work_dir_name="hand_interaction_work",
        video_processing_config=DEFAULT_VIDEO_PROCESSING_CONFIG,
        hand_object_contact_runtime_spec=DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
        host_uid=os.getuid(),
        host_gid=os.getgid(),
    )
)


class HandInteractionRuntimeError(RuntimeError):
    """Raised when standalone hand-interaction runtime execution fails."""


def _ignore_progress(_: str) -> None:
    """Default no-op progress reporter."""


def _container_resource_dir() -> Path:
    return Path(
        str(files("egomodelkit").joinpath("resources/containers/hand_interaction"))
    )


def hand_interaction_image_identity(
    runtime_spec: HandInteractionRuntimeSpec = DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
) -> DockerImageIdentity:
    """Return the deterministic identity for the frame-extraction image."""
    return build_runtime_image_identity(
        runtime_name="hand-interaction",
        repository=runtime_spec.image_repository,
        context_dir=_container_resource_dir(),
        build_arguments=[],
    )


def ensure_runtime_image(
    *,
    runtime_spec: HandInteractionRuntimeSpec = DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    streaming_command_runner: StreamingCommandRunner | None = None,
    capture_runner: CaptureRunner = capturing_subprocess_runner,
    progress: ProgressReporter = _ignore_progress,
) -> None:
    """Build the lightweight frame-extraction image only when missing."""
    progress("Checking packaged hand-interaction runtime image.")
    image_identity = hand_interaction_image_identity(runtime_spec)
    inspect_command = [
        runtime_spec.docker_executable,
        "image",
        "inspect",
        image_identity.tag,
    ]

    if command_runner(inspect_command) == 0:
        progress("Packaged hand-interaction runtime image is already available.")
        return

    progress(
        "Packaged hand-interaction runtime image is missing; preparing it now. "
        "The first run may take longer."
    )
    context_dir = _container_resource_dir()
    build_command = [
        runtime_spec.docker_executable,
        "build",
        "-f",
        str(context_dir / "Dockerfile"),
        "-t",
        image_identity.tag,
        *image_identity.label_arguments,
        str(context_dir),
    ]
    exit_code = (
        command_runner(build_command)
        if streaming_command_runner is None
        else streaming_command_runner(build_command, progress)
    )

    if exit_code != 0:
        raise HandInteractionRuntimeError(
            f"hand-interaction runtime image build failed with exit code {exit_code}."
        )

    progress("Packaged hand-interaction runtime image is ready.")
    remove_stale_runtime_images(
        docker_executable=runtime_spec.docker_executable,
        current_image=image_identity,
        capture_runner=capture_runner,
        progress=progress,
    )


def build_extract_run_command(
    request: HandInteractionRequest,
    *,
    runtime_spec: HandInteractionRuntimeSpec = DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
) -> list[str]:
    """Build the CPU Docker command that extracts Bandini-configured frames."""
    config = runtime_spec.video_processing_config
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "-v",
        _input_mount_argument(request, runtime_spec=runtime_spec),
        "-v",
        f"{request.output_dir.resolve()}:{runtime_spec.container_output_dir}",
        runtime_spec.image_tag,
        "--input-path",
        str(_container_input_path(request, runtime_spec=runtime_spec)),
        "--output-dir",
        str(runtime_spec.container_output_dir),
        "--work-dir-name",
        runtime_spec.work_dir_name,
        "--subclip-length",
        str(config.subclip_length_seconds),
        "--processing-fps",
        str(config.frame_fps),
        "--resize-width",
        str(config.resize_width),
        "--resize-height",
        str(config.resize_height),
        "--pooling-window-seconds",
        str(config.pooling_window_seconds),
        "--interaction-contact-state-threshold",
        str(config.interaction_contact_state_threshold),
        "--dominant-hand",
        config.dominant_hand,
    ]


def build_output_ownership_repair_command(
    output_dir: Path,
    *,
    runtime_spec: HandInteractionRuntimeSpec = DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
) -> list[str]:
    """Build a Docker command that restores host ownership of mounted output."""
    return [
        runtime_spec.docker_executable,
        "run",
        "--rm",
        "--entrypoint",
        "chown",
        "-v",
        f"{output_dir.resolve()}:{runtime_spec.container_output_dir}",
        runtime_spec.image_tag,
        "-R",
        f"{runtime_spec.host_uid}:{runtime_spec.host_gid}",
        str(runtime_spec.container_output_dir),
    ]


def run_hand_interaction(
    request: HandInteractionRequest,
    *,
    runtime_spec: HandInteractionRuntimeSpec = DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
    command_runner: CommandRunner = subprocess_runner,
    streaming_command_runner: StreamingCommandRunner | None = None,
    capture_runner: CaptureRunner = capturing_subprocess_runner,
    executable_locator: ExecutableLocator | None = None,
    platform_detector: PlatformDetector | None = None,
    progress: ProgressReporter = _ignore_progress,
) -> list[list[str]]:
    """Run frame extraction and HOC inference without any ADL stages."""
    progress("Validating hand-interaction request.")
    validate_hand_interaction_request(request)
    runtime_spec = replace(
        runtime_spec,
        video_processing_config=replace(
            runtime_spec.video_processing_config,
            dominant_hand=request.dominant_hand,
        ),
    )
    runtime_check_kwargs = _runtime_check_overrides(
        executable_locator=executable_locator,
        platform_detector=platform_detector,
    )
    ensure_host_runtime_ready(
        docker_executable=runtime_spec.docker_executable,
        command_runner=command_runner,
        require_linux_nvidia_gpu=True,
        progress=progress,
        **runtime_check_kwargs,
    )

    request.output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Using output directory: {request.output_dir}")
    ensure_runtime_image(
        runtime_spec=runtime_spec,
        command_runner=command_runner,
        streaming_command_runner=streaming_command_runner,
        capture_runner=capture_runner,
        progress=progress,
    )

    executed_commands: list[list[str]] = []
    extract_command = build_extract_run_command(request, runtime_spec=runtime_spec)
    _run_stage(
        extract_command,
        command_runner=command_runner,
        streaming_command_runner=streaming_command_runner,
        stage_name="hand-interaction video frame extraction",
        progress=progress,
    )
    executed_commands.append(extract_command)
    executed_commands.append(
        _repair_output_ownership(
            request.output_dir,
            runtime_spec=runtime_spec,
            command_runner=command_runner,
            streaming_command_runner=streaming_command_runner,
            progress=progress,
        )
    )

    frame_dirs = _subclip_frame_dirs(request.output_dir, runtime_spec=runtime_spec)
    if not frame_dirs:
        raise HandInteractionRuntimeError(
            "Hand-interaction frame extraction completed but no frame directories were found."
        )

    global_frame_total = sum(_frame_count(frame_dir) for frame_dir in frame_dirs)
    if global_frame_total <= 0:
        raise HandInteractionRuntimeError(
            "Hand-interaction frame extraction completed but no frame images were found."
        )

    progress(
        external_progress_line(
            "hand_interaction_frame_extracted",
            current=global_frame_total,
            total=global_frame_total,
        )
    )
    processed_frames = 0

    for frame_dir in frame_dirs:
        frame_count = _frame_count(frame_dir)
        shan_output_dir = _shan_output_dir(
            request.output_dir,
            frame_dir.name,
            runtime_spec=runtime_spec,
        )
        next_processed_frames = processed_frames + frame_count
        command_result = run_hand_object_contact(
            HandObjectContactRequest(
                input_path=frame_dir,
                output_dir=shan_output_dir,
            ),
            runtime_spec=runtime_spec.hand_object_contact_runtime_spec,
            command_runner=command_runner,
            streaming_command_runner=streaming_command_runner,
            capture_runner=capture_runner,
            executable_locator=executable_locator,
            platform_detector=platform_detector,
            progress=_global_frame_progress(
                progress,
                offset=processed_frames,
                total=global_frame_total,
            ),
        )
        progress(
            external_progress_line(
                "hand_interaction_hoc_frame_processed",
                current=next_processed_frames,
                total=global_frame_total,
            )
        )
        processed_frames = next_processed_frames
        _append_executed_command_result(executed_commands, command_result)
        executed_commands.append(
            _repair_output_ownership(
                request.output_dir,
                runtime_spec=runtime_spec,
                command_runner=command_runner,
                streaming_command_runner=streaming_command_runner,
                progress=progress,
            )
        )

    progress("hand-interaction inference completed.")
    return executed_commands


def _run_stage(
    command: list[str],
    *,
    command_runner: CommandRunner,
    streaming_command_runner: StreamingCommandRunner | None,
    stage_name: str,
    progress: ProgressReporter,
) -> None:
    progress(f"Starting {stage_name}.")
    exit_code = (
        command_runner(command)
        if streaming_command_runner is None
        else streaming_command_runner(command, progress)
    )
    if exit_code != 0:
        raise HandInteractionRuntimeError(
            f"{stage_name} failed with exit code {exit_code}."
        )
    progress(f"Finished {stage_name}.")


def _repair_output_ownership(
    output_dir: Path,
    *,
    runtime_spec: HandInteractionRuntimeSpec,
    command_runner: CommandRunner,
    streaming_command_runner: StreamingCommandRunner | None,
    progress: ProgressReporter,
) -> list[str]:
    command = build_output_ownership_repair_command(output_dir, runtime_spec=runtime_spec)
    _run_stage(
        command,
        command_runner=command_runner,
        streaming_command_runner=streaming_command_runner,
        stage_name="hand-interaction output ownership repair",
        progress=progress,
    )
    return command


def _global_frame_progress(
    progress: ProgressReporter,
    *,
    offset: int,
    total: int,
) -> ProgressReporter:
    """Convert per-subclip HOC progress into global frame progress."""

    def report(message: str) -> None:
        update = parse_external_progress_line(message)
        if update is None or update.kind != "hand_object_image_processed":
            progress(message)
            return

        progress(
            external_progress_line(
                "hand_interaction_hoc_frame_processed",
                current=offset + _payload_int(update.payload, "current"),
                total=total,
            )
        )

    return report


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _append_executed_command_result(
    executed_commands: list[list[str]],
    command_result: list[str] | list[list[str]],
) -> None:
    if not command_result:
        return
    if all(isinstance(part, str) for part in command_result):
        executed_commands.append(command_result)
        return
    executed_commands.extend(command_result)


def _runtime_check_overrides(
    *,
    executable_locator: ExecutableLocator | None,
    platform_detector: PlatformDetector | None,
) -> dict[str, ExecutableLocator | PlatformDetector]:
    overrides: dict[str, ExecutableLocator | PlatformDetector] = {}
    if executable_locator is not None:
        overrides["executable_locator"] = executable_locator
    if platform_detector is not None:
        overrides["platform_detector"] = platform_detector
    return overrides


def _container_input_path(
    request: HandInteractionRequest,
    *,
    runtime_spec: HandInteractionRuntimeSpec,
) -> PurePosixPath:
    if request.input_path.is_dir():
        return runtime_spec.container_input_dir
    return runtime_spec.container_input_dir / request.input_path.name


def _input_mount_argument(
    request: HandInteractionRequest,
    *,
    runtime_spec: HandInteractionRuntimeSpec,
) -> str:
    container_input_path = _container_input_path(request, runtime_spec=runtime_spec)
    return f"{request.input_path.resolve()}:{container_input_path}:ro"


def _work_dir(output_dir: Path, *, runtime_spec: HandInteractionRuntimeSpec) -> Path:
    return output_dir / runtime_spec.work_dir_name


def _subclip_frame_dirs(
    output_dir: Path,
    *,
    runtime_spec: HandInteractionRuntimeSpec,
) -> list[Path]:
    extracted_frames_dir = _work_dir(output_dir, runtime_spec=runtime_spec) / "extracted_frames"
    if not extracted_frames_dir.is_dir():
        return []
    return sorted(
        child
        for child in extracted_frames_dir.iterdir()
        if child.is_dir() and any(frame.suffix.lower() == ".jpg" for frame in child.iterdir())
    )


def _shan_output_dir(
    output_dir: Path,
    clip_name: str,
    *,
    runtime_spec: HandInteractionRuntimeSpec,
) -> Path:
    return _work_dir(output_dir, runtime_spec=runtime_spec) / "shan_outputs" / clip_name


def _frame_count(frame_dir: Path) -> int:
    return sum(1 for frame in frame_dir.iterdir() if frame.suffix.lower() == ".jpg")
