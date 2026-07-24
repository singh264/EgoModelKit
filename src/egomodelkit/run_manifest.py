"""Reproducibility manifests for EgoModelKit runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Final

from egomodelkit import __version__
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.runtime.adl_recognition import (
    DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC,
    adl_core_image_identity,
    adl_detic_image_identity,
)
from egomodelkit.runtime.commands import CommandResult, capturing_subprocess_runner
from egomodelkit.runtime.docker_images import DockerImageIdentity
from egomodelkit.runtime.external_code import (
    DETECTRON2_PIN,
    DETIC_PIN,
    DETIC_WEIGHTS_PIN,
    EGOVIZML_PIN,
    HAND_OBJECT_DETECTOR_PIN,
    HAND_OBJECT_DETECTOR_WEIGHTS_PIN,
    ExternalModelAssetPin,
    ExternalModelCodePin,
)
from egomodelkit.runtime.hand_interaction import (
    DEFAULT_HAND_INTERACTION_RUNTIME_SPEC,
    hand_interaction_image_identity,
)
from egomodelkit.runtime.hand_object_contact import (
    DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC,
    HandObjectContactRuntimeSpec,
    hand_object_contact_image_identity,
)
from egomodelkit.runtime.host_platform import is_wsl

CaptureRunner = Callable[[list[str]], CommandResult]

RUN_MANIFEST_SCHEMA_VERSION: Final[int] = 2
EGOMODELKIT_REPOSITORY_URL: Final[str] = "https://github.com/singh264/EgoModelKit"
EGOVIZML_CLASSIFIER_REPOSITORY_PATH: Final[str] = (
    "models/binary_active_logreg.joblib"
)
TERMINAL_RUN_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "cancelled"}
)


def build_run_manifest(
    *,
    run_id: str,
    model_id: str,
    input_path: Path,
    input_names: tuple[str, ...],
    scenario: str,
    status: str,
    output_folder: str,
    output_contract_version: int,
    model_configuration: dict[str, object],
    invocation_interface: str | None = None,
    invocation_arguments: tuple[str, ...] | None = None,
    previous_manifest: dict[str, object] | None = None,
    error_message: str | None = None,
    collect_runtime_state: bool = False,
    capture_runner: CaptureRunner | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Build a stable, model-specific reproducibility manifest."""
    timestamp = _utc_timestamp(now)
    previous_run = _mapping_value(previous_manifest, "run")
    previous_invocation = _mapping_value(previous_manifest, "invocation")
    created_at = _string_value(previous_run, "created_at_utc") or timestamp
    completed_at = (
        timestamp
        if status in TERMINAL_RUN_STATUSES
        else _string_value(previous_run, "completed_at_utc")
    )

    warnings: list[str] = []
    runtime, code_pins, asset_pins = _runtime_definition(
        model_id=model_id,
        scenario=scenario,
    )
    runner = capture_runner or capturing_subprocess_runner

    if collect_runtime_state:
        runtime["host_docker"] = _docker_version(
            docker_executable=str(runtime["docker_executable"]),
            capture_runner=runner,
            warnings=warnings,
        )
        runtime["host_nvidia"] = _nvidia_details(
            capture_runner=runner,
            warnings=warnings,
        )
        runtime["images"] = [
            _with_image_inspection(
                image,
                docker_executable=str(runtime["docker_executable"]),
                capture_runner=runner,
                warnings=warnings,
            )
            for image in _image_entries(runtime)
        ]

    run_payload: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "created_at_utc": created_at,
        "updated_at_utc": timestamp,
        "completed_at_utc": completed_at,
        "output_folder": output_folder,
    }
    if error_message is not None:
        run_payload["error_message"] = error_message

    model_artifacts = [_asset_entry(pin) for pin in asset_pins]
    if model_id == ADL_RECOGNITION_MODEL_ID:
        model_artifacts.insert(0, _egovizml_classifier_entry())

    return {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "model_id": model_id,
        "input_names": list(input_names),
        "scenario": scenario,
        "output_contract_version": output_contract_version,
        "model_configuration": model_configuration,
        "run": run_payload,
        "invocation": {
            "interface": (
                invocation_interface
                or _string_value(previous_invocation, "interface")
                or "unknown"
            ),
            "arguments": (
                list(invocation_arguments)
                if invocation_arguments is not None
                else _string_list_value(previous_invocation, "arguments")
            ),
        },
        "egomodelkit": _egomodelkit_provenance(runner),
        "host": _host_details(),
        "inputs": _input_details(input_path, input_names, warnings),
        "runtime": runtime,
        "external_code": [
            _code_entry(pin, scenario=scenario) for pin in code_pins
        ],
        "model_artifacts": model_artifacts,
        "collection_warnings": warnings,
        "notes": (
            "Absolute paths describe the machine at execution time. Docker image IDs "
            "are collected at terminal run states when the images are available."
        ),
    }


def _runtime_definition(
    *,
    model_id: str,
    scenario: str,
) -> tuple[
    dict[str, object],
    tuple[ExternalModelCodePin, ...],
    tuple[ExternalModelAssetPin, ...],
]:
    hoc_spec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC
    hoc_checkpoint_path = (
        Path("/opt/hand_object_detector")
        / hoc_spec.shan_load_dir
        / hoc_spec.shan_model_subdir
        / hoc_spec.checkpoint_filename
    ).as_posix()

    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        return (
            {
                "docker_executable": hoc_spec.docker_executable,
                "images": [_image_entry(hand_object_contact_image_identity(hoc_spec))],
                "parameters": _hoc_runtime_parameters(hoc_spec),
                "container_paths": {
                    "hand_object_detector_repository": "/opt/hand_object_detector",
                    "hand_object_detector_checkpoint": hoc_checkpoint_path,
                },
            },
            (HAND_OBJECT_DETECTOR_PIN,),
            (HAND_OBJECT_DETECTOR_WEIGHTS_PIN,),
        )

    if model_id == HAND_INTERACTION_MODEL_ID:
        interaction_spec = DEFAULT_HAND_INTERACTION_RUNTIME_SPEC
        return (
            {
                "docker_executable": interaction_spec.docker_executable,
                "images": [
                    _image_entry(hand_interaction_image_identity(interaction_spec)),
                    _image_entry(hand_object_contact_image_identity(hoc_spec)),
                ],
                "parameters": {
                    "work_dir_name": interaction_spec.work_dir_name,
                    "container_input_dir": str(interaction_spec.container_input_dir),
                    "container_output_dir": str(interaction_spec.container_output_dir),
                    "hand_object_contact": _hoc_runtime_parameters(hoc_spec),
                },
                "container_paths": {
                    "hand_interaction_entrypoint": (
                        "/opt/egomodelkit_hand_interaction_entrypoint.py"
                    ),
                    "hand_object_detector_repository": "/opt/hand_object_detector",
                    "hand_object_detector_checkpoint": hoc_checkpoint_path,
                },
            },
            (HAND_OBJECT_DETECTOR_PIN,),
            (HAND_OBJECT_DETECTOR_WEIGHTS_PIN,),
        )

    if model_id == ADL_RECOGNITION_MODEL_ID:
        adl_spec = DEFAULT_ADL_RECOGNITION_RUNTIME_SPEC
        combined_predictions = scenario == "adl-combined-predictions"
        images = [_image_entry(adl_core_image_identity(adl_spec))]
        code_pins: tuple[ExternalModelCodePin, ...] = (EGOVIZML_PIN,)
        asset_pins: tuple[ExternalModelAssetPin, ...] = ()

        if not combined_predictions:
            images.extend(
                [
                    _image_entry(adl_detic_image_identity(adl_spec)),
                    _image_entry(hand_object_contact_image_identity(hoc_spec)),
                ]
            )
            code_pins = (
                EGOVIZML_PIN,
                DETIC_PIN,
                DETECTRON2_PIN,
                HAND_OBJECT_DETECTOR_PIN,
            )
            asset_pins = (DETIC_WEIGHTS_PIN, HAND_OBJECT_DETECTOR_WEIGHTS_PIN)

        return (
            {
                "docker_executable": adl_spec.docker_executable,
                "images": images,
                "parameters": {
                    "work_dir_name": adl_spec.work_dir_name,
                    "staged_adl_dir_name": adl_spec.staged_adl_dir_name,
                    "container_input_dir": str(adl_spec.container_input_dir),
                    "container_output_dir": str(adl_spec.container_output_dir),
                    "detic_confidence_threshold": adl_spec.detic_confidence_threshold,
                    "detic_num_workers": adl_spec.detic_num_workers,
                    "segment_length_seconds": adl_spec.segment_length_seconds,
                    "subclip_encoding_fps": adl_spec.subclip_encoding_fps,
                    "inference_frame_fps": adl_spec.inference_frame_fps,
                    "active_object_iou_threshold": adl_spec.active_iou,
                    "hand_object_contact": (
                        None
                        if combined_predictions
                        else _hoc_runtime_parameters(hoc_spec)
                    ),
                },
                "container_paths": {
                    "egovizml_repository": [
                        "adl-recognition-core:/opt/EgoVizML",
                        *(
                            []
                            if combined_predictions
                            else ["adl-recognition-detic:/opt/EgoVizML"]
                        ),
                    ],
                    "egovizml_classifier": (
                        "adl-recognition-core:/opt/EgoVizML/"
                        f"{EGOVIZML_CLASSIFIER_REPOSITORY_PATH}"
                    ),
                    "detic_repository": (
                        None if combined_predictions else "/opt/Detic"
                    ),
                    "detectron2_repository": (
                        None if combined_predictions else "/opt/detectron2"
                    ),
                    "detic_checkpoint": (
                        None
                        if combined_predictions
                        else f"/opt/Detic/models/{adl_spec.detic_weights_filename}"
                    ),
                    "hand_object_detector_repository": (
                        None if combined_predictions else "/opt/hand_object_detector"
                    ),
                    "hand_object_detector_checkpoint": (
                        None if combined_predictions else hoc_checkpoint_path
                    ),
                },
            },
            code_pins,
            asset_pins,
        )

    raise ValueError(f"Unsupported model id: {model_id}")


def _hoc_runtime_parameters(
    runtime_spec: HandObjectContactRuntimeSpec,
) -> dict[str, object]:
    return {
        "checkpoint_filename": runtime_spec.checkpoint_filename,
        "checkpoint_session": runtime_spec.checkpoint_session,
        "checkpoint_epoch": runtime_spec.checkpoint_epoch,
        "checkpoint_step": runtime_spec.checkpoint_step,
        "shan_network_name": runtime_spec.shan_network_name,
        "shan_dataset_name": runtime_spec.shan_dataset_name,
        "shan_load_dir": runtime_spec.shan_load_dir,
        "pytorch_version": runtime_spec.pytorch_version,
        "torchvision_version": runtime_spec.torchvision_version,
        "torchaudio_version": runtime_spec.torchaudio_version,
        "pytorch_cuda_index_url": runtime_spec.pytorch_cuda_index_url,
        "torch_cuda_arch_list": runtime_spec.torch_cuda_arch_list,
    }


def _image_entry(identity: DockerImageIdentity) -> dict[str, object]:
    return {
        "runtime_name": identity.runtime_name,
        "repository": identity.repository,
        "tag": identity.tag,
        "build_fingerprint_sha256": identity.fingerprint,
        "inspection": {"status": "not_collected"},
    }


def _image_entries(runtime: dict[str, object]) -> list[dict[str, object]]:
    value = runtime.get("images")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _with_image_inspection(
    image: dict[str, object],
    *,
    docker_executable: str,
    capture_runner: CaptureRunner,
    warnings: list[str],
) -> dict[str, object]:
    updated = dict(image)
    tag = str(image["tag"])
    result = _safe_capture(
        [docker_executable, "image", "inspect", tag],
        capture_runner=capture_runner,
    )
    if result is None or result.returncode != 0:
        updated["inspection"] = {"status": "unavailable"}
        warnings.append(f"Docker image inspection was unavailable for {tag}.")
        return updated

    try:
        payload = json.loads(result.stdout)
        details = payload[0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        details = None

    if not isinstance(details, dict):
        updated["inspection"] = {"status": "invalid_response"}
        warnings.append(f"Docker image inspection returned invalid JSON for {tag}.")
        return updated

    config = details.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    provenance_labels = {
        key: value
        for key, value in (labels.items() if isinstance(labels, dict) else ())
        if key.startswith("org.egomodelkit.")
    }
    updated["inspection"] = {
        "status": "available",
        "image_id": details.get("Id"),
        "repo_tags": details.get("RepoTags") or [],
        "repo_digests": details.get("RepoDigests") or [],
        "created_at": details.get("Created"),
        "os": details.get("Os"),
        "architecture": details.get("Architecture"),
        "size_bytes": details.get("Size"),
        "provenance_labels": provenance_labels,
    }
    return updated


def _docker_version(
    *,
    docker_executable: str,
    capture_runner: CaptureRunner,
    warnings: list[str],
) -> dict[str, object]:
    result = _safe_capture(
        [docker_executable, "version", "--format", "{{json .}}"],
        capture_runner=capture_runner,
    )
    if result is None or result.returncode != 0:
        warnings.append("Docker version information was unavailable.")
        return {"status": "unavailable"}

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        warnings.append("Docker version information was not valid JSON.")
        return {"status": "invalid_response"}

    return {"status": "available", "details": payload}


def _nvidia_details(
    *,
    capture_runner: CaptureRunner,
    warnings: list[str],
) -> dict[str, object]:
    result = _safe_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_runner=capture_runner,
    )
    if result is None or result.returncode != 0:
        warnings.append("NVIDIA GPU information was unavailable.")
        return {"status": "unavailable", "gpus": []}

    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            continue
        memory_text = values[2]
        memory_mib = int(memory_text) if memory_text.isdigit() else memory_text
        gpus.append(
            {
                "name": values[0],
                "driver_version": values[1],
                "memory_total_mib": memory_mib,
            }
        )

    return {"status": "available", "gpus": gpus}


def _safe_capture(
    command: list[str],
    *,
    capture_runner: CaptureRunner,
) -> CommandResult | None:
    try:
        return capture_runner(command)
    except OSError:
        return None


def _egomodelkit_provenance(capture_runner: CaptureRunner) -> dict[str, object]:
    package_path = Path(__file__).resolve().parent
    git_root = _find_git_root(package_path)
    package_version = _installed_version()

    if git_root is None:
        direct_url = _installed_direct_url()
        vcs_info = _mapping_value(direct_url, "vcs_info")
        commit_sha = _string_value(vcs_info, "commit_id")
        return {
            "version": package_version,
            "repository_url": EGOMODELKIT_REPOSITORY_URL,
            "package_path": str(package_path),
            "installation_source": direct_url,
            "git": {
                "status": (
                    "installed_distribution" if commit_sha is not None else "unavailable"
                ),
                "branch": _string_value(vcs_info, "requested_revision"),
                "commit_sha": commit_sha,
                "commit_timestamp": None,
                "remote_url": _string_value(direct_url, "url"),
                "dirty": None,
                "repository_root": None,
            },
        }

    def git_output(*arguments: str) -> str | None:
        result = _safe_capture(
            ["git", "-C", str(git_root), *arguments],
            capture_runner=capture_runner,
        )
        if result is None or result.returncode != 0:
            return None
        return result.stdout.strip()

    status_output = git_output("status", "--porcelain=v1")
    commit_sha = git_output("rev-parse", "HEAD")
    return {
        "version": package_version,
        "repository_url": EGOMODELKIT_REPOSITORY_URL,
        "package_path": str(package_path),
        "installation_source": _installed_direct_url(),
        "git": {
            "status": "available" if commit_sha is not None else "partial",
            "branch": git_output("branch", "--show-current") or None,
            "commit_sha": commit_sha,
            "commit_timestamp": git_output("show", "-s", "--format=%cI", "HEAD"),
            "remote_url": git_output("remote", "get-url", "origin"),
            "dirty": None if status_output is None else bool(status_output),
            "repository_root": str(git_root),
        },
    }


def _installed_version() -> str:
    try:
        return version("egomodelkit")
    except PackageNotFoundError:
        return __version__


def _installed_direct_url() -> dict[str, object]:
    try:
        direct_url_text = distribution("egomodelkit").read_text("direct_url.json")
    except PackageNotFoundError:
        return {}

    if direct_url_text is None:
        return {}

    try:
        payload = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _find_git_root(start_path: Path) -> Path | None:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _host_details() -> dict[str, object]:
    os_release: dict[str, str] = {}
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        pass

    return {
        "hostname": socket.gethostname(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "distribution": os_release,
            "is_wsl": is_wsl(),
            "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
        },
        "hardware": {
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        },
        "executables": {
            "docker": shutil.which("docker"),
            "git": shutil.which("git"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
    }


def _input_details(
    input_path: Path,
    input_names: tuple[str, ...],
    warnings: list[str],
) -> dict[str, object]:
    resolved_input = input_path.resolve()
    files = _selected_input_files(input_path, input_names)
    entries: list[dict[str, object]] = []

    for path in files:
        try:
            stat = path.stat()
            entries.append(
                {
                    "name": path.name,
                    "relative_path": _relative_input_path(path, input_path),
                    "size_bytes": stat.st_size,
                    "modified_at_utc": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                    "sha256": _sha256(path),
                }
            )
        except OSError as exc:
            warnings.append(f"Input provenance could not be read for {path.name}: {exc}")
            entries.append(
                {
                    "name": path.name,
                    "relative_path": _relative_input_path(path, input_path),
                    "size_bytes": None,
                    "modified_at_utc": None,
                    "sha256": None,
                }
            )

    return {
        "selected_path_at_execution": str(resolved_input),
        "selected_path_type": "directory" if input_path.is_dir() else "file",
        "hash_algorithm": "sha256",
        "files": entries,
    }


def _selected_input_files(
    input_path: Path,
    input_names: tuple[str, ...],
) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    return [input_path / name for name in input_names if (input_path / name).is_file()]


def _relative_input_path(path: Path, input_path: Path) -> str:
    if input_path.is_file():
        return path.name
    return path.relative_to(input_path).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_entry(
    pin: ExternalModelCodePin,
    *,
    scenario: str,
) -> dict[str, object]:
    entry: dict[str, object] = pin.as_manifest_entry()
    locations = {
        HAND_OBJECT_DETECTOR_PIN.model_id: [
            "hand-object-contact:/opt/hand_object_detector"
        ],
        EGOVIZML_PIN.model_id: [
            "adl-recognition-core:/opt/EgoVizML",
            "adl-recognition-detic:/opt/EgoVizML",
        ],
        DETIC_PIN.model_id: ["adl-recognition-detic:/opt/Detic"],
        DETECTRON2_PIN.model_id: ["adl-recognition-detic:/opt/detectron2"],
    }
    runtime_locations = locations[pin.model_id]
    if pin == EGOVIZML_PIN and scenario == "adl-combined-predictions":
        runtime_locations = runtime_locations[:1]
    entry["runtime_locations"] = runtime_locations
    return entry


def _asset_entry(pin: ExternalModelAssetPin) -> dict[str, object]:
    entry: dict[str, object] = pin.as_manifest_entry()
    if pin.asset_id == HAND_OBJECT_DETECTOR_WEIGHTS_PIN.asset_id:
        spec = DEFAULT_HAND_OBJECT_CONTACT_RUNTIME_SPEC
        location = (
            Path("/opt/hand_object_detector")
            / spec.shan_load_dir
            / spec.shan_model_subdir
            / spec.checkpoint_filename
        ).as_posix()
        entry["runtime_locations"] = [f"hand-object-contact:{location}"]
    else:
        entry["runtime_locations"] = [
            f"adl-recognition-detic:/opt/Detic/models/{pin.filename}"
        ]
    return entry


def _egovizml_classifier_entry() -> dict[str, object]:
    """Describe the repository-tracked production classifier used by ADL runs."""
    return {
        "asset_id": "binary-active-logreg",
        "source_type": "repository_file",
        "repository_url": EGOVIZML_PIN.fork_repository_url,
        "repository_commit_sha": EGOVIZML_PIN.commit_sha,
        "repository_path": EGOVIZML_CLASSIFIER_REPOSITORY_PATH,
        "filename": Path(EGOVIZML_CLASSIFIER_REPOSITORY_PATH).name,
        "runtime_locations": [
            "adl-recognition-core:/opt/EgoVizML/"
            f"{EGOVIZML_CLASSIFIER_REPOSITORY_PATH}"
        ],
        "sha256": None,
        "sha256_note": (
            "The classifier is pinned by its containing EgoVizML repository commit. "
            "The terminal Docker image ID identifies the built runtime containing it."
        ),
    }


def _utc_timestamp(now: datetime | None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _mapping_value(
    payload: dict[str, object] | None,
    key: str,
) -> dict[str, object]:
    if payload is None:
        return {}
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _string_value(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _string_list_value(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
