import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from egomodelkit import run_manifest
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.runtime.commands import CommandResult
from egomodelkit.runtime.external_code import DETIC_WEIGHTS_PIN


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult(returncode=returncode, stdout=stdout, stderr=stderr)


def _stub_static_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_manifest,
        "_egomodelkit_provenance",
        lambda _runner: {"version": "test", "git": {"commit_sha": "a" * 40}},
    )
    monkeypatch.setattr(
        run_manifest,
        "_host_details",
        lambda: {"operating_system": {"system": "Linux"}},
    )


def test_build_adl_video_manifest_collects_complete_runtime_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_static_environment(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")

    def capture(command: list[str]) -> CommandResult:
        if command[0] == "nvidia-smi":
            return _result(stdout="GPU A, 555.42, 8192\nmalformed\nGPU B, 555.42, unknown\n")
        if command[1] == "version":
            return _result(stdout=json.dumps({"Client": {"Version": "29.0"}}))
        if command[1:3] == ["image", "inspect"]:
            return _result(
                stdout=json.dumps(
                    [
                        {
                            "Id": "sha256:image",
                            "RepoTags": [command[-1]],
                            "RepoDigests": None,
                            "Created": "2026-07-24T12:00:00Z",
                            "Os": "linux",
                            "Architecture": "amd64",
                            "Size": 123,
                            "Config": {
                                "Labels": {
                                    "org.egomodelkit.runtime.managed": "true",
                                    "unrelated": "ignored",
                                }
                            },
                        }
                    ]
                )
            )
        raise AssertionError(command)

    now = datetime(2026, 7, 24, 16, 0, tzinfo=timezone.utc)
    payload = run_manifest.build_run_manifest(
        run_id="run-adl",
        model_id=ADL_RECOGNITION_MODEL_ID,
        input_path=video,
        input_names=(video.name,),
        scenario="adl-single-video",
        status="completed",
        output_folder="/results/run-adl",
        output_contract_version=1,
        model_configuration={"segment_length_seconds": 60},
        invocation_interface="cli",
        invocation_arguments=("egomodelkit", "run", "adl-recognition"),
        error_message="recorded for coverage",
        collect_runtime_state=True,
        capture_runner=capture,
        now=now,
    )

    assert payload["manifest_schema_version"] == 2
    assert payload["run"] == {
        "run_id": "run-adl",
        "status": "completed",
        "created_at_utc": now.isoformat(),
        "updated_at_utc": now.isoformat(),
        "completed_at_utc": now.isoformat(),
        "output_folder": "/results/run-adl",
        "error_message": "recorded for coverage",
    }
    assert payload["invocation"] == {
        "interface": "cli",
        "arguments": ["egomodelkit", "run", "adl-recognition"],
    }
    assert payload["inputs"]["files"][0]["sha256"] == (
        "79fd615a866fe7f9eb4da8d9c41ab57e3bd48056df42fd2c13e4d461a87afbe3"
    )
    assert {item["model_id"] for item in payload["external_code"]} == {
        "egovizml",
        "detic",
        "detectron2",
        "hand-object-detector",
    }
    assert {item["asset_id"] for item in payload["model_artifacts"]} == {
        "binary-active-logreg",
        "detic-lcocoi21k-clip-swinb-896b32-4x-ft4x-max-size",
        "faster_rcnn_1_8_132028",
    }
    classifier = payload["model_artifacts"][0]
    assert classifier == {
        "asset_id": "binary-active-logreg",
        "source_type": "repository_file",
        "repository_url": "https://github.com/singh264/EgoVizML",
        "repository_commit_sha": "c129075eef8f818947b250d1116d00267c4a9455",
        "repository_path": "models/binary_active_logreg.joblib",
        "filename": "binary_active_logreg.joblib",
        "runtime_locations": [
            "adl-recognition-core:/opt/EgoVizML/models/"
            "binary_active_logreg.joblib"
        ],
        "sha256": None,
        "sha256_note": (
            "The classifier is pinned by its containing EgoVizML repository commit. "
            "The terminal Docker image ID identifies the built runtime containing it."
        ),
    }
    assert payload["runtime"]["container_paths"]["egovizml_classifier"] == (
        "adl-recognition-core:/opt/EgoVizML/models/binary_active_logreg.joblib"
    )
    assert [item["runtime_name"] for item in payload["runtime"]["images"]] == [
        "adl-recognition-core",
        "adl-recognition-detic",
        "hand-object-contact",
    ]
    assert all(
        item["inspection"]["status"] == "available"
        for item in payload["runtime"]["images"]
    )
    assert payload["runtime"]["images"][0]["inspection"]["repo_digests"] == []
    assert payload["runtime"]["images"][0]["inspection"]["provenance_labels"] == {
        "org.egomodelkit.runtime.managed": "true"
    }
    assert payload["runtime"]["host_docker"]["status"] == "available"
    assert payload["runtime"]["host_nvidia"] == {
        "status": "available",
        "gpus": [
            {"name": "GPU A", "driver_version": "555.42", "memory_total_mib": 8192},
            {
                "name": "GPU B",
                "driver_version": "555.42",
                "memory_total_mib": "unknown",
            },
        ],
    }
    assert payload["collection_warnings"] == []


def test_combined_predictions_manifest_preserves_previous_run_and_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_static_environment(monkeypatch)
    predictions = tmp_path / "all_preds.pkl"
    predictions.write_bytes(b"predictions")
    created = "2026-07-23T12:00:00+00:00"
    previous = {
        "run": {
            "created_at_utc": created,
            "completed_at_utc": "2026-07-23T13:00:00+00:00",
        },
        "invocation": {
            "interface": "gui",
            "arguments": ["kept", 4],
        },
    }

    payload = run_manifest.build_run_manifest(
        run_id="run-combined",
        model_id=ADL_RECOGNITION_MODEL_ID,
        input_path=predictions,
        input_names=(predictions.name,),
        scenario="adl-combined-predictions",
        status="running",
        output_folder="/results/run-combined",
        output_contract_version=1,
        model_configuration={},
        previous_manifest=previous,
        now=datetime(2026, 7, 24, 12, 0),
    )

    assert payload["run"]["created_at_utc"] == created
    assert payload["run"]["completed_at_utc"] == "2026-07-23T13:00:00+00:00"
    assert payload["run"]["updated_at_utc"] == "2026-07-24T12:00:00+00:00"
    assert payload["invocation"] == {"interface": "gui", "arguments": ["kept"]}
    assert [item["runtime_name"] for item in payload["runtime"]["images"]] == [
        "adl-recognition-core"
    ]
    assert payload["runtime"]["parameters"]["hand_object_contact"] is None
    assert payload["runtime"]["container_paths"]["detic_repository"] is None
    assert payload["external_code"][0]["runtime_locations"] == [
        "adl-recognition-core:/opt/EgoVizML"
    ]
    assert [item["asset_id"] for item in payload["model_artifacts"]] == [
        "binary-active-logreg"
    ]


def test_hand_object_and_hand_interaction_runtime_definitions_and_unknown_model() -> None:
    hoc_runtime, hoc_code, hoc_assets = run_manifest._runtime_definition(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        scenario="hand-object-single-image",
    )
    assert hoc_runtime["images"][0]["runtime_name"] == "hand-object-contact"
    assert hoc_runtime["parameters"]["checkpoint_step"] == 132028
    assert hoc_code[0].model_id == "hand-object-detector"
    assert hoc_assets[0].filename == "faster_rcnn_1_8_132028.pth"

    interaction_runtime, _, _ = run_manifest._runtime_definition(
        model_id=HAND_INTERACTION_MODEL_ID,
        scenario="hand-interaction-single-video",
    )
    assert [item["runtime_name"] for item in interaction_runtime["images"]] == [
        "hand-interaction",
        "hand-object-contact",
    ]

    with pytest.raises(ValueError, match="Unsupported model id"):
        run_manifest._runtime_definition(model_id="unknown", scenario="unknown")


def test_runtime_inspection_failure_and_invalid_response_branches() -> None:
    image = {
        "runtime_name": "test",
        "repository": "repo",
        "tag": "repo:test",
        "build_fingerprint_sha256": "a" * 64,
        "inspection": {"status": "not_collected"},
    }
    warnings: list[str] = []

    unavailable = run_manifest._with_image_inspection(
        image,
        docker_executable="docker",
        capture_runner=lambda _command: _result(returncode=1),
        warnings=warnings,
    )
    assert unavailable["inspection"] == {"status": "unavailable"}

    invalid = run_manifest._with_image_inspection(
        image,
        docker_executable="docker",
        capture_runner=lambda _command: _result(stdout="{}"),
        warnings=warnings,
    )
    assert invalid["inspection"] == {"status": "invalid_response"}

    no_labels = run_manifest._with_image_inspection(
        image,
        docker_executable="docker",
        capture_runner=lambda _command: _result(
            stdout=json.dumps([{"Id": "id", "Config": "invalid"}])
        ),
        warnings=warnings,
    )
    assert no_labels["inspection"]["provenance_labels"] == {}
    assert len(warnings) == 2

    assert run_manifest._image_entries({"images": "invalid"}) == []
    assert run_manifest._image_entries({"images": [image, "invalid"]}) == [image]


def test_docker_nvidia_and_safe_capture_error_branches() -> None:
    warnings: list[str] = []
    def failed(_command: list[str]) -> CommandResult:
        return _result(returncode=1)

    assert run_manifest._docker_version(
        docker_executable="docker",
        capture_runner=failed,
        warnings=warnings,
    ) == {"status": "unavailable"}
    assert run_manifest._nvidia_details(
        capture_runner=failed,
        warnings=warnings,
    ) == {"status": "unavailable", "gpus": []}

    def invalid_json(_command: list[str]) -> CommandResult:
        return _result(stdout="not-json")

    assert run_manifest._docker_version(
        docker_executable="docker",
        capture_runner=invalid_json,
        warnings=warnings,
    ) == {"status": "invalid_response"}

    def raises_os_error(_command: list[str]) -> CommandResult:
        raise FileNotFoundError("missing")

    assert run_manifest._safe_capture(
        ["missing"],
        capture_runner=raises_os_error,
    ) is None
    assert len(warnings) == 3


def test_git_package_host_and_input_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_find_git_root = run_manifest._find_git_root
    monkeypatch.setattr(run_manifest, "_find_git_root", lambda _path: None)
    unavailable_git = run_manifest._egomodelkit_provenance(lambda _command: _result())
    assert unavailable_git["git"]["status"] == "unavailable"

    git_root = tmp_path / "repo"
    git_root.mkdir()
    (git_root / ".git").mkdir()
    monkeypatch.setattr(run_manifest, "_find_git_root", lambda _path: git_root)
    partial_git = run_manifest._egomodelkit_provenance(
        lambda _command: _result(returncode=1)
    )
    assert partial_git["git"]["commit_sha"] is None
    assert partial_git["git"]["dirty"] is None

    monkeypatch.setattr(
        run_manifest,
        "version",
        lambda _name: (_ for _ in ()).throw(run_manifest.PackageNotFoundError),
    )
    assert run_manifest._installed_version() == run_manifest.__version__

    monkeypatch.setattr(run_manifest, "_find_git_root", original_find_git_root)
    no_repo = tmp_path / "no-repo" / "child"
    no_repo.mkdir(parents=True)
    monkeypatch.chdir(no_repo)
    assert run_manifest._find_git_root(no_repo) is None
    (tmp_path / "no-repo" / ".git").mkdir()
    assert run_manifest._find_git_root(no_repo) == tmp_path / "no-repo"

    monkeypatch.setattr(
        run_manifest.platform,
        "freedesktop_os_release",
        lambda: (_ for _ in ()).throw(OSError("unsupported")),
    )
    assert run_manifest._host_details()["operating_system"]["distribution"] == {}

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_file = input_dir / "one.mp4"
    input_file.write_bytes(b"one")
    warnings: list[str] = []
    monkeypatch.setattr(
        run_manifest,
        "_sha256",
        lambda _path: (_ for _ in ()).throw(OSError("read failed")),
    )
    details = run_manifest._input_details(input_dir, (input_file.name,), warnings)
    assert details["selected_path_type"] == "directory"
    assert details["files"][0]["sha256"] is None
    assert warnings and "read failed" in warnings[0]


def test_small_manifest_helpers_and_detic_asset_location(tmp_path: Path) -> None:
    input_file = tmp_path / "one.bin"
    input_file.write_bytes(b"one")
    assert run_manifest._selected_input_files(input_file, ("ignored",)) == [input_file]
    assert run_manifest._relative_input_path(input_file, input_file) == "one.bin"
    assert run_manifest._mapping_value(None, "run") == {}
    assert run_manifest._mapping_value({"run": "bad"}, "run") == {}
    assert run_manifest._string_value({"value": 2}, "value") is None
    assert run_manifest._string_list_value({"value": "bad"}, "value") == []
    assert run_manifest._string_list_value({"value": ["a", 2]}, "value") == ["a"]

    detic_entry = run_manifest._asset_entry(DETIC_WEIGHTS_PIN)
    assert detic_entry["runtime_locations"] == [
        "adl-recognition-detic:/opt/Detic/models/"
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    ]

    aware = datetime(2026, 7, 24, 12, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert run_manifest._utc_timestamp(aware) == "2026-07-24T16:00:00+00:00"


def test_git_success_and_installed_direct_url_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_root = tmp_path / "repo"
    git_root.mkdir()
    (git_root / ".git").mkdir()
    monkeypatch.setattr(run_manifest, "_find_git_root", lambda _path: git_root)

    def git_capture(command: list[str]) -> CommandResult:
        arguments = tuple(command[3:])
        outputs = {
            ("status", "--porcelain=v1"): " M file.py\n",
            ("rev-parse", "HEAD"): "b" * 40 + "\n",
            ("branch", "--show-current"): "feature/provenance\n",
            ("show", "-s", "--format=%cI", "HEAD"): "2026-07-24T12:00:00-04:00\n",
            ("remote", "get-url", "origin"): "git@github.com:singh264/EgoModelKit.git\n",
        }
        return _result(stdout=outputs[arguments])

    provenance = run_manifest._egomodelkit_provenance(git_capture)
    assert provenance["git"]["status"] == "available"
    assert provenance["git"]["branch"] == "feature/provenance"
    assert provenance["git"]["dirty"] is True

    monkeypatch.setattr(
        run_manifest,
        "distribution",
        lambda _name: (_ for _ in ()).throw(run_manifest.PackageNotFoundError),
    )
    assert run_manifest._installed_direct_url() == {}

    class FakeDistribution:
        def __init__(self, value: str | None) -> None:
            self.value = value

        def read_text(self, _name: str) -> str | None:
            return self.value

    monkeypatch.setattr(
        run_manifest,
        "distribution",
        lambda _name: FakeDistribution(None),
    )
    assert run_manifest._installed_direct_url() == {}

    monkeypatch.setattr(
        run_manifest,
        "distribution",
        lambda _name: FakeDistribution("not-json"),
    )
    assert run_manifest._installed_direct_url() == {}

    monkeypatch.setattr(
        run_manifest,
        "distribution",
        lambda _name: FakeDistribution("[]"),
    )
    assert run_manifest._installed_direct_url() == {}

    direct_url_payload = {
        "url": "https://github.com/singh264/EgoModelKit.git",
        "vcs_info": {
            "commit_id": "c" * 40,
            "requested_revision": "main",
        },
    }
    monkeypatch.setattr(
        run_manifest,
        "distribution",
        lambda _name: FakeDistribution(json.dumps(direct_url_payload)),
    )
    assert run_manifest._installed_direct_url() == direct_url_payload

    monkeypatch.setattr(run_manifest, "_find_git_root", lambda _path: None)
    installed = run_manifest._egomodelkit_provenance(lambda _command: _result())
    assert installed["git"] == {
        "status": "installed_distribution",
        "branch": "main",
        "commit_sha": "c" * 40,
        "commit_timestamp": None,
        "remote_url": "https://github.com/singh264/EgoModelKit.git",
        "dirty": None,
        "repository_root": None,
    }
