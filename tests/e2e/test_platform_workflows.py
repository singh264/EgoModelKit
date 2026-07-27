"""Manual Linux GPU and Windows-WSL end-to-end workflow checks."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
from fastapi.testclient import TestClient

from egomodelkit.gui_backend import create_app
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID
from egomodelkit.models.hand_object_contact import HandObjectContactRequest
from egomodelkit.runtime.commands import streaming_subprocess_runner, subprocess_runner
from egomodelkit.runtime.hand_object_contact import run_hand_object_contact
from egomodelkit.runtime.preflight import HostPrerequisiteError, ensure_host_runtime_ready

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

RUN_E2E_ENV: Final[str] = "EGOMODELKIT_RUN_E2E"
FIXTURE_ROOT_ENV: Final[str] = "EGOMODELKIT_E2E_FIXTURES"
TIMEOUT_ENV: Final[str] = "EGOMODELKIT_E2E_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 14_400.0
ADL_PROBABILITY_ABS_TOLERANCE: Final[float] = 1e-6
# Allow minor cross-GPU floating-point variation without weakening structure checks.
HOC_REL_TOLERANCE: Final[float] = 5e-4
HOC_ABS_TOLERANCE: Final[float] = 5e-4


@dataclass(frozen=True, slots=True)
class FridgeInputs:
    """Single-video and split-video fridge inputs."""

    single_video: Path
    multi_video_dir: Path


@dataclass(frozen=True, slots=True)
class HandUseReferences:
    """Approved Statepool profiles for hand-interaction checks."""

    single: Path
    multi: Path


@dataclass(frozen=True, slots=True)
class AdlReferences:
    """Approved full-video ADL prediction reference."""

    single: Path


@dataclass(frozen=True, slots=True)
class HandObjectE2EFixtures:
    """Image inputs and references for the isolated internal HOC check."""

    input_dir: Path
    expected_dir: Path


@pytest.fixture(scope="session")
def e2e_runtime_ready() -> None:
    """Require explicit opt-in and a supported Docker GPU runtime."""
    if os.environ.get(RUN_E2E_ENV) != "1":
        pytest.skip(f"Set {RUN_E2E_ENV}=1 to run manual GPU E2E tests.")

    if platform.system() != "Linux":
        pytest.skip("GPU E2E tests require Linux directly or Windows through WSL2.")

    try:
        ensure_host_runtime_ready(
            docker_executable="docker",
            command_runner=_quiet_command_runner,
            require_linux_nvidia_gpu=True,
            allow_runtime_recovery=False,
        )
    except HostPrerequisiteError as exc:
        pytest.skip(f"GPU E2E runtime is unavailable: {exc}")


@pytest.fixture(scope="session")
def fridge_inputs(e2e_runtime_ready: None) -> FridgeInputs:
    """Load the local fridge inputs shared by the public pipelines."""
    del e2e_runtime_ready
    root = _fixture_root()
    inputs = FridgeInputs(
        single_video=root / "inputs" / "fridge_17s.mp4",
        multi_video_dir=root / "inputs" / "fridge_split",
    )
    _skip_missing_fixtures(
        [
            inputs.single_video,
            inputs.multi_video_dir / "fridge_part1_10s.mp4",
            inputs.multi_video_dir / "fridge_part2_7s.mp4",
        ]
    )
    return inputs


@pytest.fixture(scope="session")
def hand_use_references(e2e_runtime_ready: None) -> HandUseReferences:
    """Load the approved hand-use Statepool references."""
    del e2e_runtime_ready
    root = _fixture_root() / "expected" / "hand_use"
    references = HandUseReferences(
        single=root / "fridge_single_statepool.csv",
        multi=root / "fridge_multi_statepool.csv",
    )
    _skip_missing_fixtures([references.single, references.multi])
    return references


@pytest.fixture(scope="session")
def adl_references(e2e_runtime_ready: None) -> AdlReferences:
    """Load the approved full-video ADL prediction reference."""
    del e2e_runtime_ready
    root = _fixture_root() / "expected" / "adl"
    references = AdlReferences(single=root / "fridge_single_predictions.csv")
    _skip_missing_fixtures([references.single])
    return references


@pytest.fixture(scope="session")
def hand_object_e2e_fixtures(e2e_runtime_ready: None) -> HandObjectE2EFixtures:
    """Load the local image fixtures used by the internal HOC runtime."""
    del e2e_runtime_ready
    root = _fixture_root()
    fixtures = HandObjectE2EFixtures(
        input_dir=root / "inputs" / "hand_object_contact",
        expected_dir=root / "expected" / "hand_object_contact",
    )
    _skip_missing_fixtures(
        [
            fixtures.input_dir / "hand_holding_ball.jpg",
            fixtures.input_dir / "hand_holding_cup.jpg",
            fixtures.expected_dir / "hand_holding_ball_shan.json",
            fixtures.expected_dir / "hand_holding_cup_shan.json",
        ]
    )
    return fixtures


def _fixture_root() -> Path:
    return Path(
        os.environ.get(FIXTURE_ROOT_ENV, Path(__file__).parent / "fixtures")
    ).expanduser()


def _skip_missing_fixtures(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    joined = "\n  - ".join(str(path) for path in missing)
    pytest.skip(
        "E2E fixtures are incomplete. Add the approved fixtures under "
        f"{_fixture_root()} or set {FIXTURE_ROOT_ENV}. Missing:\n"
        f"  - {joined}"
    )


def _quiet_command_runner(command: list[str]) -> int:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return 1


def _timeout_seconds() -> float:
    value = os.environ.get(TIMEOUT_ENV)
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS

    try:
        timeout = float(value)
    except ValueError as exc:
        raise AssertionError(f"{TIMEOUT_ENV} must be a number.") from exc

    if timeout <= 0:
        raise AssertionError(f"{TIMEOUT_ENV} must be greater than zero.")

    return timeout


def _run_cli(
    *,
    model_id: str,
    input_path: Path,
    output_root: Path,
    dominant_hand: str | None = None,
) -> Path:
    executable = shutil.which("egomodelkit")
    if executable is None:
        pytest.skip("Install the project so the egomodelkit command is available.")

    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "run",
        model_id,
        "--input",
        str(input_path),
        "--output",
        str(output_root),
    ]
    if dominant_hand is not None:
        command.extend(["--dominant-hand", dominant_hand])

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"CLI E2E run exceeded the configured timeout: {exc}")

    combined_output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    assert completed.returncode == 0, combined_output

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("Outputs: "):
            output_dir = Path(line.removeprefix("Outputs: ").strip())
            assert output_dir.is_dir(), combined_output
            return output_dir

    raise AssertionError(f"CLI output did not report its run directory.\n{combined_output}")


def _run_gui(
    *,
    model_id: str,
    input_paths: list[Path],
    output_root: Path,
    dominant_hand: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    data = {"modelId": model_id, "outputRoot": str(output_root)}
    if dominant_hand is not None:
        data["dominantHand"] = dominant_hand

    opened_files = [path.open("rb") for path in input_paths]
    try:
        files = [
            ("files", (path.name, stream, "video/mp4"))
            for path, stream in zip(input_paths, opened_files, strict=True)
        ]
        with TestClient(create_app()) as client:
            response = client.post("/api/runs", data=data, files=files)
            assert response.status_code == 200, response.text
            run_id = response.json()["runId"]
            deadline = time.monotonic() + _timeout_seconds()

            while time.monotonic() < deadline:
                progress_response = client.get(f"/api/runs/{run_id}/progress")
                assert progress_response.status_code == 200, progress_response.text
                body = progress_response.json()
                status = body["status"]

                if status == "completed":
                    output_dir = Path(body["outputFolder"])
                    assert output_dir.is_dir(), body
                    return output_dir

                if status in {"failed", "cancelled"}:
                    raise AssertionError(
                        f"GUI run ended with status {status}: "
                        f"{body.get('errorMessage') or body}"
                    )

                time.sleep(1.0)
    finally:
        for stream in opened_files:
            stream.close()

    raise AssertionError("GUI run did not complete before the E2E timeout.")


def _assert_hand_interaction(
    *,
    output_dir: Path,
    expected_statepool_path: Path,
    dominant_hand: str,
) -> None:
    shan_outputs_dir = (
        output_dir / "technical" / "intermediate_files" / "shan_outputs"
    )
    assert any(shan_outputs_dir.rglob("*_shan.json")), (
        f"No internal HOC JSON outputs found under {shan_outputs_dir}."
    )
    _assert_statepool_outputs(
        actual_path=(
            output_dir
            / "technical"
            / "post_processing"
            / "frame_level_predictions.csv"
        ),
        expected_path=expected_statepool_path,
        dominant_hand=dominant_hand,
    )


def _assert_statepool_outputs(
    *,
    actual_path: Path,
    expected_path: Path,
    dominant_hand: str,
) -> None:
    actual_rows = [
        row for row in _read_csv(actual_path) if row["is_valid_source_frame"] == "true"
    ]
    actual_rows.sort(key=lambda row: int(row["session_frame_index"]))
    expected_rows = _read_csv(expected_path)
    assert len(actual_rows) == len(expected_rows)

    non_dominant_hand = "left" if dominant_hand == "right" else "right"
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        assert int(actual["session_frame_index"]) == int(expected["Frame_number"])
        assert int(actual["right_interaction"]) == int(expected["Interaction_R"])
        assert int(actual["left_interaction"]) == int(expected["Interaction_L"])
        assert int(actual["dominant_interaction"]) == int(
            actual[f"{dominant_hand}_interaction"]
        )
        assert int(actual["non_dominant_interaction"]) == int(
            actual[f"{non_dominant_hand}_interaction"]
        )


def _assert_adl_outputs(*, output_dir: Path, expected_path: Path) -> None:
    actual_rows = _read_csv(output_dir / "results" / "adl_segment_predictions.csv")
    expected_rows = _read_csv(expected_path)
    assert actual_rows and expected_rows
    assert len(actual_rows) == len(expected_rows)

    actual_by_segment = {row["segment_name"]: row for row in actual_rows}
    expected_by_segment = {row["video"]: row for row in expected_rows}
    assert actual_by_segment.keys() == expected_by_segment.keys()

    probability_columns = sorted(
        column for column in expected_rows[0] if column.startswith("prob_")
    )
    assert probability_columns

    for segment_name in sorted(actual_by_segment):
        actual = actual_by_segment[segment_name]
        expected = expected_by_segment[segment_name]
        assert actual["prediction_status"] == "predicted"
        assert actual["predicted_adl"] == expected["predicted_label_readable"]
        assert int(float(actual["predicted_class"])) == int(
            float(expected["predicted_class"])
        )

        for column in probability_columns:
            assert float(actual[column]) == pytest.approx(
                float(expected[column]),
                abs=ADL_PROBABILITY_ABS_TOLERANCE,
            )

        assert float(actual["predicted_probability"]) == pytest.approx(
            float(expected["predicted_probability"]),
            abs=ADL_PROBABILITY_ABS_TOLERANCE,
        )


def _assert_hand_object_outputs(*, actual_dir: Path, expected_dir: Path) -> None:
    actual_files = {path.name: path for path in actual_dir.glob("*_shan.json")}
    expected_files = {path.name: path for path in expected_dir.glob("*_shan.json")}
    assert actual_files.keys() == expected_files.keys()

    for filename in sorted(expected_files):
        actual = json.loads(actual_files[filename].read_text(encoding="utf-8"))
        expected = json.loads(expected_files[filename].read_text(encoding="utf-8"))
        _assert_json_close(actual, expected, location=filename)

        stem = filename.removesuffix("_shan.json")
        assert (actual_dir / f"{stem}_shan.pkl").is_file()
        assert (actual_dir / f"{stem}_det.png").is_file()


def _assert_json_close(actual: Any, expected: Any, *, location: str) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, location
        return

    if isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), location
        assert math.isclose(
            float(actual),
            float(expected),
            rel_tol=HOC_REL_TOLERANCE,
            abs_tol=HOC_ABS_TOLERANCE,
        ), location
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), location
        assert len(actual) == len(expected), location
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_json_close(
                actual_item,
                expected_item,
                location=f"{location}[{index}]",
            )
        return

    assert isinstance(expected, dict), location
    assert isinstance(actual, dict), location
    assert actual.keys() == expected.keys(), location
    for key in expected:
        _assert_json_close(actual[key], expected[key], location=f"{location}.{key}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _multi_inputs(fixtures: FridgeInputs) -> list[Path]:
    return [
        fixtures.multi_video_dir / "fridge_part1_10s.mp4",
        fixtures.multi_video_dir / "fridge_part2_7s.mp4",
    ]


@pytest.mark.parametrize(
    ("interface", "model_id", "multi_video", "dominant_hand"),
    [
        pytest.param("gui", HAND_INTERACTION_MODEL_ID, False, "right", id="01-gui-hi-single-right"),
        pytest.param("gui", HAND_INTERACTION_MODEL_ID, False, "left", id="02-gui-hi-single-left"),
        pytest.param("gui", HAND_INTERACTION_MODEL_ID, True, "right", id="03-gui-hi-multi-right"),
        pytest.param("gui", HAND_INTERACTION_MODEL_ID, True, "left", id="04-gui-hi-multi-left"),
        pytest.param("gui", ADL_RECOGNITION_MODEL_ID, False, None, id="05-gui-adl-single"),
        pytest.param("cli", HAND_INTERACTION_MODEL_ID, False, "right", id="06-cli-hi-single-right"),
        pytest.param("cli", HAND_INTERACTION_MODEL_ID, False, "left", id="07-cli-hi-single-left"),
        pytest.param("cli", HAND_INTERACTION_MODEL_ID, True, "right", id="08-cli-hi-multi-right"),
        pytest.param("cli", HAND_INTERACTION_MODEL_ID, True, "left", id="09-cli-hi-multi-left"),
        pytest.param("cli", ADL_RECOGNITION_MODEL_ID, False, None, id="10-cli-adl-single"),
    ],
)
def test_platform_workflow(
    fridge_inputs: FridgeInputs,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    interface: str,
    model_id: str,
    multi_video: bool,
    dominant_hand: str | None,
) -> None:
    input_paths = (
        _multi_inputs(fridge_inputs)
        if multi_video
        else [fridge_inputs.single_video]
    )

    if interface == "gui":
        output_dir = _run_gui(
            model_id=model_id,
            input_paths=input_paths,
            output_root=tmp_path,
            dominant_hand=dominant_hand,
        )
    else:
        output_dir = _run_cli(
            model_id=model_id,
            input_path=(
                fridge_inputs.multi_video_dir
                if multi_video
                else input_paths[0]
            ),
            output_root=tmp_path,
            dominant_hand=dominant_hand,
        )

    if model_id == HAND_INTERACTION_MODEL_ID:
        references = request.getfixturevalue("hand_use_references")
        assert isinstance(references, HandUseReferences)
        assert dominant_hand is not None
        _assert_hand_interaction(
            output_dir=output_dir,
            expected_statepool_path=(references.multi if multi_video else references.single),
            dominant_hand=dominant_hand,
        )
        return

    assert not multi_video
    references = request.getfixturevalue("adl_references")
    assert isinstance(references, AdlReferences)
    _assert_adl_outputs(
        output_dir=output_dir,
        expected_path=references.single,
    )


def test_internal_hand_object_contact_matches_reference(
    hand_object_e2e_fixtures: HandObjectE2EFixtures,
    tmp_path: Path,
) -> None:
    """Run the hidden image model once for the supplied ball/cup reference set."""
    output_dir = tmp_path / "hand_object_contact"
    run_hand_object_contact(
        HandObjectContactRequest(
            input_path=hand_object_e2e_fixtures.input_dir,
            output_dir=output_dir,
        ),
        command_runner=subprocess_runner,
        streaming_command_runner=streaming_subprocess_runner,
    )
    _assert_hand_object_outputs(
        actual_dir=output_dir,
        expected_dir=hand_object_e2e_fixtures.expected_dir,
    )
