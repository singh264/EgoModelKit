from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

RESOURCE_ROOT = (
    Path(__file__).parents[2]
    / "src"
    / "egomodelkit"
    / "resources"
    / "containers"
)


def _load_script(relative_path: str, module_name: str):
    script_path = RESOURCE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hand_interaction_ffmpeg_progress_fills_every_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "hand_interaction/entrypoint.py",
        "egomodelkit_test_hand_interaction_entrypoint_progress",
    )
    commands: list[list[str]] = []

    class FakeProcess:
        stdout = iter(
            [
                "frame=1\n",
                "frame=3\n",
                "progress=end\n",
            ]
        )

        @staticmethod
        def wait() -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    reported: list[int] = []

    frame_count = module._extract_frames(
        video_path=tmp_path / "input.mp4",
        output_pattern=tmp_path / "frame_%06d.jpg",
        processing_fps=30,
        resize_width=720,
        resize_height=405,
        progress=reported.append,
    )

    assert frame_count == 3
    assert reported == [1, 2, 3]
    assert commands[0][commands[0].index("-progress") + 1] == "pipe:1"


def test_initial_multi_video_frame_totals_round_the_session_once() -> None:
    hand_interaction = _load_script(
        "hand_interaction/entrypoint.py",
        "egomodelkit_test_hand_interaction_expected_frames",
    )
    adl = _load_script(
        "adl_recognition/entrypoint.py",
        "egomodelkit_test_adl_expected_frames",
    )
    videos = [
        {
            "source_duration_seconds": 1200.666667,
            "source_fps": 15.0,
            "source_total_frames": 18010,
        },
        {
            "source_duration_seconds": 1199.533333,
            "source_fps": 15.0,
            "source_total_frames": 17993,
        },
        {
            "source_duration_seconds": 1199.8,
            "source_fps": 15.0,
            "source_total_frames": 17997,
        },
    ]

    assert hand_interaction._expected_output_frame_total(videos, 30) == 108000
    assert adl._expected_inference_frame_total(videos, 1) == 3600


def test_hand_interaction_organizes_after_extraction_in_subclip_sized_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "hand_interaction/entrypoint.py",
        "egomodelkit_test_hand_interaction_organization_progress",
    )
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            input_path=str(input_path),
            output_dir=str(output_dir),
            work_dir_name="work",
            subclip_length=2,
            processing_fps=2,
            resize_width=720,
            resize_height=405,
            pooling_window_seconds=1.0,
            interaction_contact_state_threshold=3,
            dominant_hand="right",
        ),
    )
    monkeypatch.setattr(
        module,
        "_probe_source_video_metadata",
        lambda _path: {
            "source_duration_seconds": 4.5,
            "source_fps": 2.0,
            "source_total_frames": 9,
        },
    )

    def fake_extract_frames(*, output_pattern: Path, progress, **_kwargs: object) -> int:
        output_pattern.parent.mkdir(parents=True, exist_ok=True)
        for current in range(1, 10):
            (output_pattern.parent / f"frame_{current:06d}.jpg").write_bytes(b"jpg")
            progress(current)
        return 9

    monkeypatch.setattr(module, "_extract_frames", fake_extract_frames)
    updates: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        module,
        "_emit_progress",
        lambda kind, **payload: updates.append(
            (kind, int(payload["current"]), int(payload["total"]))
        ),
    )

    module.main()

    assert [update for update in updates if "organiz" in update[0]] == [
        ("hand_interaction_frames_organizing", 0, 9),
        ("hand_interaction_frame_organized", 4, 9),
        ("hand_interaction_frame_organized", 8, 9),
    ]
    extracted_updates = [
        index
        for index, update in enumerate(updates)
        if update[0] == "hand_interaction_frame_extracted"
    ]
    organizing_index = next(
        index
        for index, update in enumerate(updates)
        if update[0] == "hand_interaction_frames_organizing"
    )
    assert max(extracted_updates) < organizing_index

    extracted_dir = output_dir / "work" / "extracted_frames"
    assert [
        sum(1 for _ in path.glob("*.jpg"))
        for path in sorted(extracted_dir.iterdir())
    ] == [4, 4, 1]
    assert not (output_dir / "work" / "temporary_frames").exists()


def test_adl_frame_extraction_emits_after_each_written_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = ModuleType("cv2")
    cv2.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
    cv2.VideoCapture = object  # type: ignore[attr-defined]
    cv2.destroyAllWindows = lambda: None  # type: ignore[attr-defined]
    cv2.imwrite = lambda _path, _frame: True  # type: ignore[attr-defined]
    moviepy = ModuleType("moviepy")
    moviepy_editor = ModuleType("moviepy.editor")
    moviepy_editor.VideoFileClip = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)
    monkeypatch.setitem(sys.modules, "moviepy.editor", moviepy_editor)
    module = _load_script(
        "adl_recognition/extract_adl_frames.py",
        "egomodelkit_test_adl_extract_progress",
    )

    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [object(), object(), object(), object(), object()]
            self.index = 0
            self.released = False

        @staticmethod
        def get(_property: int) -> float:
            return 4.0

        def isOpened(self) -> bool:
            return not self.released

        def read(self) -> tuple[bool, object | None]:
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    written_paths: list[Path] = []
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        module.cv2,
        "imwrite",
        lambda path, _frame: written_paths.append(Path(path)) or True,
    )
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: None)
    updates: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        module,
        "_emit_progress",
        lambda kind, **payload: updates.append(
            (kind, int(payload["current"]), int(payload["total"]))
        ),
    )

    current = module._extract_subclip_frames(
        video_path=tmp_path / "video001--1.MP4",
        output_dir=tmp_path / "video001--1",
        frame_fps=2,
        current=7,
        progress_total=12,
    )

    assert current == 10
    assert [path.name for path in written_paths] == [
        "frame_0.jpg",
        "frame_2.jpg",
        "frame_4.jpg",
    ]
    assert updates == [
        ("adl_frame_extracted", 8, 12),
        ("adl_frame_extracted", 9, 12),
        ("adl_frame_extracted", 10, 12),
    ]


def test_adl_flatten_and_combine_progress_cover_each_frame_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _load_script(
        "adl_recognition/entrypoint.py",
        "egomodelkit_test_adl_flatten_progress",
    )
    adl_dir = tmp_path / "meal-preparation-cleanup"
    for frame_index in (0, 1):
        detic_path = (
            adl_dir
            / "detic_raw"
            / "video001--1"
            / f"frame_{frame_index}_detic.pkl"
        )
        shan_path = (
            adl_dir
            / "subclips_shan"
            / "video001--1"
            / f"frame_{frame_index}_shan.pkl"
        )
        detic_path.parent.mkdir(parents=True, exist_ok=True)
        shan_path.parent.mkdir(parents=True, exist_ok=True)
        detic_path.write_bytes(b"detic")
        shan_path.write_bytes(b"shan")

    flatten_updates: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        entrypoint,
        "_emit_progress",
        lambda kind, **payload: flatten_updates.append(
            (kind, int(payload["current"]), int(payload["total"]))
        ),
    )

    paired_count = entrypoint._flatten_nested_model_outputs(adl_dir)

    assert paired_count == 2
    assert flatten_updates == [
        ("adl_prediction_frames_discovered", 0, 4),
        ("adl_prediction_frame_processed", 1, 4),
        ("adl_prediction_frame_processed", 2, 4),
    ]

    egoviz = ModuleType("egoviz")
    models = ModuleType("egoviz.models")
    processing = ModuleType("egoviz.models.processing")

    def fake_load_pickle(path: str) -> dict[str, object]:
        if path.endswith("_shan.pkl"):
            return {"objects": []}
        return {"boxes": [], "remapped_metadata": []}

    processing.load_pickle = fake_load_pickle  # type: ignore[attr-defined]
    models.processing = processing  # type: ignore[attr-defined]
    egoviz.models = models  # type: ignore[attr-defined]
    process_detic = ModuleType("process_detic")
    process_detic._load_mapping_df = lambda: object()  # type: ignore[attr-defined]
    process_detic.process_detic_preds = (  # type: ignore[attr-defined]
        lambda predictions, _mapping: predictions
    )
    torch = ModuleType("torch")
    torchvision = ModuleType("torchvision")
    torchvision_ops = ModuleType("torchvision.ops")
    torchvision_ops.box_iou = lambda _left, _right: []  # type: ignore[attr-defined]
    torchvision.ops = torchvision_ops  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "egoviz", egoviz)
    monkeypatch.setitem(sys.modules, "egoviz.models", models)
    monkeypatch.setitem(sys.modules, "egoviz.models.processing", processing)
    monkeypatch.setitem(sys.modules, "process_detic", process_detic)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.ops", torchvision_ops)
    combine = _load_script(
        "adl_recognition/combine_adl_predictions.py",
        "egomodelkit_test_adl_combine_progress",
    )
    combine_updates: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        combine,
        "_emit_progress",
        lambda kind, **payload: combine_updates.append(
            (kind, int(payload["current"]), int(payload["total"]))
        ),
    )

    all_predictions = combine.combine_predictions(
        data_root=tmp_path,
        active_iou=0.8,
        progress_offset=2,
        progress_total=4,
    )

    assert len(all_predictions) == 2
    assert combine_updates == [
        ("adl_prediction_frame_processed", 3, 4),
        ("adl_prediction_frame_processed", 4, 4),
    ]
    with (tmp_path / "all_preds.pkl").open("rb") as stream:
        persisted = pickle.load(stream)
    assert set(persisted) == set(all_predictions)
