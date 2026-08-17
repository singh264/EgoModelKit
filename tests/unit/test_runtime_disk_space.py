import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import egomodelkit.runtime.disk_space as disk_space
from egomodelkit.models.adl_recognition import ADL_RECOGNITION_MODEL_ID
from egomodelkit.models.hand_interaction import HAND_INTERACTION_MODEL_ID
from egomodelkit.models.hand_object_contact import HAND_OBJECT_CONTACT_MODEL_ID
from egomodelkit.runtime.commands import CommandResult
from egomodelkit.runtime.disk_space import (
    DiskSpacePreflightError,
    ensure_sufficient_disk_space,
    estimate_pipeline_storage,
    format_bytes,
)


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _mp4_bytes(*, duration_seconds: int = 10, width: int = 1920, height: int = 1080) -> bytes:
    timescale = 1_000
    mvhd = bytearray(100)
    mvhd[0] = 0
    struct.pack_into(">I", mvhd, 12, timescale)
    struct.pack_into(">I", mvhd, 16, duration_seconds * timescale)

    tkhd = bytearray(84)
    tkhd[0] = 0
    struct.pack_into(">I", tkhd, 76, width << 16)
    struct.pack_into(">I", tkhd, 80, height << 16)

    moov = _box(b"moov", _box(b"mvhd", bytes(mvhd)) + _box(b"trak", _box(b"tkhd", bytes(tkhd))))
    return _box(b"ftyp", b"isom\x00\x00\x00\x00") + moov


def _write_mp4(
    path: Path,
    *,
    duration_seconds: int = 10,
    width: int = 1920,
    height: int = 1080,
) -> None:
    path.write_bytes(
        _mp4_bytes(
            duration_seconds=duration_seconds,
            width=width,
            height=height,
        )
    )


def test_estimate_hand_interaction_uses_duration_and_fixed_processing_resolution(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    _write_mp4(video, duration_seconds=10)

    estimate = estimate_pipeline_storage(
        model_id=HAND_INTERACTION_MODEL_ID,
        input_path=video,
    )

    assert estimate.estimated_output_bytes > 300 * 720 * 405
    assert estimate.peak_output_bytes > estimate.estimated_output_bytes
    assert estimate.estimated_file_count == 300 * 4 + 10
    assert estimate.input_bytes == video.stat().st_size


def test_estimate_adl_uses_source_dimensions_and_video_working_copy(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_mp4(video, duration_seconds=120, width=1280, height=720)

    estimate = estimate_pipeline_storage(
        model_id=ADL_RECOGNITION_MODEL_ID,
        input_path=video,
    )

    assert estimate.estimated_output_bytes > 120 * 1280 * 720
    assert estimate.peak_output_bytes > estimate.estimated_output_bytes
    assert estimate.estimated_file_count == 120 * 5 + 2 * 3 + 10


def test_estimate_adl_all_preds_and_hand_object_contact(tmp_path: Path) -> None:
    predictions = tmp_path / "all_preds.pkl"
    predictions.write_bytes(b"p" * 100)
    adl = estimate_pipeline_storage(
        model_id=ADL_RECOGNITION_MODEL_ID,
        input_path=predictions,
    )
    assert adl.input_bytes == 100
    assert adl.estimated_file_count == 100

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"i" * 1_000)
    hoc = estimate_pipeline_storage(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        input_path=image,
    )
    assert hoc.input_bytes == 1_000
    assert hoc.estimated_output_bytes > 6_000
    assert hoc.estimated_file_count == 23


def test_estimate_rejects_unknown_model_missing_files_and_invalid_mp4(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported model id"):
        estimate_pipeline_storage(model_id="unknown", input_path=tmp_path)

    with pytest.raises(DiskSpacePreflightError, match="no supported input"):
        estimate_pipeline_storage(
            model_id=HAND_OBJECT_CONTACT_MODEL_ID,
            input_path=tmp_path,
        )

    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not-mp4")
    with pytest.raises(DiskSpacePreflightError, match="duration"):
        estimate_pipeline_storage(
            model_id=HAND_INTERACTION_MODEL_ID,
            input_path=invalid,
        )


def test_adl_estimate_rejects_mp4_without_track_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    timescale = 1_000
    mvhd = bytearray(100)
    struct.pack_into(">I", mvhd, 12, timescale)
    struct.pack_into(">I", mvhd, 16, 10 * timescale)
    path.write_bytes(_box(b"moov", _box(b"mvhd", bytes(mvhd))))

    with pytest.raises(DiskSpacePreflightError, match="frame dimensions"):
        estimate_pipeline_storage(
            model_id=ADL_RECOGNITION_MODEL_ID,
            input_path=path,
        )


def test_format_bytes_uses_mib_and_gib() -> None:
    assert format_bytes(2 * disk_space.GIB) == "2.0 GiB"
    assert format_bytes(512 * disk_space.MIB) == "512.0 MiB"


def test_preflight_combines_output_and_docker_growth_on_same_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    messages: list[str] = []

    monkeypatch.setattr(
        disk_space.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * disk_space.GIB),
    )
    monkeypatch.setattr(disk_space, "_ensure_output_inodes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        disk_space,
        "_docker_storage_anchor",
        lambda **_kwargs: tmp_path,
    )
    monkeypatch.setattr(
        disk_space,
        "_storage_anchor",
        lambda *_args, **_kwargs: tmp_path,
    )

    def capture(command: list[str]) -> CommandResult:
        if command[1:3] == ["image", "inspect"]:
            return CommandResult(0, "123456\n", "")
        raise AssertionError(command)

    report = ensure_sufficient_disk_space(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        input_path=image,
        output_dir=tmp_path / "results",
        capture_runner=capture,
        progress=messages.append,
    )

    assert report.images_to_build == ()
    assert report.docker_incremental_bytes == disk_space.DOCKER_RUNTIME_TRANSIENT_BYTES
    assert report.output_free_bytes == 100 * disk_space.GIB
    assert report.docker_free_bytes == 100 * disk_space.GIB
    assert messages[-1] == "Disk-space preflight passed."


def test_preflight_plans_missing_images_removes_stale_tags_and_uses_old_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    stale_tag = "egomodelkit-hand-object-contact:sha-old12345678"

    monkeypatch.setattr(
        disk_space.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * disk_space.GIB),
    )
    monkeypatch.setattr(disk_space, "_ensure_output_inodes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(disk_space, "_docker_storage_anchor", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(
        disk_space,
        "_storage_anchor",
        lambda *_args, **_kwargs: tmp_path,
    )

    commands: list[list[str]] = []

    def capture(command: list[str]) -> CommandResult:
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            tag = command[3]
            if tag == stale_tag:
                return CommandResult(0, str(30 * disk_space.GIB), "")
            return CommandResult(1, "", "missing")
        if command[1:3] == ["image", "ls"]:
            return CommandResult(0, stale_tag + "\n", "")
        if command[1:3] == ["image", "rm"]:
            return CommandResult(0, "", "")
        raise AssertionError(command)

    report = ensure_sufficient_disk_space(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        input_path=image,
        output_dir=tmp_path / "results",
        capture_runner=capture,
    )

    assert report.images_to_build == ("hand-object-contact",)
    assert report.removed_images == (stale_tag,)
    assert report.docker_incremental_bytes > 30 * disk_space.GIB
    assert ["docker", "image", "rm", stale_tag] in commands


def test_preflight_can_plan_missing_images_without_removing_stale_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    stale_tag = "egomodelkit-hand-object-contact:sha-old12345678"

    monkeypatch.setattr(
        disk_space.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * disk_space.GIB),
    )
    monkeypatch.setattr(disk_space, "_ensure_output_inodes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(disk_space, "_docker_storage_anchor", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(
        disk_space,
        "_storage_anchor",
        lambda *_args, **_kwargs: tmp_path,
    )

    commands: list[list[str]] = []

    def capture(command: list[str]) -> CommandResult:
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            tag = command[3]
            if tag == stale_tag:
                return CommandResult(0, str(30 * disk_space.GIB), "")
            return CommandResult(1, "", "missing")
        if command[1:3] == ["image", "ls"]:
            return CommandResult(0, stale_tag + "\n", "")
        raise AssertionError(command)

    report = ensure_sufficient_disk_space(
        model_id=HAND_OBJECT_CONTACT_MODEL_ID,
        input_path=image,
        output_dir=tmp_path / "results",
        capture_runner=capture,
        cleanup_stale_images=False,
    )

    assert report.images_to_build == ("hand-object-contact",)
    assert report.removed_images == ()
    assert report.docker_incremental_bytes > 30 * disk_space.GIB
    assert not any(command[1:3] == ["image", "rm"] for command in commands)


def test_preflight_rejects_insufficient_shared_free_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(
        disk_space.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1 * disk_space.GIB),
    )
    monkeypatch.setattr(disk_space, "_ensure_output_inodes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(disk_space, "_docker_storage_anchor", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(
        disk_space,
        "_storage_anchor",
        lambda *_args, **_kwargs: tmp_path,
    )

    with pytest.raises(DiskSpacePreflightError, match="Insufficient disk space"):
        ensure_sufficient_disk_space(
            model_id=HAND_OBJECT_CONTACT_MODEL_ID,
            input_path=image,
            output_dir=tmp_path / "results",
            capture_runner=lambda _command: CommandResult(0, "123\n", ""),
        )


def test_inode_preflight_rejects_low_file_entry_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        disk_space.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_favail=5),
    )
    with pytest.raises(DiskSpacePreflightError, match="filesystem inodes"):
        disk_space._ensure_output_inodes(tmp_path, estimated_file_count=10)

    monkeypatch.setattr(
        disk_space.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_favail=0),
    )
    disk_space._ensure_output_inodes(tmp_path, estimated_file_count=10)


def test_storage_anchor_and_docker_anchor_platform_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "missing" / "results"
    assert disk_space._nearest_existing_path(nested) == tmp_path

    docker_root = tmp_path / "docker"
    docker_root.mkdir()
    result = CommandResult(0, str(docker_root) + "\n", "")
    assert disk_space._docker_storage_anchor(
        docker_executable="docker",
        capture_runner=lambda _command: result,
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: False,
    ) == docker_root

    assert disk_space._docker_storage_anchor(
        docker_executable="docker",
        capture_runner=lambda _command: CommandResult(1, "", ""),
        platform_detector=lambda: "Darwin",
        wsl_detector=lambda: False,
    ) == Path.home()

    anchor = disk_space._storage_anchor(
        nested,
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: False,
    )
    assert anchor == tmp_path


def test_private_docker_helpers_handle_invalid_inspect_and_listing() -> None:
    invalid = disk_space._docker_image_size_bytes(
        docker_executable="docker",
        image_tag="example:tag",
        capture_runner=lambda _command: CommandResult(0, "not-a-number", ""),
    )
    assert invalid is None

    empty = disk_space._repository_image_tags(
        docker_executable="docker",
        repository="example",
        capture_runner=lambda _command: CommandResult(1, "", "error"),
    )
    assert empty == ()


def test_iter_boxes_handles_extended_zero_and_invalid_sizes(tmp_path: Path) -> None:
    extended_payload = b"abcd"
    extended = struct.pack(">I4sQ", 1, b"uuid", 20) + extended_payload
    zero = struct.pack(">I4s", 0, b"free") + b"tail"
    path = tmp_path / "boxes.bin"
    path.write_bytes(extended + zero)

    with path.open("rb") as stream:
        boxes = list(disk_space._iter_boxes(stream, 0, path.stat().st_size))
    assert [box[0] for box in boxes] == [b"uuid", b"free"]

    bad = tmp_path / "bad.bin"
    bad.write_bytes(struct.pack(">I4s", 4, b"bad!"))
    with bad.open("rb") as stream:
        assert list(disk_space._iter_boxes(stream, 0, bad.stat().st_size)) == []


def test_version_one_mvhd_and_tkhd_helpers(tmp_path: Path) -> None:
    payload = bytearray(96)
    payload[0] = 1
    struct.pack_into(">I", payload, 20, 1_000)
    struct.pack_into(">Q", payload, 24, 5_000)
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    with path.open("rb") as stream:
        assert disk_space._mvhd_duration(stream, 0, len(payload)) == 5.0

    tkhd = bytearray(96)
    tkhd[0] = 1
    struct.pack_into(">I", tkhd, 88, 640 << 16)
    struct.pack_into(">I", tkhd, 92, 480 << 16)
    trak = _box(b"tkhd", bytes(tkhd))
    path.write_bytes(trak)
    with path.open("rb") as stream:
        assert disk_space._trak_dimensions(stream, 0, len(trak)) == (640, 480)


def test_inode_preflight_noops_when_statvfs_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "statvfs", raising=False)
    disk_space._ensure_output_inodes(tmp_path, estimated_file_count=1)


def test_runtime_image_identities_cover_all_models_and_unknown() -> None:
    hoc = disk_space._runtime_image_identities(HAND_OBJECT_CONTACT_MODEL_ID)
    hand = disk_space._runtime_image_identities(HAND_INTERACTION_MODEL_ID)
    adl = disk_space._runtime_image_identities(ADL_RECOGNITION_MODEL_ID)

    assert [identity.runtime_name for identity in hoc] == ["hand-object-contact"]
    assert [identity.runtime_name for identity in hand] == [
        "hand-interaction",
        "hand-object-contact",
    ]
    assert [identity.runtime_name for identity in adl] == [
        "adl-recognition-core",
        "adl-recognition-detic",
        "hand-object-contact",
    ]
    with pytest.raises(ValueError, match="Unsupported model id"):
        disk_space._runtime_image_identities("unknown")


def test_probe_mp4_ignores_non_metadata_children_and_zero_dimension_track(
    tmp_path: Path,
) -> None:
    timescale = 1_000
    mvhd = bytearray(100)
    struct.pack_into(">I", mvhd, 12, timescale)
    struct.pack_into(">I", mvhd, 16, 2 * timescale)

    zero_tkhd = bytearray(84)
    video_tkhd = bytearray(84)
    struct.pack_into(">I", video_tkhd, 76, 320 << 16)
    struct.pack_into(">I", video_tkhd, 80, 240 << 16)
    moov_payload = (
        _box(b"free", b"ignored")
        + _box(b"trak", _box(b"tkhd", bytes(zero_tkhd)))
        + _box(b"mvhd", bytes(mvhd))
        + _box(b"trak", _box(b"tkhd", bytes(video_tkhd)))
    )
    path = tmp_path / "clip.mp4"
    path.write_bytes(_box(b"free", b"top") + _box(b"moov", moov_payload))

    metadata = disk_space._probe_mp4(path, require_dimensions=True)
    assert metadata.duration_seconds == 2.0
    assert (metadata.width, metadata.height) == (320, 240)


def test_box_and_metadata_helpers_handle_truncated_payloads() -> None:
    import io

    with io.BytesIO(b"1234") as stream:
        assert list(disk_space._iter_boxes(stream, 0, 12)) == []

    extended_header_only = struct.pack(">I4s", 1, b"uuid") + b"1234"
    with io.BytesIO(extended_header_only) as stream:
        assert list(disk_space._iter_boxes(stream, 0, 16)) == []

    with io.BytesIO(b"short") as stream:
        assert disk_space._mvhd_duration(stream, 0, 5) == 0.0

    version_one_short = bytes([1]) + b"\x00" * 20
    with io.BytesIO(version_one_short) as stream:
        assert disk_space._mvhd_duration(stream, 0, len(version_one_short)) == 0.0

    short_tkhd = _box(b"tkhd", b"short")
    with io.BytesIO(short_tkhd) as stream:
        assert disk_space._trak_dimensions(stream, 0, len(short_tkhd)) == (0, 0)

    short_v1_tkhd_payload = bytes([1]) + b"\x00" * 83
    short_v1_tkhd = _box(b"tkhd", short_v1_tkhd_payload)
    with io.BytesIO(short_v1_tkhd) as stream:
        assert disk_space._trak_dimensions(stream, 0, len(short_v1_tkhd)) == (0, 0)

    no_tkhd = _box(b"free", b"nothing")
    with io.BytesIO(no_tkhd) as stream:
        assert disk_space._trak_dimensions(stream, 0, len(no_tkhd)) == (0, 0)


def test_docker_storage_anchor_wsl_and_linux_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path) == "/mnt/c":
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert disk_space._docker_storage_anchor(
        docker_executable="docker",
        capture_runner=lambda _command: CommandResult(1, "", ""),
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: True,
    ) == Path("/mnt/c")

    monkeypatch.setattr(Path, "exists", original_exists)
    fallback = disk_space._docker_storage_anchor(
        docker_executable="docker",
        capture_runner=lambda _command: CommandResult(1, "", ""),
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: False,
    )
    assert fallback.exists()


def test_storage_anchor_wsl_drive_wsl_backing_and_no_c_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_exists = Path.exists
    original_resolve = Path.resolve

    def drive_resolve(path: Path, *args, **kwargs) -> Path:
        if path == tmp_path:
            return Path("/mnt/d/results")
        return original_resolve(path, *args, **kwargs)

    def drive_exists(path: Path) -> bool:
        if str(path) == "/mnt/d":
            return True
        if str(path) == "/mnt/c":
            return False
        return original_exists(path)

    monkeypatch.setattr(disk_space, "_nearest_existing_path", lambda _path: tmp_path)
    monkeypatch.setattr(Path, "resolve", drive_resolve)
    monkeypatch.setattr(Path, "exists", drive_exists)
    assert disk_space._storage_anchor(
        tmp_path / "out",
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: True,
    ) == Path("/mnt/d")

    monkeypatch.setattr(Path, "resolve", lambda self, *args, **kwargs: Path("/home/user"))
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True if str(self) == "/mnt/c" else original_exists(self),
    )
    assert disk_space._storage_anchor(
        tmp_path / "out",
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: True,
    ) == Path("/mnt/c")

    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: False if str(self) == "/mnt/c" else original_exists(self),
    )
    assert disk_space._storage_anchor(
        tmp_path / "out",
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: True,
    ) == tmp_path


def test_inode_preflight_passes_with_sufficient_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        disk_space.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_favail=100_000),
    )
    disk_space._ensure_output_inodes(tmp_path, estimated_file_count=10)


def test_disk_space_report_serializes_for_gui() -> None:
    report = disk_space.DiskSpaceReport(
        estimated_output_bytes=1,
        peak_output_bytes=2,
        output_free_bytes=3,
        docker_incremental_bytes=4,
        docker_free_bytes=5,
        images_to_build=("image-a",),
        removed_images=("old:tag",),
        estimated_file_count=6,
    )
    assert report.as_dict() == {
        "estimatedOutputBytes": 1,
        "peakOutputBytes": 2,
        "outputFreeBytes": 3,
        "dockerIncrementalBytes": 4,
        "dockerFreeBytes": 5,
        "imagesToBuild": ["image-a"],
        "removedImages": ["old:tag"],
        "estimatedFileCount": 6,
    }


def test_storage_anchor_wsl_mount_falls_back_to_c_when_drive_root_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_exists = Path.exists
    monkeypatch.setattr(disk_space, "_nearest_existing_path", lambda _path: tmp_path)
    monkeypatch.setattr(Path, "resolve", lambda self, *args, **kwargs: Path("/mnt/d/results"))

    def fake_exists(path: Path) -> bool:
        if str(path) == "/mnt/d":
            return False
        if str(path) == "/mnt/c":
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert disk_space._storage_anchor(
        tmp_path / "out",
        platform_detector=lambda: "Linux",
        wsl_detector=lambda: True,
    ) == Path("/mnt/c")
