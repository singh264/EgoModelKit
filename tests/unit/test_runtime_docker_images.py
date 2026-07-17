from pathlib import Path

from egomodelkit.runtime.commands import CommandResult
from egomodelkit.runtime.docker_images import (
    MANAGED_RUNTIME_LABEL,
    DockerImageIdentity,
    build_runtime_image_identity,
    remove_stale_runtime_images,
)


def _identity(context_dir: Path, *, build_arguments: list[str] | None = None):
    return build_runtime_image_identity(
        runtime_name="example-runtime",
        repository="egomodelkit-example",
        context_dir=context_dir,
        build_arguments=build_arguments or ["--build-arg", "VALUE=one"],
    )


def test_runtime_image_identity_is_deterministic_and_labeled(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    identity = _identity(tmp_path)

    assert identity == _identity(tmp_path)
    assert identity.tag.startswith("egomodelkit-example:sha-")
    assert len(identity.tag.rsplit("-", 1)[1]) == 12
    assert f"{MANAGED_RUNTIME_LABEL}=true" in identity.label_arguments


def test_runtime_image_identity_changes_with_context_and_build_arguments(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    initial = _identity(tmp_path)

    dockerfile.write_text("FROM scratch\nLABEL changed=true\n", encoding="utf-8")
    changed_context = _identity(tmp_path)
    changed_argument = _identity(
        tmp_path,
        build_arguments=["--build-arg", "VALUE=two"],
    )

    assert changed_context.tag != initial.tag
    assert changed_argument.tag != changed_context.tag


def test_runtime_image_identity_ignores_python_cache_files(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    initial = _identity(tmp_path)
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "ignored.pyc").write_bytes(b"cache")

    assert _identity(tmp_path) == initial


def test_stale_runtime_images_are_removed_but_current_image_is_kept() -> None:
    current = DockerImageIdentity(
        runtime_name="example-runtime",
        repository="egomodelkit-example",
        fingerprint="a" * 64,
    )
    commands: list[list[str]] = []

    def capture(command: list[str]) -> CommandResult:
        commands.append(command)
        if command[1:3] == ["image", "ls"]:
            return CommandResult(
                returncode=0,
                stdout=(
                    f"{current.tag}\n"
                    "egomodelkit-example:dev\n"
                    "egomodelkit-example:sha-old12345678\n"
                ),
                stderr="",
            )
        return CommandResult(returncode=0, stdout="", stderr="")

    removed = remove_stale_runtime_images(
        docker_executable="docker",
        current_image=current,
        capture_runner=capture,
    )

    assert removed == (
        "egomodelkit-example:dev",
        "egomodelkit-example:sha-old12345678",
    )
    assert ["docker", "image", "rm", current.tag] not in commands


def test_stale_image_cleanup_is_best_effort() -> None:
    messages: list[str] = []
    current = DockerImageIdentity(
        runtime_name="example-runtime",
        repository="egomodelkit-example",
        fingerprint="b" * 64,
    )

    removed = remove_stale_runtime_images(
        docker_executable="docker",
        current_image=current,
        capture_runner=lambda _command: CommandResult(1, "", "unavailable"),
        progress=messages.append,
    )

    assert removed == ()
    assert any("current image remains usable" in message for message in messages)
