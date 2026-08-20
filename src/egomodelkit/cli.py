""" EgoModelKit command-line interface. """

import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Final, cast

import typer

from egomodelkit.bandini_metrics import (
    DEFAULT_DOMINANT_HAND,
    HandLabel,
    VideoProcessingConfig,
)
from egomodelkit.models.catalog import cli_model_ids, get_model_definition
from egomodelkit.output_contract import (
    build_run_id,
    build_run_output_layout,
    create_output_scaffold,
    finalize_runtime_outputs,
    infer_input_scenario,
    write_run_summary,
)
from egomodelkit.progress import (
    ProgressEvent,
    parse_external_progress_line,
    write_progress_event,
    write_runtime_log_line,
)
from egomodelkit.runtime.adapters import ModelRequest, get_runtime_adapter
from egomodelkit.runtime.commands import (
    streaming_subprocess_runner,
    subprocess_runner,
)
from egomodelkit.runtime.disk_space import ensure_sufficient_disk_space
from egomodelkit.runtime.preflight import HostPrerequisiteError, ensure_host_runtime_ready

app = typer.Typer(
    help = "EgoModelKit: reproducible egocentric-video model packaging and inference."
)
CLI_RUNTIME_ERROR_EXIT_CODE: Final[int] = 1
CLI_UNSUPPORTED_MODEL_EXIT_CODE: Final[int] = 2

def _report_progress(message: str) -> None:
    """ Print one user-facing runtime progress message. """
    typer.echo(f"EgoModelKit: {message}")


def _build_unique_cli_run_id(output_root: Path) -> str:
    """ Return a run id that does not collide with an existing output folder. """
    base_run_id = build_run_id()

    for index in range(1000):
        run_id = base_run_id if index == 0 else f"{base_run_id}-{index + 1:03d}"
        layout = build_run_output_layout(output_root, run_id = run_id)

        if not layout.run_dir.exists():
            return run_id

    raise ValueError("Unable to create a unique run id.")


def _payload_int(payload: dict[str, object], key: str) -> int | None:
    """ Return one optional integer from an external progress payload. """
    value = payload.get(key)

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str) and value.isdigit():
        return int(value)

    return None


def _cli_progress_reporter(layout) -> Callable[[str], None]:
    """ Return a reporter that mirrors GUI console, progress, and runtime logs. """
    def report(message: str) -> None:
        update = parse_external_progress_line(message)

        if update is None:
            write_runtime_log_line(layout.runtime_log_path, message)
            _report_progress(message)
            return

        payload = update.payload
        event = ProgressEvent(
            stage = update.kind,
            message = update.kind.replace("_", " "),
            current = _payload_int(payload, "current"),
            total = _payload_int(payload, "total"),
            unit = (
                str(payload["unit"])
                if isinstance(payload.get("unit"), str)
                else None
            ),
        )
        write_progress_event(layout.progress_log_path, event)
        _report_progress(event.display_text)

    return report


def _run_model_with_output_contract(
    *,
    model_id: str,
    request: ModelRequest,
) -> Path:
    """ Run one validated CLI model with the same output contract used by the GUI. """
    if model_id not in cli_model_ids():
        raise ValueError(f"Unsupported model id: {model_id}")

    adapter = get_runtime_adapter(model_id)
    adapter.validate(request)

    ensure_sufficient_disk_space(
        model_id=model_id,
        input_path=request.input_path,
        output_dir=request.output_dir,
        progress=_report_progress,
        cleanup_stale_images=True,
    )

    output_root = request.output_dir
    run_id = _build_unique_cli_run_id(output_root)
    layout = build_run_output_layout(output_root, run_id = run_id)
    scenario = infer_input_scenario(model_id = model_id, input_path = request.input_path)
    definition = get_model_definition(model_id)
    video_processing_config = VideoProcessingConfig(
        dominant_hand=(
            request.dominant_hand
            if definition.uses_dominant_hand and hasattr(request, "dominant_hand")
            else DEFAULT_DOMINANT_HAND
        ),
    )

    create_output_scaffold(
        layout = layout,
        model_id = model_id,
        input_path = request.input_path,
        scenario = scenario,
        status = "running",
        video_processing_config = video_processing_config,
        invocation_interface = "cli",
        invocation_arguments = tuple(sys.argv),
    )

    progress = _cli_progress_reporter(layout)

    try:
        adapter.run(
            adapter.with_output_dir(request, layout.run_dir),
            command_runner=subprocess_runner,
            streaming_command_runner=streaming_subprocess_runner,
            progress=progress,
        )

        finalize_runtime_outputs(
            layout = layout,
            model_id = model_id,
            input_path = request.input_path,
            scenario=scenario,
            progress=progress,
        )

        write_run_summary(
            layout = layout,
            model_id = model_id,
            input_path = request.input_path,
            scenario = scenario,
            status = "completed",
        )
    except Exception as exc:
        write_runtime_log_line(layout.runtime_log_path, f"Run failed: {exc}")
        write_run_summary(
            layout = layout,
            model_id = model_id,
            input_path = request.input_path,
            scenario = scenario,
            status = "failed",
            error_message = str(exc),
        )
        raise

    return layout.run_dir

@app.callback()
def main() -> None:
    """ EgoModelKit command-line interface. """


@app.command()
def gui(
    port: int = typer.Option(
        7860,
        "--port",
        help = "Local port for the browser GUI.",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help = "Do not automatically open a browser window."
    ),
) -> None:
    """ Launch the local browser GUI. """
    try:
        if platform.system() == "Linux":
            ensure_host_runtime_ready(
                docker_executable = "docker",
                command_runner = subprocess_runner,
                progress = _report_progress,
            )

        from egomodelkit.gui import launch_gui

        launch_gui(server_port = port, inbrowser = not no_browser)
    except (HostPrerequisiteError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(code = CLI_RUNTIME_ERROR_EXIT_CODE) from exc

@app.command()
def run(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists = True,
            file_okay = True,
            dir_okay = True,
            readable = True,
            help = "Path to a model input file or directory.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help = "Directory for model outputs.",
        ),
    ],
    model_id: str = typer.Argument(
        ...,
        help = f"Public model id. Supported: {', '.join(cli_model_ids())}.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help = "Validate the request without executing the model.",
    ),
    dominant_hand: Annotated[
        str | None,
        typer.Option(
            "--dominant-hand",
            help="Dominant hand for hand-interaction only: left or right.",
        ),
    ] = None,
) -> None:
    """ Run one packaged model adapter. """
    if model_id not in cli_model_ids():
        typer.echo(f"Unsupported model: {model_id}", err = True)
        raise typer.Exit(code=CLI_UNSUPPORTED_MODEL_EXIT_CODE)

    try:
        definition = get_model_definition(model_id)
        if dominant_hand is not None and not definition.uses_dominant_hand:
            raise ValueError(
                "--dominant-hand is only supported for hand-interaction."
            )

        adapter = get_runtime_adapter(model_id)
        request = adapter.build_request(
            input_path=input_path,
            output_dir=output_dir,
            dominant_hand=cast(
                HandLabel,
                dominant_hand or DEFAULT_DOMINANT_HAND,
            ),
        )

        if dry_run:
            adapter.validate(request)
            ensure_host_runtime_ready(
                docker_executable=adapter.docker_executable,
                command_runner=subprocess_runner,
                require_linux_nvidia_gpu=False,
                progress=_report_progress,
            )
            ensure_sufficient_disk_space(
                model_id=model_id,
                input_path=input_path,
                output_dir=output_dir,
                progress=_report_progress,
            )
            typer.echo(adapter.dry_run_validation_message)
            typer.echo(f"Input: {input_path}")
            typer.echo(f"Output: {output_dir}")
            if definition.uses_dominant_hand and hasattr(request, "dominant_hand"):
                typer.echo(f"Dominant hand: {request.dominant_hand}")
            return

        completed_output_dir = _run_model_with_output_contract(
            model_id=model_id,
            request=request,
        )
        typer.echo(f"Completed: {model_id}")
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err = True)

        raise typer.Exit(code=CLI_RUNTIME_ERROR_EXIT_CODE) from exc

    typer.echo(f"Outputs: {completed_output_dir}")
