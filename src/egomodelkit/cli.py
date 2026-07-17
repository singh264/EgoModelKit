""" EgoModelKit command-line interface. """

import platform
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Final, cast

import typer

from egomodelkit.bandini_metrics import (
    DEFAULT_DOMINANT_HAND,
    HandLabel,
    VideoProcessingConfig,
)
from egomodelkit.models.adl_recognition import (
    ADL_RECOGNITION_DRY_RUN_VALIDATION_MESSAGE,
    ADL_RECOGNITION_MODEL_ID,
    AdlRecognitionInputError,
    AdlRecognitionRequest,
    validate_adl_recognition_request,
)
from egomodelkit.models.hand_object_contact import (
    HAND_OBJECT_CONTACT_DRY_RUN_VALIDATION_MESSAGE,
    HAND_OBJECT_CONTACT_MODEL_ID,
    HandObjectContactInputError,
    HandObjectContactRequest,
    validate_hand_object_contact_request,
)
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
from egomodelkit.runtime.adl_recognition import (
    AdlRecognitionRuntimeError,
    run_adl_recognition,
)
from egomodelkit.runtime.commands import (
    streaming_subprocess_runner,
    subprocess_runner,
)
from egomodelkit.runtime.hand_object_contact import (
    HandObjectContactRuntimeError,
    run_hand_object_contact,
)
from egomodelkit.runtime.preflight import (
    HostPrerequisiteError,
    ensure_host_runtime_ready,
)

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
        else:
            payload = update.payload
            write_progress_event(
                layout.progress_log_path,
                ProgressEvent(
                    stage = update.kind,
                    message = update.kind.replace("_", " "),
                    current = _payload_int(payload, "current"),
                    total = _payload_int(payload, "total"),
                    unit = (
                        str(payload["unit"])
                        if isinstance(payload.get("unit"), str)
                        else None
                    ),
                ),
            )

        _report_progress(message)

    return report


def _run_model_with_output_contract(
    *,
    model_id: str,
    request: HandObjectContactRequest | AdlRecognitionRequest,
) -> Path:
    """ Run one validated CLI model with the same output contract used by the GUI. """
    if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
        if not isinstance(request, HandObjectContactRequest):
            raise TypeError(
                "Hand-object contact requires a HandObjectContactRequest."
            )

        validate_hand_object_contact_request(request)
    elif model_id == ADL_RECOGNITION_MODEL_ID:
        if not isinstance(request, AdlRecognitionRequest):
            raise TypeError("ADL recognition requires an AdlRecognitionRequest.")

        validate_adl_recognition_request(request)
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

    output_root = request.output_dir
    run_id = _build_unique_cli_run_id(output_root)
    layout = build_run_output_layout(output_root, run_id = run_id)
    scenario = infer_input_scenario(model_id = model_id, input_path = request.input_path)
    video_processing_config = VideoProcessingConfig(
        dominant_hand = (
            request.dominant_hand
            if isinstance(request, AdlRecognitionRequest)
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
    )

    progress = _cli_progress_reporter(layout)

    try:
        if isinstance(request, HandObjectContactRequest):
            run_hand_object_contact(
                HandObjectContactRequest(
                    input_path = request.input_path,
                    output_dir = layout.run_dir,
                ),
                command_runner = subprocess_runner,
                streaming_command_runner = streaming_subprocess_runner,
                progress = progress,
            )
        else:
            run_adl_recognition(
                AdlRecognitionRequest(
                    input_path = request.input_path,
                    output_dir = layout.run_dir,
                    dominant_hand = request.dominant_hand,
                ),
                command_runner = subprocess_runner,
                streaming_command_runner = streaming_subprocess_runner,
                progress = progress,
            )

        finalize_runtime_outputs(
            layout = layout,
            model_id = model_id,
            input_path = request.input_path,
            scenario = scenario,
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
        )
        raise

    return layout.run_dir


def _run_hand_object_contact_with_output_contract(
    request: HandObjectContactRequest,
) -> Path:
    """ Run hand-object contact through the shared CLI output contract. """
    return _run_model_with_output_contract(
        model_id = HAND_OBJECT_CONTACT_MODEL_ID,
        request = request,
    )


def _run_adl_recognition_with_output_contract(
    request: AdlRecognitionRequest,
) -> Path:
    """ Run ADL recognition through the shared CLI output contract. """
    return _run_model_with_output_contract(
        model_id = ADL_RECOGNITION_MODEL_ID,
        request = request,
    )

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
        help = "Public model id. Supported: hand-object-contact, adl-recognition.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help = "Validate the request without executing the model.",
    ),
    dominant_hand: Annotated[
        str,
        typer.Option(
            "--dominant-hand",
            help = "Dominant hand after injury for ADL metrics: left or right.",
        ),
    ] = DEFAULT_DOMINANT_HAND,
) -> None:
    """ Run one packaged model adapter. """
    if (
        model_id != HAND_OBJECT_CONTACT_MODEL_ID and
        model_id != ADL_RECOGNITION_MODEL_ID
    ):
        typer.echo(f"Unsupported model: {model_id}", err = True)
        raise typer.Exit(code=CLI_UNSUPPORTED_MODEL_EXIT_CODE)
        
    try:
        if model_id == HAND_OBJECT_CONTACT_MODEL_ID:
            request = HandObjectContactRequest(
                input_path = input_path,
                output_dir = output_dir,
            )

            if dry_run:
                validate_hand_object_contact_request(request)
                typer.echo(HAND_OBJECT_CONTACT_DRY_RUN_VALIDATION_MESSAGE)
                typer.echo(f"Input: {input_path}")
                typer.echo(f"Output: {output_dir}")
                
                return

            completed_output_dir = _run_hand_object_contact_with_output_contract(request)

            typer.echo("Completed: hand-object-contact")
        else:
            request = AdlRecognitionRequest(
                input_path = input_path,
                output_dir = output_dir,
                dominant_hand = cast(HandLabel, dominant_hand),
            )

            if dry_run:
                validate_adl_recognition_request(request)
                typer.echo(ADL_RECOGNITION_DRY_RUN_VALIDATION_MESSAGE)
                typer.echo(f"Input: {input_path}")
                typer.echo(f"Output: {output_dir}")
                typer.echo(f"Dominant hand: {request.dominant_hand}")
                
                return

            completed_output_dir = _run_adl_recognition_with_output_contract(request)

            typer.echo("Completed: adl-recognition")
    except (
        HandObjectContactInputError,
        AdlRecognitionInputError,
        HostPrerequisiteError,
        HandObjectContactRuntimeError,
        AdlRecognitionRuntimeError,
        RuntimeError,
    ) as exc:
        typer.echo(f"Error: {exc}", err = True)
        
        raise typer.Exit(code=CLI_RUNTIME_ERROR_EXIT_CODE) from exc
    
    typer.echo(f"Outputs: {completed_output_dir}")
