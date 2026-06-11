""" EgoModelKit command-line interface. """

from pathlib import Path
from typing import Annotated, Final

import typer

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
from egomodelkit.runtime.adl_recognition import (
    AdlRecognitionRuntimeError,
    run_adl_recognition,
)
from egomodelkit.runtime.commands import subprocess_runner
from egomodelkit.runtime.hand_object_contact import (
    HandObjectContactRuntimeError,
    run_hand_object_contact,
)
from egomodelkit.runtime.preflight import HostPrerequisiteError

app = typer.Typer(
    help = "EgoModelKit: reproducible egocentric-video model packaging and inference."
)

CLI_RUNTIME_ERROR_EXIT_CODE: Final[int] = 1
CLI_UNSUPPORTED_MODEL_EXIT_CODE: Final[int] = 2

def _report_progress(message: str) -> None:
    """ Print one user-facing runtime progress message. """
    typer.echo(f"EgoModelKit: {message}")

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
        from egomodelkit.gui import launch_gui
        
        launch_gui(server_port = port, inbrowser = not no_browser)
    except RuntimeError as exc:
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

            run_hand_object_contact(
                request,
                command_runner = subprocess_runner,
                progress = _report_progress,
            )

            typer.echo("Completed: hand-object-contact")
        else:
            request = AdlRecognitionRequest(
                input_path = input_path,
                output_dir = output_dir,
            )

            if dry_run:
                validate_adl_recognition_request(request)
                typer.echo(ADL_RECOGNITION_DRY_RUN_VALIDATION_MESSAGE)
                typer.echo(f"Input: {input_path}")
                typer.echo(f"Output: {output_dir}")
                return

            run_adl_recognition(
                request,
                command_runner = subprocess_runner,
                progress = _report_progress,
            )

            typer.echo("Completed: adl-recognition")
    except (
        HandObjectContactInputError,
        AdlRecognitionInputError,
        HostPrerequisiteError,
        HandObjectContactRuntimeError,
        AdlRecognitionRuntimeError,
    ) as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(code=CLI_RUNTIME_ERROR_EXIT_CODE) from exc
    
    typer.echo(f"Outputs: {output_dir}")
