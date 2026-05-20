""" EgoModelKit command-line interface. """

from pathlib import Path
from typing import Annotated, Final

import typer

from egomodelkit.models.hand_object_contact import (
    DRY_RUN_VALIDATION_MESSAGE,
    HAND_OBJECT_CONTACT_MODEL_ID,
    HandObjectContactInputError,
    HandObjectContactRequest,
    validate_request,
)
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
def run(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists = True,
            file_okay = True,
            dir_okay = True,
            readable = True,
            help = "Path to one image file or an images directory.",
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
        help = "Public model id. Currently supported: hand-object-contact.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help = "Validate the request without executing the model.",
    ),
) -> None:
    """ Run one packaged model adapter. """
    if model_id != HAND_OBJECT_CONTACT_MODEL_ID:
        typer.echo(f"Unsupported model: {model_id}", err = True)
        raise typer.Exit(code=CLI_UNSUPPORTED_MODEL_EXIT_CODE)
    
    request = HandObjectContactRequest(
        input_path = input_path,
        output_dir = output_dir,
    )
    
    try:
        if dry_run:
            validate_request(request)
            typer.echo(DRY_RUN_VALIDATION_MESSAGE)
            typer.echo(f"Input: {input_path}")
            typer.echo(f"Output: {output_dir}")
            return
        
        run_hand_object_contact(
            request,
            progress=_report_progress,
        )
    except (
        HandObjectContactInputError,
        HandObjectContactRuntimeError,
        HostPrerequisiteError,
    ) as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(code=CLI_RUNTIME_ERROR_EXIT_CODE) from exc
    
    typer.echo("Completed: hand-object-contact")
    typer.echo(f"Outputs: {output_dir}")
