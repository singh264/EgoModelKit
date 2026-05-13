""" EgoModelKit command-line interface. """

import typer

app = typer.Typer(
    help = "EgoModelKit: reproducible egocentric-video model packaging and inference."
)

@app.callback()
def main() -> None:
    """ EgoModeKit command-line interface. """
