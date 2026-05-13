from typer.testing import CliRunner

from egomodelkit.cli import app

runner = CliRunner()

def test_cli_help_renders():
    result = runner.invoke(app, ["--help"])
        
    assert result.exit_code == 0
    assert "EgoModelKit" in result.stdout
