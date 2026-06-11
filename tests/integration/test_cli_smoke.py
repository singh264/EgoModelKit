from typer.testing import CliRunner

from egomodelkit.cli import app

runner = CliRunner()

def test_cli_help_renders():
    result = runner.invoke(app, ["--help"])
        
    assert result.exit_code == 0
    assert "EgoModelKit" in result.stdout

def test_gui_command_launches_without_browser(monkeypatch):
    captured = {}
    
    def fake_launch_gui(*, server_port, inbrowser):
        captured["server_port"] = server_port
        captured["inbrowser"] = inbrowser
    
    monkeypatch.setattr("egomodelkit.gui.launch_gui", fake_launch_gui)
    
    result = runner.invoke(app, ["gui", "--port", "9000", "--no-browser"])
    
    assert result.exit_code == 0
    assert captured == {"server_port": 9000, "inbrowser": False}

def test_gui_command_reports_runtime_error(monkeypatch):
    def fake_launch_gui(*, server_port, inbrowser):
        raise RuntimeError("GUI missing")
    
    monkeypatch.setattr("egomodelkit.gui.launch_gui", fake_launch_gui)
    
    result = runner.invoke(app, ["gui"])
    
    assert result.exit_code == 1
    assert "GUI missing" in result.output
