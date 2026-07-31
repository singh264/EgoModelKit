@echo off
set "SCRIPT_NAME=egomodelkit_machine_setup.py"
set "TARGET_PATH=%USERPROFILE%\Downloads\%SCRIPT_NAME%"

if exist "%TARGET_PATH%" (
    python "%TARGET_PATH%"
) else (
    echo Error: %SCRIPT_NAME% was not found in your Downloads folder.
    echo Expected path: %TARGET_PATH%
)

pause
