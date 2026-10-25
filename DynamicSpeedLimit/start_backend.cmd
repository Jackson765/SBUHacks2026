@echo off
setlocal
pushd "%~dp0"

if defined FLOW_PYTHON goto run
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "FLOW_PYTHON=%CONDA_PREFIX%\python.exe"
    goto run
)
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "FLOW_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
    goto run
)
if exist "%USERPROFILE%\anaconda3\envs\dynamic-speed\python.exe" (
    set "FLOW_PYTHON=%USERPROFILE%\anaconda3\envs\dynamic-speed\python.exe"
    goto run
)
if exist "%USERPROFILE%\miniconda3\envs\dynamic-speed\python.exe" (
    set "FLOW_PYTHON=%USERPROFILE%\miniconda3\envs\dynamic-speed\python.exe"
    goto run
)
set "FLOW_PYTHON=python"

:run
echo Python: %FLOW_PYTHON%
echo Open http://127.0.0.1:8000 after the server starts. Press Ctrl+C to stop.
"%FLOW_PYTHON%" -m backend.app
set "FLOW_EXIT=%ERRORLEVEL%"
if not "%FLOW_EXIT%"=="0" (
    echo.
    echo If dependencies are missing, run this in PowerShell from DynamicSpeedLimit:
    echo ^& "%FLOW_PYTHON%" -m pip install -r backend/requirements.txt
)
popd
exit /b %FLOW_EXIT%
