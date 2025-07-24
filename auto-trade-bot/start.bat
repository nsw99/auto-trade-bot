@echo off
:: ----------------------------------------
:: 파이썬이 모듈을 찾을 수 있도록 프로젝트 루트 폴더를 경로에 추가 (가장 중요)
set PYTHONPATH=%~dp0
:: ----------------------------------------
ECHO Starting Flask dashboard in the background...
start /b python dashboard/app.py

ECHO ----------------------------------------
timeout /t 2 /nobreak > nul

ECHO Opening dashboard in your browser at http://127.0.0.1:5000
start http://127.0.0.1:5000

ECHO ----------------------------------------
ECHO Starting DCA bot in the foreground...
python -m src.main

ECHO ----------------------------------------
ECHO Bot stopped.Please close the dashboard window manually if it's still running.
pause
