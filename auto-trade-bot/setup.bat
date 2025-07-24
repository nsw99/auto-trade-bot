@echo off
:: UTF-8 모드로 변경하여 한글 깨짐을 방지합니다.
chcp 65001 > nul

setlocal

:: =================================================================
::               사용자 설정 부분 (파일 이름 확인!)
:: =================================================================
:: 아래 변수에 프로젝트 폴더에 넣어둔 설치 파일들의 정확한 이름을 적어주세요.
set PYTHON_INSTALLER=python-3.11.9-amd64.exe
set REDIS_INSTALLER=Redis-x64-3.0.504.msi


:: =================================================================
:: 1. 관리자 권한 확인 및 요청
:: =================================================================
ECHO Checking for administrator permissions...
net session >nul 2>&1
if %errorLevel% == 0 (
    ECHO Success: Administrator permissions confirmed.
) else (
    ECHO Failure: Administrator permissions not found.
    ECHO Attempting to re-launch with administrator permissions...
    powershell -Command "Start-Process -FilePath '%~dpnx0' -Verb RunAs"
    exit
)
echo.


:: =================================================================
:: 2. Redis 설치 확인 및 자동 설치 (추가된 부분)
:: =================================================================
ECHO Searching for Redis...
where redis-server >nul 2>nul
if %errorLevel% == 0 (
    ECHO Redis is already installed.
) else (
    ECHO Redis not found. Starting automatic installation...
    ECHO Installer: %REDIS_INSTALLER%
    
    :: MSI 설치 파일을 '조용히(quiet)' 실행하고, 로그는 남기지 않습니다.
    :: 설치가 끝날 때까지 기다립니다.
    start /wait msiexec /i %REDIS_INSTALLER% /quiet /qn /norestart
    
    ECHO Redis installation finished.
)
echo.


:: =================================================================
:: 3. 파이썬 설치 확인 및 자동 설치
:: =================================================================
ECHO Searching for Python...
where py >nul 2>nul
if %errorLevel% == 0 (
    ECHO Python is already installed.
) else (
    ECHO Python not found. Starting automatic installation...
    ECHO Installer: %PYTHON_INSTALLER%
    start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_py=1
    ECHO Python installation finished.
)
echo.


:: =================================================================
:: 4. 필요한 라이브러리 자동 설치 (requirements.txt)
:: =================================================================
ECHO Installing required libraries from requirements.txt...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
ECHO Library installation finished.
echo.


:: =================================================================
:: 5. 자동매매 봇 및 대시보드 실행
:: =================================================================
ECHO All preparations complete. Starting the application...
echo.

set PYTHONPATH=%~dp0

ECHO Starting Flask dashboard in the background...
start /b py dashboard/app.py

ECHO ----------------------------------------
timeout /t 2 /nobreak > nul

ECHO Opening dashboard in your browser at http://127.0.0.1:5000
start http://127.0.0.1:5000

ECHO ----------------------------------------
ECHO Starting DCA bot in the foreground...
py src/main.py

ECHO ----------------------------------------
ECHO Bot stopped.
pause