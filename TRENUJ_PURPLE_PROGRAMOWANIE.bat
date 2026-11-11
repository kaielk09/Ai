@echo off
chcp 65001 >nul
setlocal
title Purple AI - trening programowania
echo ============================================================
echo       PURPLE AI - TRENING PROGRAMOWANIA (20 000 KROKOW)
echo ============================================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo BLAD: Nie znaleziono Pythona. Zainstaluj Python 3.11 lub 3.12
  echo i zaznacz opcje "Add Python to PATH".
  pause
  exit /b 1
)
set "WORK=%LOCALAPPDATA%\PurpleAI_Programming_Training"
set "OUT=%USERPROFILE%\Desktop\PURPLE_AI_PROGRAMOWANIE"
if not exist "%WORK%" mkdir "%WORK%"
if not exist "%OUT%" mkdir "%OUT%"
set "PURPLE_WORK=%WORK%"
set "PURPLE_OUT=%OUT%"
cd /d "%WORK%"
echo [1/4] Tworzenie srodowiska...
if not exist "%WORK%\.venv\Scripts\python.exe" (
  python -m venv "%WORK%\.venv"
  if errorlevel 1 goto fail
)
set "PY=%WORK%\.venv\Scripts\python.exe"
echo [2/4] Instalowanie bibliotek...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%PY%" -m pip install requests huggingface_hub
if errorlevel 1 goto fail
echo [3/4] Instalowanie PyTorch z obsluga CUDA...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo Nie udalo sie pobrac PyTorch CUDA.
  goto fail
)
echo [4/4] Pobieranie skryptu treningowego...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/kaielk09/Ai/main/Purple%%20programowanie/train_from_base.py' -OutFile '%WORK%\train_from_base.py'"
if errorlevel 1 goto fail
echo.
echo Trening pobierze bazowe wagi z Hugging Face oraz legalne zrodla kodu z GitHuba.
echo Wyniki beda zapisywane na pulpicie do: %OUT%
echo Nie zamykaj tego okna podczas treningu.
echo.
"%PY%" "%WORK%\train_from_base.py"
if errorlevel 1 goto fail
echo.
echo ============================================================
echo TRENING ZAKONCZONY. WYNIKI: %OUT%
echo ============================================================
pause
exit /b 0
:fail
echo.
echo WYSTAPIL BLAD. Trening mogl nie zostac uruchomiony lub zostal przerwany.
echo Szczegoly sa powyzej. Pliki wynikowe (jesli powstaly): %OUT%
pause
exit /b 1
