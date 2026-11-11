@echo off
REM Jalankan Contact Angle Analyzer. Pertama kali (atau bila virtualenv rusak): membuat virtualenv & install paket.
cd /d "%~dp0"

REM Virtualenv yang disalin dari PC lain punya python.exe tetapi tidak berfungsi -> cek benar-benar bisa impor.
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python -c "import streamlit, cv2, scipy.optimize, pandas, matplotlib, openpyxl" >nul 2>&1
    if errorlevel 1 (
        echo Virtualenv rusak atau tidak lengkap, dibuat ulang...
        rmdir /s /q .venv
    )
)

if not exist .venv\Scripts\python.exe (
    echo Membuat virtualenv...
    python -m venv .venv || goto :error
    .venv\Scripts\python -m pip install --upgrade pip
    REM Folder OneDrive bisa mengunci file sesaat saat sinkronisasi (WinError 32): coba sekali lagi.
    .venv\Scripts\python -m pip install -r requirements.txt || .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)

REM --server.address 0.0.0.0 agar bisa dibuka dari PC lain di jaringan lab
.venv\Scripts\streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false
goto :eof

:error
echo Gagal menyiapkan lingkungan Python. Pastikan Python 3.11+ terpasang.
echo Bila muncul "An Application Control policy has blocked this file", matikan Smart App Control di Windows Security.
pause
