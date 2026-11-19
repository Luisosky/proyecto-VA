@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Demo En Vivo - Defensa Matematica (40%%)

echo ======================================================================
echo    DEMO EN VIVO: ATAQUE ADVERSARIO Y DEFENSA MATEMATICA (40%%)
echo    Controles: [1-5] Ataques - [d] Defensa ON/OFF - [m] Vista
echo               [n/p] Siguiente/Anterior foto - [q] Salir
echo ======================================================================
echo.
echo [INFO] Cargando modelos (puede tardar 2-4 segundos la primera vez)...
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo_en_vivo.py --fuente test %*
) else (
    python demo_en_vivo.py --fuente test %*
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] El script termino con codigo %errorlevel%.
    pause
)
