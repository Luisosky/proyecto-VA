@echo off
chcp 65001 > nul
title Demo En Vivo - Defensa Matematica (40%)

echo ======================================================================
echo    DEMO EN VIVO: ATAQUE ADVERSARIO Y DEFENSA MATEMATICA (40%%)
echo    Controles: [1-5] Ataques | [d] Defensa ON/OFF | [m] Vista | [q] Salir
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo_en_vivo.py --fuente test %*
) else (
    python demo_en_vivo.py --fuente test %*
)

if %errorlevel% neq 0 pause
