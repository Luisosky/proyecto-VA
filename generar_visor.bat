@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Visor Didactico de Transformaciones - Seguridad Canina

echo ======================================================================
echo    LABORATORIO DIDACTICO: SEGURIDAD CANINA (TAMANO + BOZAL)
echo    Auditoria Hibrida de Modelos ante Desafios de Robustez Reales
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Ejecutando generador con el entorno virtual .venv...
    ".venv\Scripts\python.exe" generar_visor.py %1
    goto ABRIR_HTML
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Ejecutando generador con python del sistema...
    python generar_visor.py %1
    goto ABRIR_HTML
)

echo [AVISO] Python no detectado. Abriendo visor HTML existente...
echo.

:ABRIR_HTML
if exist "visor_transformaciones.html" (
    echo [EXITO] Abriendo visor_transformaciones.html en tu navegador...
    start "" "visor_transformaciones.html"
) else (
    echo [ERROR] No se encontro el archivo visor_transformaciones.html.
    pause
    exit /b 1
)

timeout /t 2 > nul
exit /b 0
