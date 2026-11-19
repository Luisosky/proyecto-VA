@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Panel de Clasificacion Canina - Tamano y Bozal

echo ======================================================================
echo    PANEL DE CLASIFICACION CANINA (TAMANO + BOZAL + SEGURIDAD)
echo    Evaluacion real en paralelo: Clasico (OpenCV) vs. YOLOv8
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Ejecutando clasificacion real sobre dataset_seguridad_canina\test...
    ".venv\Scripts\python.exe" generar_clasificador_html.py
) else (
    python generar_clasificador_html.py
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Hubo un error al generar la pagina de clasificacion.
    pause
    exit /b 1
)

if exist "visor_clasificacion_perros.html" (
    echo [EXITO] Abriendo visor_clasificacion_perros.html en tu navegador...
    start "" "visor_clasificacion_perros.html"
)

timeout /t 2 > nul
exit /b 0
