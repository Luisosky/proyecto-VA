@echo off
chcp 65001 > nul
title Visor Didactico de Transformaciones - Vision Artificial

echo ======================================================================
echo    LABORATORIO DIDACTICO: TRANSFORMACIONES DE VISION ARTIFICIAL
echo    Auditoria Hibrida de Modelos ante Desafios de Robustez Reales
echo ======================================================================
echo.

REM Intentar ejecutar con python o py si esta disponible
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Ejecutando generador con python...
    python generar_visor.py
    goto ABRIR_HTML
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Ejecutando generador con lanzador py...
    py generar_visor.py
    goto ABRIR_HTML
)

echo [AVISO] Python no se detecto en el PATH del sistema o requiere configuracion.
echo Utilizando el visor HTML pregenerado directamente...
echo.

:ABRIR_HTML
if exist "visor_transformaciones.html" (
    echo [EXITO] Abriendo visor_transformaciones.html en tu navegador predeterminado...
    start "" "visor_transformaciones.html"
) else (
    echo [ERROR] No se encontro el archivo visor_transformaciones.html.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo   Visor abierto con exito. Puedes cerrar esta ventana.
echo ======================================================================
timeout /t 3 > nul
exit /b 0
