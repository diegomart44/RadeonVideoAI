@echo off
title Compilar RadeonVideoAI Suite Portable
echo ======================================================================
echo    Iniciando Compilacion de RadeonVideoAI Suite para Windows 11
echo ======================================================================
echo.

python build_portable.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Se produjo un error durante la compilacion.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [EXITO] Compilacion completada. La carpeta dist\RadeonVideoAI esta lista para ser distribuida.
pause

