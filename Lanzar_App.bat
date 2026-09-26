@echo off
setlocal enabledelayedexpansion
title RadeonVideoAI - Suite de Restauración y Reescalado con IA

echo ======================================================================
echo    RadeonVideoAI Suite - Motor Generativo para Windows
echo    GPU: cualquier NVIDIA/AMD/Intel con DirectX 12 (DirectML) / CPU
echo ======================================================================
echo.

:: Obtener directorio base del script
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

:: Si se ejecuta desde la raíz del proyecto, buscar en dist/RadeonVideoAI
if exist "%APP_DIR%dist\RadeonVideoAI\RadeonVideoAI.exe" (
    set "EXE_PATH=%APP_DIR%dist\RadeonVideoAI\RadeonVideoAI.exe"
    set "BIN_DIR=%APP_DIR%dist\RadeonVideoAI"
) else if exist "%APP_DIR%RadeonVideoAI.exe" (
    set "EXE_PATH=%APP_DIR%RadeonVideoAI.exe"
    set "BIN_DIR=%APP_DIR%"
) else (
    echo [MODO DESARROLLO] No se encontro binario compilado. Ejecutando mediante Python local...
    python main.py
    goto :fin
)

:: Configurar variables de entorno y PATH para aislar el runtime portable
set "PATH=%BIN_DIR%;%PATH%"
set "DML_VISIBLE_DEVICES=0"
set "KMP_DUPLICATE_LIB_OK=TRUE"

echo [*] Iniciando RadeonVideoAI en modo portable independiente...
start "" "%EXE_PATH%"

:fin
exit /b 0

