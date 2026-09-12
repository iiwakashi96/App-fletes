@echo off
REM Lanzador de la calculadora de fletes. Doble clic aqui para abrirla.
cd /d "%~dp0"

REM pythonw.exe abre la ventana sin dejar una consola negra detras.
set PY=C:\Users\f\AppData\Local\Python\pythoncore-3.14-64\pythonw.exe
if exist "%PY%" (
  start "" "%PY%" app_fletes.py
) else (
  echo No se encontro Python en:
  echo   %PY%
  echo Abre la app con:  py app_fletes.py
  pause
)
