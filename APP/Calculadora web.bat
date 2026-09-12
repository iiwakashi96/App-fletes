@echo off
REM Abre la calculadora de fletes en el navegador. Doble clic aqui.
REM Se usa 'python -m streamlit' porque la carpeta Scripts de Python no
REM esta en el PATH y el comando 'streamlit' a secas no se reconoce.
cd /d "%~dp0"

set PY=C:\Users\f\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PY%" (
  echo No se encontro Python en: %PY%
  pause
  exit /b 1
)

echo Abriendo la calculadora... cierra esta ventana para detenerla.
"%PY%" -m streamlit run streamlit_app.py
pause
