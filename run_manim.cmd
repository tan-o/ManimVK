@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" -m manim %*
exit /b %ERRORLEVEL%
