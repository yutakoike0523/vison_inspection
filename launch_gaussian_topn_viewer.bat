@echo off
cd /d "%~dp0"
python gaussian_topn_viewer.py
if errorlevel 1 pause
