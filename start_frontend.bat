@echo off
cd /d "%~dp0"
py -3.11 -m http.server 5500 --directory frontend
