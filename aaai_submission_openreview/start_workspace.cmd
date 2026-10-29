@echo off
cd /d "%~dp0"
echo Keep this window open while editing. Saving writes local files; Git is separate.
python -X utf8 "primer\workspace.py" serve --open
if errorlevel 1 pause
