@echo off
TITLE Text Summarizer
ECHO Installing required dependencies...
python -m pip install --upgrade pip
ECHO.
ECHO Running the Text Summarizer...
python Text-Summarizer.py
python Model_trainer.py

:: Wait for Enter key before exiting
echo Press Enter to exit...
pause >nul
exit
