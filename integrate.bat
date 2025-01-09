@echo off
:: Define the folder path and Python scripts
set FOLDER_PATH=audio
set SCRIPT1=youtube.py
set SCRIPT2=extract_pod.py
set SCRIPT3=audio_concatenator.py

if "%1"=="" (
    echo Usage: %0 <link_argument>
    exit /b 1
)

set LINK_ARG=%1

echo Running %SCRIPT1% with argument: %LINK_ARG%...
python %SCRIPT1% %LINK_ARG%

if errorlevel 1 (
    echo Error: %SCRIPT1% failed. Exiting.
    exit /b 1
)

echo %SCRIPT1% executed successfully.

if not exist "%FOLDER_PATH%\*" (
    echo Error: Folder %FOLDER_PATH% is empty. Exiting.
    exit /b 1
)

echo Running %SCRIPT2%...
python %SCRIPT2%

if errorlevel 1 (
    echo Error: %SCRIPT2% failed. Exiting.
    exit /b 1
)

echo %SCRIPT2% executed successfully.

echo Running %SCRIPT3%...
python %SCRIPT3%

if errorlevel 1 (
    echo Error: %SCRIPT3% failed. Exiting.
    exit /b 1
)

echo %SCRIPT3% executed successfully.

echo Emptying folder %FOLDER_PATH%...
del /q "%FOLDER_PATH%\*"

if errorlevel 1 (
    echo Error: Failed to empty folder %FOLDER_PATH%.
    exit /b 1
)

echo Folder %FOLDER_PATH% emptied successfully.
echo All scripts executed successfully. Exiting.
exit /b 0