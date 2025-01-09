@echo off
:: Colors for output
set RED=
set GREEN=
set YELLOW=
set NC=

echo 🚀 Starting environment setup...

:: Function to check and install system dependencies
call :check_system_dependencies

:: Function to create and activate virtual environment
if not exist "venv" (
    call :setup_virtual_environment
)

echo Installing Python dependencies...
pip install -r requirements.txt

call :validate_python_packages

:: Create necessary directories
mkdir output 2>nul
mkdir temp 2>nul
mkdir audio 2>nul
mkdir logs 2>nul

:: Check for .env file
if not exist ".env" (
    echo Creating .env file template...
    echo HUGGINGFACE_API_TOKEN= > .env
    echo Please add your Hugging Face API token to .env file
)

echo ✓ Setup completed successfully!
echo You can now run the summarizer with: python youtube_summarizer.py <youtube_url>
exit /b

:check_system_dependencies
set dependencies=ffmpeg python3 pip3

for %%d in (%dependencies%) do (
    where %%d >nul 2>&1
    if errorlevel 1 (
        echo Installing %%d...
        if exist "C:\Program Files\Git\usr\bin\apt-get.exe" (
            "C:\Program Files\Git\usr\bin\apt-get.exe" update
            "C:\Program Files\Git\usr\bin\apt-get.exe" install -y %%d
        ) else (
            echo Cannot install %%d. Please install manually.
            exit /b 1
        )
    ) else (
        echo ✓ %%d is installed
    )
)
exit /b

:setup_virtual_environment
echo Setting up Python virtual environment...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
exit /b

:validate_python_packages
echo Validating Python packages...
python -c "
import sys
required_packages = {
    'torch': 'torch',
    'transformers': 'transformers',
    'pysrt': 'pysrt',
    'requests': 'requests',
    'pydub': 'pydub',
    'youtube_transcript_api': 'youtube_transcript_api'
}

missing_packages = []
for package, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f'✓ {package} is installed')
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {", ".join(missing_packages)}')
    sys.exit(1)
"
exit /b