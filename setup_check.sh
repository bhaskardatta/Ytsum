#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🚀 Starting environment setup..."

# Function to check and install system dependencies
check_system_dependencies() {
    local dependencies=("ffmpeg" "python3" "pip3")
    
    for dep in "${dependencies[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            echo -e "${YELLOW}Installing $dep...${NC}"
            if [ -x "$(command -v apt-get)" ]; then
                sudo apt-get update && sudo apt-get install -y "$dep"
            elif [ -x "$(command -v yum)" ]; then
                sudo yum install -y "$dep"
            elif [ -x "$(command -v brew)" ]; then
                brew install "$dep"
            else
                echo -e "${RED}Cannot install $dep. Please install manually.${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}✓ $dep is installed${NC}"
        fi
    done
}

# Function to create and activate virtual environment
setup_virtual_environment() {
    echo "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip3 install --upgrade pip
}

# Function to validate Python packages
validate_python_packages() {
    echo "Validating Python packages..."
    python3 -c "
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
}

# Main setup process
main() {
    check_system_dependencies
    
    if [ ! -d "venv" ]; then
        setup_virtual_environment
    fi
    
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    validate_python_packages
    
    # Create necessary directories
    mkdir -p output temp audio logs
    
    # Check for .env file
    if [ ! -f ".env" ]; then
        echo "Creating .env file template..."
        echo "HUGGINGFACE_API_TOKEN=" > .env
        echo -e "${YELLOW}Please add your Hugging Face API token to .env file${NC}"
    fi
    
    echo -e "${GREEN}✓ Setup completed successfully!${NC}"
    echo "You can now run the summarizer with: python3 youtube_summarizer.py <youtube_url>"
}

main