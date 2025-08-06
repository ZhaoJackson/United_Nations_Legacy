# 🛠️ Installation & Setup Guide

Comprehensive installation guide for development, testing, and production environments.

## 📋 **System Requirements**

### **Minimum Requirements**
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.11.0+ (recommended: 3.11.9)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet for Azure OpenAI API calls

### **Recommended Development Setup**
- **OS**: macOS 12+ or Ubuntu 20.04+
- **Python**: 3.11.9
- **RAM**: 16GB+
- **Storage**: 20GB+ SSD
- **IDE**: VS Code with Python extension
- **Terminal**: iTerm2 (macOS) or Windows Terminal

## 🐍 **Python Environment Setup**

### **Option 1: Using pyenv (Recommended)**

```bash
# Install pyenv (macOS)
brew install pyenv

# Install pyenv (Linux)
curl https://pyenv.run | bash

# Install Python 3.11.9
pyenv install 3.11.9
pyenv local 3.11.9

# Verify version
python --version  # Should show Python 3.11.9
```

### **Option 2: Using conda**

```bash
# Create conda environment
conda create -n un_dashboard python=3.11.9

# Activate environment
conda activate un_dashboard

# Verify installation
python --version
which python
```

### **Option 3: Using system Python**

```bash
# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

## 📦 **Dependencies Installation**

### **Core Dependencies**

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify critical packages
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
```

### **Development Dependencies**

```bash
# Install development tools
pip install pytest pytest-cov black flake8 mypy

# Install Jupyter for notebook development
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=un_dashboard
```

### **Optional Dependencies**

```bash
# For advanced visualization
pip install kaleido  # Static image export for Plotly

# For development
pip install pre-commit  # Git hooks
pip install sphinx     # Documentation generation
```

## 🔑 **Credentials Configuration**

### **Azure OpenAI Setup**

1. **Obtain Azure OpenAI Credentials**:
   - Azure subscription with OpenAI service
   - GPT-4o deployment
   - O1 model deployment (if available)

2. **Create Secrets File**:

```bash
# Create .streamlit directory
mkdir -p .streamlit

# Create secrets.toml file
cat > .streamlit/secrets.toml << 'EOF'
# Azure OpenAI GPT-4o Configuration
AZURE_OPENAI_4O_API_KEY = "your-api-key-here"
AZURE_OPENAI_4O_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_OPENAI_4O_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_4O_DEPLOYMENT = "gpt-4o"

# Azure OpenAI O1 Configuration
AZURE_OPENAI_O1_API_KEY = "your-api-key-here"
AZURE_OPENAI_O1_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_OPENAI_O1_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_O1_DEPLOYMENT = "o1"
EOF
```

3. **Secure Permissions**:

```bash
# Set appropriate permissions
chmod 600 .streamlit/secrets.toml

# Add to .gitignore (already included)
echo ".streamlit/secrets.toml" >> .gitignore
```

### **Environment Variables (Alternative)**

```bash
# Export environment variables
export AZURE_OPENAI_4O_API_KEY="your-api-key"
export AZURE_OPENAI_4O_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_4O_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_4O_DEPLOYMENT="gpt-4o"

# Add to ~/.bashrc or ~/.zshrc for persistence
```

## 📊 **Data Setup**

### **Verify Existing Data**

```bash
# Check data structure
tree src/outputs/ -L 3

# Expected structure:
# src/outputs/
# ├── data_output/
# │   ├── Financial_Cleaned.csv (9.5MB)
# │   ├── SDG_Goals_Cleaned.csv (1.4MB)
# │   ├── UN_Agencies_Cleaned.csv (1.4MB)
# │   └── unified_data.csv (48MB)
# └── model_output/
#     ├── Agency_model.pkl (33MB)
#     ├── SDG_model.pkl (11MB)
#     ├── anomaly_detection.csv (9.9MB)
#     ├── funding_prediction.csv (9.5MB)
#     └── un_agency.csv (9.6MB)
```

### **Data Regeneration (if needed)**

```bash
# Navigate to notebooks directory
cd src/notebooks/

# Run data cleaning pipeline
jupyter notebook data_cleaning.ipynb

# Run model training notebooks (in order)
jupyter notebook sdg.ipynb
jupyter notebook agency.ipynb
jupyter notebook funding.ipynb
jupyter notebook anomaly.ipynb
jupyter notebook assistance.ipynb
```

### **Data Validation**

```bash
# Validate data integrity
python -c "
import pandas as pd
import os

# Check file sizes and basic structure
files = {
    'Financial_Cleaned.csv': 'src/outputs/data_output/Financial_Cleaned.csv',
    'SDG_Goals_Cleaned.csv': 'src/outputs/data_output/SDG_Goals_Cleaned.csv',
    'UN_Agencies_Cleaned.csv': 'src/outputs/data_output/UN_Agencies_Cleaned.csv'
}

for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f'{name}: {df.shape[0]:,} rows, {df.shape[1]} columns')
    else:
        print(f'Missing: {name}')
"
```

## 🔧 **Development Environment**

### **VS Code Setup**

1. **Install Extensions**:
   - Python
   - Jupyter
   - Streamlit Snippets
   - GitLens

2. **Configure Settings** (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/venv": true
    }
}
```

3. **Create Launch Configuration** (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit App",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "app.py"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### **Git Configuration**

```bash
# Configure Git hooks
pre-commit install

# Set up Git configuration
git config core.autocrlf false  # Prevent line ending issues
git config push.default simple
```

## 🧪 **Testing Setup**

### **Unit Tests**

```bash
# Create test directory structure
mkdir -p tests/{unit,integration,e2e}

# Install testing dependencies
pip install pytest pytest-cov pytest-mock

# Run tests
pytest tests/ -v --cov=src/
```

### **Streamlit Testing**

```bash
# Install Streamlit testing framework
pip install streamlit[test]

# Create test file
cat > tests/test_app.py << 'EOF'
import pytest
from streamlit.testing.v1 import AppTest

def test_app_loads():
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception
EOF
```

## 🚀 **Running the Application**

### **Development Mode**

```bash
# Standard development server
streamlit run app.py

# With custom configuration
streamlit run app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
```

### **Debug Mode**

```bash
# Enable debug logging
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run app.py

# Or with Python debugging
python -c "
import streamlit as st
st.set_option('client.showErrorDetails', True)
exec(open('app.py').read())
"
```

## 📊 **Performance Optimization**

### **Memory Management**

```bash
# Monitor memory usage
pip install memory-profiler

# Profile memory usage
mprof run streamlit run app.py
mprof plot
```

### **Caching Configuration**

Create `config.toml` in `.streamlit/`:

```toml
[global]
maxUploadSize = 200

[server]
maxMessageSize = 200
enableCORS = false
enableXsrfProtection = false

[client]
caching = true
```

## 🔒 **Security Setup**

### **File Permissions**

```bash
# Set appropriate permissions
chmod 755 app.py
chmod 755 -R pages/
chmod 755 -R src/
chmod 600 .streamlit/secrets.toml
```

### **Environment Isolation**

```bash
# Verify isolation
python -c "
import sys
print('Python executable:', sys.executable)
print('Python path:', sys.path[:3])
"
```

## ✅ **Installation Verification**

### **Comprehensive Test Script**

```bash
# Create verification script
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""Installation verification script"""

import sys
import os
import importlib
import pandas as pd

def check_python_version():
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} (need 3.11+)")
        return False

def check_dependencies():
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'openai',
        'sklearn', 'xgboost', 'joblib', 'pycountry'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def check_data_files():
    required_files = [
        'src/outputs/data_output/Financial_Cleaned.csv',
        'src/outputs/data_output/SDG_Goals_Cleaned.csv',
        'src/outputs/data_output/UN_Agencies_Cleaned.csv',
        'src/outputs/model_output/SDG_model.pkl',
        'src/outputs/model_output/Agency_model.pkl'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({size:.1f}MB)")
        else:
            print(f"❌ {file_path}")
            missing.append(file_path)
    
    return len(missing) == 0

def check_secrets():
    secrets_path = '.streamlit/secrets.toml'
    if os.path.exists(secrets_path):
        print(f"✅ {secrets_path}")
        return True
    else:
        print(f"⚠️  {secrets_path} (AI features will be disabled)")
        return False

if __name__ == "__main__":
    print("🔍 Verifying UN Dashboard Installation\n")
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Data Files", check_data_files()),
        ("Secrets Configuration", check_secrets())
    ]
    
    print(f"\n📊 Installation Summary:")
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")
    
    all_passed = all(passed for _, passed in checks[:3])  # Secrets optional
    
    if all_passed:
        print(f"\n🎉 Installation successful! Run: streamlit run app.py")
    else:
        print(f"\n❌ Installation incomplete. Check failed items above.")
        sys.exit(1)
EOF

# Run verification
python verify_installation.py
```

## 🆘 **Common Issues & Solutions**

### **Import Errors**
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **Port Conflicts**
```bash
# Find and kill processes using port 8501
lsof -ti:8501 | xargs kill -9
```

### **Memory Issues**
```bash
# Increase available memory
ulimit -v 8388608  # 8GB virtual memory limit
```

### **Permission Errors**
```bash
# Fix file permissions
chmod -R 755 .
chmod 600 .streamlit/secrets.toml
```

---

**🎯 Next Steps**: Once installation is complete, proceed to [Architecture Overview](./04_ARCHITECTURE.md) to understand the system design, or jump to [Development Workflow](./06_DEVELOPMENT_WORKFLOW.md) to start contributing.