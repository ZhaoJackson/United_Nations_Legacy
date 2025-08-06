# 🔧 Troubleshooting Guide

Comprehensive troubleshooting guide for common issues and their solutions in the UN Financial Intelligence Dashboard.

## 🚨 **Common Issues & Quick Fixes**

### **1. Application Won't Start**

#### **Issue: Import Errors**
```bash
Error: ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Fix Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to .bashrc/.zshrc permanently
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> ~/.bashrc

# Alternative: Install in development mode
pip install -e .
```

#### **Issue: Missing Dependencies**
```bash
Error: No module named 'streamlit'
```

**Solution:**
```bash
# Verify virtual environment is activated
which python
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check for conflicting packages
pip check
```

#### **Issue: Port Already in Use**
```bash
Error: Port 8501 is already in use
```

**Solution:**
```bash
# Find and kill process using port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app.py --server.port 8502

# Check all Streamlit processes
ps aux | grep streamlit
```

---

### **2. Data Loading Issues**

#### **Issue: File Not Found Errors**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'src/outputs/data_output/Financial_Cleaned.csv'
```

**Solution:**
```bash
# Check file existence
ls -la src/outputs/data_output/

# Verify file permissions
chmod 755 -R src/outputs/

# Regenerate missing data files
cd src/notebooks/
jupyter notebook data_cleaning.ipynb

# Run complete data pipeline
python scripts/regenerate_data.py
```

#### **Issue: Data Format Errors**
```bash
Error: ParserError: Error tokenizing data
```

**Solution:**
```python
# Debug data loading
import pandas as pd

try:
    df = pd.read_csv('problematic_file.csv')
except pd.errors.ParserError as e:
    print(f"Parser error: {e}")
    # Try with different encoding
    df = pd.read_csv('problematic_file.csv', encoding='utf-8')
    # Or skip bad lines
    df = pd.read_csv('problematic_file.csv', error_bad_lines=False)
```

#### **Issue: Memory Errors with Large Datasets**
```bash
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Use chunked loading
def load_large_file_in_chunks(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)
    return pd.concat(chunks, ignore_index=True)

# Optimize data types
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df
```

---

### **3. Azure OpenAI Issues**

#### **Issue: API Key Not Working**
```bash
Error: Incorrect API key provided
```

**Solution:**
```bash
# Verify credentials in secrets.toml
cat .streamlit/secrets.toml

# Test API key manually
python -c "
import openai
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key='your-key',
    api_version='2024-12-01-preview',
    azure_endpoint='your-endpoint'
)
print('API key is valid')
"

# Check environment variables
echo $AZURE_OPENAI_4O_API_KEY
```

#### **Issue: Rate Limiting**
```bash
Error: Rate limit exceeded
```

**Solution:**
```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                        continue
                    raise e
            return None
        return wrapper
    return decorator

@retry_with_backoff()
def make_api_call(prompt):
    # Your API call here
    pass
```

#### **Issue: Quota Exceeded**
```bash
Error: You exceeded your current quota
```

**Solution:**
1. Check Azure OpenAI usage in Azure portal
2. Upgrade to higher tier if needed
3. Implement usage monitoring:

```python
def monitor_api_usage():
    """Monitor API usage and warn before limits"""
    # Implement usage tracking
    usage_count = get_current_usage()
    quota_limit = get_quota_limit()
    
    if usage_count > quota_limit * 0.8:
        st.warning("⚠️ Approaching API quota limit")
```

---

### **4. Performance Issues**

#### **Issue: Slow Page Loading**
```bash
Page takes >10 seconds to load
```

**Solution:**
```python
# Add performance monitoring
import time
import streamlit as st

def profile_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        if execution_time > 2:  # Log slow functions
            st.warning(f"Slow function: {func.__name__} took {execution_time:.2f}s")
        
        return result
    return wrapper

# Optimize caching
@st.cache_data(ttl=3600, max_entries=50)
def load_expensive_data():
    # Expensive operation
    return data

# Use fragments for expensive components
@st.fragment
def render_expensive_chart():
    # Chart rendering code
    pass
```

#### **Issue: Memory Usage Too High**
```bash
Process killed due to memory usage
```

**Solution:**
```python
import gc
import psutil

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    memory_percent = process.memory_percent()
    
    if memory_percent > 80:
        st.warning(f"High memory usage: {memory_percent:.1f}%")
        # Force garbage collection
        gc.collect()

# Optimize data structures
def optimize_dataframe(df):
    """Reduce DataFrame memory usage"""
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    # Downcast numeric types
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df
```

---

### **5. UI/UX Issues**

#### **Issue: Charts Not Displaying**
```bash
Plotly charts show as blank
```

**Solution:**
```python
# Ensure Plotly is properly configured
import plotly.graph_objects as go
import plotly.express as px

# Check for data issues
def debug_chart_data(df):
    print(f"Data shape: {df.shape}")
    print(f"Data types: {df.dtypes}")
    print(f"Null values: {df.isnull().sum()}")
    print(f"Sample data:\n{df.head()}")

# Add error handling to chart creation
def create_safe_chart(df, x_col, y_col):
    try:
        if df.empty:
            st.warning("No data available for chart")
            return None
        
        if x_col not in df.columns or y_col not in df.columns:
            st.error(f"Missing columns: {x_col}, {y_col}")
            return None
        
        fig = px.bar(df, x=x_col, y=y_col)
        return fig
    
    except Exception as e:
        st.error(f"Chart creation failed: {e}")
        return None
```

#### **Issue: Sidebar Not Working**
```bash
Sidebar elements not responding
```

**Solution:**
```python
# Debug sidebar state
def debug_sidebar():
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Session state keys: {list(st.session_state.keys())}")
    
    # Reset sidebar state if needed
    if st.sidebar.button("Reset Sidebar"):
        for key in list(st.session_state.keys()):
            if key.startswith('sidebar_'):
                del st.session_state[key]
        st.rerun()

# Ensure proper widget keys
selected_year = st.sidebar.selectbox(
    "Select Year",
    options=[2020, 2021, 2022, 2023, 2024, 2025],
    key="sidebar_year"  # Unique key prevents conflicts
)
```

#### **Issue: Session State Conflicts**
```bash
Widget values not persisting
```

**Solution:**
```python
# Proper session state management
def init_session_state():
    """Initialize session state with defaults"""
    defaults = {
        'selected_country': 'Afghanistan',
        'selected_year': 2025,
        'selected_theme': 'education',
        'chat_messages': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Debug session state
def debug_session_state():
    if st.checkbox("Show Session State Debug"):
        st.write("Session State:", st.session_state)
```

---

### **6. Model & ML Issues**

#### **Issue: Model Loading Fails**
```bash
Error: cannot load model from pickle file
```

**Solution:**
```python
import joblib
import pickle

def safe_model_loading(model_path):
    """Safely load ML models with error handling"""
    try:
        # Try joblib first (recommended for sklearn)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Joblib loading failed: {e}")
        
        try:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e2:
            st.error(f"Pickle loading failed: {e2}")
            return None

# Validate model compatibility
def validate_model(model, test_data):
    """Validate model works with current data"""
    try:
        # Test prediction
        sample_prediction = model.predict(test_data.iloc[:1])
        return True
    except Exception as e:
        st.error(f"Model validation failed: {e}")
        return False
```

#### **Issue: Prediction Errors**
```bash
Error: Input contains NaN values
```

**Solution:**
```python
def prepare_prediction_data(data):
    """Clean data for model prediction"""
    # Handle missing values
    data = data.fillna(data.median(numeric_only=True))
    
    # Handle categorical missing values
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].fillna('Unknown')
    
    # Ensure correct data types
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)
    
    return data

def safe_prediction(model, data):
    """Make predictions with error handling"""
    try:
        cleaned_data = prepare_prediction_data(data)
        predictions = model.predict(cleaned_data)
        return predictions
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None
```

---

### **7. Deployment Issues**

#### **Issue: Streamlit Cloud Build Fails**
```bash
Build failed: Package installation error
```

**Solution:**
1. **Check requirements.txt format:**
```bash
# Ensure proper formatting
streamlit>=1.47.0,<2.0.0
pandas>=2.0.0,<2.3.0

# No space around operators
# Use specific version ranges
```

2. **Add Python version specification:**
```bash
# Create .python-version file
echo "3.11.9" > .python-version
```

3. **Check for conflicting dependencies:**
```bash
pip-check-reqs requirements.txt
pip list --outdated
```

#### **Issue: Application Crashes on Streamlit Cloud**
```bash
App crashes with MemoryError
```

**Solution:**
```python
# Optimize for cloud deployment
import streamlit as st

# Reduce cache size for cloud
@st.cache_data(ttl=3600, max_entries=10)  # Reduced from 50
def load_cloud_optimized_data():
    # Load only essential data
    columns_needed = ['Country', 'Region', 'Total required resources']
    return pd.read_csv('data.csv', usecols=columns_needed)

# Implement data sampling for large datasets
def sample_data_for_cloud(df, max_rows=10000):
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df
```

#### **Issue: Secrets Not Working on Cloud**
```bash
KeyError: 'AZURE_OPENAI_4O_API_KEY'
```

**Solution:**
1. **Verify secrets configuration in Streamlit Cloud:**
   - Go to app settings
   - Check secrets are properly formatted
   - No trailing spaces or quotes

2. **Add fallback handling:**
```python
def get_api_key_safely():
    """Get API key with fallback"""
    try:
        return st.secrets["AZURE_OPENAI_4O_API_KEY"]
    except KeyError:
        st.warning("⚠️ Azure OpenAI API key not configured. AI features disabled.")
        return None
```

---

## 🔍 **Debugging Tools & Techniques**

### **1. Streamlit Debug Mode**

```python
# Enable debug mode
import streamlit as st

# Show debug information
if st.checkbox("Debug Mode"):
    st.write("**Session State:**", st.session_state)
    st.write("**App Mode:**", st.get_option("global.developmentMode"))
    
    # Memory usage
    import psutil
    memory = psutil.virtual_memory()
    st.write(f"**Memory Usage:** {memory.percent}%")
```

### **2. Logging Configuration**

```python
import logging
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def debug_function_call(func_name, *args, **kwargs):
    """Log function calls for debugging"""
    logger.info(f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
```

### **3. Performance Profiling**

```python
import cProfile
import pstats
import io

def profile_streamlit_app():
    """Profile Streamlit app performance"""
    pr = cProfile.Profile()
    pr.enable()
    
    # Your app code here
    main_app()
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    st.text(s.getvalue())
```

---

## 🛠️ **Maintenance Scripts**

### **1. Health Check Script**

Create `scripts/health_check.py`:

```python
#!/usr/bin/env python3
"""Comprehensive health check for the application"""

import os
import sys
import pandas as pd
import requests
import joblib
from pathlib import Path

def check_file_integrity():
    """Check all required files exist and are readable"""
    required_files = [
        'src/outputs/data_output/Financial_Cleaned.csv',
        'src/outputs/data_output/SDG_Goals_Cleaned.csv',
        'src/outputs/data_output/UN_Agencies_Cleaned.csv',
        'src/outputs/model_output/SDG_model.pkl',
        'src/outputs/model_output/Agency_model.pkl'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing: {file_path}")
            return False
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=1)
            elif file_path.endswith('.pkl'):
                model = joblib.load(file_path)
            print(f"✅ Valid: {file_path}")
        except Exception as e:
            print(f"❌ Corrupted: {file_path} - {e}")
            return False
    
    return True

def check_dependencies():
    """Check all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'openai',
        'sklearn', 'xgboost', 'joblib', 'pycountry'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ Missing package: {package}")
            return False
    
    return True

def check_app_startup():
    """Test if app can start without errors"""
    try:
        # Test imports
        from src.commonconst import financial_df, get_theme_list
        from src.dynamic_analysis import DynamicDataProcessor
        
        # Test data loading
        themes = get_theme_list()
        if not themes:
            print("❌ No themes found")
            return False
        
        if financial_df.empty:
            print("❌ Financial data is empty")
            return False
        
        print("✅ App startup test passed")
        return True
    
    except Exception as e:
        print(f"❌ App startup failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Running health check...\n")
    
    checks = [
        ("File Integrity", check_file_integrity()),
        ("Dependencies", check_dependencies()),
        ("App Startup", check_app_startup())
    ]
    
    print("\n📊 Health Check Results:")
    all_passed = True
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All health checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some health checks failed!")
        sys.exit(1)
```

### **2. Data Validation Script**

Create `scripts/validate_data.py`:

```python
#!/usr/bin/env python3
"""Validate data integrity and format"""

import pandas as pd
import numpy as np

def validate_financial_data():
    """Validate financial dataset"""
    df = pd.read_csv('src/outputs/data_output/Financial_Cleaned.csv')
    
    # Check required columns
    required_cols = [
        'Country', 'Region', 'Theme', 'Total required resources',
        'Total available resources', 'Total expenditure resources'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['Total required resources']):
        print("❌ Total required resources is not numeric")
        return False
    
    # Check for negative values where inappropriate
    if (df['Total required resources'] < 0).any():
        print("❌ Negative values in required resources")
        return False
    
    # Check for reasonable data ranges
    max_funding = df['Total required resources'].max()
    if max_funding > 1e12:  # 1 trillion
        print(f"⚠️ Suspiciously high funding amount: {max_funding}")
    
    print("✅ Financial data validation passed")
    return True

if __name__ == "__main__":
    validate_financial_data()
```

---

## 📞 **Getting Help**

### **1. Log Collection**

Before reaching out for help, collect these logs:

```bash
# Streamlit logs
streamlit run app.py --logger.level debug > streamlit.log 2>&1

# System information
python --version > system_info.txt
pip list >> system_info.txt
uname -a >> system_info.txt

# Memory and disk usage
free -h >> system_info.txt
df -h >> system_info.txt
```

### **2. Error Reporting Template**

When reporting issues, include:

```markdown
## Issue Description
Brief description of the problem

## Environment
- OS: [macOS/Windows/Linux]
- Python Version: [3.11.9]
- Streamlit Version: [1.47.0]
- Deployment: [Local/Streamlit Cloud/Docker]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Messages
```
Full error traceback here
```

## Additional Context
Screenshots, logs, or other relevant information
```

### **3. Contact Information**

- **Developer**: Zichen Zhao (Jackson)
- **Email**: ziche.zhao@un.org | zichen.zhao@columbia.edu
- **GitHub Issues**: https://github.com/ZhaoJackson/United_Nations_Legacy/issues
- **Emergency Contact**: For critical production issues

---

**🔄 Next Steps**: Explore [Future Development](./17_FUTURE_DEVELOPMENT.md) for roadmap and enhancement opportunities, or check [API Reference](./11_API_REFERENCE.md) for detailed function documentation.