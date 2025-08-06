# 🔄 Development Workflow

Comprehensive guide for developers contributing to the UN Financial Intelligence Dashboard.

## 🎯 **Development Philosophy**

### **Core Principles**
1. **Data-Driven**: All features should serve analytical insights
2. **User-Centric**: Design for both technical and non-technical users
3. **Performance-First**: Optimize for large dataset handling
4. **Maintainable**: Write clear, documented, testable code
5. **Secure**: Protect sensitive data and API credentials

### **Quality Standards**
- **Code Coverage**: Minimum 80% for new features
- **Documentation**: All public functions must be documented
- **Performance**: Page load times < 5 seconds
- **Accessibility**: WCAG 2.1 AA compliance
- **Security**: No hardcoded credentials or sensitive data

---

## 🚀 **Getting Started**

### **1. Development Environment Setup**

```bash
# Clone repository
git clone https://github.com/ZhaoJackson/United_Nations_Legacy.git
cd United_Nations_Legacy

# Create development branch
git checkout -b feature/your-feature-name

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

### **2. Development Configuration**

Create `.env` file for local development:
```bash
# Development environment variables
DEBUG=True
STREAMLIT_LOGGER_LEVEL=debug
PYTHONPATH=${PYTHONPATH}:$(pwd)
```

Configure VS Code settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

---

## 📝 **Coding Standards**

### **Python Style Guide**

**Follow PEP 8 with these specific guidelines:**

```python
# Function naming: snake_case
def process_financial_data():
    pass

# Class naming: PascalCase
class DataProcessor:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3

# File naming: snake_case.py
# data_processor.py, financial_analysis.py
```

### **Streamlit-Specific Patterns**

```python
# Page structure pattern
def render_page():
    """Standard page rendering pattern"""
    setup_page_config()
    render_sidebar()
    render_main_content()
    render_footer()

# Component pattern with caching
@st.cache_data(ttl=3600)
def load_data_component():
    """Cached data loading for components"""
    return expensive_data_operation()

# State management pattern
def init_session_state():
    """Initialize session state variables"""
    if 'key' not in st.session_state:
        st.session_state.key = default_value
```

### **Documentation Standards**

```python
def analyze_funding_gap(
    data: pd.DataFrame, 
    year: int, 
    region: str = "All"
) -> Dict[str, float]:
    """
    Analyze funding gaps for specified parameters.
    
    Args:
        data: Financial dataset with required columns
        year: Analysis year (2020-2026)
        region: Region filter or "All" for global analysis
        
    Returns:
        Dictionary containing gap analysis metrics:
        - total_required: Total funding requirements
        - total_available: Available funding
        - gap_amount: Funding shortfall
        - coverage_ratio: Percentage of requirements met
        
    Raises:
        ValueError: If year is outside valid range
        KeyError: If required columns are missing
        
    Example:
        >>> result = analyze_funding_gap(df, 2025, "Africa")
        >>> print(f"Gap: ${result['gap_amount']:,.0f}")
    """
    # Implementation here
```

---

## 🏗️ **Feature Development Workflow**

### **1. Planning Phase**

Before starting development:

1. **Create GitHub Issue**: Document feature requirements
2. **Design Review**: Discuss architecture with team
3. **Data Impact**: Assess data pipeline changes needed
4. **UI/UX Mockup**: Create wireframes for UI changes
5. **Performance Plan**: Consider impact on load times

### **2. Development Phase**

#### **Branch Strategy**
```bash
# Feature branches
git checkout -b feature/sdg-prediction-enhancement
git checkout -b bugfix/memory-optimization
git checkout -b hotfix/critical-security-fix

# Branch naming convention:
# feature/{feature-name}
# bugfix/{issue-description}
# hotfix/{critical-fix}
```

#### **Development Cycle**
```bash
# 1. Start development
git checkout -b feature/new-analytics-page

# 2. Make changes
# ... develop your feature ...

# 3. Test locally
streamlit run app.py
pytest tests/

# 4. Code quality checks
black .
flake8 .
mypy src/

# 5. Commit changes
git add .
git commit -m "feat: add advanced analytics page with ML insights"

# 6. Push to remote
git push origin feature/new-analytics-page
```

### **3. Testing Requirements**

#### **Unit Tests**
```python
# tests/test_data_processing.py
import pytest
import pandas as pd
from src.commonconst import filter_by_agency

def test_filter_by_agency():
    """Test agency filtering functionality"""
    # Arrange
    test_data = pd.DataFrame({
        'Country': ['USA', 'Kenya', 'Brazil'],
        'Agencies': ['UNDP', 'UNICEF; WHO', 'UNDP; UNICEF']
    })
    
    # Act
    result = filter_by_agency(test_data, 'UNDP')
    
    # Assert
    assert len(result) == 2
    assert all(result['Agencies'].str.contains('UNDP'))
```

#### **Integration Tests**
```python
# tests/test_ml_pipeline.py
def test_sdg_prediction_pipeline():
    """Test complete SDG prediction workflow"""
    # Test model loading, prediction, and output format
    model = load_sdg_model()
    prediction = predict_sdg_goals(model, "Kenya", "education", 1.0)
    
    assert isinstance(prediction, list)
    assert len(prediction) > 0
```

#### **UI Tests**
```python
# tests/test_streamlit_app.py
from streamlit.testing.v1 import AppTest

def test_main_page_loads():
    """Test main dashboard page loads without errors"""
    at = AppTest.from_file("pages/main_page.py")
    at.run()
    assert not at.exception
```

### **4. Code Review Process**

#### **Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Screenshots
(If UI changes)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data exposed
```

#### **Review Criteria**
- **Functionality**: Does it work as intended?
- **Performance**: Impact on load times and memory usage
- **Security**: No credentials or sensitive data exposed
- **Maintainability**: Clear, documented, testable code
- **User Experience**: Intuitive interface design

---

## 🔄 **Data Development Workflow**

### **1. Data Pipeline Changes**

When modifying data processing:

```bash
# 1. Backup existing data
cp -r src/outputs/data_output/ backup/data_output_$(date +%Y%m%d)/

# 2. Modify notebooks
jupyter notebook src/notebooks/data_cleaning.ipynb

# 3. Test pipeline
cd src/notebooks/
python -c "
import pandas as pd
df = pd.read_csv('../outputs/data_output/Financial_Cleaned.csv')
print(f'Data shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
"

# 4. Validate output
python scripts/validate_data_integrity.py
```

### **2. ML Model Development**

#### **Model Training Workflow**
```python
# 1. Data preparation
def prepare_training_data():
    """Prepare data for model training"""
    df = pd.read_csv('src/outputs/data_output/Financial_Cleaned.csv')
    # Feature engineering
    # Data splitting
    return X_train, X_test, y_train, y_test

# 2. Model training with versioning
def train_model_with_tracking():
    """Train model with performance tracking"""
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Performance evaluation
    performance = evaluate_model(model, X_test, y_test)
    
    # Model versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'src/outputs/model_output/SDG_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    
    return model, performance

# 3. Model validation
def validate_model_performance(model, threshold=0.8):
    """Validate model meets performance requirements"""
    test_score = model.score(X_test, y_test)
    assert test_score >= threshold, f"Model score {test_score} below threshold {threshold}"
```

### **3. Data Quality Assurance**

```python
# scripts/validate_data_integrity.py
import pandas as pd
import numpy as np

def validate_financial_data():
    """Comprehensive data validation"""
    df = pd.read_csv('src/outputs/data_output/Financial_Cleaned.csv')
    
    # Schema validation
    required_columns = [
        'Country', 'Region', 'Theme', 'Total required resources',
        'Total available resources', 'Total expenditure resources'
    ]
    assert all(col in df.columns for col in required_columns)
    
    # Data quality checks
    assert df['Total required resources'].min() >= 0
    assert not df['Country'].isnull().any()
    assert df['Region'].isin(['Africa', 'Asia Pacific', 'Arab States', 
                             'Europe and Central Asia', 
                             'Latin America and the Caribbean']).all()
    
    print("✅ Data validation passed")

if __name__ == "__main__":
    validate_financial_data()
```

---

## 🚀 **Performance Optimization**

### **1. Code Performance**

#### **Streamlit Optimization**
```python
# Use caching for expensive operations
@st.cache_data(ttl=3600)
def load_large_dataset():
    """Cache large dataset loading"""
    return pd.read_csv('large_file.csv')

# Optimize dataframe operations
def optimize_dataframe_operations(df):
    """Vectorized operations for better performance"""
    # Use vectorized operations instead of loops
    df['calculated_field'] = df['field1'] * df['field2']
    
    # Use categorical data types for memory efficiency
    df['Region'] = df['Region'].astype('category')
    
    return df

# Fragment expensive components
@st.fragment
def render_expensive_chart():
    """Fragment to isolate expensive rendering"""
    # Expensive chart rendering code
    pass
```

#### **Memory Management**
```python
def memory_efficient_processing():
    """Process large datasets with memory efficiency"""
    # Use chunked processing for large files
    chunk_size = 10000
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        process_chunk(chunk)
        
    # Explicit garbage collection
    import gc
    gc.collect()
```

### **2. Database Optimization**

```python
# Efficient data loading patterns
def load_data_efficiently():
    """Load data with optimal performance"""
    # Load only required columns
    columns = ['Country', 'Region', 'Total required resources']
    df = pd.read_csv('data.csv', usecols=columns)
    
    # Use appropriate data types
    dtypes = {
        'Country': 'category',
        'Region': 'category',
        'Total required resources': 'float32'
    }
    df = df.astype(dtypes)
    
    return df
```

---

## 🔒 **Security Guidelines**

### **1. Credential Management**

```python
# ✅ CORRECT: Use secrets management
import streamlit as st

api_key = st.secrets["AZURE_OPENAI_4O_API_KEY"]

# ❌ INCORRECT: Never hardcode credentials
api_key = "sk-1234567890abcdef"  # DON'T DO THIS

# Environment variable fallback
import os
api_key = os.getenv('AZURE_OPENAI_4O_API_KEY')
if not api_key:
    st.error("API key not configured")
    st.stop()
```

### **2. Input Validation**

```python
def validate_user_input(country: str, year: int) -> bool:
    """Validate user inputs for security"""
    # Validate country name
    valid_countries = get_country_list()
    if country not in valid_countries:
        raise ValueError(f"Invalid country: {country}")
    
    # Validate year range
    if not 2020 <= year <= 2026:
        raise ValueError(f"Invalid year: {year}")
    
    return True

def sanitize_string_input(user_input: str) -> str:
    """Sanitize string inputs"""
    # Remove potentially dangerous characters
    import re
    sanitized = re.sub(r'[<>"\']', '', user_input)
    return sanitized.strip()
```

### **3. Error Handling**

```python
def safe_api_call(prompt: str) -> str:
    """Make API calls with proper error handling"""
    try:
        client = get_azure_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        # Log error securely (don't log sensitive data)
        logger.error(f"API call failed: {type(e).__name__}")
        
        # Return user-friendly error
        return "I'm sorry, I'm experiencing technical difficulties. Please try again later."
```

---

## 📊 **Monitoring & Debugging**

### **1. Logging Configuration**

```python
# Set up structured logging
import logging
import sys

def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Use logging throughout application
logger = logging.getLogger(__name__)

def process_data():
    logger.info("Starting data processing")
    try:
        # Processing logic
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
```

### **2. Performance Monitoring**

```python
# Monitor function execution time
import time
import functools

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if execution_time > 1.0:  # Log slow operations
            logger.warning(f"{func.__name__} took {execution_time:.2f}s")
        
        return result
    return wrapper

@monitor_performance
def slow_operation():
    """Example of monitored function"""
    time.sleep(2)
```

### **3. Debugging Streamlit Applications**

```python
# Debug mode configuration
if st.secrets.get("DEBUG", False):
    import streamlit as st
    
    # Show debug information
    st.write("Debug Mode Enabled")
    st.write(f"Session State: {st.session_state}")
    
    # Memory usage
    import psutil
    memory_usage = psutil.virtual_memory().percent
    st.write(f"Memory Usage: {memory_usage}%")

# Debug specific components
def debug_data_loading():
    """Debug data loading issues"""
    try:
        df = pd.read_csv('data.csv')
        st.success(f"Data loaded: {df.shape}")
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.write("Debug info:", str(e))
```

---

## 🚀 **Deployment Workflow**

### **1. Pre-deployment Checklist**

```bash
# Run complete test suite
pytest tests/ -v --cov=src/

# Code quality checks
black --check .
flake8 .
mypy src/

# Security scan
bandit -r src/

# Performance test
python scripts/performance_test.py

# Validate all data files exist
python scripts/validate_deployment.py
```

### **2. Streamlit Cloud Deployment**

```yaml
# .streamlit/config.toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false

[client]
caching = true
```

### **3. Environment-Specific Configuration**

```python
# Detect environment
def get_environment():
    """Detect current environment"""
    if "streamlit.app" in os.environ.get("HOSTNAME", ""):
        return "production"
    elif os.environ.get("DEBUG") == "true":
        return "development"
    else:
        return "local"

# Environment-specific settings
ENV = get_environment()

if ENV == "production":
    CACHE_TTL = 3600
    DEBUG_MODE = False
elif ENV == "development":
    CACHE_TTL = 300
    DEBUG_MODE = True
else:
    CACHE_TTL = 60
    DEBUG_MODE = True
```

---

## 📈 **Release Management**

### **1. Version Control**

```bash
# Semantic versioning
git tag -a v1.2.3 -m "Release version 1.2.3"

# Version format:
# MAJOR.MINOR.PATCH
# 1.0.0 - Initial release
# 1.1.0 - New features
# 1.1.1 - Bug fixes
```

### **2. Release Notes**

```markdown
# Release Notes v1.2.3

## New Features
- Added advanced anomaly detection with new algorithms
- Enhanced UI responsiveness for mobile devices

## Improvements
- Optimized data loading performance by 40%
- Updated Azure OpenAI integration to latest API

## Bug Fixes
- Fixed memory leak in large dataset processing
- Resolved filtering issues in dashboard

## Breaking Changes
- None

## Migration Guide
- No migration required for this release
```

---

**🎯 Next Steps**: Ready to deploy? Check out the [Deployment Guide](./12_DEPLOYMENT.md) for production setup, or explore [Testing & QA](./13_TESTING_QA.md) for comprehensive testing strategies.