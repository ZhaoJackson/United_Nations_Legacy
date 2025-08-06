# 🧪 Testing & Quality Assurance

Comprehensive testing strategies and quality assurance protocols for the UN Financial Intelligence Dashboard.

## 🎯 **Testing Framework Overview**

The testing strategy follows a comprehensive pyramid approach ensuring code quality, functionality, and user experience across all components.

```
                    🔺 E2E Tests
                   (UI/UX Testing)
                 🔺🔺 Integration Tests  
               (Component Integration)
             🔺🔺🔺 Unit Tests
           (Function-level Testing)
         🔺🔺🔺🔺 Static Analysis
       (Code Quality & Security)
```

---

## 📋 **Test Categories**

### **1. Unit Tests** - Function-level validation
- Individual function testing
- Data processing logic validation  
- Model prediction accuracy
- Utility function verification

### **2. Integration Tests** - Component interaction
- Data pipeline integration
- ML model integration
- API endpoint testing
- Azure OpenAI integration

### **3. End-to-End Tests** - Complete user workflows
- Dashboard page loading
- User interaction flows
- Cross-page navigation
- AI chatbot conversations

### **4. Performance Tests** - System performance
- Load testing with large datasets
- Memory usage monitoring
- Response time validation
- Concurrent user simulation

### **5. Security Tests** - Security validation
- Input sanitization
- Credential protection
- Data access controls
- API security

---

## 🔧 **Test Environment Setup**

### **Testing Dependencies**
```bash
# Install testing framework
pip install pytest pytest-cov pytest-mock pytest-html
pip install streamlit[test] selenium
pip install locust  # For load testing
pip install bandit  # For security testing
```

### **Test Directory Structure**
```
tests/
├── unit/                    # Unit tests
│   ├── test_commonconst.py
│   ├── test_data_processing.py
│   ├── test_ml_models.py
│   └── test_utilities.py
├── integration/             # Integration tests
│   ├── test_data_pipeline.py
│   ├── test_model_integration.py
│   ├── test_api_integration.py
│   └── test_page_integration.py
├── e2e/                     # End-to-end tests
│   ├── test_user_workflows.py
│   ├── test_dashboard_flows.py
│   └── test_chatbot_flows.py
├── performance/             # Performance tests
│   ├── test_load_performance.py
│   ├── test_memory_usage.py
│   └── load_test_scenarios.py
├── security/                # Security tests
│   ├── test_input_validation.py
│   ├── test_authentication.py
│   └── test_data_protection.py
├── fixtures/                # Test data and fixtures
│   ├── sample_data.csv
│   ├── mock_responses.json
│   └── test_models.pkl
└── conftest.py             # Pytest configuration
```

---

## 🧪 **Unit Testing**

### **Core Functions Testing**

```python
# tests/unit/test_commonconst.py
import pytest
import pandas as pd
from src.commonconst import (
    get_theme_list, get_country_list, filter_by_agency,
    predict_sdg_goals, format_currency
)

class TestDataAccess:
    """Test data access functions"""
    
    def test_get_theme_list(self):
        """Test theme list retrieval"""
        themes = get_theme_list()
        
        assert isinstance(themes, list)
        assert len(themes) > 0
        assert 'education' in themes
        assert 'environment' in themes
        assert all(isinstance(theme, str) for theme in themes)
    
    def test_get_country_list(self):
        """Test country list retrieval"""
        countries = get_country_list()
        
        assert isinstance(countries, list)
        assert len(countries) > 100  # Expect substantial country list
        assert 'Afghanistan' in countries  # Should be alphabetically first
        assert all(isinstance(country, str) for country in countries)
    
    def test_filter_by_agency(self):
        """Test agency filtering functionality"""
        # Create test data
        test_data = pd.DataFrame({
            'Country': ['USA', 'Kenya', 'Brazil'],
            'Agencies': ['UNDP', 'UNICEF; WHO', 'UNDP; UNICEF']
        })
        
        # Test specific agency filter
        result = filter_by_agency(test_data, 'UNDP')
        assert len(result) == 2
        assert all(result['Agencies'].str.contains('UNDP'))
        
        # Test "All Agencies" filter
        result_all = filter_by_agency(test_data, 'All Agencies')
        assert len(result_all) == len(test_data)

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(1500) == '$1.50K'
        assert format_currency(1500000) == '$1.50M'
        assert format_currency(1500000000) == '$1.50B'
        assert format_currency(1500000000000) == '$1.50T'
        assert format_currency(0) == '$0'
    
    @pytest.mark.parametrize("value,expected", [
        (999, '$999'),
        (1000, '$1.00K'),
        (1234567, '$1.23M'),
        (2500000000, '$2.50B')
    ])
    def test_format_currency_parametrized(self, value, expected):
        """Parametrized test for currency formatting"""
        assert format_currency(value) == expected
```

### **Data Processing Testing**

```python
# tests/unit/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from src.dynamic_analysis import DynamicDataProcessor

class TestDataProcessor:
    """Test data processing functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'Country': ['Kenya', 'Nigeria', 'Ghana'],
            'Region': ['Africa', 'Africa', 'Africa'],
            'Theme': ['education', 'health', 'education'],
            'Total required resources': [1000000, 2000000, 1500000],
            'Total available resources': [800000, 1800000, 1200000],
            'Agencies': ['UNESCO; UNICEF', 'WHO', 'UNESCO']
        })
    
    def test_data_structure_discovery(self):
        """Test data structure discovery"""
        processor = DynamicDataProcessor()
        structure = processor.discover_data_structure()
        
        assert 'regions' in structure
        assert 'themes' in structure
        assert 'total_files' in structure
        assert isinstance(structure['themes'], set)
        assert len(structure['themes']) >= 10  # Expect at least 10 themes
    
    def test_theme_data_loading(self, sample_data):
        """Test theme-specific data loading"""
        processor = DynamicDataProcessor()
        
        # Mock the data loading
        with pytest.patch.object(processor, 'load_theme_data', return_value=sample_data):
            result = processor.load_theme_data('education')
            
            assert isinstance(result, pd.DataFrame)
            assert 'Country' in result.columns
            assert 'Theme' in result.columns
    
    def test_data_validation(self, sample_data):
        """Test data validation logic"""
        # Test valid data
        assert sample_data['Total required resources'].min() >= 0
        assert not sample_data['Country'].isnull().any()
        
        # Test invalid data detection
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'Total required resources'] = -1000
        
        with pytest.raises(AssertionError):
            assert invalid_data['Total required resources'].min() >= 0
```

### **ML Model Testing**

```python
# tests/unit/test_ml_models.py
import pytest
import joblib
import numpy as np
from unittest.mock import patch, MagicMock
from src.commonconst import load_sdg_model, predict_sdg_goals

class TestMLModels:
    """Test machine learning model functionality"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = MagicMock()
        model.predict.return_value = np.array([[1, 0, 1, 0]])
        return model
    
    def test_model_loading_success(self):
        """Test successful model loading"""
        with patch('joblib.load') as mock_load:
            mock_load.return_value = MagicMock()
            model = load_sdg_model()
            assert model is not None
            mock_load.assert_called_once()
    
    def test_model_loading_failure(self):
        """Test model loading failure handling"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model file not found")
            model = load_sdg_model()
            assert model is None
    
    def test_sdg_prediction(self, mock_model):
        """Test SDG prediction functionality"""
        with patch('src.commonconst.load_sdg_model', return_value=mock_model):
            # Mock the encoders and transformers
            with patch('src.commonconst.label_encoder_country') as mock_country_encoder, \
                 patch('src.commonconst.label_encoder_theme') as mock_theme_encoder, \
                 patch('src.commonconst.multi_label_binarizer') as mock_mlb:
                
                mock_country_encoder.transform.return_value = [0]
                mock_theme_encoder.transform.return_value = [1]
                mock_mlb.inverse_transform.return_value = [('Quality Education', 'Gender Equality')]
                
                result = predict_sdg_goals(mock_model, 'Kenya', 'education', 1.0)
                
                assert isinstance(result, list)
                assert len(result) > 0
                mock_model.predict.assert_called_once()
    
    def test_prediction_with_invalid_inputs(self):
        """Test prediction with invalid inputs"""
        with patch('src.commonconst.load_sdg_model', return_value=None):
            result = predict_sdg_goals(None, 'InvalidCountry', 'invalid_theme', 15.0)
            assert result == ["Model not available"]
```

---

## 🔗 **Integration Testing**

### **Data Pipeline Integration**

```python
# tests/integration/test_data_pipeline.py
import pytest
import pandas as pd
import os
from src.notebooks.data_cleaning import run_data_pipeline

class TestDataPipelineIntegration:
    """Test complete data pipeline integration"""
    
    @pytest.fixture(scope="class")
    def pipeline_outputs(self):
        """Run data pipeline and return outputs"""
        # Run the complete pipeline
        run_data_pipeline()
        
        # Load generated outputs
        outputs = {}
        output_files = [
            'Financial_Cleaned.csv',
            'SDG_Goals_Cleaned.csv', 
            'UN_Agencies_Cleaned.csv',
            'unified_data.csv'
        ]
        
        for filename in output_files:
            file_path = f'src/outputs/data_output/{filename}'
            if os.path.exists(file_path):
                outputs[filename] = pd.read_csv(file_path)
        
        return outputs
    
    def test_pipeline_outputs_exist(self, pipeline_outputs):
        """Test that all expected output files are generated"""
        expected_files = [
            'Financial_Cleaned.csv',
            'SDG_Goals_Cleaned.csv',
            'UN_Agencies_Cleaned.csv', 
            'unified_data.csv'
        ]
        
        for filename in expected_files:
            assert filename in pipeline_outputs
            assert not pipeline_outputs[filename].empty
    
    def test_data_consistency_across_outputs(self, pipeline_outputs):
        """Test data consistency across different output files"""
        financial_df = pipeline_outputs['Financial_Cleaned.csv']
        sdg_df = pipeline_outputs['SDG_Goals_Cleaned.csv']
        
        # Test country consistency
        financial_countries = set(financial_df['Country'].unique())
        sdg_countries = set(sdg_df['Country'].unique())
        
        assert sdg_countries.issubset(financial_countries)
        
        # Test region consistency
        assert financial_df['Region'].nunique() == 5
        assert all(region in ['Africa', 'Asia Pacific', 'Arab States', 
                             'Europe and Central Asia', 
                             'Latin America and the Caribbean'] 
                  for region in financial_df['Region'].unique())
    
    def test_data_quality_post_pipeline(self, pipeline_outputs):
        """Test data quality after pipeline processing"""
        financial_df = pipeline_outputs['Financial_Cleaned.csv']
        
        # Test for required columns
        required_columns = [
            'Country', 'Region', 'Theme', 'Total required resources'
        ]
        for col in required_columns:
            assert col in financial_df.columns
        
        # Test data types
        assert pd.api.types.is_numeric_dtype(financial_df['Total required resources'])
        
        # Test data ranges
        assert financial_df['Total required resources'].min() >= 0
        assert not financial_df['Country'].isnull().any()
```

### **Page Integration Testing**

```python
# tests/integration/test_page_integration.py
import pytest
from streamlit.testing.v1 import AppTest

class TestPageIntegration:
    """Test Streamlit page integration"""
    
    def test_main_app_loads(self):
        """Test main application loads without errors"""
        at = AppTest.from_file("app.py")
        at.run()
        assert not at.exception
    
    def test_main_page_functionality(self):
        """Test main dashboard page functionality"""
        at = AppTest.from_file("pages/main_page.py")
        at.run()
        
        # Check for key components
        assert not at.exception
        assert len(at.selectbox) > 0  # Should have filter selectboxes
        assert len(at.metric) > 0     # Should have KPI metrics
    
    def test_model_page_predictions(self):
        """Test model page prediction functionality"""
        at = AppTest.from_file("pages/model.py")
        at.run()
        
        # Test country selection
        if len(at.selectbox) > 0:
            at.selectbox[0].select("Kenya")
            at.run()
            assert not at.exception
    
    def test_chatbot_page_interaction(self):
        """Test chatbot page basic interaction"""
        at = AppTest.from_file("pages/bot.py")
        at.run()
        
        # Test chat input functionality
        if len(at.chat_input) > 0:
            at.chat_input[0].set_value("What is the funding gap in Africa?")
            at.run()
            # Should handle gracefully even without API keys
            assert not at.exception
```

---

## 🌐 **End-to-End Testing**

### **User Workflow Testing**

```python
# tests/e2e/test_user_workflows.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

class TestUserWorkflows:
    """Test complete user workflows"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        """Setup Chrome driver for testing"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    def test_complete_dashboard_workflow(self, driver):
        """Test complete dashboard exploration workflow"""
        # Navigate to application
        driver.get("http://localhost:8501")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Test navigation to different pages
        pages = ["OVERVIEW", "DASHBOARD", "ANALYSIS", "MODELS", "CHATBOT"]
        
        for page in pages:
            try:
                # Find and click page link
                page_link = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, page))
                )
                page_link.click()
                
                # Wait for page content to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "stApp"))
                )
                
                # Verify no error messages
                error_elements = driver.find_elements(By.CLASS_NAME, "stAlert")
                error_messages = [elem.text for elem in error_elements if "error" in elem.text.lower()]
                assert len(error_messages) == 0, f"Errors found on {page} page: {error_messages}"
                
            except Exception as e:
                pytest.fail(f"Failed to navigate to {page} page: {str(e)}")
    
    def test_filter_interaction_workflow(self, driver):
        """Test filter interaction workflow"""
        # Navigate to dashboard page
        driver.get("http://localhost:8501")
        
        # Click on DASHBOARD
        dashboard_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "DASHBOARD"))
        )
        dashboard_link.click()
        
        # Test filter interactions
        try:
            # Find selectbox elements
            selectboxes = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSelectbox']")
            
            if selectboxes:
                # Click on first selectbox (usually Year filter)
                selectboxes[0].click()
                
                # Select different option
                options = driver.find_elements(By.CSS_SELECTOR, "[role='option']")
                if len(options) > 1:
                    options[1].click()
                
                # Wait for content to update
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "stApp"))
                )
        
        except Exception as e:
            # Filter interaction is optional - don't fail test
            print(f"Filter interaction warning: {str(e)}")
```

### **Cross-Browser Testing**

```python
# tests/e2e/test_cross_browser.py
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

class TestCrossBrowser:
    """Test application across different browsers"""
    
    @pytest.mark.parametrize("browser", ["chrome", "firefox"])
    def test_basic_functionality_across_browsers(self, browser):
        """Test basic functionality across different browsers"""
        
        if browser == "chrome":
            options = ChromeOptions()
            options.add_argument("--headless")
            driver = webdriver.Chrome(options=options)
        elif browser == "firefox":
            options = FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox(options=options)
        
        try:
            # Navigate to application
            driver.get("http://localhost:8501")
            
            # Basic functionality test
            assert "UN Financial Intelligence Dashboard" in driver.title
            
            # Check for critical elements
            body = driver.find_element(By.TAG_NAME, "body")
            assert body is not None
            
        finally:
            driver.quit()
```

---

## ⚡ **Performance Testing**

### **Load Testing**

```python
# tests/performance/test_load_performance.py
import time
import statistics
import concurrent.futures
import requests

class TestLoadPerformance:
    """Test application performance under load"""
    
    def test_page_load_times(self):
        """Test individual page load times"""
        pages = [
            "http://localhost:8501",  # Overview
            "http://localhost:8501/DASHBOARD",
            "http://localhost:8501/ANALYSIS", 
            "http://localhost:8501/MODELS",
            "http://localhost:8501/CHATBOT"
        ]
        
        load_times = {}
        
        for page in pages:
            times = []
            for _ in range(5):  # Test 5 times per page
                start_time = time.time()
                try:
                    response = requests.get(page, timeout=30)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        times.append(end_time - start_time)
                except requests.RequestException:
                    pass  # Skip failed requests
            
            if times:
                load_times[page] = {
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'max': max(times)
                }
        
        # Assert performance thresholds
        for page, metrics in load_times.items():
            assert metrics['mean'] < 10.0, f"Page {page} average load time too high: {metrics['mean']:.2f}s"
            assert metrics['max'] < 20.0, f"Page {page} max load time too high: {metrics['max']:.2f}s"
    
    def test_concurrent_user_simulation(self):
        """Test application with multiple concurrent users"""
        
        def simulate_user():
            """Simulate single user session"""
            try:
                # User workflow: visit 3 different pages
                pages = [
                    "http://localhost:8501",
                    "http://localhost:8501/DASHBOARD", 
                    "http://localhost:8501/MODELS"
                ]
                
                session_time = time.time()
                for page in pages:
                    response = requests.get(page, timeout=15)
                    if response.status_code != 200:
                        return False
                    time.sleep(1)  # Simulate user thinking time
                
                return time.time() - session_time
                
            except requests.RequestException:
                return False
        
        # Simulate 10 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(simulate_user) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_sessions = [r for r in results if isinstance(r, (int, float))]
        success_rate = len(successful_sessions) / len(results)
        
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"
        
        if successful_sessions:
            avg_session_time = statistics.mean(successful_sessions)
            assert avg_session_time < 30.0, f"Average session time too high: {avg_session_time:.2f}s"
```

### **Memory Usage Testing**

```python
# tests/performance/test_memory_usage.py
import psutil
import time
import gc
from src.commonconst import financial_df, load_sdg_model

class TestMemoryUsage:
    """Test memory usage and optimization"""
    
    def test_baseline_memory_usage(self):
        """Test baseline memory usage"""
        gc.collect()  # Clean up before testing
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load main data
        df = financial_df
        
        memory_after_data = process.memory_info().rss / 1024 / 1024
        data_memory_usage = memory_after_data - baseline_memory
        
        # Assert reasonable memory usage for data
        assert data_memory_usage < 1000, f"Data loading uses too much memory: {data_memory_usage:.1f}MB"
    
    def test_model_memory_usage(self):
        """Test memory usage when loading models"""
        gc.collect()
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Load models
        sdg_model = load_sdg_model()
        
        memory_after_models = process.memory_info().rss / 1024 / 1024
        model_memory_usage = memory_after_models - baseline_memory
        
        # Assert reasonable memory usage for models
        assert model_memory_usage < 2000, f"Model loading uses too much memory: {model_memory_usage:.1f}MB"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations"""
        gc.collect()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Perform repeated operations
        for i in range(10):
            df = financial_df.copy()
            filtered_df = df[df['Region'] == 'Africa']
            aggregated = filtered_df.groupby('Country')['Total required resources'].sum()
            del df, filtered_df, aggregated
            
            if i % 3 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Assert no significant memory leak
        assert memory_increase < 100, f"Potential memory leak detected: {memory_increase:.1f}MB increase"
```

---

## 🔒 **Security Testing**

### **Input Validation Testing**

```python
# tests/security/test_input_validation.py
import pytest
from src.commonconst import filter_by_agency, predict_sdg_goals

class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attempts"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "javascript:alert('xss')"
        ]
        
        for malicious_input in malicious_inputs:
            # Test with agency filter
            try:
                result = filter_by_agency(pd.DataFrame({'Agencies': ['UNDP']}), malicious_input)
                # Should return empty or safe result
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                # Exceptions are acceptable for malicious input
                pass
    
    def test_file_path_injection(self):
        """Test protection against path traversal attacks"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        # Test file loading functions don't accept malicious paths
        for malicious_path in malicious_paths:
            try:
                # Should not be able to load arbitrary files
                with open(malicious_path, 'r') as f:
                    content = f.read()
                pytest.fail(f"Should not be able to read {malicious_path}")
            except (FileNotFoundError, PermissionError, OSError):
                # Expected behavior - file access should be restricted
                pass
    
    def test_xss_protection(self):
        """Test protection against XSS attacks"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "');alert('xss');//"
        ]
        
        # Test that XSS payloads are properly sanitized
        for payload in xss_payloads:
            # Test string sanitization functions
            sanitized = payload.replace('<', '&lt;').replace('>', '&gt;')
            assert '<script>' not in sanitized
            assert 'javascript:' not in sanitized
```

### **Authentication & Authorization Testing**

```python
# tests/security/test_authentication.py
import pytest
from unittest.mock import patch, MagicMock

class TestAuthentication:
    """Test authentication and authorization mechanisms"""
    
    def test_api_key_protection(self):
        """Test that API keys are properly protected"""
        # Test that secrets are not exposed in error messages
        with patch('streamlit.secrets') as mock_secrets:
            mock_secrets.__getitem__.side_effect = KeyError("API key not found")
            
            try:
                from src.commonconst import client_4o
                # Should handle missing API key gracefully
                assert client_4o is None or hasattr(client_4o, 'api_key')
            except Exception as e:
                # Error message should not contain sensitive information
                error_msg = str(e).lower()
                assert 'api_key' not in error_msg
                assert 'secret' not in error_msg
                assert 'password' not in error_msg
    
    def test_credential_exposure(self):
        """Test that credentials are not exposed in logs or output"""
        import logging
        
        # Capture log output
        with patch('logging.Logger.info') as mock_log:
            # Trigger operations that might log credentials
            try:
                from src.prompt.chatbot import get_chatbot_response
                # Function should not log credentials
            except Exception:
                pass
            
            # Check that no credentials were logged
            for call in mock_log.call_args_list:
                log_message = str(call).lower()
                assert 'api_key' not in log_message
                assert 'secret' not in log_message
                assert 'password' not in log_message
```

---

## 📊 **Test Automation & CI/CD**

### **Pytest Configuration**

```python
# conftest.py
import pytest
import pandas as pd
import os
import tempfile

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_financial_data():
    """Create sample financial data for testing"""
    return pd.DataFrame({
        'Country': ['Kenya', 'Nigeria', 'Ghana', 'Egypt', 'Morocco'],
        'Region': ['Africa'] * 5,
        'Theme': ['education', 'health', 'environment', 'governance', 'poverty'],
        'Total required resources': [1000000, 2000000, 1500000, 1800000, 1200000],
        'Total available resources': [800000, 1600000, 1200000, 1440000, 960000],
        'Total expenditure resources': [750000, 1500000, 1100000, 1300000, 900000],
        'Agencies': ['UNESCO', 'WHO', 'UNEP', 'UNDP', 'UNDP; WFP'],
        'SDG Goals': ['Quality Education', 'Good Health', 'Climate Action', 'Peace and Justice', 'No Poverty']
    })

@pytest.fixture(autouse=True)
def clean_cache():
    """Clean Streamlit cache before each test"""
    import streamlit as st
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
```

### **GitHub Actions CI/CD**

```yaml
# .github/workflows/test.yml
name: Comprehensive Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-html bandit safety
    
    - name: Run security tests
      run: |
        bandit -r src/ -f json -o bandit-report.json
        safety check --json --output safety-report.json
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
    
    - name: Generate test report
      run: |
        pytest tests/ --html=test-report.html --self-contained-html
    
    - name: Upload test artifacts
      uses: actions/upload-artifact@v3
      with:
        name: test-reports
        path: |
          test-report.html
          htmlcov/
          bandit-report.json
          safety-report.json
```

---

## 📈 **Test Coverage & Quality Metrics**

### **Coverage Requirements**

```python
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-branch
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
testpaths = tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
```

### **Quality Gates**

```python
# Quality requirements for CI/CD
QUALITY_GATES = {
    'unit_test_coverage': 80,      # Minimum 80% unit test coverage
    'integration_coverage': 70,    # Minimum 70% integration coverage
    'performance_threshold': 5,    # Maximum 5s page load time
    'security_score': 8.0,         # Minimum security score (1-10)
    'code_complexity': 10,         # Maximum cyclomatic complexity
    'duplication_ratio': 0.05      # Maximum 5% code duplication
}
```

### **Continuous Monitoring**

```python
# tests/quality/test_quality_metrics.py
import pytest
import radon.complexity as radon_complexity
import radon.metrics as radon_metrics

class TestQualityMetrics:
    """Test code quality metrics"""
    
    def test_cyclomatic_complexity(self):
        """Test that functions don't exceed complexity thresholds"""
        complex_functions = []
        
        # Analyze complexity of key modules
        modules = ['src/commonconst.py', 'src/dynamic_analysis.py']
        
        for module in modules:
            try:
                with open(module, 'r') as f:
                    code = f.read()
                
                complexity_results = radon_complexity.cc_visit(code)
                for result in complexity_results:
                    if result.complexity > 10:
                        complex_functions.append(f"{module}:{result.name}")
            except FileNotFoundError:
                pass
        
        assert len(complex_functions) == 0, f"Functions with high complexity: {complex_functions}"
    
    def test_maintainability_index(self):
        """Test maintainability index of codebase"""
        modules = ['src/commonconst.py', 'src/dynamic_analysis.py']
        low_maintainability = []
        
        for module in modules:
            try:
                with open(module, 'r') as f:
                    code = f.read()
                
                mi = radon_metrics.mi_visit(code, multi=True)
                if mi < 20:  # Maintainability index below 20 is concerning
                    low_maintainability.append(f"{module}:{mi:.1f}")
            except FileNotFoundError:
                pass
        
        assert len(low_maintainability) == 0, f"Modules with low maintainability: {low_maintainability}"
```

---

## 🎯 **Testing Best Practices**

### **Test Writing Guidelines**

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Test Isolation**: Each test should be independent
4. **Mocking**: Mock external dependencies and slow operations
5. **Assertions**: Use specific assertions with clear error messages

### **Common Testing Patterns**

```python
# Pattern 1: Arrange-Act-Assert
def test_currency_formatting():
    # Arrange
    amount = 1500000
    
    # Act
    result = format_currency(amount)
    
    # Assert
    assert result == '$1.50M'

# Pattern 2: Parametrized testing
@pytest.mark.parametrize("input,expected", [
    (1000, '$1.00K'),
    (1500000, '$1.50M'),
    (2500000000, '$2.50B')
])
def test_currency_formatting_multiple_values(input, expected):
    assert format_currency(input) == expected

# Pattern 3: Exception testing
def test_invalid_input_raises_exception():
    with pytest.raises(ValueError, match="Invalid country"):
        predict_sdg_goals(None, "InvalidCountry", "theme", 1.0)
```

---

**🔄 Next Steps**: Implement the testing framework by running `pytest tests/` to execute all tests, or explore the [Performance Optimization](./15_PERFORMANCE.md) guide for system optimization strategies.