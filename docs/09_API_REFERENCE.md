# 📚 API Reference

Comprehensive API documentation for the UN Financial Intelligence Dashboard modules and functions.

## 🎯 **Module Overview**

The dashboard consists of several interconnected modules:

- **`src.commonconst`**: Core constants, utilities, and data access functions
- **`src.dynamic_analysis`**: Advanced data processing and analysis
- **`src.prompt.*`**: AI prompt engineering modules (6 modules)
- **`pages.*`**: Streamlit dashboard pages (5 pages)

---

## 🔧 **Core Module: `src.commonconst`**

Central module providing shared constants, utilities, and data access functions.

### **Configuration Constants**

```python
# File Paths
FINANCIAL_PATH = "src/outputs/data_output/Financial_Cleaned.csv"
UN_AGENCIES_PATH = "src/outputs/data_output/UN_Agencies_Cleaned.csv"
SDG_GOALS_PATH = "src/outputs/data_output/SDG_Goals_Cleaned.csv"
SDG_MODEL_PATH = "src/outputs/model_output/SDG_model.pkl"
AGENCY_MODEL_PATH = "src/outputs/model_output/Agency_model.pkl"

# Display Configuration
PLOT_YEAR_RANGE = list(range(2020, 2026))
DEFAULT_YEAR_INDEX = len(PLOT_YEAR_RANGE) - 1
UN_BLUE_GRADIENT = ["#f0f9ff", "#e0f2fe", ..., "#0c4a6e"]

# Page Configuration
PAGE_CONFIG = {
    "page_title": "UN Financial Intelligence Dashboard",
    "page_icon": "🇺🇳",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
```

### **Data Access Functions**

#### **`get_theme_list() -> List[str]`**
Get available themes from both data files and filesystem.

```python
def get_theme_list() -> List[str]:
    """
    Get comprehensive theme list from data and filesystem.
    
    Returns:
        List[str]: Sorted list of available themes
        
    Example:
        >>> themes = get_theme_list()
        >>> print(themes[:3])
        ['crime', 'digital', 'education']
    """
```

#### **`get_region_list() -> List[str]`**
Get available regions from data sources.

```python
def get_region_list() -> List[str]:
    """
    Get comprehensive region list from data and filesystem.
    
    Returns:
        List[str]: Sorted list of available regions
        
    Example:
        >>> regions = get_region_list()
        >>> print(regions)
        ['Africa', 'Arab States', 'Asia Pacific', 
         'Europe and Central Asia', 
         'Latin America and the Caribbean']
    """
```

#### **`get_country_list() -> List[str]`**
Get available countries from financial data.

```python
def get_country_list() -> List[str]:
    """
    Get sorted list of countries from financial dataset.
    
    Returns:
        List[str]: Alphabetically sorted country names
        
    Example:
        >>> countries = get_country_list()
        >>> print(countries[:5])
        ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina']
    """
```

#### **`get_agencies_list() -> List[str]`**
Extract unique UN agencies from financial data.

```python
def get_agencies_list() -> List[str]:
    """
    Extract unique UN agencies from the financial data.
    
    Returns:
        List[str]: Sorted list of UN agency names
        
    Example:
        >>> agencies = get_agencies_list()
        >>> print(agencies[:3])
        ['FAO', 'ILO', 'IOM']
    """
```

#### **`get_sdg_goals_list() -> List[str]`**
Extract unique SDG goals from financial data.

```python
def get_sdg_goals_list() -> List[str]:
    """
    Extract unique SDG goals from the financial data.
    
    Returns:
        List[str]: Sorted list of SDG goal descriptions
        
    Example:
        >>> sdgs = get_sdg_goals_list()
        >>> print(sdgs[:2])
        ['No Poverty', 'Zero Hunger']
    """
```

### **Data Filtering Functions**

#### **`filter_by_agency(df: pd.DataFrame, selected_agency: str) -> pd.DataFrame`**
Filter dataframe by selected UN agency.

```python
def filter_by_agency(df: pd.DataFrame, selected_agency: str) -> pd.DataFrame:
    """
    Filter dataframe by selected UN agency.
    
    Args:
        df: Input DataFrame with 'Agencies' column
        selected_agency: Agency name or "All Agencies"
        
    Returns:
        pd.DataFrame: Filtered dataframe
        
    Example:
        >>> filtered_df = filter_by_agency(financial_df, "UNDP")
        >>> print(len(filtered_df))
        1250
    """
```

#### **`filter_by_sdg(df: pd.DataFrame, selected_sdg: str) -> pd.DataFrame`**
Filter dataframe by selected SDG goal.

```python
def filter_by_sdg(df: pd.DataFrame, selected_sdg: str) -> pd.DataFrame:
    """
    Filter dataframe by selected SDG goal.
    
    Args:
        df: Input DataFrame with 'SDG Goals' column
        selected_sdg: SDG goal name or "All SDG Goals"
        
    Returns:
        pd.DataFrame: Filtered dataframe
        
    Example:
        >>> filtered_df = filter_by_sdg(financial_df, "Quality Education")
        >>> print(len(filtered_df))
        890
    """
```

### **Model Loading Functions**

#### **`@st.cache_resource load_sdg_model()`**
Load the saved SDG prediction model.

```python
@st.cache_resource
def load_sdg_model():
    """
    Load the saved SDG prediction model.
    
    Returns:
        sklearn.Pipeline or None: Trained SDG classification model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        
    Example:
        >>> model = load_sdg_model()
        >>> if model:
        ...     predictions = model.predict(input_data)
    """
```

#### **`@st.cache_resource load_agency_model()`**
Load the saved Agency prediction model.

```python
@st.cache_resource  
def load_agency_model():
    """
    Load the saved Agency prediction model.
    
    Returns:
        sklearn.Pipeline or None: Trained agency recommendation model
        
    Example:
        >>> model = load_agency_model()
        >>> if model:
        ...     agencies = model.predict(country_data)
    """
```

### **Prediction Functions**

#### **`predict_sdg_goals(model, country: str, theme: str, strategic_priority_code: float) -> List[str]`**
Make SDG predictions using trained model.

```python
def predict_sdg_goals(
    model, 
    country: str, 
    theme: str, 
    strategic_priority_code: float
) -> List[str]:
    """
    Make prediction using SDG model.
    
    Args:
        model: Trained SDG prediction model
        country: Country name
        theme: Theme category
        strategic_priority_code: Strategic priority code (1-10)
        
    Returns:
        List[str]: List of predicted SDG goal names
        
    Example:
        >>> predictions = predict_sdg_goals(
        ...     model, "Kenya", "education", 1.0
        ... )
        >>> print(predictions)
        ['Quality Education', 'Gender Equality']
    """
```

#### **`predict_agencies(model, country: str, theme: str, strategic_priority_code: float) -> List[str]`**
Make agency predictions using trained model.

```python
def predict_agencies(
    model, 
    country: str, 
    theme: str, 
    strategic_priority_code: float
) -> List[str]:
    """
    Make prediction using Agency model.
    
    Args:
        model: Trained agency prediction model
        country: Country name
        theme: Theme category
        strategic_priority_code: Strategic priority code (1-10)
        
    Returns:
        List[str]: List of predicted agency names
        
    Example:
        >>> agencies = predict_agencies(
        ...     model, "Kenya", "education", 1.0
        ... )
        >>> print(agencies)
        ['UNESCO', 'UNICEF', 'UNDP']
    """
```

### **Utility Functions**

#### **`format_currency(value: float) -> str`**
Format large currency values for display.

```python
def format_currency(value: float) -> str:
    """
    Format large currency values for display.
    
    Args:
        value: Numeric value to format
        
    Returns:
        str: Formatted currency string with appropriate units
        
    Example:
        >>> format_currency(1500000)
        '$1.50M'
        >>> format_currency(2500000000)
        '$2.50B'
    """
```

#### **`safe_load_csv(file_path: str, default_df: pd.DataFrame = None) -> pd.DataFrame`**
Safely load CSV files with error handling.

```python
def safe_load_csv(
    file_path: str, 
    default_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Safely load CSV files with error handling.
    
    Args:
        file_path: Path to CSV file
        default_df: Default DataFrame if loading fails
        
    Returns:
        pd.DataFrame: Loaded data or default DataFrame
        
    Example:
        >>> df = safe_load_csv("data.csv")
        >>> print(df.shape)
        (1000, 15)
    """
```

---

## 🔄 **Dynamic Analysis Module: `src.dynamic_analysis`**

Advanced data processing and analysis capabilities.

### **`class DynamicDataProcessor`**
Handles dynamic data processing for new themes and regions.

```python
class DynamicDataProcessor:
    """
    Handles dynamic data processing for new themes and regions.
    
    Attributes:
        data_dir (Path): Data directory path
        output_dir (Path): Output directory path
        data_output_dir (Path): Data output subdirectory
        model_output_dir (Path): Model output subdirectory
    """
    
    def __init__(self, data_dir: str = "src/data", output_dir: str = "src/outputs"):
        """
        Initialize data processor.
        
        Args:
            data_dir: Path to raw data directory
            output_dir: Path to outputs directory
        """
```

#### **`discover_data_structure() -> Dict[str, Any]`**
Discover all available themes and regions from the file system.

```python
def discover_data_structure(self) -> Dict[str, Any]:
    """
    Discover all available themes and regions from the file system.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - regions: Dict mapping region names to theme lists
            - themes: Set of all available themes
            - total_files: Total number of data files found
            - missing_files: List of expected but missing files
            
    Example:
        >>> processor = DynamicDataProcessor()
        >>> structure = processor.discover_data_structure()
        >>> print(structure['themes'])
        {'crime', 'digital', 'education', 'environment', ...}
    """
```

#### **`load_theme_data(theme: str, regions: Optional[List[str]] = None) -> pd.DataFrame`**
Load data for a specific theme across regions.

```python
def load_theme_data(
    self, 
    theme: str, 
    regions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load data for a specific theme across all or specified regions.
    
    Args:
        theme: Theme name to load
        regions: Optional list of regions to include
        
    Returns:
        pd.DataFrame: Combined data for the theme
        
    Example:
        >>> processor = DynamicDataProcessor()
        >>> education_data = processor.load_theme_data("education")
        >>> print(education_data.shape)
        (5000, 25)
    """
```

#### **`create_unified_dataset(save_to_file: bool = True) -> pd.DataFrame`**
Create a unified dataset from all themes and regions.

```python
def create_unified_dataset(self, save_to_file: bool = True) -> pd.DataFrame:
    """
    Create a unified dataset from all themes and regions.
    
    Args:
        save_to_file: Whether to save the result to CSV
        
    Returns:
        pd.DataFrame: Unified dataset
        
    Example:
        >>> processor = DynamicDataProcessor()
        >>> unified = processor.create_unified_dataset()
        >>> print(f"Unified dataset: {unified.shape}")
        Unified dataset: (50000, 30)
    """
```

### **`class DynamicModelManager`**
Manages dynamic model training and updating for new themes.

```python
class DynamicModelManager:
    """
    Manages dynamic model training and updating for new themes.
    
    Attributes:
        output_dir (Path): Model output directory
    """
    
    def __init__(self, output_dir: str = "src/outputs/model_output"):
        """
        Initialize model manager.
        
        Args:
            output_dir: Path to model output directory
        """
```

#### **`needs_retraining(theme_list: List[str]) -> bool`**
Check if models need retraining based on available themes.

```python
def needs_retraining(self, theme_list: List[str]) -> bool:
    """
    Check if models need retraining based on available themes.
    
    Args:
        theme_list: List of available themes
        
    Returns:
        bool: True if retraining is needed
        
    Example:
        >>> manager = DynamicModelManager()
        >>> themes = get_theme_list()
        >>> needs_update = manager.needs_retraining(themes)
        >>> print(f"Retraining needed: {needs_update}")
    """
```

---

## 🤖 **AI Prompt Modules: `src.prompt.*`**

Specialized modules for AI integration with different analysis types.

### **`src.prompt.chatbot`**

#### **`get_chatbot_response(user_input: str, data: pd.DataFrame, region: str, theme: str) -> str`**
Generate conversational AI response for user queries.

```python
def get_chatbot_response(
    user_input: str, 
    data: pd.DataFrame, 
    region: str, 
    theme: str
) -> str:
    """
    Generate conversational AI response for user queries.
    
    Args:
        user_input: User's natural language query
        data: Filtered financial dataset
        region: Selected region filter
        theme: Selected theme filter
        
    Returns:
        str: AI-generated response with data insights
        
    Example:
        >>> response = get_chatbot_response(
        ...     "What are the funding gaps in education?",
        ...     filtered_df, "Africa", "education"
        ... )
        >>> print(response[:100])
        "Based on the education data for Africa, there are significant..."
    """
```

#### **`create_financial_data_summary(df: pd.DataFrame, region: str, theme: str) -> str`**
Create concise summary of filtered financial data.

```python
def create_financial_data_summary(
    df: pd.DataFrame, 
    region: str, 
    theme: str
) -> str:
    """
    Create a concise summary of the filtered financial data.
    
    Args:
        df: Financial DataFrame
        region: Region filter applied
        theme: Theme filter applied
        
    Returns:
        str: Formatted data summary with key statistics
        
    Example:
        >>> summary = create_financial_data_summary(df, "Africa", "education")
        >>> print(summary)
        **CURRENT DATA SUMMARY** (Filters: Africa | education)
        - Total Projects: 1,250
        - Required Resources: $2,500,000,000
        ...
    """
```

### **`src.prompt.dashboard`**

#### **`get_dashboard_insights(analysis_type: str, filters: Dict, data_context: Dict) -> str`**
Generate strategic dashboard insights using O1 model.

```python
def get_dashboard_insights(
    analysis_type: str, 
    filters: Dict, 
    data_context: Dict
) -> str:
    """
    Generate strategic dashboard insights using O1 model.
    
    Args:
        analysis_type: Type of analysis ("financial", "regional", etc.)
        filters: Dictionary of current filter selections
        data_context: Dictionary with data summary and metrics
        
    Returns:
        str: O1-generated strategic insights and recommendations
        
    Example:
        >>> insights = get_dashboard_insights(
        ...     "financial",
        ...     {"region": "Africa", "theme": "education"},
        ...     {"total_required": 2500000000}
        ... )
    """
```

### **`src.prompt.models`**

#### **`get_model_analysis(country: str, theme: str, predictions: Dict, context: Dict) -> str`**
Generate AI analysis of ML model predictions.

```python
def get_model_analysis(
    country: str, 
    theme: str, 
    predictions: Dict, 
    context: Dict
) -> str:
    """
    Generate AI analysis of ML model predictions.
    
    Args:
        country: Selected country
        theme: Selected theme
        predictions: Dictionary with SDG and agency predictions
        context: Historical context and performance data
        
    Returns:
        str: GPT-4o generated strategic analysis
        
    Example:
        >>> analysis = get_model_analysis(
        ...     "Kenya", "education",
        ...     {"sdgs": ["Quality Education"], "agencies": ["UNESCO"]},
        ...     {"historical_funding": 50000000}
        ... )
    """
```

---

## 📱 **Page Modules: `pages.*`**

Streamlit dashboard page implementations.

### **`pages.main_page`**

#### **`format_number(n: float) -> str`**
Format numbers with appropriate units for better readability.

```python
def format_number(n: float) -> str:
    """
    Format numbers with appropriate units for better readability.
    
    Args:
        n: Number to format
        
    Returns:
        str: Formatted number with units (K, M, B, T)
        
    Example:
        >>> format_number(1500000)
        '$1.5M'
        >>> format_number(2500000000)
        '$2.5B'
    """
```

#### **`get_iso_alpha(country_name: str) -> str`**
Get ISO alpha-3 code for country name.

```python
def get_iso_alpha(country_name: str) -> str:
    """
    Get ISO alpha-3 code for country name.
    
    Args:
        country_name: Full country name
        
    Returns:
        str: ISO alpha-3 code or None if not found
        
    Example:
        >>> get_iso_alpha("United States")
        'USA'
        >>> get_iso_alpha("Kenya")
        'KEN'
    """
```

### **`pages.prediction`**

The prediction page contains advanced analytics with three main tabs:

1. **Funding Predictions**: RandomForest forecasting analysis
2. **Anomaly Detection**: Outlier identification and investigation
3. **Agency Performance**: Clustering and efficiency analysis

### **`pages.model`**

#### **`get_model_countries() -> List[str]`**
Get countries available for model prediction.

```python
def get_model_countries() -> List[str]:
    """
    Get countries available for model prediction.
    
    Returns:
        List[str]: Sorted list of countries with model compatibility
        
    Example:
        >>> countries = get_model_countries()
        >>> print(len(countries))
        150
    """
```

#### **`get_model_themes() -> List[str]`**
Get themes available for model prediction.

```python
def get_model_themes() -> List[str]:
    """
    Get themes available for model prediction.
    
    Returns:
        List[str]: Sorted list of themes with model compatibility
        
    Example:
        >>> themes = get_model_themes()
        >>> print(themes)
        ['crime', 'digital', 'education', ...]
    """
```

### **`pages.bot`**

The bot page implements a conversational AI interface with:

- Persistent chat history
- Real-time statistics sidebar
- Dynamic data filtering
- Quick question templates

---

## 🔧 **Azure OpenAI Integration**

### **Client Configuration**

```python
# Global clients (from commonconst.py)
client_4o: AzureOpenAI  # GPT-4o client for conversational AI
client_o1: AzureOpenAI  # O1 client for strategic reasoning

DEPLOYMENT_4O: str  # GPT-4o deployment name
DEPLOYMENT_O1: str  # O1 deployment name

def get_azure_openai_client(model_type: str = "4o") -> AzureOpenAI:
    """
    Get Azure OpenAI client with fallback handling.
    
    Args:
        model_type: "4o" for GPT-4o or "o1" for O1 model
        
    Returns:
        AzureOpenAI: Configured client or None if unavailable
        
    Example:
        >>> client = get_azure_openai_client("4o")
        >>> if client:
        ...     response = client.chat.completions.create(...)
    """
```

---

## 📊 **Data Structures**

### **Financial Data Schema**

```python
# Main financial dataset columns
FINANCIAL_COLUMNS = {
    'Country': 'str',           # Country name
    'Region': 'str',            # UN region
    'Theme': 'str',             # Thematic area
    'Plan name': 'str',         # Development plan name
    'Strategic priority code': 'float',  # Priority code (1-10)
    'Strategic priority': 'str', # Priority description
    'Agencies': 'str',          # Semicolon-separated UN agencies
    'SDG Goals': 'str',         # Semicolon-separated SDG goals
    'Total required resources': 'float',   # Total funding needed
    'Total available resources': 'float',  # Total funding available
    'Total expenditure resources': 'float', # Total spent
    # Year-specific columns (2020-2026)
    '2025 Required': 'float',   # 2025 requirements
    '2025 Available': 'float',  # 2025 available funding
    '2025 Expenditure': 'float', # 2025 expenditure
    # ... similar for other years
}
```

### **Model Prediction Schema**

```python
# SDG Prediction Input
SDG_INPUT_SCHEMA = {
    'Country': str,              # Required
    'Theme': str,                # Required  
    'Strategic priority code': float  # Required (1-10)
}

# SDG Prediction Output
SDG_OUTPUT_SCHEMA = List[str]  # List of predicted SDG goal names

# Agency Prediction Input/Output
AGENCY_INPUT_SCHEMA = SDG_INPUT_SCHEMA  # Same input schema
AGENCY_OUTPUT_SCHEMA = List[str]        # List of predicted agency names
```

---

## ⚠️ **Error Handling**

### **Common Exceptions**

```python
# Data loading errors
FileNotFoundError: "Required data file not found"
pd.errors.EmptyDataError: "No data in file"
pd.errors.ParserError: "Error parsing CSV data"

# Model errors  
AttributeError: "Model not properly loaded"
ValueError: "Invalid input parameters for prediction"

# API errors
openai.AuthenticationError: "Invalid Azure OpenAI credentials"
openai.RateLimitError: "API rate limit exceeded"
openai.APIError: "Azure OpenAI API error"

# Streamlit errors
st.errors.StreamlitAPIException: "Streamlit configuration error"
```

### **Error Handling Patterns**

```python
# Safe data loading
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"Data file not found: {file_path}")
    st.stop()
except pd.errors.EmptyDataError:
    st.warning("Data file is empty")
    df = pd.DataFrame()

# Safe model loading
try:
    model = joblib.load(model_path)
except Exception as e:
    st.warning("Model not available. Some features disabled.")
    model = None

# Safe API calls
try:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
except openai.AuthenticationError:
    return "API authentication failed. Please check credentials."
except openai.RateLimitError:
    return "Service temporarily busy. Please try again."
except Exception as e:
    logger.error(f"API error: {e}")
    return "Service temporarily unavailable."
```

---

## 🔧 **Configuration Options**

### **Streamlit Configuration**

```python
# Page configuration (from PAGE_CONFIG)
st.set_page_config(
    page_title="UN Financial Intelligence Dashboard",
    page_icon="🇺🇳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching configuration
@st.cache_data(ttl=3600, max_entries=50)  # 1 hour cache, 50 entries max
@st.cache_resource  # For models and expensive resources
```

### **Performance Tuning**

```python
# Memory optimization
import gc
gc.collect()  # Force garbage collection

# DataFrame optimization
df = df.astype({
    'Region': 'category',
    'Theme': 'category',
    'Country': 'category'
})

# Chunked processing for large datasets
chunk_size = 10000
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    process_chunk(chunk)
```

---

## 📈 **Performance Metrics**

### **Timing Benchmarks**

```python
# Expected performance metrics
PERFORMANCE_BENCHMARKS = {
    'page_load_time': '<5 seconds',
    'data_loading': '<2 seconds',
    'chart_rendering': '<1 second',
    'ml_prediction': '<3 seconds',
    'ai_response': '<10 seconds'
}

# Memory usage
MEMORY_BENCHMARKS = {
    'base_application': '<500MB',
    'with_full_dataset': '<2GB',
    'peak_usage': '<4GB'
}
```

---

## 🧪 **Testing Utilities**

### **Test Helper Functions**

```python
# Test data generation
def create_test_dataframe(rows: int = 100) -> pd.DataFrame:
    """Create test DataFrame for unit testing"""
    return pd.DataFrame({
        'Country': ['TestCountry'] * rows,
        'Region': ['TestRegion'] * rows,
        'Theme': ['education'] * rows,
        'Total required resources': np.random.randint(1000, 100000, rows)
    })

# Mock API responses
def mock_openai_response(prompt: str) -> str:
    """Mock OpenAI API response for testing"""
    return f"Mock response for: {prompt[:50]}..."

# Performance testing
def benchmark_function(func, *args, **kwargs) -> float:
    """Benchmark function execution time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return execution_time
```

---

## 🔄 **Version Compatibility**

### **Python Version Requirements**
- **Minimum**: Python 3.11.0
- **Recommended**: Python 3.11.9
- **Maximum**: Python 3.12.x

### **Key Dependencies**
- **Streamlit**: 1.47.0+ (required for navigation features)
- **Pandas**: 2.0.0+ (for performance improvements)
- **OpenAI**: 1.35.0+ (for Azure integration)
- **Scikit-learn**: 1.3.0+ (for model compatibility)

---

**🎯 Next Steps**: Explore [Development Workflow](./06_DEVELOPMENT_WORKFLOW.md) for contribution guidelines, or check [Troubleshooting Guide](./14_TROUBLESHOOTING.md) for common issues and solutions.