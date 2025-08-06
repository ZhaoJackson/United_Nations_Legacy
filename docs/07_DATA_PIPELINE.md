# 📊 Data Pipeline Documentation

Comprehensive guide to the UN Financial Intelligence Dashboard data processing pipeline.

## 🎯 **Pipeline Overview**

The data pipeline transforms **55+ Excel files** (48MB+ total) from UN regional/thematic data into clean, analysis-ready datasets through a sophisticated ETL process.

### **Pipeline Flow Summary**
```
Raw UN Data (55 Excel files)
├── Regional Structure (5 regions × 11 themes)
├── ETL Processing (data_cleaning.ipynb)
├── Data Validation & Cleaning
├── Feature Engineering & Standardization
├── Output Generation (4 clean CSV files)
└── ML Model Training (5 specialized models)
```

---

## 📁 **Data Sources**

### **Raw Data Structure**
```
src/data/
├── Africa/ (11 theme files)
│   ├── crime.xlsx
│   ├── digital.xlsx
│   ├── education.xlsx
│   └── ... (8 more themes)
├── Arab States/ (11 theme files)
├── Asia Pacific/ (11 theme files)
├── Europe and Central Asia/ (11 theme files)
└── Latin America and the Caribbean/ (11 theme files)

Total: 5 regions × 11 themes = 55 Excel files
```

### **Theme Categories**
- **crime**: Crime prevention and criminal justice
- **digital**: Digital transformation and technology
- **education**: Education and skills development
- **environment**: Climate change and environmental sustainability
- **food**: Food security and nutrition
- **gender**: Gender equality and women's empowerment
- **governance**: Governance and institutional development
- **poverty**: Poverty reduction and social protection
- **water**: Water, sanitation and hygiene (WASH)
- **work**: Decent work and economic growth
- **youth**: Youth development and empowerment

### **Regional Coverage**
- **Africa**: 54 countries
- **Arab States**: 22 countries
- **Asia Pacific**: 36 countries
- **Europe and Central Asia**: 17 countries
- **Latin America and the Caribbean**: 33 countries

---

## 🔄 **ETL Process: `data_cleaning.ipynb`**

### **Stage 1: Data Ingestion**

```python
# Load and combine Excel files across regions/themes
def load_regional_theme_data():
    """Load all Excel files from regional theme structure"""
    data_frames = []
    
    for region in REGIONS:
        for theme in THEMES:
            file_path = f"src/data/{region}/{theme}.xlsx"
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df['Region'] = region
                df['Theme'] = theme
                data_frames.append(df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df
```

**Key Operations:**
- Load 55+ Excel files systematically
- Add region and theme metadata
- Combine into unified DataFrame
- Handle missing files gracefully

### **Stage 2: Schema Standardization**

```python
# Standardize column structures across files
STANDARD_COLUMNS = [
    'Country', 'Region', 'Theme', 'Plan name',
    'Strategic priority code', 'Strategic priority',
    'Agencies', 'SDG Goals',
    'Total required resources', 'Total available resources', 
    'Total expenditure resources',
    # Year-specific columns (2020-2026)
    '2020 Required', '2020 Available', '2020 Expenditure',
    '2021 Required', '2021 Available', '2021 Expenditure',
    # ... continues for all years
]
```

**Standardization Steps:**
1. **Column Mapping**: Map varied column names to standard schema
2. **Data Type Conversion**: Ensure numeric columns are properly typed
3. **Missing Column Handling**: Add missing columns with default values
4. **Column Ordering**: Arrange columns in consistent order

### **Stage 3: Data Cleaning & Validation**

```python
def clean_financial_data(df):
    """Comprehensive data cleaning pipeline"""
    
    # 1. Handle missing values
    df['Total required resources'] = df['Total required resources'].fillna(0)
    df['Total available resources'] = df['Total available resources'].fillna(0)
    df['Total expenditure resources'] = df['Total expenditure resources'].fillna(0)
    
    # 2. Remove duplicates
    df = df.drop_duplicates(subset=['Country', 'Region', 'Theme', 'Plan name'])
    
    # 3. Data validation
    assert df['Total required resources'].min() >= 0, "Negative funding amounts found"
    assert not df['Country'].isnull().any(), "Missing country names"
    
    # 4. Standardize text fields
    df['Country'] = df['Country'].str.strip().str.title()
    df['Agencies'] = df['Agencies'].str.replace(';', '; ').str.strip()
    
    return df
```

**Cleaning Operations:**
- **Missing Value Imputation**: Financial zeros, categorical defaults
- **Duplicate Removal**: Based on key identifier columns
- **Data Validation**: Range checks, null value assertions
- **Text Standardization**: Consistent formatting and separators

### **Stage 4: Feature Engineering**

```python
def engineer_features(df):
    """Create derived features for analysis"""
    
    # 1. Funding gap calculation
    df['Funding Gap'] = df['Total required resources'] - df['Total available resources']
    df['Funding Coverage %'] = (df['Total available resources'] / 
                               df['Total required resources'] * 100).fillna(0)
    
    # 2. Strategic Priority Labels (SP_Label)
    df['SP_Label'] = df['Strategic priority code'].astype(str) + ' - ' + df['Strategic priority']
    
    # 3. Multi-value field expansion
    df = expand_agencies_and_sdgs(df)
    
    # 4. Temporal aggregation
    df = aggregate_yearly_data(df)
    
    return df
```

**Feature Engineering Steps:**
1. **Calculated Metrics**: Funding gaps, coverage percentages
2. **Label Creation**: Strategic priority labels for grouping
3. **Multi-value Expansion**: Split semicolon-separated fields
4. **Temporal Features**: Year-over-year calculations

### **Stage 5: Output Generation**

```python
# Generate 4 clean output files
def generate_outputs(processed_df):
    """Generate clean CSV outputs"""
    
    # 1. Main financial dataset
    financial_clean = processed_df[FINANCIAL_COLUMNS]
    financial_clean.to_csv('src/outputs/data_output/Financial_Cleaned.csv', index=False)
    
    # 2. SDG goals mapping
    sdg_df = extract_sdg_mappings(processed_df)
    sdg_df.to_csv('src/outputs/data_output/SDG_Goals_Cleaned.csv', index=False)
    
    # 3. UN agencies mapping
    agencies_df = extract_agency_mappings(processed_df)
    agencies_df.to_csv('src/outputs/data_output/UN_Agencies_Cleaned.csv', index=False)
    
    # 4. Unified dataset
    unified_df = create_unified_dataset(processed_df)
    unified_df.to_csv('src/outputs/data_output/unified_data.csv', index=False)
```

---

## 📊 **Output Datasets**

### **1. Financial_Cleaned.csv (9.5MB)**
**Purpose**: Main financial dataset for dashboard analysis

```python
# Schema
FINANCIAL_SCHEMA = {
    'Country': str,           # Country name
    'Region': str,            # UN region
    'Theme': str,             # Thematic area  
    'Plan name': str,         # Development plan
    'Strategic priority code': float,  # Priority code (1-10)
    'Strategic priority': str, # Priority description
    'Agencies': str,          # Semicolon-separated agencies
    'SDG Goals': str,         # Semicolon-separated SDG goals
    'Total required resources': float,   # Total funding needed
    'Total available resources': float,  # Total funding available
    'Total expenditure resources': float, # Total expenditure
    # Year-specific columns for 2020-2026
    '2025 Required': float,   # 2025 requirements
    '2025 Available': float,  # 2025 available
    '2025 Expenditure': float # 2025 expenditure
}
```

**Usage**: Primary dataset for all dashboard visualizations and analysis

### **2. SDG_Goals_Cleaned.csv (1.4MB)**
**Purpose**: SDG goal mappings for ML model training

```python
# Key columns
SDG_COLUMNS = [
    'Country', 'Region', 'Theme', 
    'SDG Goals',  # Individual SDG goal names
    'Strategic priority code',
    'Total required resources'
]
```

**Usage**: Training data for SDG prediction model

### **3. UN_Agencies_Cleaned.csv (1.4MB)**
**Purpose**: UN agency collaboration mappings

```python
# Key columns  
AGENCY_COLUMNS = [
    'Country', 'Region', 'Theme',
    'Agencies',  # Individual agency names
    'Strategic priority code', 
    'Total required resources'
]
```

**Usage**: Training data for agency recommendation model

### **4. unified_data.csv (48MB)**
**Purpose**: Complete unified dataset with all features

**Usage**: Comprehensive dataset for advanced analytics and model training

---

## 🔄 **Data Pipeline Execution**

### **Manual Execution**

```bash
# Navigate to notebooks directory
cd src/notebooks/

# Start Jupyter notebook
jupyter notebook

# Open and run data_cleaning.ipynb
# Follow cell-by-cell execution
# Monitor output for errors or warnings
```

### **Automated Execution**

```python
# Create automated pipeline script
def run_data_pipeline():
    """Execute complete data pipeline"""
    
    print("🔍 Starting data pipeline...")
    
    # Stage 1: Load raw data
    print("📥 Loading raw data files...")
    raw_df = load_regional_theme_data()
    
    # Stage 2: Clean and validate
    print("🧹 Cleaning and validating data...")
    clean_df = clean_financial_data(raw_df)
    
    # Stage 3: Feature engineering
    print("⚙️ Engineering features...")
    processed_df = engineer_features(clean_df)
    
    # Stage 4: Generate outputs
    print("💾 Generating output files...")
    generate_outputs(processed_df)
    
    # Stage 5: Validation
    print("✅ Validating outputs...")
    validate_pipeline_outputs()
    
    print("🎉 Data pipeline completed successfully!")

if __name__ == "__main__":
    run_data_pipeline()
```

### **Pipeline Monitoring**

```python
def validate_pipeline_outputs():
    """Validate pipeline output quality"""
    
    # Check file existence and sizes
    required_files = [
        ('Financial_Cleaned.csv', 8_000_000),  # ~8MB minimum
        ('SDG_Goals_Cleaned.csv', 1_000_000),  # ~1MB minimum
        ('UN_Agencies_Cleaned.csv', 1_000_000), # ~1MB minimum
        ('unified_data.csv', 40_000_000)       # ~40MB minimum
    ]
    
    for filename, min_size in required_files:
        file_path = f'src/outputs/data_output/{filename}'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing output file: {filename}")
        
        file_size = os.path.getsize(file_path)
        if file_size < min_size:
            raise ValueError(f"Output file too small: {filename} ({file_size} bytes)")
    
    # Data quality checks
    df = pd.read_csv('src/outputs/data_output/Financial_Cleaned.csv')
    
    assert len(df) > 10000, f"Insufficient records: {len(df)}"
    assert df['Country'].nunique() > 100, f"Insufficient countries: {df['Country'].nunique()}"
    assert df['Region'].nunique() == 5, f"Incorrect regions: {df['Region'].nunique()}"
    
    print("✅ All pipeline outputs validated successfully")
```

---

## ⚡ **Performance Optimization**

### **Memory Optimization**

```python
def optimize_memory_usage(df):
    """Optimize DataFrame memory usage"""
    
    # Convert categorical columns
    categorical_cols = ['Country', 'Region', 'Theme']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df
```

### **Processing Optimization**

```python
def chunked_processing(file_paths, chunk_size=10000):
    """Process large files in chunks"""
    
    for file_path in file_paths:
        chunks = []
        for chunk in pd.read_excel(file_path, chunksize=chunk_size):
            processed_chunk = process_chunk(chunk)
            chunks.append(processed_chunk)
        
        combined = pd.concat(chunks, ignore_index=True)
        yield combined
```

---

## 🔍 **Data Quality Assurance**

### **Quality Metrics**

```python
DATA_QUALITY_METRICS = {
    'completeness': {
        'country_coverage': '>150 countries',
        'region_coverage': '5 regions',
        'theme_coverage': '11 themes',
        'year_coverage': '2020-2026'
    },
    'accuracy': {
        'funding_amounts': 'Non-negative values',
        'country_names': 'Valid country names',
        'agency_names': 'Valid UN agency codes'
    },
    'consistency': {
        'column_formats': 'Standardized schemas',
        'data_types': 'Proper numeric/categorical types',
        'text_formatting': 'Consistent case and separators'
    }
}
```

### **Automated Quality Checks**

```python
def run_quality_checks(df):
    """Run comprehensive data quality checks"""
    
    issues = []
    
    # Completeness checks
    if df['Country'].isnull().any():
        issues.append("Missing country names found")
    
    if df['Total required resources'].isnull().any():
        issues.append("Missing funding amounts found")
    
    # Accuracy checks
    if (df['Total required resources'] < 0).any():
        issues.append("Negative funding amounts found")
    
    if df['Region'].nunique() != 5:
        issues.append(f"Expected 5 regions, found {df['Region'].nunique()}")
    
    # Consistency checks
    for col in ['Total required resources', 'Total available resources']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column {col} is not numeric")
    
    if issues:
        raise ValueError(f"Data quality issues found: {issues}")
    
    print("✅ All data quality checks passed")
```

---

## 🛠️ **Troubleshooting Data Pipeline**

### **Common Issues**

#### **Issue 1: Missing Excel Files**
```python
# Solution: Check file existence before processing
def check_data_availability():
    """Check availability of all expected data files"""
    missing_files = []
    
    for region in REGIONS:
        for theme in THEMES:
            file_path = f"src/data/{region}/{theme}.xlsx"
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️ Missing {len(missing_files)} data files:")
        for file in missing_files[:10]:  # Show first 10
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print("✅ All expected data files found")
```

#### **Issue 2: Schema Inconsistencies**
```python
# Solution: Robust schema standardization
def standardize_schema(df, expected_columns):
    """Ensure DataFrame has expected schema"""
    
    # Add missing columns with defaults
    for col in expected_columns:
        if col not in df.columns:
            if 'resources' in col.lower():
                df[col] = 0.0
            else:
                df[col] = ''
    
    # Remove unexpected columns
    df = df[expected_columns]
    
    return df
```

#### **Issue 3: Memory Issues**
```python
# Solution: Chunked processing
def process_large_datasets():
    """Handle large datasets with memory constraints"""
    
    # Process in chunks
    chunk_size = 10000
    processed_chunks = []
    
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = process_data_chunk(chunk)
        processed_chunks.append(processed_chunk)
        
        # Explicit garbage collection
        import gc
        gc.collect()
    
    final_df = pd.concat(processed_chunks, ignore_index=True)
    return final_df
```

---

## 📅 **Pipeline Maintenance Schedule**

### **Regular Maintenance Tasks**

| Frequency | Task | Description |
|-----------|------|-------------|
| **Monthly** | Data Refresh | Update with new UN data releases |
| **Quarterly** | Schema Review | Check for new columns/themes |
| **Bi-annually** | Performance Audit | Optimize pipeline performance |
| **Annually** | Complete Refresh | Full pipeline review and update |

### **Data Refresh Process**

```bash
# Monthly data refresh workflow
1. Backup existing data: cp -r src/outputs/data_output/ backup/
2. Update raw data files in src/data/
3. Run pipeline: python scripts/run_data_pipeline.py
4. Validate outputs: python scripts/validate_outputs.py
5. Deploy updates: git add . && git commit -m "Data refresh $(date)"
```

---

## 🎯 **Next Steps**

### **For New Developers**
1. **Understand the Flow**: Read through this document completely
2. **Run the Pipeline**: Execute `data_cleaning.ipynb` step by step
3. **Validate Outputs**: Verify all 4 output files are generated correctly
4. **Test Modifications**: Make small changes and observe effects

### **For Data Updates**
1. **Backup Current Data**: Always backup before modifications
2. **Update Raw Files**: Replace Excel files in `src/data/`
3. **Run Full Pipeline**: Execute complete ETL process
4. **Validate Results**: Run quality checks and validation
5. **Update Models**: Retrain ML models if significant data changes

### **For Pipeline Enhancement**
1. **Performance Profiling**: Identify bottlenecks
2. **Automation**: Create scheduled pipeline execution
3. **Error Handling**: Improve robustness and error recovery
4. **Monitoring**: Add pipeline health monitoring

---

**🔄 Next Steps**: Explore [ML Models Documentation](./08_ML_MODELS.md) to understand how the cleaned data is used for model training, or check [Testing & QA](./13_TESTING_QA.md) for comprehensive testing strategies.