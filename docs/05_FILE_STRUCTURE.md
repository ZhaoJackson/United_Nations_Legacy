# 📁 File Structure Guide

Comprehensive guide to the project's file organization and purpose of each component.

## 🌳 **Complete Project Structure**

```
United_Nations_Legacy/
├── 📄 app.py                          # Main application entry point (16 lines)
├── 📄 requirements.txt                # Python dependencies (37 lines)
├── 📄 README.md                       # Project documentation (387 lines)
├── 📄 LICENSE.md                      # MIT license (11 lines)
├── 📄 .gitignore                      # Git exclusions (90 lines)
├── 📁 .streamlit/                     # Streamlit configuration
│   ├── 📄 secrets.toml               # API credentials (encrypted)
│   └── 📄 config.toml                # App configuration
├── 📁 docs/                          # Developer documentation
│   ├── 📄 README.md                  # Documentation index
│   ├── 📄 02_QUICK_START.md          # Quick start guide
│   ├── 📄 03_INSTALLATION_SETUP.md   # Installation instructions
│   ├── 📄 04_ARCHITECTURE.md         # System architecture
│   └── 📄 05_FILE_STRUCTURE.md       # This file
├── 📁 pages/                         # Streamlit dashboard pages
│   ├── 📄 overview.py                # Platform overview (1,023 lines)
│   ├── 📄 main_page.py              # Interactive dashboard (818 lines)
│   ├── 📄 prediction.py             # ML analytics (1,253 lines)
│   ├── 📄 model.py                  # Real-time predictions (375 lines)
│   ├── 📄 bot.py                    # AI chatbot (371 lines)
│   └── 📁 style/
│       └── 📄 style.css             # Custom UN-themed styling (407 lines)
└── 📁 src/                          # Source code and data
    ├── 📄 __init__.py               # Package initialization (18 lines)
    ├── 📄 commonconst.py            # Core constants & utilities (763 lines)
    ├── 📄 dynamic_analysis.py       # Advanced data processing (448 lines)
    ├── 📁 prompt/                   # AI prompt engineering modules
    │   ├── 📄 __init__.py           # Prompt package init (36 lines)
    │   ├── 📄 chatbot.py           # GPT-4o conversational AI (409 lines)
    │   ├── 📄 dashboard.py         # O1 strategic insights (219 lines)
    │   ├── 📄 models.py            # GPT-4o prediction analysis (329 lines)
    │   ├── 📄 funding_prediction.py # O1 funding insights (219 lines)
    │   ├── 📄 anomaly_detection.py  # O1 anomaly analysis (216 lines)
    │   └── 📄 agency_performance.py # O1 performance insights (216 lines)
    ├── 📁 notebooks/                # Data science pipeline
    │   ├── 📄 data_cleaning.ipynb   # ETL pipeline (2,591 lines)
    │   ├── 📄 sdg.ipynb            # SDG prediction model (947 lines)
    │   ├── 📄 agency.ipynb         # Agency recommendation model (1,283 lines)
    │   ├── 📄 funding.ipynb        # Funding forecasting (355 lines)
    │   ├── 📄 anomaly.ipynb        # Anomaly detection (256 lines)
    │   ├── 📄 assistance.ipynb     # Performance clustering (373 lines)
    │   ├── 📄 Financial.csv        # Raw financial data (10.0 MB)
    │   ├── 📄 SDG_Goals.csv        # Raw SDG data (2.2 MB)
    │   └── 📄 UN_Agencies.csv      # Raw agency data (2.4 MB)
    ├── 📁 data/                     # Regional thematic data (55 files)
    │   ├── 📁 Africa/              # 11 Excel files by theme
    │   ├── 📁 Arab States/         # 11 Excel files by theme
    │   ├── 📁 Asia Pacific/        # 11 Excel files by theme
    │   ├── 📁 Europe and Central Asia/ # 11 Excel files by theme
    │   └── 📁 Latin America and the Caribbean/ # 11 Excel files by theme
    └── 📁 outputs/                  # Generated analysis results
        ├── 📁 data_output/          # Cleaned datasets (60.3 MB total)
        │   ├── 📄 unified_data.csv  # Combined dataset (48 MB)
        │   ├── 📄 Financial_Cleaned.csv # Financial data (9.5 MB)
        │   ├── 📄 SDG_Goals_Cleaned.csv # SDG mappings (1.4 MB)
        │   └── 📄 UN_Agencies_Cleaned.csv # Agency data (1.4 MB)
        └── 📁 model_output/         # ML models & predictions (82 MB total)
            ├── 📄 SDG_model.pkl     # SDG classifier (11 MB)
            ├── 📄 Agency_model.pkl  # Agency recommender (33 MB)
            ├── 📄 funding_prediction.csv # 2026 forecasts (9.5 MB)
            ├── 📄 anomaly_detection.csv # Anomaly results (9.9 MB)
            └── 📄 un_agency.csv     # Performance clusters (9.6 MB)
```

## 🎯 **Core Application Files**

### **`app.py` - Application Entry Point**
```python
# Purpose: Main router and page navigation
# Lines: 16
# Key Functionality:
- Streamlit page configuration
- Multi-page navigation setup  
- Page routing (5 pages total)
```

**Responsibilities:**
- Initialize Streamlit configuration
- Setup page navigation structure
- Route to appropriate dashboard pages
- Load shared configurations

**Dependencies:**
- `src.commonconst` for shared constants
- `src.dynamic_analysis` for data processing
- Individual page modules

---

### **`requirements.txt` - Dependencies**
```bash
# Purpose: Python package dependencies with version constraints
# Lines: 37
# Categories:
- Core framework (Streamlit, OpenAI)
- Data processing (Pandas, NumPy) 
- ML libraries (Scikit-learn, XGBoost, CatBoost)
- Visualization (Plotly, Seaborn)
- Utilities (pycountry, requests)
```

**Key Dependencies:**
- **Streamlit 1.47+**: Web framework
- **OpenAI 1.35+**: Azure OpenAI integration
- **Pandas 2.0+**: Data manipulation
- **XGBoost 1.7+**: Machine learning models
- **Plotly 5.15+**: Interactive visualizations

---

## 📱 **Dashboard Pages**

### **`pages/overview.py` - Platform Overview**
```python
# Purpose: Executive dashboard with platform introduction
# Lines: 1,023
# Tabs: Platform Overview, Data Sources, Acknowledgments
```

**Key Features:**
- Real-time platform statistics
- System capabilities showcase
- ML model performance metrics
- Data source documentation
- Team acknowledgments

**Components:**
- Statistics grid (projects, countries, funding)
- Feature cards (capabilities overview)
- Model showcase (accuracy metrics)
- Acknowledgments section

---

### **`pages/main_page.py` - Interactive Dashboard**
```python
# Purpose: Core financial analysis with interactive filtering
# Lines: 818
# Features: Global map, KPIs, trends, AI insights
```

**Key Components:**
- **Global Funding Map**: Choropleth visualization
- **KPI Metrics**: Required/Available/Expenditure totals
- **Top 10 Rankings**: Countries by funding needs
- **Multi-Year Trends**: Country-specific analysis
- **AI Strategic Intelligence**: O1 model insights

**Filtering Options:**
- Year selection (2020-2025)
- Theme selection (11 themes)
- Region selection (5 regions + all)
- UN Agency filtering
- SDG Goal filtering

---

### **`pages/prediction.py` - ML Analytics**
```python
# Purpose: Advanced analytics with ML predictions
# Lines: 1,253
# Tabs: Funding Predictions, Anomaly Detection, Performance
```

**Tab Structure:**
1. **🔮 Funding Predictions**: RandomForest forecasting
2. **🚨 Anomaly Detection**: LocalOutlierFactor analysis
3. **📊 Agency Performance**: KMeans clustering

**Analytics Features:**
- Model performance metrics
- Interactive predictions
- Anomaly investigation
- Performance optimization insights

---

### **`pages/model.py` - Real-time Predictions**
```python
# Purpose: Interactive ML predictions with AI analysis
# Lines: 375
# Models: SDG classifier, Agency recommender
```

**Core Features:**
- **Interactive Country Selection**: World map interface
- **Real-time Predictions**: SDG goals and agency recommendations
- **Historical Context**: Past collaboration analysis
- **AI Strategic Analysis**: GPT-4o powered insights

**Prediction Flow:**
1. Select country, theme, strategic priority
2. Generate ML predictions (SDG goals, agencies)
3. Fetch historical context
4. Generate AI strategic analysis

---

### **`pages/bot.py` - Conversational AI**
```python
# Purpose: Natural language interface for data exploration
# Lines: 371
# AI Model: Azure GPT-4o with context awareness
```

**Interface Components:**
- **Chat Interface**: Persistent conversation history
- **Quick Statistics**: Real-time metrics sidebar
- **Filter Integration**: Dynamic data filtering
- **Quick Questions**: Pre-defined query templates

**AI Capabilities:**
- Natural language query processing
- Contextual data analysis
- Dynamic response generation
- Multi-turn conversation support

---

## 🎨 **Styling & Assets**

### **`pages/style/style.css` - UN Theme Styling**
```css
/* Purpose: Custom CSS for UN branding and responsive design */
/* Lines: 407 */
/* Features: UN color scheme, responsive layouts, animations */
```

**Styling Categories:**
- **UN Branding**: Official color palette (#009edb)
- **Component Styling**: Cards, metrics, charts
- **Responsive Design**: Mobile and desktop layouts
- **Interactive Elements**: Hover effects, transitions

---

## 🧠 **Core Business Logic**

### **`src/commonconst.py` - Central Constants**
```python
# Purpose: Shared utilities, constants, and configurations
# Lines: 763
# Role: Core module imported by all pages
```

**Key Sections:**
- **Path Constants**: File and directory paths
- **Configuration**: Page settings, color schemes
- **Data Functions**: Loading, filtering, processing
- **Azure OpenAI**: Client configuration
- **Model Functions**: ML model loading and prediction

**Critical Functions:**
- `get_theme_list()`: Dynamic theme discovery
- `get_region_list()`: Region enumeration
- `filter_by_agency()`: Agency-based filtering
- `load_sdg_model()`: ML model loading
- `predict_sdg_goals()`: SDG prediction pipeline

---

### **`src/dynamic_analysis.py` - Advanced Processing**
```python
# Purpose: Dynamic data processing and analysis capabilities
# Lines: 448
# Classes: DynamicDataProcessor, DynamicModelManager
```

**Core Classes:**
- **DynamicDataProcessor**: Handles new theme discovery and processing
- **DynamicModelManager**: Manages model retraining and updates

**Key Features:**
- Automatic theme detection
- Data structure discovery
- Model retraining triggers
- System adaptation capabilities

---

## 🤖 **AI Prompt Engineering**

### **`src/prompt/` - AI Integration Modules**

Each module specializes in different types of AI analysis:

#### **`chatbot.py` - Conversational AI** (409 lines)
- **Purpose**: Natural language processing for user queries
- **Model**: Azure GPT-4o
- **Features**: Context awareness, data summarization

#### **`dashboard.py` - Strategic Insights** (219 lines)
- **Purpose**: Executive-level strategic analysis
- **Model**: Azure OpenAI O1
- **Output**: 4-section strategic reports

#### **`models.py` - Prediction Analysis** (329 lines)
- **Purpose**: ML prediction interpretation
- **Model**: Azure GPT-4o
- **Features**: Historical context integration

#### **`funding_prediction.py` - Financial Insights** (219 lines)
- **Purpose**: Funding forecast analysis
- **Model**: Azure OpenAI O1
- **Output**: Risk assessment and planning recommendations

#### **`anomaly_detection.py` - Risk Analysis** (216 lines)
- **Purpose**: Anomaly investigation priorities
- **Model**: Azure OpenAI O1
- **Output**: Investigation roadmap and risk mitigation

#### **`agency_performance.py` - Performance Optimization** (216 lines)
- **Purpose**: Agency efficiency recommendations
- **Model**: Azure OpenAI O1
- **Output**: Performance improvement strategies

---

## 📓 **Data Science Pipeline**

### **`src/notebooks/` - Jupyter Notebooks**

#### **`data_cleaning.ipynb` - ETL Pipeline** (2,591 lines)
```python
# Purpose: Master data preprocessing pipeline
# Input: Raw Excel files (55 files across regions/themes)
# Output: 4 cleaned CSV files ready for analysis
```

**Processing Steps:**
1. Load and combine Excel files
2. Standardize column structures
3. Handle missing values and duplicates
4. Create unified strategic priority labels
5. Generate clean datasets for ML

#### **ML Model Notebooks:**

- **`sdg.ipynb`** (947 lines): SDG prediction model training
- **`agency.ipynb`** (1,283 lines): Agency recommendation model
- **`funding.ipynb`** (355 lines): Funding gap forecasting
- **`anomaly.ipynb`** (256 lines): Anomaly detection pipeline
- **`assistance.ipynb`** (373 lines): Agency performance clustering

**Common Pattern:**
```python
1. Data Loading & Preprocessing
2. Feature Engineering
3. Model Training & Validation
4. Performance Evaluation
5. Model Serialization
6. Prediction Generation
```

---

## 📊 **Data Files**

### **Raw Data (`src/notebooks/` and `src/data/`)**

#### **Core Datasets:**
- **`Financial.csv`** (10.0 MB): Main financial transactions and allocations
- **`SDG_Goals.csv`** (2.2 MB): Sustainable Development Goal mappings
- **`UN_Agencies.csv`** (2.4 MB): UN agency collaboration data

#### **Regional Data (`src/data/`):**
```
55 Excel files organized as:
- 5 regions × 11 themes = 55 files
- Each file: ~1-5 MB
- Themes: crime, digital, education, environment, food, gender, governance, poverty, water, work, youth
```

### **Processed Data (`src/outputs/data_output/`)**

#### **Cleaned Datasets:**
- **`unified_data.csv`** (48 MB): Combined dataset from all sources
- **`Financial_Cleaned.csv`** (9.5 MB): Cleaned financial data
- **`SDG_Goals_Cleaned.csv`** (1.4 MB): Processed SDG mappings
- **`UN_Agencies_Cleaned.csv`** (1.4 MB): Cleaned agency data

### **Model Outputs (`src/outputs/model_output/`)**

#### **Trained Models:**
- **`SDG_model.pkl`** (11 MB): XGBoost SDG classifier
- **`Agency_model.pkl`** (33 MB): XGBoost agency recommender

#### **Predictions & Analysis:**
- **`funding_prediction.csv`** (9.5 MB): 2026 funding forecasts
- **`anomaly_detection.csv`** (9.9 MB): Anomaly scores and flags
- **`un_agency.csv`** (9.6 MB): Performance clusters and metrics

---

## 🔧 **Configuration Files**

### **`.streamlit/secrets.toml` - API Credentials**
```toml
# Purpose: Secure storage of Azure OpenAI credentials
# Security: Excluded from version control
# Required for: AI features (chatbot, insights generation)
```

### **`.streamlit/config.toml` - App Configuration**
```toml
# Purpose: Streamlit application settings
# Features: Performance tuning, security options
# Customization: Server settings, UI preferences
```

### **`.gitignore` - Version Control Exclusions**
```bash
# Purpose: Exclude sensitive and generated files
# Excludes: Credentials, cache files, large datasets
# Lines: 90
```

---

## 📂 **Directory Responsibilities**

| Directory | Purpose | Size | Key Files |
|-----------|---------|------|-----------|
| **`/`** | Project root | - | app.py, requirements.txt, README.md |
| **`pages/`** | Dashboard UI | 62KB | 5 main pages + styling |
| **`src/`** | Core logic | - | commonconst.py, dynamic_analysis.py |
| **`src/prompt/`** | AI modules | 16KB | 6 specialized prompt modules |
| **`src/notebooks/`** | Data pipeline | 14.6MB | 6 Jupyter notebooks + raw data |
| **`src/data/`** | Regional data | ~55MB | 55 Excel files (5 regions × 11 themes) |
| **`src/outputs/`** | Generated files | 142MB | Cleaned data + trained models |
| **`docs/`** | Documentation | - | Developer guides and API docs |

---

## 🎯 **File Naming Conventions**

### **Code Files:**
- **Pages**: `{page_name}.py` (e.g., `main_page.py`)
- **Modules**: `{module_name}.py` (e.g., `commonconst.py`)
- **Notebooks**: `{purpose}.ipynb` (e.g., `data_cleaning.ipynb`)

### **Data Files:**
- **Raw**: `{Source}_{Type}.csv` (e.g., `Financial.csv`)
- **Cleaned**: `{Source}_Cleaned.csv` (e.g., `Financial_Cleaned.csv`)
- **Models**: `{Type}_model.pkl` (e.g., `SDG_model.pkl`)
- **Predictions**: `{type}_prediction.csv` (e.g., `funding_prediction.csv`)

### **Documentation:**
- **Guides**: `{NN}_{TITLE}.md` (e.g., `02_QUICK_START.md`)
- **Numbered**: Sequential ordering for logical flow
- **Descriptive**: Clear, actionable titles

---

## 🔄 **File Dependencies**

### **Import Hierarchy:**
```
app.py
├── src.commonconst (central imports)
├── src.dynamic_analysis
└── pages.*
    ├── src.commonconst
    ├── src.prompt.*
    └── src.dynamic_analysis
```

### **Data Dependencies:**
```
Raw Data (src/data/ + src/notebooks/)
├── data_cleaning.ipynb
├── Cleaned Data (src/outputs/data_output/)
├── ML Notebooks (src/notebooks/)
├── Model Outputs (src/outputs/model_output/)
└── Dashboard Pages (pages/)
```

---

**🔄 Next Steps**: Explore the [Development Workflow](./06_DEVELOPMENT_WORKFLOW.md) to understand how to contribute to and extend this codebase, or jump to [Data Pipeline Documentation](./07_DATA_PIPELINE.md) for data processing details.