# 🇺🇳 UN Financial Intelligence Dashboard

> **A comprehensive AI-powered analytics platform for UN JointWork Plans financial data analysis, featuring predictive modeling, anomaly detection, and intelligent chatbot assistance.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.47+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

---

## 🌟 **Project Overview**

This dashboard provides comprehensive analysis of UN JointWork Plans financial data, enabling insights into funding patterns, resource allocation, and collaboration trends across different regions, themes, and UN agencies. Built with advanced machine learning models and an AI-powered chatbot for intelligent data exploration.

### **🎯 Key Features**
- 💰 **Funding Gap Analysis** - Identify and predict funding shortfalls across regions and themes
- 🌍 **Regional Intelligence** - Compare performance across Africa, Asia Pacific, Arab States, Europe & Central Asia, Latin America
- 🎯 **Thematic Analysis** - Deep dive into education, governance, environment, gender equality, and more
- 🤖 **AI-Powered Chat** - Interactive assistant for data exploration and insights
- 📊 **Predictive Modeling** - ML-driven predictions for SDG goals and agency recommendations
- 🔍 **Anomaly Detection** - Identify unusual patterns in strategic priorities

---

## 🏗️ **Complete Workflow Architecture**

### **1. Data Pipeline & Processing**
```mermaid
graph TD
    A[Raw Data Sources] --> B[Data Cleaning]
    B --> C[Feature Engineering] 
    C --> D[Model Training]
    D --> E[Prediction Generation]
    E --> F[Streamlit Dashboard]
```

#### **📥 Data Sources**
- **Raw Regional Data**: [`src/data/`](src/data/) - Regional Excel files by theme (Africa, Asia Pacific, Arab States, Europe & Central Asia, Latin America)
- **Financial Dataset**: [`src/notebooks/Financial.csv`](src/notebooks/Financial.csv) - Core funding and expenditure data
- **SDG Mapping**: [`src/notebooks/SDG_Goals.csv`](src/notebooks/SDG_Goals.csv) - Sustainable Development Goals alignment
- **Agency Data**: [`src/notebooks/UN_Agencies.csv`](src/notebooks/UN_Agencies.csv) - UN agency collaboration metrics

#### **🔧 Data Cleaning & Processing**
- **Main Processor**: [`src/notebooks/data_cleaning.ipynb`](src/notebooks/data_cleaning.ipynb) - Comprehensive data preprocessing pipeline
- **Configuration**: [`src/commonconst.py`](src/commonconst.py) - Central constants, paths, and utility functions
- **Outputs**: [`src/outputs/data_output/`](src/outputs/data_output/) - Cleaned and processed datasets

### **2. Machine Learning Models**

#### **🎯 SDG Prediction Model**
- **Training Notebook**: [`src/notebooks/sdg.ipynb`](src/notebooks/sdg.ipynb)
- **Model File**: [`src/outputs/model_output/SDG_model.pkl`](src/outputs/model_output/SDG_model.pkl)
- **Purpose**: Predict relevant Sustainable Development Goals based on country, theme, and strategic priorities

#### **🏢 Agency Recommendation Model**  
- **Training Notebook**: [`src/notebooks/agency.ipynb`](src/notebooks/agency.ipynb)
- **Model File**: [`src/outputs/model_output/Agency_model.pkl`](src/outputs/model_output/Agency_model.pkl)
- **Purpose**: Recommend optimal UN agencies for collaboration based on project characteristics

#### **💰 Funding Prediction System**
- **Analysis Notebook**: [`src/notebooks/funding.ipynb`](src/notebooks/funding.ipynb)
- **Output**: [`src/outputs/model_output/funding_prediction.csv`](src/outputs/model_output/funding_prediction.csv)
- **Purpose**: Forecast funding requirements and identify potential gaps

#### **🔍 Anomaly Detection Engine**
- **Detection Notebook**: [`src/notebooks/anomaly.ipynb`](src/notebooks/anomaly.ipynb)
- **Output**: [`src/outputs/model_output/anomaly_detection.csv`](src/outputs/model_output/anomaly_detection.csv)
- **Purpose**: Identify unusual patterns in strategic priorities and resource allocation

### **3. Streamlit Application Architecture**

#### **🚀 Main Application**
- **Entry Point**: [`app.py`](app.py) - Main application with navigation and configuration
- **Dependencies**: [`requirements.txt`](requirements.txt) - Python package dependencies
- **Secrets Management**: [`.streamlit/secrets.toml`](.streamlit/secrets.toml) - Azure OpenAI API configuration

#### **📱 Dashboard Pages**
- **Overview**: [`pages/overview.py`](pages/overview.py) - Executive summary and key metrics
- **Main Dashboard**: [`pages/main_page.py`](pages/main_page.py) - Interactive financial analysis and visualizations  
- **Predictive Analysis**: [`pages/prediction.py`](pages/prediction.py) - ML model results and trend analysis
- **Interactive Models**: [`pages/model.py`](pages/model.py) - Real-time predictions and scenario modeling
- **AI Chatbot**: [`pages/bot.py`](pages/bot.py) - Intelligent conversational interface

#### **🎨 Styling & Assets**
- **CSS Styling**: [`pages/style/style.css`](pages/style/style.css) - Custom UI styling and themes
- **AI Integration**: [`src/prompt.py`](src/prompt.py) - LLM prompt engineering and response handling

---

## ⚡ **Quick Start Guide**

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/your-username/United_Nations_Legacy.git
cd United_Nations_Legacy

# Install dependencies  
pip install -r requirements.txt

# Configure Azure OpenAI credentials
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys
```

### **2. Data Setup**
Ensure you have the required data files (see [`DATA_README.md`](DATA_README.md) for details):
- Place CSV files in `src/notebooks/`
- Organize regional Excel files in `src/data/`
- Run data processing notebooks if needed

### **3. Launch Application**
```bash
streamlit run app.py
```

### **4. Access Dashboard**
- **Local**: http://localhost:8501
- **Web Deployment**: Your Streamlit Cloud URL

---

## 🎯 **How to Use the Dashboard**

### **📊 Overview Page**
- View executive summary of funding gaps and performance metrics
- Explore high-level regional and thematic trends
- Access quick insights and recommendations

### **🏠 Main Dashboard** 
- **Filter by**: Country, region, theme, year, UN agency
- **Visualizations**: Interactive maps, trend charts, funding gap analysis
- **Export**: Download filtered data and visualizations

### **📈 Predictive Analysis**
- **Model Performance**: View accuracy metrics and validation results
- **Trend Analysis**: Explore funding and performance trends over time
- **Anomaly Detection**: Identify unusual patterns and outliers

### **🎯 Interactive Models**
- **SDG Prediction**: Input project details to get relevant SDG recommendations
- **Agency Matching**: Find optimal UN agencies for collaboration
- **Scenario Modeling**: Test different funding and resource scenarios

### **🤖 AI Chatbot**
- **Natural Language Queries**: Ask questions in plain English
- **Data Exploration**: "What are the funding gaps in Africa for education?"
- **Comparative Analysis**: "Which agencies work most on climate action?"
- **Trend Insights**: "Show me resource allocation trends for governance projects"

---

## 🔧 **Technical Architecture**

### **🛠️ Tech Stack**
- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: Plotly, Seaborn
- **AI Integration**: Azure OpenAI (GPT-4, O1 models)
- **Data Processing**: Jupyter Notebooks, Excel/CSV handling

### **📦 Project Structure**
```
United_Nations_Legacy/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies & deployment config
├── DATA_README.md                  # Data files documentation
├── pages/                          # Streamlit pages
│   ├── overview.py                 # Executive dashboard
│   ├── main_page.py               # Interactive analysis
│   ├── prediction.py              # ML model results  
│   ├── model.py                   # Real-time predictions
│   ├── bot.py                     # AI chatbot interface
│   └── style/
│       └── style.css              # Custom styling
├── src/
│   ├── commonconst.py             # Configuration and utilities
│   ├── prompt.py                  # AI prompt engineering
│   ├── notebooks/                 # Data science workflows
│   │   ├── data_cleaning.ipynb    # Data preprocessing
│   │   ├── sdg.ipynb             # SDG prediction model
│   │   ├── agency.ipynb          # Agency recommendation model  
│   │   ├── funding.ipynb         # Funding analysis
│   │   └── anomaly.ipynb         # Anomaly detection
│   ├── data/                      # Raw regional data (excluded from git)
│   └── outputs/                   # Processed data and model outputs
│       ├── data_output/           # Cleaned datasets
│       └── model_output/          # ML model files and predictions
└── .streamlit/
    └── secrets.toml               # API configuration (excluded from git)
```

---

## 🌍 **Use Cases & Applications**

### **👥 For UN Staff & Policy Experts**
- **Resource Planning**: Identify funding gaps and optimize allocation
- **Performance Monitoring**: Track agency collaboration and project outcomes  
- **Strategic Decision Making**: Data-driven insights for program planning
- **Cross-Regional Learning**: Compare best practices across regions

### **📊 For Data Analysts & Researchers**
- **Advanced Analytics**: Access to comprehensive UN financial datasets
- **Predictive Modeling**: ML-powered forecasting and scenario analysis
- **Anomaly Detection**: Identify unusual patterns requiring investigation
- **Custom Analysis**: Interactive filtering and data exploration

### **🎯 For Program Managers**
- **Project Planning**: SDG alignment and agency collaboration recommendations
- **Budget Optimization**: Funding gap analysis and resource prioritization  
- **Impact Assessment**: Performance tracking and trend analysis
- **Reporting**: Automated insights and visualization generation

---

## 🚀 **Development & Deployment**

### **🔄 Data Science Workflow**
1. **Data Collection**: Regional Excel files and financial datasets
2. **Preprocessing**: Clean, standardize, and engineer features
3. **Model Training**: SDG prediction, agency recommendation, anomaly detection
4. **Validation**: Cross-validation and performance testing  
5. **Deployment**: Model integration into Streamlit dashboard

### **☁️ Deployment Options**
- **Streamlit Cloud**: Automated deployment from GitHub
- **Local Development**: Run locally with `streamlit run app.py`
- **Enterprise**: Deploy on UN infrastructure with proper security

> 📋 **For detailed deployment instructions and troubleshooting**: See [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)

### **🔒 Security & Privacy**
- **API Security**: Azure OpenAI credentials via environment variables
- **Data Privacy**: Sensitive data excluded from version control
- **Access Control**: Configure authentication as needed for deployment

---

## 📞 **Contact & Support**

- **Developer**: Zichen Zhao (Jackson)
- **Email**: ziche.zhao@un.org  
- **Organization**: UN Development Coordination Office (UNDCO)
- **GitHub**: [Repository Issues](https://github.com/your-username/United_Nations_Legacy/issues)

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

## 🙏 **Acknowledgments**

- UN Development Coordination Office (UNDCO)
- UN Country Teams contributing data
- Azure AI Foundry for enterprise AI infrastructure
- Open source community for tools and libraries

---

*Built with ❤️ for evidence-based decision-making in UN programming*