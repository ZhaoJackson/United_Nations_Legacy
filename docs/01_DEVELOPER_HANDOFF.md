# 🎯 Developer Handoff Guide

**Essential guide for internal developers taking over the UN Financial Intelligence Dashboard project.**

## 🚀 **Immediate Actions Required**

### **Day 1: Environment Setup (1-2 hours)**

1. **Clone Repository & Setup Environment**
```bash
git clone https://github.com/ZhaoJackson/United_Nations_Legacy.git
cd United_Nations_Legacy
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Configure Azure OpenAI Credentials**
```bash
# Create secrets file
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit with your actual Azure OpenAI credentials
```

3. **Verify Installation**
```bash
# Run verification script
python docs/scripts/verify_installation.py

# Test application
streamlit run app.py
# Navigate to http://localhost:8501
```

### **Day 1: Critical Validation (30 minutes)**

✅ **Test Core Functions:**
- [ ] Overview page loads with platform statistics
- [ ] Dashboard shows interactive world map
- [ ] Models page generates predictions
- [ ] Chatbot responds to queries (requires Azure credentials)
- [ ] All filters work correctly

✅ **Verify Data Pipeline:**
- [ ] `src/outputs/data_output/` contains 4 CSV files (60.3MB total)
- [ ] `src/outputs/model_output/` contains 5 model files (82MB total)
- [ ] No critical errors in browser console

### **Week 1: System Understanding (4-6 hours)**

📚 **Read Documentation (Priority Order):**
1. [Quick Start Guide](./02_QUICK_START.md) - Get running immediately
2. [Architecture Overview](./04_ARCHITECTURE.md) - Understand system design
3. [File Structure Guide](./05_FILE_STRUCTURE.md) - Navigate codebase
4. [API Reference](./11_API_REFERENCE.md) - Understand functions

🧪 **Hands-on Exploration:**
1. **Test Each Page**: Spend 15 minutes on each of the 5 dashboard pages
2. **Modify Filters**: Understand how data filtering works across components
3. **Run Notebooks**: Execute `src/notebooks/data_cleaning.ipynb` to understand data flow

---

## 🏗️ **Project Architecture Summary**

### **High-Level Components**
```
UN Financial Intelligence Dashboard
├── 📱 Frontend (Streamlit) - 5 interactive pages
├── 🧠 AI Layer (Azure OpenAI) - 6 specialized prompt modules  
├── 🤖 ML Pipeline (Scikit-learn) - 5 trained models
├── 📊 Data Layer (142MB) - Cleaned datasets + model outputs
└── 🔧 Core Logic (Python) - commonconst.py + dynamic_analysis.py
```

### **Key Data Flow**
```
Raw UN Data (55 Excel files) 
→ data_cleaning.ipynb (ETL)
→ Cleaned CSVs (60MB)
→ ML Training (5 notebooks)
→ Model Outputs (82MB)
→ Streamlit Dashboard (Real-time analytics)
```

### **Critical Files You Must Understand**
1. **`app.py`** (16 lines) - Main entry point and navigation
2. **`src/commonconst.py`** (763 lines) - Core utilities and constants
3. **`pages/main_page.py`** (818 lines) - Primary dashboard interface
4. **`src/notebooks/data_cleaning.ipynb`** (2,591 lines) - ETL pipeline

---

## 🎯 **Feature Priorities**

### **Immediate Maintenance (Week 1-2)**
- **Azure OpenAI Rate Limiting**: Implement retry logic and usage monitoring
- **Performance Optimization**: Add loading indicators and optimize caching
- **Error Handling**: Improve user-facing error messages
- **Data Validation**: Ensure data integrity checks are robust

### **Short-term Enhancements (Month 1-3)**
- **Mobile Responsiveness**: Optimize for tablet/mobile viewing
- **Advanced Filtering**: Add date range and multi-select filters
- **Export Functionality**: PDF reports and CSV data exports
- **User Analytics**: Track usage patterns and popular features

### **Medium-term Development (Month 3-6)**
- **Real-time Data**: Integration with live UN data sources
- **Advanced AI**: GPT-4 Turbo integration and custom prompts
- **Performance**: Database migration from CSV to PostgreSQL
- **Multi-language**: UN official language support

---

## 🔧 **Critical Maintenance Tasks**

### **Monthly Tasks**
```bash
# Update dependencies (first Monday of month)
pip list --outdated
pip install --upgrade streamlit pandas plotly

# Data refresh (if new UN data available)
cd src/notebooks/
jupyter notebook data_cleaning.ipynb

# Model retraining (quarterly)
jupyter notebook sdg.ipynb
jupyter notebook agency.ipynb
```

### **Azure OpenAI Monitoring**
```python
# Add to your monitoring script
def monitor_api_usage():
    """Track Azure OpenAI usage and costs"""
    # Monitor daily request counts
    # Alert when approaching quota limits  
    # Track response times and errors
    pass
```

### **Performance Monitoring**
```python
# Key metrics to track
PERFORMANCE_THRESHOLDS = {
    'page_load_time': 5.0,      # seconds
    'memory_usage': 2.0,        # GB
    'api_response_time': 10.0,  # seconds
    'error_rate': 0.05          # 5%
}
```

---

## 🚨 **Common Issues & Quick Fixes**

### **Issue 1: Application Won't Start**
```bash
# Most common cause: Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or dependency conflicts
pip install --force-reinstall streamlit
```

### **Issue 2: Charts Not Loading**
```python
# Debug data issues
def debug_chart_data(df):
    print(f"Shape: {df.shape}")
    print(f"Nulls: {df.isnull().sum()}")
    print(f"Data types: {df.dtypes}")
```

### **Issue 3: Azure OpenAI Errors**
```python
# Implement retry logic
@retry_with_backoff(max_retries=3)
def safe_api_call(prompt):
    try:
        return client.chat.completions.create(...)
    except openai.RateLimitError:
        time.sleep(60)  # Wait 1 minute
        raise
```

### **Issue 4: Memory Issues**
```python
# Optimize large dataset handling
@st.cache_data(ttl=3600, max_entries=10)
def load_data_optimized():
    # Load only required columns
    # Use categorical data types
    # Sample large datasets if needed
    pass
```

---

## 📊 **Data Understanding**

### **Core Datasets (Must Know)**
1. **Financial_Cleaned.csv** (9.5MB)
   - Main financial transactions and allocations
   - Countries, regions, themes, funding amounts
   - Years 2020-2026 with Required/Available/Expenditure

2. **SDG_Goals_Cleaned.csv** (1.4MB)  
   - SDG goal mappings for projects
   - Used for SDG prediction model

3. **UN_Agencies_Cleaned.csv** (1.4MB)
   - UN agency collaboration data
   - Used for agency recommendation model

### **Model Outputs (Generated)**
1. **SDG_model.pkl** (11MB) - XGBoost classifier for SDG predictions
2. **Agency_model.pkl** (33MB) - XGBoost classifier for agency recommendations
3. **funding_prediction.csv** (9.5MB) - 2026 funding forecasts
4. **anomaly_detection.csv** (9.9MB) - Anomaly scores and flags
5. **un_agency.csv** (9.6MB) - Agency performance clusters

---

## 🔑 **Key Configuration Files**

### **`.streamlit/secrets.toml`** (CRITICAL - Never commit!)
```toml
# Required for AI features
AZURE_OPENAI_4O_API_KEY = "your-key"
AZURE_OPENAI_4O_ENDPOINT = "your-endpoint"  
AZURE_OPENAI_4O_DEPLOYMENT = "gpt-4o"

AZURE_OPENAI_O1_API_KEY = "your-key"
AZURE_OPENAI_O1_ENDPOINT = "your-endpoint"
AZURE_OPENAI_O1_DEPLOYMENT = "o1"
```

### **`requirements.txt`** (Deployment-Critical)
```bash
# Core framework
streamlit>=1.47.0,<2.0.0
openai>=1.35.0,<2.0.0

# ML and data
pandas>=2.0.0,<2.3.0
scikit-learn>=1.3.0,<1.6.0
xgboost>=1.7.0,<3.0.0
plotly>=5.15.0,<6.0.0
```

---

## 🚀 **Deployment Guide**

### **Streamlit Cloud (Current Production)**
1. **Repository**: https://github.com/ZhaoJackson/United_Nations_Legacy
2. **Live URL**: https://united-nations-legacy.streamlit.app/
3. **Deployment**: Automatic on main branch push
4. **Secrets**: Configured in Streamlit Cloud dashboard

### **Emergency Deployment Procedure**
```bash
# If production is down
1. Check Streamlit Cloud status
2. Verify GitHub repository health  
3. Check Azure OpenAI service status
4. Restart application in Streamlit Cloud
5. If critical: Deploy to backup environment
```

### **Backup Deployment (Docker)**
```bash
# Quick Docker deployment
docker build -t un-dashboard .
docker run -p 8501:8501 un-dashboard
```

---

## 📈 **Success Metrics**

### **User Experience Metrics**
- **Page Load Time**: <5 seconds (currently ~3s)
- **Monthly Active Users**: Track via Streamlit analytics
- **Session Duration**: Target >10 minutes average
- **Error Rate**: <1% of user sessions

### **Technical Health Metrics**
- **Uptime**: >99.5% (monitor via uptimeRobot)
- **Memory Usage**: <2GB peak usage
- **API Response Time**: <10 seconds for AI features
- **Data Freshness**: Update monthly with new UN data

---

## 🎯 **Next Steps Action Plan**

### **Week 1-2: Stabilization**
- [ ] Complete environment setup and testing
- [ ] Review all documentation thoroughly  
- [ ] Test all features and identify any immediate issues
- [ ] Set up monitoring and alerting
- [ ] Create backup deployment environment

### **Month 1: Enhancement**
- [ ] Implement performance optimizations
- [ ] Add comprehensive error handling
- [ ] Improve mobile responsiveness
- [ ] Add user analytics tracking
- [ ] Create automated testing suite

### **Month 2-3: Growth**
- [ ] Integrate new data sources
- [ ] Enhance AI capabilities
- [ ] Add advanced visualization features
- [ ] Implement user feedback system
- [ ] Plan database migration strategy

---

## 🆘 **Support & Resources**

### **Immediate Help**
- **Original Developer**: Zichen Zhao (Jackson)
- **Email**: ziche.zhao@un.org | zichen.zhao@columbia.edu  
- **GitHub Issues**: https://github.com/ZhaoJackson/United_Nations_Legacy/issues

### **Technical Resources**
- **Streamlit Docs**: https://docs.streamlit.io/
- **Azure OpenAI**: https://azure.microsoft.com/en-us/products/ai-services/openai-service
- **UN Data Source**: https://uninfo.org/data-explorer/cooperation-framework/activity-report

### **Internal Documentation**
- **Architecture**: [docs/04_ARCHITECTURE.md](./04_ARCHITECTURE.md)
- **API Reference**: [docs/11_API_REFERENCE.md](./11_API_REFERENCE.md)
- **Troubleshooting**: [docs/14_TROUBLESHOOTING.md](./14_TROUBLESHOOTING.md)
- **Development Workflow**: [docs/06_DEVELOPMENT_WORKFLOW.md](./06_DEVELOPMENT_WORKFLOW.md)

---

## ✅ **Handoff Checklist**

### **Technical Handoff**
- [ ] Repository access granted
- [ ] Azure OpenAI credentials transferred
- [ ] Streamlit Cloud admin access provided
- [ ] Local environment successfully configured
- [ ] All 5 dashboard pages tested and working
- [ ] AI features functional with provided credentials

### **Knowledge Transfer**
- [ ] Architecture overview session completed
- [ ] Data pipeline walkthrough completed  
- [ ] AI integration explanation provided
- [ ] Common issues and solutions reviewed
- [ ] Future roadmap discussion held

### **Operational Handoff**
- [ ] Monitoring and alerting setup
- [ ] Backup procedures documented
- [ ] Emergency contacts established
- [ ] Maintenance schedule agreed upon
- [ ] Performance benchmarks established

---

**🎯 Ready to Begin**: Start with the [Quick Start Guide](./02_QUICK_START.md), then dive into [Architecture Overview](./04_ARCHITECTURE.md) to understand the system design. The comprehensive documentation will guide you through every aspect of the platform.

**🚨 Critical**: Ensure Azure OpenAI credentials are properly configured - this enables 80% of the platform's AI features. Without them, the dashboard will still function but with limited intelligence capabilities.