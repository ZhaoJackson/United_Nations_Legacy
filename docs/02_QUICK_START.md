# 🚀 Quick Start Guide

This guide gets you up and running with the UN Financial Intelligence Dashboard in **under 10 minutes**.

## ⚡ **Prerequisites Check**

Before starting, ensure you have:
- **Python 3.11.9** (recommended) or Python 3.11+
- **Git** for repository cloning
- **Azure OpenAI credentials** (for AI features)
- **10GB+ free disk space** (for data and models)

## 📦 **Step 1: Clone & Setup (2 minutes)**

```bash
# Clone the repository
git clone https://github.com/ZhaoJackson/United_Nations_Legacy.git
cd United_Nations_Legacy

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 **Step 2: Configure Credentials (1 minute)**

Create the secrets configuration file:

```bash
# Create secrets directory
mkdir -p .streamlit

# Create secrets file
touch .streamlit/secrets.toml
```

Add your Azure OpenAI credentials to `.streamlit/secrets.toml`:

```toml
# Azure OpenAI GPT-4o Configuration
AZURE_OPENAI_4O_API_KEY = "your-gpt4o-api-key"
AZURE_OPENAI_4O_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_OPENAI_4O_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_4O_DEPLOYMENT = "gpt-4o"

# Azure OpenAI O1 Configuration  
AZURE_OPENAI_O1_API_KEY = "your-o1-api-key"
AZURE_OPENAI_O1_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_OPENAI_O1_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_O1_DEPLOYMENT = "o1"
```

⚠️ **Important**: Without these credentials, AI features will be disabled but the dashboard will still function.

## 📊 **Step 3: Verify Data Files (1 minute)**

Check that essential data files exist:

```bash
# Check data directory structure
ls -la src/outputs/data_output/

# Expected files:
# Financial_Cleaned.csv (9.5MB)
# SDG_Goals_Cleaned.csv (1.4MB) 
# UN_Agencies_Cleaned.csv (1.4MB)
# unified_data.csv (48MB)

# Check model files
ls -la src/outputs/model_output/

# Expected files:
# SDG_model.pkl (11MB)
# Agency_model.pkl (33MB)
# funding_prediction.csv (9.5MB)
# anomaly_detection.csv (9.9MB)
# un_agency.csv (9.6MB)
```

If files are missing, see [Data Setup Guide](./07_DATA_PIPELINE.md#data-regeneration).

## 🌐 **Step 4: Launch Application (30 seconds)**

```bash
# Launch the Streamlit application
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XXX:8501
```

## ✅ **Step 5: Verify Installation (2 minutes)**

1. **Open Browser**: Navigate to http://localhost:8501
2. **Check Overview Page**: Should load with platform statistics
3. **Test Dashboard**: Navigate to DASHBOARD page, select filters
4. **Verify Data**: Ensure financial data loads in visualizations
5. **Test AI Features**: Try the CHATBOT page (requires Azure credentials)

## 🎯 **What You Should See**

### **Overview Page**
- Platform statistics (projects, countries, funding amounts)
- System capabilities showcase
- Model performance metrics

### **Dashboard Page**  
- Interactive world map with funding data
- KPI metrics (Required/Available/Expenditure)
- Filtering controls (Year/Theme/Region/Agency/SDG)

### **Analysis Page**
- Funding predictions and forecasts
- Anomaly detection results
- Agency performance clustering

### **Models Page**
- Interactive country selection
- Real-time SDG and Agency predictions
- AI-generated strategic insights

### **Chatbot Page**
- Natural language query interface
- Real-time financial statistics
- Quick question templates

## 🔧 **Quick Troubleshooting**

### **Port Already in Use**
```bash
# Kill existing process
pkill -f streamlit
# Or use different port
streamlit run app.py --server.port 8502
```

### **Module Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### **Data Loading Issues**
```bash
# Check file permissions
chmod -R 755 src/outputs/

# Verify file integrity
python -c "import pandas as pd; print(pd.read_csv('src/outputs/data_output/Financial_Cleaned.csv').shape)"
```

### **AI Features Not Working**
- Verify Azure credentials in `.streamlit/secrets.toml`
- Check API key permissions and quotas
- See [AI Integration Guide](./09_AI_INTEGRATION.md) for details

## 🎯 **Next Steps**

Once the application is running:

1. **Explore the Interface**: Navigate through all 5 pages
2. **Test Filtering**: Try different combinations of filters
3. **Ask Questions**: Use the chatbot with sample queries
4. **Check Performance**: Monitor loading times and responsiveness
5. **Review Logs**: Check terminal output for any warnings

## 📚 **Further Reading**

- **Detailed Setup**: [Installation & Setup Guide](./03_INSTALLATION_SETUP.md)
- **Development**: [Development Workflow](./06_DEVELOPMENT_WORKFLOW.md)  
- **Deployment**: [Production Deployment](./12_DEPLOYMENT.md)
- **Troubleshooting**: [Common Issues](./14_TROUBLESHOOTING.md)

## 💡 **Development Tips**

- **Auto-reload**: Streamlit automatically reloads on file changes
- **Debugging**: Add `st.write()` statements for debugging
- **Performance**: Monitor memory usage with large datasets
- **Caching**: Use `@st.cache_data` for expensive operations

---

**🎉 Congratulations!** You now have a fully functional UN Financial Intelligence Dashboard running locally. The platform is ready for development, testing, and deployment.

*Need help? Check the [Troubleshooting Guide](./14_TROUBLESHOOTING.md) or contact the development team.*