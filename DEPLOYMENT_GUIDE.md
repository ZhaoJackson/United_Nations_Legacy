# 🚀 Deployment Guide

## ⚡ Quick Deploy (Current Setup)

The application is configured to deploy with a single `requirements.txt` file containing all dependencies and configuration.

### **Streamlit Cloud Deployment**
1. **Connect Repository**: Link your GitHub repository to Streamlit Cloud
2. **Deploy**: Streamlit Cloud will automatically detect `requirements.txt`
3. **Configure Secrets**: Add your Azure OpenAI credentials in Streamlit Cloud settings

---

## 🔧 If Deployment Fails

If Streamlit Cloud deployment fails with the consolidated configuration, you may need to restore the separate configuration files:

### **Create runtime.txt**
```bash
echo "python-3.11.9" > runtime.txt
```

### **Create packages.txt**  
```bash
cat > packages.txt << EOF
build-essential
python3-dev
EOF
```

### **Keep .python-version (Optional for local development)**
```bash
echo "3.11.9" > .python-version
```

---

## 📋 Configuration Files Purpose

| File | Purpose | Required For |
|------|---------|-------------|
| `requirements.txt` | Python packages | All deployments |
| `runtime.txt` | Python version specification | Streamlit Cloud |
| `packages.txt` | System-level dependencies (apt packages) | Streamlit Cloud |
| `.python-version` | Local development Python version | pyenv users |

---

## 🌐 Alternative Deployment Options

### **Local Development**
```bash
pip install -r requirements.txt
streamlit run app.py
```

### **Docker Deployment**
```dockerfile
FROM python:3.11.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### **Heroku Deployment**
For Heroku, you'll need:
- `runtime.txt` with Python version
- `Procfile` with: `web: streamlit run app.py --server.port=$PORT`

---

## 🔐 Environment Variables

Ensure these secrets are configured in your deployment platform:

```bash
AZURE_OPENAI_4O_API_KEY=your-key-here
AZURE_OPENAI_4O_API_VERSION=2024-02-15-preview  
AZURE_OPENAI_4O_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_4O_DEPLOYMENT=your-deployment-name

AZURE_OPENAI_O1_API_KEY=your-key-here
AZURE_OPENAI_O1_API_VERSION=2024-09-01-preview
AZURE_OPENAI_O1_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_O1_DEPLOYMENT=your-deployment-name
```

---

## 🆘 Troubleshooting

### **Common Issues**

1. **Missing Python Version**
   - **Error**: No Python version specified
   - **Solution**: Create `runtime.txt` with `python-3.11.9`

2. **System Dependencies Missing**
   - **Error**: Package installation fails  
   - **Solution**: Create `packages.txt` with system dependencies

3. **Memory Issues**
   - **Error**: Out of memory during deployment
   - **Solution**: Consider reducing ML model complexity or use model loading optimization

4. **Secrets Not Found**
   - **Error**: KeyError for Azure OpenAI credentials
   - **Solution**: Configure secrets in deployment platform settings

### **Verification Steps**

1. **Local Test**: `streamlit run app.py` 
2. **Dependencies Check**: `pip install -r requirements.txt`
3. **Secrets Test**: Verify API calls work in chatbot
4. **Data Access**: Ensure all required data files are present

---

## 📞 Support

If deployment issues persist:
- Check [Streamlit Community Forum](https://discuss.streamlit.io/)
- Review deployment platform documentation
- Contact: ziche.zhao@un.org 