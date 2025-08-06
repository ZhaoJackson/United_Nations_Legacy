# 🚀 Deployment Guide

Comprehensive deployment guide for production environments, focusing on Streamlit Cloud and alternative deployment options.

## 🎯 **Deployment Overview**

The UN Financial Intelligence Dashboard supports multiple deployment strategies:

1. **🌐 Streamlit Cloud** (Recommended) - Zero-config cloud deployment
2. **🐳 Docker** - Containerized deployment for any environment
3. **☁️ Cloud Platforms** - AWS, Azure, GCP deployment
4. **🖥️ Local Server** - On-premises deployment

---

## 🌐 **Streamlit Cloud Deployment (Recommended)**

### **1. Prerequisites**

- GitHub repository with the codebase
- Streamlit Cloud account (free tier available)
- Azure OpenAI credentials
- Access to required data files

### **2. Repository Preparation**

```bash
# Ensure repository is clean and up-to-date
git status
git add .
git commit -m "Prepare for production deployment"
git push origin main

# Verify essential files exist
ls -la requirements.txt
ls -la app.py
ls -la src/outputs/data_output/
ls -la src/outputs/model_output/
```

### **3. Streamlit Cloud Setup**

#### **Step 1: Connect Repository**
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub account
3. Click "New app"
4. Select repository: `United_Nations_Legacy`
5. Set main file path: `app.py`
6. Choose branch: `main`

#### **Step 2: Configure Secrets**
Add the following to Streamlit Cloud secrets management:

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

# Optional: Debug settings
DEBUG = false
```

#### **Step 3: Configure Advanced Settings**

Create `.streamlit/config.toml` for production:

```toml
[global]
developmentMode = false
maxUploadSize = 200

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[client]
caching = true
displayEnabled = true

[logger]
level = "info"

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
```

### **4. Deployment Process**

```bash
# 1. Final verification
python verify_deployment.py

# 2. Create deployment tag
git tag -a v1.0.0-prod -m "Production deployment v1.0.0"
git push origin v1.0.0-prod

# 3. Monitor deployment in Streamlit Cloud dashboard
# - Build logs
# - Runtime metrics
# - Error reports
```

### **5. Post-Deployment Verification**

```bash
# Test all pages load correctly
curl -I https://your-app-name.streamlit.app/

# Test API endpoints work
python scripts/test_production_endpoints.py

# Monitor performance
python scripts/monitor_production.py
```

**Expected Deployment URL**: `https://united-nations-legacy.streamlit.app/`

---

## 🐳 **Docker Deployment**

### **1. Dockerfile Configuration**

Create `Dockerfile`:

```dockerfile
# Use official Python runtime
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/.streamlit

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **2. Docker Compose Setup**

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  un-dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./src/outputs:/app/src/outputs:ro
      - ./.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - un-dashboard
    restart: unless-stopped
```

### **3. Build and Deploy**

```bash
# Build Docker image
docker build -t un-dashboard:latest .

# Run with Docker Compose
docker-compose up -d

# Verify deployment
docker-compose ps
docker-compose logs un-dashboard

# Test access
curl -I http://localhost:8501
```

### **4. Production Docker Setup**

```bash
# Multi-stage build for production
cat > Dockerfile.prod << 'EOF'
# Build stage
FROM python:3.11.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11.9-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Set up application
WORKDIR /app
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
```

---

## ☁️ **Cloud Platform Deployment**

### **1. AWS Deployment**

#### **Using AWS App Runner**

Create `apprunner.yaml`:

```yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
  env:
    - name: PYTHONPATH
      value: /app
run:
  runtime-version: 3.11.9
  command: streamlit run app.py --server.port=8080 --server.address=0.0.0.0
  network:
    port: 8080
    env:
      - name: PORT
        value: "8080"
```

#### **Using AWS ECS with Fargate**

```json
{
  "family": "un-dashboard",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "un-dashboard",
      "image": "your-account.dkr.ecr.region.amazonaws.com/un-dashboard:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PYTHONPATH",
          "value": "/app"
        }
      ],
      "secrets": [
        {
          "name": "AZURE_OPENAI_4O_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:azure-openai-credentials:AZURE_OPENAI_4O_API_KEY::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/un-dashboard",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### **2. Azure Deployment**

#### **Using Azure Container Instances**

```bash
# Create resource group
az group create --name un-dashboard-rg --location eastus

# Create container instance
az container create \
  --resource-group un-dashboard-rg \
  --name un-dashboard \
  --image your-registry.azurecr.io/un-dashboard:latest \
  --dns-name-label un-dashboard-unique \
  --ports 8501 \
  --environment-variables PYTHONPATH=/app \
  --secure-environment-variables \
    AZURE_OPENAI_4O_API_KEY=$AZURE_OPENAI_4O_API_KEY \
  --cpu 2 \
  --memory 4
```

#### **Using Azure App Service**

Create `azure-pipelines.yml`:

```yaml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  dockerRegistryServiceConnection: 'your-registry-connection'
  imageRepository: 'un-dashboard'
  containerRegistry: 'yourregistry.azurecr.io'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    steps:
    - task: Docker@2
      displayName: Build and push an image to container registry
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Deploy
  displayName: Deploy stage
  jobs:
  - deployment: Deploy
    displayName: Deploy
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureWebAppContainer@1
            displayName: 'Azure Web App on Container Deploy'
            inputs:
              azureSubscription: 'your-subscription'
              appName: 'un-dashboard-app'
              containers: '$(containerRegistry)/$(imageRepository):$(tag)'
```

### **3. Google Cloud Platform**

#### **Using Cloud Run**

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/un-dashboard

# Deploy to Cloud Run
gcloud run deploy un-dashboard \
  --image gcr.io/PROJECT-ID/un-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars PYTHONPATH=/app \
  --set-secrets AZURE_OPENAI_4O_API_KEY=azure-openai-key:latest
```

---

## 🖥️ **Local Server Deployment**

### **1. Ubuntu Server Setup**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install system dependencies
sudo apt install git nginx supervisor -y

# Create application user
sudo useradd -m -s /bin/bash streamlit
sudo su - streamlit
```

### **2. Application Setup**

```bash
# Clone repository
git clone https://github.com/ZhaoJackson/United_Nations_Legacy.git
cd United_Nations_Legacy

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets
mkdir -p .streamlit
echo '[secrets]
AZURE_OPENAI_4O_API_KEY = "your-key"' > .streamlit/secrets.toml

# Test application
streamlit run app.py --server.port 8501
```

### **3. Supervisor Configuration**

Create `/etc/supervisor/conf.d/streamlit.conf`:

```ini
[program:streamlit]
command=/home/streamlit/United_Nations_Legacy/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
directory=/home/streamlit/United_Nations_Legacy
user=streamlit
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/streamlit.log
environment=PYTHONPATH="/home/streamlit/United_Nations_Legacy"
```

### **4. Nginx Configuration**

Create `/etc/nginx/sites-available/streamlit`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl restart supervisor
```

---

## 🔒 **Security Configuration**

### **1. HTTPS Setup**

#### **Using Let's Encrypt**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal setup
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### **SSL Nginx Configuration**

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://127.0.0.1:8501;
        # ... other proxy settings
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### **2. Firewall Configuration**

```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### **3. Environment Security**

```bash
# Secure file permissions
chmod 600 .streamlit/secrets.toml
chmod 755 -R src/
chmod 644 requirements.txt

# Create deployment user with limited privileges
sudo useradd -r -s /bin/false streamlit-app
sudo chown -R streamlit-app:streamlit-app /opt/streamlit
```

---

## 📊 **Monitoring & Maintenance**

### **1. Health Checks**

Create `scripts/health_check.py`:

```python
#!/usr/bin/env python3
"""Health check script for production deployment"""

import requests
import sys
import time

def check_app_health(url="http://localhost:8501"):
    """Check if Streamlit app is responding"""
    try:
        response = requests.get(f"{url}/_stcore/health", timeout=10)
        if response.status_code == 200:
            print("✅ Application is healthy")
            return True
        else:
            print(f"❌ Application unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def check_data_files():
    """Verify essential data files exist"""
    required_files = [
        'src/outputs/data_output/Financial_Cleaned.csv',
        'src/outputs/model_output/SDG_model.pkl',
        'src/outputs/model_output/Agency_model.pkl'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return False
    
    print("✅ All data files present")
    return True

if __name__ == "__main__":
    all_checks_passed = all([
        check_app_health(),
        check_data_files()
    ])
    
    sys.exit(0 if all_checks_passed else 1)
```

### **2. Automated Monitoring**

```bash
# Cron job for health monitoring
crontab -e

# Add these lines:
*/5 * * * * /usr/bin/python3 /path/to/health_check.py >> /var/log/health_check.log 2>&1
0 2 * * * /usr/bin/python3 /path/to/backup_script.py >> /var/log/backup.log 2>&1
```

### **3. Log Management**

```bash
# Configure log rotation
sudo cat > /etc/logrotate.d/streamlit << 'EOF'
/var/log/streamlit.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 streamlit streamlit
    postrotate
        supervisorctl restart streamlit
    endscript
}
EOF
```

---

## 🚀 **Performance Optimization**

### **1. Application Optimization**

```python
# Production configuration
import streamlit as st

# Cache configuration for production
@st.cache_data(ttl=3600, max_entries=100)
def load_production_data():
    """Optimized data loading for production"""
    return pd.read_csv('data.csv')

# Session state optimization
def optimize_session_state():
    """Clear unnecessary session state"""
    keys_to_keep = ['user_preferences', 'current_filters']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
```

### **2. Server Optimization**

```bash
# Optimize Python performance
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Optimize memory usage
ulimit -v 4194304  # 4GB virtual memory limit

# Configure Streamlit for production
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
export STREAMLIT_CLIENT_CACHING=true
```

---

## 🔄 **Deployment Automation**

### **1. GitHub Actions CI/CD**

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Streamlit Cloud
      run: |
        # Trigger deployment webhook
        curl -X POST "${{ secrets.STREAMLIT_WEBHOOK_URL }}"
```

### **2. Deployment Script**

Create `scripts/deploy.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Starting deployment process..."

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
python scripts/health_check.py
python scripts/validate_data_integrity.py

# Create backup
echo "💾 Creating backup..."
tar -czf backup-$(date +%Y%m%d-%H%M%S).tar.gz src/outputs/

# Update application
echo "📦 Updating application..."
git pull origin main
pip install -r requirements.txt

# Restart services
echo "🔄 Restarting services..."
sudo supervisorctl restart streamlit
sudo nginx -s reload

# Post-deployment verification
echo "✅ Verifying deployment..."
sleep 10
python scripts/health_check.py

echo "🎉 Deployment completed successfully!"
```

---

**🎯 Next Steps**: Explore [Testing & QA](./13_TESTING_QA.md) for comprehensive testing strategies, or check [Troubleshooting Guide](./14_TROUBLESHOOTING.md) for common deployment issues and solutions.