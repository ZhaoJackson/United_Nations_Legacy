# commonconst.py
from openai import AzureOpenAI
import numpy as np
from pathlib import Path
import streamlit as st
import pandas as pd
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from xgboost import XGBRegressor
import pycountry
from pathlib import Path

# Azure OpenAI Credentials - with fallback for missing secrets
try:
    client_4o = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_4O_API_KEY"],
        api_version=st.secrets["AZURE_OPENAI_4O_API_VERSION"],
        azure_endpoint=st.secrets["AZURE_OPENAI_4O_ENDPOINT"]
    )
    DEPLOYMENT_4O = st.secrets["AZURE_OPENAI_4O_DEPLOYMENT"]
    
    client_o1 = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_O1_API_KEY"],
        api_version=st.secrets["AZURE_OPENAI_O1_API_VERSION"],
        azure_endpoint=st.secrets["AZURE_OPENAI_O1_ENDPOINT"]
    )
    DEPLOYMENT_O1 = st.secrets["AZURE_OPENAI_O1_DEPLOYMENT"]
    
    # Flag to indicate if AI services are available
    AI_SERVICES_AVAILABLE = True
    
except KeyError as e:
    st.warning(f"⚠️ AI services not configured: Missing secret {e}. Some features may be limited.")
    
    # Create dummy clients to prevent import errors
    client_4o = None
    client_o1 = None
    DEPLOYMENT_4O = "dummy"
    DEPLOYMENT_O1 = "dummy"
    AI_SERVICES_AVAILABLE = False

# Directory paths
FINANCIAL_PATH = "src/outputs/data_output/Financial_Cleaned.csv"
UN_AGENCIES_PATH = "src/outputs/data_output/UN_Agencies_Cleaned.csv"
SDG_GOALS_PATH = "src/outputs/data_output/SDG_Goals_Cleaned.csv"

# Model output paths
FUNDING_PREDICTION_PATH = "src/outputs/model_output/funding_prediction.csv"
ANOMALY_DETECTION_PATH = "src/outputs/model_output/anomaly_detection.csv"
UN_AGENCY_PERFORMANCE_PATH = "src/outputs/model_output/un_agency.csv"

# Model file paths
SDG_MODEL_PATH = "src/outputs/model_output/SDG_model.pkl"
AGENCY_MODEL_PATH = "src/outputs/model_output/Agency_model.pkl"

# Style and layout constants
STYLE_CSS_PATH = "pages/style/style.css"

# Page configuration constants
PAGE_CONFIG = {
    "page_title": "UN Financial Intelligence Dashboard",
    "page_icon": "🇺🇳",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        'Get Help': 'https://github.com/your-username/united-nations-legacy',
        'Report a bug': "https://github.com/your-username/united-nations-legacy/issues",
        'About': """
        # UN Financial Intelligence Dashboard
        
        This dashboard provides comprehensive analysis of UN JointWork Plans financial data, 
        enabling insights into funding patterns, resource allocation, and collaboration trends 
        across different regions, themes, and UN agencies.
        
        **Key Features:**
        - 💰 Funding gap analysis and predictions
        - 🌍 Regional and country-specific insights  
        - 🎯 Thematic area analysis
        - 🏢 UN agency collaboration patterns
        - 🤖 AI-powered chat assistant
        """
    }
}

# Social Media Meta Tags for LinkedIn sharing
def get_social_meta_tags(app_url="https://your-app-name.streamlit.app/"):
    """Generate social media meta tags with custom app URL"""
    # Create a professional preview image using a service like og-image or similar
    og_image_url = "https://og-image.vercel.app/**UN%20Financial%20Intelligence**%20Dashboard.png?theme=light&md=1&fontSize=100px&images=https%3A%2F%2Fassets.vercel.com%2Fimage%2Fupload%2Ffront%2Fassets%2Fdesign%2Fvercel-triangle-black.svg"
    
    return f"""
<meta property="og:title" content="UN Financial Intelligence Dashboard - JointWork Plans Analysis">
<meta property="og:description" content="Comprehensive analysis of UN JointWork Plans financial data with AI-powered insights into funding patterns, resource allocation, and collaboration trends across regions and themes.">
<meta property="og:type" content="website">
<meta property="og:url" content="{app_url}">
<meta property="og:image" content="{og_image_url}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:site_name" content="UN Financial Intelligence Dashboard">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="UN Financial Intelligence Dashboard">
<meta name="twitter:description" content="AI-powered analysis of UN JointWork Plans financial data with comprehensive insights into funding patterns and collaboration trends.">
<meta name="twitter:image" content="{og_image_url}">

<meta name="description" content="UN Financial Intelligence Dashboard provides comprehensive analysis of JointWork Plans financial data, enabling insights into funding patterns, resource allocation, and collaboration trends across different regions, themes, and UN agencies.">
<meta name="keywords" content="UN, United Nations, Financial Analysis, JointWork Plans, Funding, Resource Allocation, Dashboard, AI, Analytics">
<meta name="author" content="UN Financial Intelligence Team">

<link rel="canonical" href="{app_url}">
"""

# Default social media meta tags (will be updated when app is deployed)
SOCIAL_META_TAGS = get_social_meta_tags()

# Year ranges for analysis
PLOT_YEAR_RANGE = list(range(2020, 2026))
DEFAULT_YEAR_INDEX = len(PLOT_YEAR_RANGE) - 1

# Color schemes and styling
UN_BLUE_GRADIENT = ["#f0f9ff", "#e0f2fe", "#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9", "#0284c7", "#0369a1", "#075985", "#0c4a6e"]

# Performance color mapping
PERFORMANCE_COLORS = {
    'Top Performer': '#22c55e',
    'Moderate Performer': '#3b82f6', 
    'Execution Gap': '#f59e0b',
    'Low Performer': '#dc2626'
}

# Metric card styling
METRIC_CARD_COLORS = {
    'required': '#dc2626',
    'available': '#009edb',
    'expenditure': '#22c55e',
    'countries': '#6366f1',
    'coverage': '#059669',
    'gap': '#dc2626',
    'agencies': '#0891b2',
    'projects': '#7c3aed'
}

# Chat interface constants
CHAT_WELCOME_MESSAGE = '''
<h4 style="color: #7c3aed; margin: 0 0 1rem 0;">👋 Welcome to the UN Financial Intelligence Assistant!</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; line-height: 1.6;">
    I'm here to help you analyze UN JointWork Plans financial data. You can ask me questions about:
</p>
<ul style="color: #64748b; margin: 1rem 0; padding-left: 1.5rem; line-height: 1.6;">
    <li>💰 <strong>Funding patterns</strong> and resource allocation</li>
    <li>🌍 <strong>Regional and country-specific</strong> insights</li>
    <li>🎯 <strong>Thematic area</strong> analysis</li>
    <li>🏢 <strong>UN agency</strong> collaboration patterns</li>
    <li>📊 <strong>Trends and comparisons</strong> across time periods</li>
</ul>
<p style="color: #64748b; margin: 1rem 0 0.5rem 0; font-weight: 600;">Try asking questions like:</p>
<ul style="color: #64748b; margin: 0.5rem 0; padding-left: 1.5rem; font-style: italic;">
    <li>"What are the funding gaps in Africa for education?"</li>
    <li>"Which agencies work most on climate action?"</li>
    <li>"Show me resource allocation trends for governance projects"</li>
</ul>
'''

# Quick question templates for chatbot
QUICK_QUESTIONS = [
    "What are the top funding gaps by region?",
    "Which themes have the highest resource requirements?", 
    "Show me agency collaboration patterns",
    "What are the latest funding trends?"
]

# Dynamic Data Discovery Functions
def discover_available_themes_from_filesystem():
    """Discover all available themes from the regional data directories"""
    themes_set = set()
    data_dir = Path("src/data")
    
    if data_dir.exists():
        for region_dir in data_dir.iterdir():
            if region_dir.is_dir() and not region_dir.name.startswith('.'):
                for theme_file in region_dir.iterdir():
                    if theme_file.suffix == '.xlsx' and not theme_file.name.startswith('.'):
                        theme_name = theme_file.stem
                        themes_set.add(theme_name)
    
    return sorted(list(themes_set))

def discover_available_regions_from_filesystem():
    """Discover all available regions from the data directory structure"""
    regions_set = set()
    data_dir = Path("src/data")
    
    if data_dir.exists():
        for region_dir in data_dir.iterdir():
            if region_dir.is_dir() and not region_dir.name.startswith('.'):
                regions_set.add(region_dir.name)
    
    return sorted(list(regions_set))

def get_theme_file_mapping():
    """Create a mapping of themes to their file paths across regions"""
    theme_mapping = {}
    data_dir = Path("src/data")
    
    if data_dir.exists():
        for region_dir in data_dir.iterdir():
            if region_dir.is_dir() and not region_dir.name.startswith('.'):
                region_name = region_dir.name
                for theme_file in region_dir.iterdir():
                    if theme_file.suffix == '.xlsx' and not theme_file.name.startswith('.'):
                        theme_name = theme_file.stem
                        if theme_name not in theme_mapping:
                            theme_mapping[theme_name] = {}
                        theme_mapping[theme_name][region_name] = str(theme_file)
    
    return theme_mapping

# Safe data loading with error handling and fallback data
def safe_load_csv(file_path, default_df=None, use_fallback=True):
    """Safely load CSV files with error handling and fallback data"""
    try:
        if Path(file_path).exists():
            return pd.read_csv(file_path)
        else:
            if use_fallback:
                # Try to use fallback data for common file types
                if "Financial_Cleaned.csv" in file_path:
                    from src.fallback_data import get_fallback_data, create_demo_message
                    st.info("📋 Using sample financial data for demonstration purposes.")
                    return get_fallback_data('financial')
                elif "SDG_Goals_Cleaned.csv" in file_path:
                    from src.fallback_data import get_fallback_data
                    st.info("📋 Using sample SDG data for demonstration purposes.")
                    return get_fallback_data('sdg')
                elif "UN_Agencies_Cleaned.csv" in file_path:
                    from src.fallback_data import get_fallback_data
                    st.info("📋 Using sample agency data for demonstration purposes.")
                    return get_fallback_data('agency')
            
            st.warning(f"Data file not found: {file_path}. Using empty DataFrame.")
            return default_df if default_df is not None else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return default_df if default_df is not None else pd.DataFrame()

# Load Data with error handling
financial_df = safe_load_csv(FINANCIAL_PATH)
un_agencies_df = safe_load_csv(UN_AGENCIES_PATH)
sdg_goals_df = safe_load_csv(SDG_GOALS_PATH)

# Load Model Outputs with error handling
funding_prediction_df = safe_load_csv(FUNDING_PREDICTION_PATH)
anomaly_detection_df = safe_load_csv(ANOMALY_DETECTION_PATH)
un_agency_performance_df = safe_load_csv(UN_AGENCY_PERFORMANCE_PATH)

# Funding Gap Calculation
for year in range(2020, 2027):
    req_col = f"{year} Required"
    avail_col = f"{year} Available"
    if req_col in financial_df.columns and avail_col in financial_df.columns:
        financial_df[f"{year} Gap"] = financial_df[req_col] - financial_df[avail_col]

# ---------- Page 1 ----------
# Extract dynamic year-based columns
financial_year_cols = [col for col in financial_df.columns if any(year in col for year in ['2023', '2024', '2025', '2026'])]
required_cols = [col for col in financial_year_cols if "Required" in col]
available_cols = [col for col in financial_year_cols if "Available" in col]
expenditure_cols = [col for col in financial_year_cols if "Expenditure" in col]

def get_country_list():
    return sorted(financial_df["Country"].dropna().unique())

def get_theme_list():
    """Get comprehensive theme list from both data files and filesystem"""
    themes_from_data = set()
    themes_from_filesystem = set(discover_available_themes_from_filesystem())
    
    # Get themes from financial data if available
    if not financial_df.empty and "Theme" in financial_df.columns:
        themes_from_data = set(financial_df["Theme"].dropna().unique())
    
    # Combine both sources
    all_themes = themes_from_data.union(themes_from_filesystem)
    return sorted(list(all_themes))

def get_region_list():
    """Get comprehensive region list from both data files and filesystem"""
    regions_from_data = set()
    regions_from_filesystem = set(discover_available_regions_from_filesystem())
    
    # Get regions from financial data if available
    if not financial_df.empty and "Region" in financial_df.columns:
        regions_from_data = set(financial_df["Region"].dropna().unique())
    
    # Combine both sources
    all_regions = regions_from_data.union(regions_from_filesystem)
    return sorted(list(all_regions))

def get_available_themes_for_region(region):
    """Get available themes for a specific region"""
    theme_mapping = get_theme_file_mapping()
    available_themes = []
    
    for theme, regions in theme_mapping.items():
        if region in regions:
            available_themes.append(theme)
    
    return sorted(available_themes)

def check_data_completeness():
    """Check which themes and regions have complete data coverage"""
    theme_mapping = get_theme_file_mapping()
    all_regions = get_region_list()
    all_themes = get_theme_list()
    
    completeness_report = {
        'complete_themes': [],
        'partial_themes': [],
        'missing_themes': [],
        'total_files': 0,
        'missing_files': 0
    }
    
    for theme in all_themes:
        if theme in theme_mapping:
            regions_with_theme = set(theme_mapping[theme].keys())
            all_regions_set = set(all_regions)
            
            if regions_with_theme == all_regions_set:
                completeness_report['complete_themes'].append(theme)
            elif regions_with_theme:
                completeness_report['partial_themes'].append(theme)
            else:
                completeness_report['missing_themes'].append(theme)
                
            completeness_report['total_files'] += len(regions_with_theme)
            completeness_report['missing_files'] += len(all_regions_set - regions_with_theme)
        else:
            completeness_report['missing_themes'].append(theme)
    
    return completeness_report

def get_agencies_list():
    """Extract unique UN agencies from the financial data"""
    agencies_set = set()
    for agencies_str in financial_df["Agencies"].dropna().unique():
        if pd.notna(agencies_str) and str(agencies_str).strip():
            # Split by semicolon and comma, then clean up
            agencies = re.split(r'[;,]', str(agencies_str))
            for agency in agencies:
                clean_agency = agency.strip()
                if clean_agency and len(clean_agency) > 3:  # Filter out very short strings
                    agencies_set.add(clean_agency)
    return sorted(list(agencies_set))

def get_sdg_goals_list():
    """Extract unique SDG goals from the financial data"""
    sdg_set = set()
    for sdg_str in financial_df["SDG Goals"].dropna().unique():
        if pd.notna(sdg_str) and str(sdg_str).strip():
            # Split by semicolon and comma, then clean up
            sdgs = re.split(r'[;,]', str(sdg_str))
            for sdg in sdgs:
                clean_sdg = sdg.strip()
                if clean_sdg and len(clean_sdg) > 3:  # Filter out very short strings
                    sdg_set.add(clean_sdg)
    return sorted(list(sdg_set))

def filter_by_agency(df, selected_agency):
    """Filter dataframe by selected UN agency"""
    if selected_agency == "All Agencies":
        return df
    return df[df["Agencies"].str.contains(selected_agency, case=False, na=False)]

def filter_by_sdg(df, selected_sdg):
    """Filter dataframe by selected SDG goal"""
    if selected_sdg == "All SDG Goals":
        return df
    return df[df["SDG Goals"].str.contains(selected_sdg, case=False, na=False)]

# ---------- Page 2 - Model Analysis Functions ----------
def get_performance_labels():
    """Get unique performance labels from agency clustering"""
    return sorted(un_agency_performance_df["Performance_Label"].dropna().unique())

def get_anomaly_countries():
    """Get countries with anomalous strategic priorities"""
    anomalous_data = anomaly_detection_df[anomaly_detection_df["SP_Anomaly_Flag"] == "Yes"]
    return sorted(anomalous_data["Country"].dropna().unique())

def format_currency(value):
    """Format large currency values for display"""
    if pd.isna(value) or value == 0:
        return "$0"
    elif abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.0f}K"
    else:
        return f"${value:.0f}"

# ---------- SDG and Agency Mapping Functions ----------
def get_sdg_mapping():
    """Get mapping of SDG codes to full descriptions"""
    return {
        1: "No Poverty",
        2: "Zero Hunger", 
        3: "Good Health and Well-being",
        4: "Quality Education",
        5: "Gender Equality",
        6: "Clean Water and Sanitation",
        7: "Affordable and Clean Energy",
        8: "Decent Jobs and Economic Growth",
        9: "Industry, Innovation and Infrastructure",
        10: "Reduced Inequalities",
        11: "Sustainable Cities and Communities",
        12: "Responsible Consumption and Production",
        13: "Climate Action",
        14: "Life Below Water",
        15: "Life on Land",
        16: "Peace and Justice and Strong Institution",
        17: "Partnerships for the Goals"
    }

def get_agency_abbreviations():
    """Get common abbreviations for UN agencies"""
    return {
        "United Nations Development Programme": "UNDP",
        "United Nations Children's Fund": "UNICEF",
        "Food and Agriculture Organization of the United Nations": "FAO",
        "United Nations Population Fund": "UNFPA",
        "United Nations Educational, Scientific and Cultural Organisation": "UNESCO",
        "UN Women": "UN Women",
        "International Labour Organisation": "ILO",
        "International Organization for Migration": "IOM",
        "World Health Organization": "WHO",
        "United Nations World Food Programme": "WFP",
        "United Nations High Commissioner for Refugees": "UNHCR",
        "United Nations Office on Drugs and Crime": "UNODC",
        "United Nations Industrial Development Organization": "UNIDO",
        "United Nations Environment Programme": "UNEP",
        "United Nations Human Settlement Programme": "UN-Habitat",
        "United Nations Joint Programme on HIV and AIDS Secretariat": "UNAIDS",
        "United Nations High Commissioner for Human Rights": "OHCHR",
        "United Nations Office for Project Services": "UNOPS",
        "International Fund for Agricultural Development": "IFAD",
        "United Nations Economic Commission for Europe": "UNECE"
    }

def format_sdg_prediction(sdg_name):
    """Format SDG prediction with code and description"""
    sdg_mapping = get_sdg_mapping()
    
    # Try to extract SDG code from the name or find matching description
    for code, description in sdg_mapping.items():
        if description.lower() == sdg_name.lower():
            return f"SDG {code}: {description}"
    
    # If no exact match, return as is but formatted
    return f"SDG: {sdg_name}"

def format_agency_prediction(agency_name):
    """Format agency prediction with abbreviation if available"""
    abbreviations = get_agency_abbreviations()
    
    # Get abbreviation if available
    abbrev = abbreviations.get(agency_name, "")
    
    if abbrev:
        return f"{agency_name} ({abbrev})"
    else:
        return agency_name

# ---------- Model Loading Functions ----------
@st.cache_resource
def load_sdg_model():
    """Load the saved SDG prediction model"""
    import joblib
    try:
        if Path(SDG_MODEL_PATH).exists():
            return joblib.load(SDG_MODEL_PATH)
        else:
            st.warning(f"SDG model not found at {SDG_MODEL_PATH}. Please retrain the model.")
            return None
    except Exception as e:
        st.error(f"Error loading SDG model: {e}")
        return None

@st.cache_resource  
def load_agency_model():
    """Load the saved Agency prediction model"""
    import joblib
    try:
        if Path(AGENCY_MODEL_PATH).exists():
            return joblib.load(AGENCY_MODEL_PATH)
        else:
            st.warning(f"Agency model not found at {AGENCY_MODEL_PATH}. Please retrain the model.")
            return None
    except Exception as e:
        st.error(f"Error loading Agency model: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def refresh_data():
    """Refresh all data sources and return updated dataframes"""
    global financial_df, un_agencies_df, sdg_goals_df
    global funding_prediction_df, anomaly_detection_df, un_agency_performance_df
    
    # Reload main datasets
    financial_df = safe_load_csv(FINANCIAL_PATH)
    un_agencies_df = safe_load_csv(UN_AGENCIES_PATH)
    sdg_goals_df = safe_load_csv(SDG_GOALS_PATH)
    
    # Reload model outputs
    funding_prediction_df = safe_load_csv(FUNDING_PREDICTION_PATH)
    anomaly_detection_df = safe_load_csv(ANOMALY_DETECTION_PATH)
    un_agency_performance_df = safe_load_csv(UN_AGENCY_PERFORMANCE_PATH)
    
    # Recalculate funding gaps
    for year in range(2020, 2027):
        req_col = f"{year} Required"
        avail_col = f"{year} Available"
        if req_col in financial_df.columns and avail_col in financial_df.columns:
            financial_df[f"{year} Gap"] = financial_df[req_col] - financial_df[avail_col]
    
    return {
        'financial_df': financial_df,
        'un_agencies_df': un_agencies_df,
        'sdg_goals_df': sdg_goals_df,
        'funding_prediction_df': funding_prediction_df,
        'anomaly_detection_df': anomaly_detection_df,
        'un_agency_performance_df': un_agency_performance_df
    }

def force_refresh_cache():
    """Force refresh all cached data and models"""
    refresh_data.clear()
    load_sdg_model.clear()
    load_agency_model.clear()
    st.success("Cache cleared! Data and models will be reloaded.")

def get_model_status():
    """Check status of all models and data files"""
    status = {
        'models': {
            'sdg_model': Path(SDG_MODEL_PATH).exists(),
            'agency_model': Path(AGENCY_MODEL_PATH).exists()
        },
        'data_files': {
            'financial_data': Path(FINANCIAL_PATH).exists(),
            'un_agencies_data': Path(UN_AGENCIES_PATH).exists(),
            'sdg_goals_data': Path(SDG_GOALS_PATH).exists(),
            'funding_predictions': Path(FUNDING_PREDICTION_PATH).exists(),
            'anomaly_detection': Path(ANOMALY_DETECTION_PATH).exists(),
            'agency_performance': Path(UN_AGENCY_PERFORMANCE_PATH).exists()
        },
        'themes_available': len(get_theme_list()),
        'regions_available': len(get_region_list()),
        'data_completeness': check_data_completeness()
    }
    return status

def check_system_updates_needed():
    """Check if the system needs updates due to new themes or data changes"""
    try:
        from src.dynamic_analysis import DynamicDataProcessor, DynamicModelManager
        
        # Initialize processors
        data_processor = DynamicDataProcessor()
        model_manager = DynamicModelManager()
        
        # Discover current data structure
        structure = data_processor.discover_data_structure()
        themes = structure['themes']
        
        # Check if models need retraining
        needs_retraining = model_manager.needs_retraining(themes)
        
        # Check data completeness
        completeness = check_data_completeness()
        has_incomplete_data = len(completeness['partial_themes']) > 0 or len(completeness['missing_themes']) > 0
        
        return {
            'needs_update': needs_retraining or has_incomplete_data,
            'needs_retraining': needs_retraining,
            'has_incomplete_data': has_incomplete_data,
            'themes_found': themes,
            'completeness_report': completeness
        }
    except Exception as e:
        return {
            'needs_update': False,
            'error': str(e)
        }

def auto_adapt_to_new_themes():
    """Automatically adapt the system when new themes are detected"""
    try:
        from src.dynamic_analysis import auto_update_analysis
        
        # Run the automatic update analysis
        results = auto_update_analysis()
        
        # Return results for display
        return {
            'success': True,
            'themes_found': results['themes_found'],
            'regions_found': results['regions_found'],
            'processing_time': results['processing_time'],
            'models_need_retraining': results['models_need_retraining'],
            'dashboard_status': results['dashboard_status']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_model_countries():
    """Get countries available for model prediction"""
    # Use the same countries from the cleaned datasets
    sdg_countries = set(sdg_goals_df["Country"].dropna().unique())
    agency_countries = set(un_agencies_df["Country"].dropna().unique())
    return sorted(list(sdg_countries.intersection(agency_countries)))

def get_model_themes():
    """Get themes available for model prediction"""
    # Use the same themes from the cleaned datasets
    sdg_themes = set(sdg_goals_df["Theme"].dropna().unique())
    agency_themes = set(un_agencies_df["Theme"].dropna().unique())
    return sorted(list(sdg_themes.intersection(agency_themes)))

def predict_sdg_goals(model, country, theme, strategic_priority_code):
    """Make prediction using SDG model"""
    if model is None:
        return []
    
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Country': [country],
            'Theme': [theme], 
            'Strategic priority code': [strategic_priority_code]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get predicted classes - handle different model structures
        predicted_classes = []
        
        # Check if it's OneVsRestClassifier
        if hasattr(model.named_steps['classifier'], 'classes_'):
            classes = model.named_steps['classifier'].classes_
            # prediction should be binary array for multi-label
            if len(prediction.shape) > 1:
                pred_binary = prediction[0]
            else:
                pred_binary = prediction
                
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    predicted_classes.append(str(classes[i]))
        
        # Remove any non-string predictions and clean up
        predicted_classes = [str(cls) for cls in predicted_classes if cls is not None and str(cls).strip()]
        
        return predicted_classes
    except Exception as e:
        st.error(f"Error making SDG prediction: {e}")
        return []

def predict_agencies(model, country, theme, strategic_priority_code):
    """Make prediction using Agency model"""
    if model is None:
        return []
    
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Country': [country],
            'Theme': [theme],
            'Strategic priority code': [strategic_priority_code]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get predicted classes - handle different model structures
        predicted_classes = []
        
        # Check if it's OneVsRestClassifier
        if hasattr(model.named_steps['classifier'], 'classes_'):
            classes = model.named_steps['classifier'].classes_
            # prediction should be binary array for multi-label
            if len(prediction.shape) > 1:
                pred_binary = prediction[0]
            else:
                pred_binary = prediction
                
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    predicted_classes.append(str(classes[i]))
        
        # Remove any non-string predictions and clean up
        predicted_classes = [str(cls) for cls in predicted_classes if cls is not None and str(cls).strip()]
        
        return predicted_classes
    except Exception as e:
        st.error(f"Error making Agency prediction: {e}")
        return []

def get_historical_context(country, theme, strategic_priority_code):
    """Get historical context from the datasets for predictions"""
    try:
        # Load the datasets using path constants with safe loading
        funding_df = safe_load_csv(FUNDING_PREDICTION_PATH)
        anomaly_df = safe_load_csv(ANOMALY_DETECTION_PATH)
        agency_df = safe_load_csv(UN_AGENCY_PERFORMANCE_PATH)
        
        # Filter for the specific inputs
        context = {}
        
        # Get matching records from each dataset
        funding_matches = funding_df[
            (funding_df['Country'] == country) & 
            (funding_df['Theme'] == theme) & 
            (funding_df['Strategic priority code'] == strategic_priority_code)
        ]
        
        anomaly_matches = anomaly_df[
            (anomaly_df['Country'] == country) & 
            (anomaly_df['Theme'] == theme) & 
            (anomaly_df['Strategic priority code'] == strategic_priority_code)
        ]
        
        agency_matches = agency_df[
            (agency_df['Country'] == country) & 
            (agency_df['Theme'] == theme) & 
            (agency_df['Strategic priority code'] == strategic_priority_code)
        ]
        
        # Extract historical SDG Goals and Agencies
        if not funding_matches.empty:
            sdg_goals = []
            agencies = []
            
            for _, row in funding_matches.iterrows():
                if pd.notna(row['SDG Goals']):
                    goals = [g.strip() for g in str(row['SDG Goals']).split(';')]
                    sdg_goals.extend(goals)
                if pd.notna(row['Agencies']):
                    agcs = [a.strip() for a in str(row['Agencies']).split(';')]
                    agencies.extend(agcs)
            
            context['historical_sdg_goals'] = list(set(sdg_goals))
            context['historical_agencies'] = list(set(agencies))
            context['funding_data'] = funding_matches
            
        if not anomaly_matches.empty:
            context['anomaly_data'] = anomaly_matches
            context['is_anomalous'] = anomaly_matches['SP_Anomaly_Flag'].iloc[0] == 'Yes'
            
        if not agency_matches.empty:
            context['performance_data'] = agency_matches
            if 'Performance_Label' in agency_matches.columns:
                context['performance_label'] = agency_matches['Performance_Label'].iloc[0]
                
        return context
        
    except Exception as e:
        st.error(f"Error getting historical context: {e}")
        return {}