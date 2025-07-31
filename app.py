import sys
import os

# Robust path handling for both local and Streamlit Cloud deployment
def setup_python_path():
    """Setup Python path to ensure src modules can be imported"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    src_dir = os.path.join(project_root, 'src')
    
    # Add both project root and src directory to Python path
    paths_to_add = [project_root, src_dir]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# Initialize path setup
setup_python_path()

try:
    from src.commonconst import *
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback import for deployment environments
    import commonconst as cc
    # Copy necessary variables
    PAGE_CONFIG = getattr(cc, 'PAGE_CONFIG', {
        "page_title": "UN Financial Intelligence Dashboard",
        "page_icon": "🇺🇳",
        "layout": "wide"
    })
    get_social_meta_tags = getattr(cc, 'get_social_meta_tags', lambda x: "")

# ---------------- Page Configuration ----------------
st.set_page_config(**PAGE_CONFIG)

# ---------------- Inject Social Media Meta Tags ----------------
# Update this URL after deployment to Streamlit Cloud
APP_URL = "https://your-app-name.streamlit.app/"  # Replace with your actual Streamlit app URL
social_meta_tags = get_social_meta_tags(APP_URL)
st.markdown(social_meta_tags, unsafe_allow_html=True)

# ---------------- Navigation ----------------
overview_page = st.Page("pages/overview.py", title="OVERVIEW", icon="🌟")
main_page = st.Page("pages/main_page.py", title="DASHBOARD", icon="🏠")
prediction_page = st.Page("pages/prediction.py", title="ANALYSIS", icon="📊")
model_page = st.Page("pages/model.py", title="MODELS", icon="🎯")
bot_page = st.Page("pages/bot.py", title="CHATBOT", icon="🤖")
# setting = st.Page("pages/help.py", title="HELP", icon="❓") #To Do, Doing, Done

# Run the navigation
pg = st.navigation([overview_page, main_page, prediction_page, model_page, bot_page])
pg.run()