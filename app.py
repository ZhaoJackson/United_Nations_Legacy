import sys
import os

# Add the current directory to Python path for Streamlit Cloud deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.commonconst import *

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