from src.commonconst import *
from src.dynamic_analysis import *

# ---------------- Page Configuration ----------------
st.set_page_config(**PAGE_CONFIG)

# ---------------- Navigation ----------------
overview_page = st.Page("pages/overview.py", title="OVERVIEW", icon="🌟")
main_page = st.Page("pages/main_page.py", title="DASHBOARD", icon="🏠")
prediction_page = st.Page("pages/prediction.py", title="ANALYSIS", icon="📊")
model_page = st.Page("pages/model.py", title="MODELS", icon="🎯")
bot_page = st.Page("pages/bot.py", title="CHATBOT", icon="🤖")

# Run the navigation
pg = st.navigation([overview_page, main_page, prediction_page, model_page, bot_page])
pg.run()