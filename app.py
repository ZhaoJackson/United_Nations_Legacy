from src.commonconst import *

# ---------------- Navigation ----------------
overview_page = st.Page("pages/overview.py", title="OVERVIEW", icon="🌟")
main_page = st.Page("pages/main_page.py", title="DASHBOARD", icon="🏠")
prediction_page = st.Page("pages/prediction.py", title="ANALYSIS", icon="📊")
model_page = st.Page("pages/model.py", title="MODELS", icon="🎯")
bot_page = st.Page("pages/bot.py", title="CHATBOT", icon="🤖")
# setting = st.Page("pages/help.py", title="HELP", icon="❓") #To Do, Doing, Done

# Set page layout to wide
pg = st.navigation([overview_page, main_page, prediction_page, model_page, bot_page])
pg.run()