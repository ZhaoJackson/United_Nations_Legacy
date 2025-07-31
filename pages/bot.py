import sys
import os

# Ensure project root is in Python path for imports
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.commonconst import *
from src.prompt.chatbot import get_chatbot_response
from src.dynamic_analysis import DynamicDataProcessor

# Initialize dynamic data processor
@st.cache_resource
def get_dynamic_processor():
    return DynamicDataProcessor()

# ---------- Enhanced Custom Styles ----------
with open(STYLE_CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Vertical Color Bar ----------
st.markdown('<div class="vertical-color-bar"></div>', unsafe_allow_html=True)

# ---------- Main Content Container ----------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ---------- Enhanced UN Theme Header ----------
st.markdown('''
<div style="background: linear-gradient(135deg, #009edb 0%, #006bb6 50%, #004c8c 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 1.5rem; box-shadow: 0 6px 25px rgba(0,158,219,0.3); border: 1px solid rgba(255,255,255,0.1); margin-top: 0;">
    <div style="text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2.8rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            🤖 UN Financial Intelligence Chatbot
        </h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: 300; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">
            AI-Powered Financial Data Analysis & Strategic Insights
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                💬 Natural Language Queries • 🌍 Regional Filtering • 📊 Real-time Financial Analysis
            </span>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Load financial data
@st.cache_data
def load_financial_data():
    """Load and cache the cleaned financial data"""
    try:
        df = pd.read_csv("src/outputs/data_output/Financial_Cleaned.csv")
        return df
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        return pd.DataFrame()

financial_df = load_financial_data()

# Initialize session state for chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = financial_df

# ---------- Sidebar Filters ----------
st.sidebar.markdown('''
<div style="background: linear-gradient(135deg, #009edb 0%, #006bb6 100%); padding: 1.5rem; margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 15px 15px;">
    <h2 style="color: white; text-align: center; margin: 0; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
        🔍 Data Filters
    </h2>
    <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 0.5rem 0 0 0; font-size: 0.85rem;">
        Filter the data for targeted analysis
    </p>
</div>
''', unsafe_allow_html=True)

# Get unique values for filters using dynamic functions
regions = ['All Regions'] + get_region_list()
themes = ['All Themes'] + get_theme_list()

# Filter controls
selected_region = st.sidebar.selectbox(
    "🌍 Select Region",
    regions,
    help="Filter data by geographical region"
)

selected_theme = st.sidebar.selectbox(
    "🎯 Select Theme",
    themes,
    help="Filter data by thematic area"
)

# Apply filters
filtered_df = financial_df.copy()
if selected_region != 'All Regions':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
if selected_theme != 'All Themes':
    filtered_df = filtered_df[filtered_df['Theme'] == selected_theme]

st.session_state.filtered_data = filtered_df

# Display filter summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Filter Summary")

if not filtered_df.empty:
    total_projects = len(filtered_df)
    total_required = filtered_df['Total required resources'].sum()
    total_available = filtered_df['Total available resources'].sum()
    unique_countries = filtered_df['Country'].nunique()
    unique_agencies = len(filtered_df['Agencies'].dropna().str.split(';').explode().unique())
    
    st.sidebar.markdown(f"""
    **📋 Projects:** {total_projects:,}  
    **🌍 Countries:** {unique_countries:,}  
    **🏢 Agencies:** {unique_agencies:,}  
    **💰 Required:** {total_required:,.0f}  
    **💵 Available:** {total_available:,.0f}  
    **📈 Coverage:** {(total_available/total_required*100):.1f}%
    """)
else:
    st.sidebar.markdown("*No data available for current filters*")

# ---------- Main Chat Interface ----------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat Header
    st.markdown('''
    <div class="chat-header">
        <h3 style="margin: 0; font-size: 1.5rem;">💬 Financial Intelligence Assistant</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
            Ask me anything about UN financial data, projects, and strategic insights
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Chat Messages Area
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_messages:
            # Welcome message
            st.markdown(f'''
            <div class="welcome-message">
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
                <p style="color: #64748b; font-size: 0.9rem; margin: 1rem 0 0 0;">
                    <strong>Current filters:</strong> {selected_region} | {selected_theme}
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Display chat messages
        for i, message in enumerate(st.session_state.chat_messages):
            if message['role'] == 'user':
                st.markdown(f'''
                <div class="user-message">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message['content']}
                </div>
                ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat Input Section
    st.markdown('<div class="chat-input-section">', unsafe_allow_html=True)
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Ask me about UN financial data, funding gaps, country analysis...",
                label_visibility="collapsed"
            )
        
        with col_button:
            send_button = st.form_submit_button("Send 🚀", use_container_width=True)
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to chat
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Show typing indicator
        with st.spinner("🤖 Analyzing data and generating response..."):
            # Get bot response
            bot_response = get_chatbot_response(
                user_input, 
                filtered_df, 
                selected_region, 
                selected_theme
            )
            
            # Add bot response to chat
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': bot_response
            })
        
        # Rerun to update chat display
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Quick Stats Dashboard
    st.markdown("### 📊 Quick Statistics")
    
    if not filtered_df.empty:
        # Key metrics
        total_projects = len(filtered_df)
        total_required = filtered_df['Total required resources'].sum()
        total_available = filtered_df['Total available resources'].sum()
        funding_gap = total_required - total_available
        coverage_ratio = (total_available / total_required * 100) if total_required > 0 else 0
        
        st.markdown(f'''
        <div class="stats-card">
            <h4 style="color: #1e40af; margin: 0;">📋 Total Projects</h4>
            <p style="font-size: 1.8rem; font-weight: bold; color: #374151; margin: 0.5rem 0;">{total_projects:,}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="stats-card">
            <h4 style="color: #dc2626; margin: 0;">💰 Required Resources</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #374151; margin: 0.5rem 0;">${total_required:,.0f}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="stats-card">
            <h4 style="color: #059669; margin: 0;">💵 Available Resources</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #374151; margin: 0.5rem 0;">${total_available:,.0f}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        gap_color = "#dc2626" if funding_gap > 0 else "#059669"
        st.markdown(f'''
        <div class="stats-card" style="border-left: 4px solid {gap_color};">
            <h4 style="color: {gap_color}; margin: 0;">📊 Funding Gap</h4>
            <p style="font-size: 1.3rem; font-weight: bold; color: #374151; margin: 0.5rem 0;">${funding_gap:,.0f}</p>
            <small style="color: #64748b;">Coverage: {coverage_ratio:.1f}%</small>
        </div>
        ''', unsafe_allow_html=True)
        
        # Top themes/regions
        st.markdown("---")
        st.markdown("### 🎯 Top Focus Areas")
        
        if selected_region == 'All Regions':
            top_regions = filtered_df.groupby('Region')['Total required resources'].sum().sort_values(ascending=False).head(5)
            for region, amount in top_regions.items():
                st.markdown(f"**{region}:** ${amount:,.0f}")
        
        if selected_theme == 'All Themes':
            st.markdown("**Top Themes by Funding:**")
            top_themes = filtered_df.groupby('Theme')['Total required resources'].sum().sort_values(ascending=False).head(5)
            for theme, amount in top_themes.items():
                st.markdown(f"• **{theme.title()}:** ${amount:,.0f}")
    
    else:
        st.info("🔍 No data available for current filters. Try adjusting your selection.")
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("### 🚀 Quick Questions")
    
    quick_questions = QUICK_QUESTIONS
    
    for question in quick_questions:
        if st.button(question, key=f"quick_{question}", use_container_width=True):
            # Add to chat and get response
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': question
            })
            
            with st.spinner("🤖 Analyzing..."):
                bot_response = get_chatbot_response(
                    question, 
                    filtered_df, 
                    selected_region, 
                    selected_theme
                )
                
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': bot_response
                })
            
            st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_messages = []
    st.rerun()

# ---------- Close Main Content Container ----------
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Close Main Content Container ----------
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Enhanced Bottom Colored Dots ----------
st.markdown('''
<div class="bottom-dots">
    <span class="dot" style="background-color: #ff6b6b;"></span>
    <span class="dot" style="background-color: #ffa500;"></span>
    <span class="dot" style="background-color: #ffeb3b;"></span>
    <span class="dot" style="background-color: #4caf50;"></span>
    <span class="dot" style="background-color: #2196f3;"></span>
    <span class="dot" style="background-color: #3f51b5;"></span>
    <span class="dot" style="background-color: #9c27b0;"></span>
    <span class="dot" style="background-color: #e91e63;"></span>
    <span class="dot" style="background-color: #795548;"></span>
    <span class="dot" style="background-color: #607d8b;"></span>
    <span class="dot" style="background-color: #ff9800;"></span>
    <span class="dot" style="background-color: #009688;"></span>
    <span class="dot" style="background-color: #8bc34a;"></span>
    <span class="dot" style="background-color: #cddc39;"></span>
    <span class="dot" style="background-color: #ffc107;"></span>
    <span class="dot" style="background-color: #ff5722;"></span>
</div>
''', unsafe_allow_html=True)

# ---------- Enhanced Footer ----------
st.markdown(
    """
    <div class='footer'>
        <p style='font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;'>
            🤖 <strong>UN Financial Intelligence Chatbot</strong>
        </p>
        <p style='margin: 0.5rem 0; color: #475569;'>
            Natural Language Queries | Real-time Analysis | Strategic Financial Insights
        </p>
        <p style='font-size: 0.85rem; color: #64748b; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,158,219,0.1);'>
            Powered by Azure OpenAI GPT-4o | Interactive Data Visualization | Professional Standards
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
