import sys
import os

# Robust path handling for both local and Streamlit Cloud deployment
def setup_python_path():
    """Setup Python path to ensure src modules can be imported"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from pages/
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
    from src.prompt.models import get_strategic_insights
    from src.dynamic_analysis import DynamicDataProcessor
except ImportError as e:
    print(f"Import error in model.py: {e}")
    # Fallback imports for deployment
    try:
        import streamlit as st
        from commonconst import *
        from prompt.models import get_strategic_insights
        from dynamic_analysis import DynamicDataProcessor
    except ImportError as fallback_error:
        st.error(f"Failed to import required modules: {fallback_error}")
        st.stop()

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
<div class="top-header">
    <h1>🎯 UN AI Strategic Insights Platform</h1>
    <p>AI-Powered Predictions with GPT-4o Strategic Analysis</p>
    <div class="header-features">
        <span>🗺️ Interactive Country Selection • 🤖 Multi-Label Predictions • 🧠 AI Strategic Insights</span>
    </div>
</div>
''', unsafe_allow_html=True)

# Load models
sdg_model = load_sdg_model()
agency_model = load_agency_model()

# ---------- Enhanced Sidebar Controls ----------
st.sidebar.markdown('''
<div style="background: linear-gradient(135deg, #009edb 0%, #006bb6 100%); padding: 1.5rem; margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 15px 15px;">
    <h2 style="color: white; text-align: center; margin: 0; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
        🎯 Model Parameters
    </h2>
    <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 0.5rem 0 0 0; font-size: 0.85rem;">
        Configure prediction inputs
    </p>
</div>
''', unsafe_allow_html=True)

# Input controls for both models
model_countries = get_model_countries()
model_themes = get_model_themes()

selected_country = st.sidebar.selectbox(
    "🌍 Selected Country",
    model_countries,
    help="Choose a country for prediction"
)

selected_theme = st.sidebar.selectbox(
    "🎯 Select Theme",
    model_themes,
    help="Choose a thematic area"
)

strategic_priority_code = st.sidebar.number_input(
    "🔢 Strategic Priority Code",
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=1.0,
    help="Enter strategic priority code (1-10)"
)

st.sidebar.markdown('---')

# Model information
st.sidebar.markdown('''
<div class="model-info">
    <h4 style="color: #166534; margin: 0 0 0.5rem 0; font-weight: 600;">🤖 Model Details</h4>
    <p style="color: #15803d; font-size: 0.75rem; margin: 0; line-height: 1.4;">
        <strong>Algorithm:</strong> XGBoost Multi-Label<br>
        <strong>SDG Model:</strong> F1: 0.522, Hamming Loss: 0.200<br>
        <strong>Agency Model:</strong> F1: 0.491, Hamming Loss: 0.042<br>
        <strong>Features:</strong> Country, Theme, Priority Code<br>
        <strong>AI Analysis:</strong> Azure OpenAI GPT-4o
    </p>
</div>
''', unsafe_allow_html=True)

# GPT-4o Integration info
st.sidebar.markdown('''
<div style="background: linear-gradient(135deg, #fef7ff, #f3e8ff); border-left: 4px solid #a855f7; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
    <h4 style="color: #7c3aed; margin: 0 0 0.5rem 0; font-weight: 600;">🧠 AI Strategic Analysis</h4>
    <p style="color: #6b46c1; font-size: 0.75rem; margin: 0; line-height: 1.4;">
        <strong>Powered by:</strong> Azure OpenAI GPT-4o<br>
        <strong>Analysis:</strong> 5-Section Strategic Review<br>
        <strong>Context:</strong> Historical Data Integration<br>
        <strong>Output:</strong> Actionable Recommendations
    </p>
</div>
''', unsafe_allow_html=True)

# ---------- Main Layout: Country Selection with Embedded Map ----------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('''
    <div class="country-selection-box">
        <h3 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">🗺️ Country Selection & Interactive Map</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display selected country prominently
    st.markdown(f'''
    <div class="prediction-form">
        <h4 style="color: #1e40af; margin: 0 0 1rem 0; text-align: center;">🌍 Selected Country</h4>
        <p style="color: #374151; font-size: 1.5rem; font-weight: 600; margin: 0; text-align: center; background: linear-gradient(135deg, #dbeafe, #bfdbfe); padding: 1rem; border-radius: 8px;">{selected_country}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Create and display world map for country selection
    @st.cache_data
    def load_country_data():
        try:
            funding_df = pd.read_csv("src/outputs/model_output/funding_prediction.csv")
            countries = funding_df['Country'].unique()
            country_data = []
            for country in countries:
                try:
                    iso_code = pycountry.countries.lookup(country).alpha_3
                    country_data.append({'Country': country, 'ISO3': iso_code, 'Available': 1})
                except:
                    pass
            return pd.DataFrame(country_data)
        except:
            return pd.DataFrame({'Country': [], 'ISO3': [], 'Available': []})

    country_map_data = load_country_data()

    if not country_map_data.empty:
        fig_map = px.choropleth(
            country_map_data,
            locations="ISO3",
            color="Available",
            hover_name="Country",
            title="Click on a Country for Analysis",
            color_continuous_scale=["#e0f2fe", "#009edb"],
            labels={"Available": "Available for Analysis"}
        )
        
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=350,
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_showscale=False
        )
        
        selected_country_map = st.plotly_chart(fig_map, use_container_width=True, key="country_map")
    
    # Prediction button
    if st.button("🎯 Generate AI Predictions & Strategic Insights", type="primary", use_container_width=True):
        with st.spinner("🤖 Running AI models and generating strategic insights..."):
            try:
                # Get predictions
                predicted_sdgs = predict_sdg_goals(sdg_model, selected_country, selected_theme, strategic_priority_code)
                predicted_agencies = predict_agencies(agency_model, selected_country, selected_theme, strategic_priority_code)
                
                # Get historical context
                context = get_historical_context(selected_country, selected_theme, strategic_priority_code)
                
                # Generate GPT-4o strategic insights
                st.info("🧠 Generating Azure OpenAI GPT-4o strategic analysis...")
                strategic_insights = get_strategic_insights(
                    selected_country, selected_theme, strategic_priority_code,
                    predicted_sdgs, predicted_agencies, context
                )
                
                # Store in session state
                st.session_state.predicted_sdgs = predicted_sdgs
                st.session_state.predicted_agencies = predicted_agencies
                st.session_state.historical_context = context
                st.session_state.strategic_insights = strategic_insights
                
                st.success("✅ AI predictions and strategic insights generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("Please try again or check the configuration.")

with col2:
    # Display predictions
    if 'predicted_sdgs' in st.session_state or 'historical_context' in st.session_state:
        context = st.session_state.get('historical_context', {})
        
        # SDG Goals Section
        st.markdown("#### 🎯 Predicted SDG Goals")
        
        if 'predicted_sdgs' in st.session_state and st.session_state.predicted_sdgs:
            # Use historical data if available, otherwise use model predictions
            if context.get('historical_sdg_goals'):
                sdg_display = context['historical_sdg_goals']
                st.markdown("*Based on historical collaboration patterns:*")
            else:
                sdg_display = st.session_state.predicted_sdgs
                st.markdown("*AI model predictions:*")
            
            sdg_tags_html = ""
            for i, sdg in enumerate(sdg_display, 1):
                formatted_sdg = format_sdg_prediction(sdg)
                sdg_tags_html += f'<span class="sdg-tag">🎯 {formatted_sdg}</span>'
            
            st.markdown(f'<div style="margin: 1rem 0;">{sdg_tags_html}</div>', unsafe_allow_html=True)
            
            # Detailed breakdown
            for i, sdg in enumerate(sdg_display, 1):
                formatted_sdg = format_sdg_prediction(sdg)
                st.markdown(f"**{i}.** {formatted_sdg}")
        else:
            st.info("🎯 Click 'Generate AI Predictions' to see SDG goal recommendations")
        
        st.markdown("---")
        
        # UN Agencies Section  
        st.markdown("#### 🏢 Recommended UN Agencies")
        
        if 'predicted_agencies' in st.session_state or context.get('historical_agencies'):
            # Use historical data if available, otherwise use model predictions
            if context.get('historical_agencies'):
                agency_display = context['historical_agencies']
                st.markdown("*Based on historical collaboration patterns:*")
            else:
                agency_display = st.session_state.get('predicted_agencies', [])
                st.markdown("*AI model predictions:*")
            
            if agency_display:
                agency_tags_html = ""
                for i, agency in enumerate(agency_display, 1):
                    formatted_agency = format_agency_prediction(str(agency))
                    display_name = formatted_agency[:60] + "..." if len(formatted_agency) > 60 else formatted_agency
                    agency_tags_html += f'<span class="agency-tag">🏢 {display_name}</span>'
                
                st.markdown(f'<div style="margin: 1rem 0;">{agency_tags_html}</div>', unsafe_allow_html=True)
                
                # Detailed breakdown
                for i, agency in enumerate(agency_display, 1):
                    formatted_agency = format_agency_prediction(str(agency))
                    st.markdown(f"**{i}.** {formatted_agency}")
            else:
                st.info("🏢 No agency recommendations available")
        else:
            st.info("🏢 Click 'Generate AI Predictions' to see agency recommendations")
    else:
        st.markdown('''
        <div class="prediction-card">
            <h4 style="color: #6b7280; margin: 0 0 0.5rem 0;">🤖 Ready for AI Predictions</h4>
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">
                Click the "Generate AI Predictions & Strategic Insights" button to get SDG goals, 
                UN agency recommendations, and GPT-4o strategic analysis based on your selected parameters.
            </p>
        </div>
        ''', unsafe_allow_html=True)

# ---------- GPT-4o Strategic Insights Section ----------
if 'strategic_insights' in st.session_state:
    st.markdown("---")
    st.markdown('''
    <div class="strategic-insights">
        <h2 style="color: #7c3aed; margin: 0 0 1rem 0; font-weight: 600;">🧠 GPT-4o Strategic Analysis</h2>
        <p style="color: #64748b; margin: 0 0 1rem 0; line-height: 1.6;">
            Advanced AI analysis combining model predictions with historical context to provide actionable strategic insights.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display the strategic insights
    st.markdown(st.session_state.strategic_insights)
    
    # Additional context metrics if available
    if 'historical_context' in st.session_state:
        context = st.session_state.historical_context
        
        st.markdown("---")
        st.markdown("### 📊 Supporting Data Context")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3, gap="medium")
        
        with insight_col1:
            anomaly_status = "🚨 Anomalous" if context.get('is_anomalous', False) else "✅ Normal"
            st.markdown(f'''
            <div class="insight-card" style="border-left: 4px solid {'#dc2626' if context.get('is_anomalous', False) else '#22c55e'};">
                <h4 style="color: {'#dc2626' if context.get('is_anomalous', False) else '#22c55e'}; margin: 0;">🔍 Anomaly Status</h4>
                <p style="color: #374151; font-size: 1.1rem; font-weight: bold; margin: 0.5rem 0;">{anomaly_status}</p>
                <small style="color: #64748b;">Funding pattern analysis</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with insight_col2:
            performance = context.get('performance_label', 'Unknown')
            color = PERFORMANCE_COLORS.get(performance, '#64748b')
            
            st.markdown(f'''
            <div class="insight-card" style="border-left: 4px solid {color};">
                <h4 style="color: {color}; margin: 0;">📈 Performance Level</h4>
                <p style="color: #374151; font-size: 1.1rem; font-weight: bold; margin: 0.5rem 0;">{performance}</p>
                <small style="color: #64748b;">Historical efficiency rating</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with insight_col3:
            funding_data = context.get('funding_data')
            if funding_data is not None and not funding_data.empty:
                total_required = funding_data['Total required resources'].sum()
                st.markdown(f'''
                <div class="insight-card" style="border-left: 4px solid #3b82f6;">
                    <h4 style="color: #3b82f6; margin: 0;">💰 Total Resources</h4>
                    <p style="color: #374151; font-size: 1.1rem; font-weight: bold; margin: 0.5rem 0;">{format_currency(total_required)}</p>
                    <small style="color: #64748b;">Required funding identified</small>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="insight-card" style="border-left: 4px solid #64748b;">
                    <h4 style="color: #64748b; margin: 0;">💰 No Data</h4>
                    <p style="color: #374151; font-size: 1.1rem; font-weight: bold; margin: 0.5rem 0;">N/A</p>
                    <small style="color: #64748b;">No funding data available</small>
                </div>
                ''', unsafe_allow_html=True)

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
            🧠 <strong>UN AI-Powered Strategic Intelligence Platform</strong>
        </p>
        <p style='margin: 0.5rem 0; color: #475569;'>
            XGBoost Predictions | GPT-4o Strategic Analysis | Interactive Decision Support
        </p>
        <p style='font-size: 0.85rem; color: #64748b; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,158,219,0.1);'>
            Advanced AI Models: Multi-Label Classification + Large Language Model Strategic Insights
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
