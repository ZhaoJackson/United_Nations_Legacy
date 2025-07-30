from src.commonconst import *
from src.dynamic_analysis import DynamicDataProcessor

# Initialize dynamic data processor
@st.cache_resource
def get_dynamic_processor():
    return DynamicDataProcessor()

# ---------- Enhanced Custom Styles ----------
with open(STYLE_CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add overview-specific custom CSS
st.markdown("""
<style>
    .vertical-color-bar {
        position: fixed;
        left: 0;
        top: 0;
        width: 8px;
        height: 100vh;
        background: linear-gradient(to bottom, 
            #ff6b6b 0%, #ffa500 14%, #ffeb3b 28%, #4caf50 42%, 
            #2196f3 56%, #3f51b5 70%, #9c27b0 84%, #e91e63 100%);
        z-index: 1000;
    }
    
    .main-content {
        margin-left: 20px;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
        border: 2px solid rgba(0,158,219,0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,158,219,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,158,219,0.25);
        border-color: rgba(0,158,219,0.4);
    }
    
    .capabilities-section {
        background: linear-gradient(135deg, rgba(0,158,219,0.08) 0%, rgba(0,107,182,0.03) 100%);
        border: 1px solid rgba(0,158,219,0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 6px 25px rgba(0,158,219,0.1);
    }
    
    .model-showcase {
        background: linear-gradient(135deg, #fef7ff, #f3e8ff);
        border: 2px solid #a855f7;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(168,85,247,0.15);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(243,244,246,0.8) 100%);
        border: 1px solid rgba(156,163,175,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(156,163,175,0.1);
        transition: transform 0.2s ease;
    }
    
    .stat-item:hover {
        transform: translateY(-2px);
    }

    .tab-content {
        padding: 2rem 0;
    }
    
    .acknowledgement-card {
        background: linear-gradient(135deg, rgba(0,158,219,0.05) 0%, rgba(0,107,182,0.02) 100%);
        border: 2px solid rgba(0,158,219,0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,158,219,0.1);
    }
    
    .prompt-info-card {
        background: linear-gradient(135deg, rgba(124,58,237,0.05) 0%, rgba(91,33,182,0.02) 100%);
        border: 2px solid rgba(124,58,237,0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(124,58,237,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar with Auto-Adapt System ----------
with st.sidebar:
    st.markdown("### 🔧 System Management")
    
    # Check system status for the sidebar
    try:
        system_status = check_system_updates_needed()
        
        if system_status.get('needs_update', False):
            st.warning("⚠️ System update needed")
            st.write(f"Themes: {len(system_status.get('themes_found', []))}")
            st.write(f"Retraining: {'Yes' if system_status.get('needs_retraining', False) else 'No'}")
            
            # Auto-adaptation button in sidebar
            if st.button("🚀 Auto-Adapt System", help="Automatically analyze new themes and update system compatibility", use_container_width=True):
                with st.spinner("Adapting system..."):
                    adaptation_results = auto_adapt_to_new_themes()
                    
                    if adaptation_results.get('success', False):
                        st.success("✅ System Adapted!")
                        st.write(f"Themes: {len(adaptation_results.get('themes_found', []))}")
                        st.write(f"Regions: {len(adaptation_results.get('regions_found', []))}")
                        st.write(f"Time: {adaptation_results.get('processing_time', 0):.2f}s")
                    else:
                        st.error(f"⚠️ Error: {adaptation_results.get('error', 'Unknown error')}")
        else:
            st.success("✅ System up to date")
            
    except Exception as e:
        st.info(f"ℹ️ System check available")

# ---------- Vertical Color Bar ----------
st.markdown('<div class="vertical-color-bar"></div>', unsafe_allow_html=True)

# ---------- Main Content Container ----------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ---------- Hero Header ----------
st.markdown('''
<div class="top-header" style="padding: 3rem; border-radius: 25px; margin-bottom: 2rem; position: relative; overflow: hidden;">
    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 50%, transparent 100%); opacity: 0.3;"></div>
    <div style="position: relative; z-index: 2;">
        <h1 style="color: white; margin: 0; font-size: 3.5rem; font-weight: 800; text-shadow: 0 4px 8px rgba(0,0,0,0.4); letter-spacing: -1px;">
            🇺🇳 UN JointWork Plans Intelligence Platform
        </h1>
        <p style="color: rgba(255,255,255,0.95); margin: 1rem 0; font-size: 1.4rem; font-weight: 300; text-shadow: 0 2px 4px rgba(0,0,0,0.3); max-width: 800px; margin-left: auto; margin-right: auto;">
            Advanced AI-Powered Analytics for Global Development Strategic Planning & Resource Optimization
        </p>
        <div class="header-features" style="margin-top: 2rem; padding-top: 2rem;">
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 2rem; color: rgba(255,255,255,0.9); font-size: 1rem;">
                <span>🤖 <strong>Machine Learning Models</strong></span>
                <span>📊 <strong>Interactive Analytics</strong></span>
                <span>🌍 <strong>Global Data Integration</strong></span>
                <span>💬 <strong>AI-Powered Insights</strong></span>
            </div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# ---------- Create Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Platform Overview", "📋 Data Sources & Prompts", "👥 Acknowledgments"])

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # ---------- Load and Display Key Platform Statistics ----------
    try:
        financial_data = pd.read_csv("src/outputs/data_output/Financial_Cleaned.csv")
        
        total_projects = len(financial_data)
        total_countries = financial_data['Country'].nunique()
        total_agencies = len(financial_data['Agencies'].dropna().str.split(';').explode().unique())
        total_funding = financial_data['Total required resources'].sum()
        total_regions = financial_data['Region'].nunique()
        total_themes = financial_data['Theme'].nunique()
        
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        total_projects = total_countries = total_agencies = total_funding = total_regions = total_themes = 0

    st.markdown("## 📊 Platform Data Overview")
    st.markdown("### Real-time statistics from the UN JointWork Plans database:")

    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5, stats_col6 = st.columns(6, gap="medium")

    with stats_col1:
        st.markdown(f'''
        <div class="stat-item" style="border-left: 4px solid #3b82f6;">
            <h3 style="color: #3b82f6; margin: 0; font-size: 2rem; font-weight: bold;">{total_projects:,}</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">Total Projects</p>
        </div>
        ''', unsafe_allow_html=True)

    with stats_col2:
        st.markdown(f'''
        <div class="stat-item" style="border-left: 4px solid #22c55e;">
            <h3 style="color: #22c55e; margin: 0; font-size: 2rem; font-weight: bold;">{total_countries}</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">Countries</p>
        </div>
        ''', unsafe_allow_html=True)

    with stats_col3:
        st.markdown(f'''
        <div class="stat-item" style="border-left: 4px solid #f59e0b;">
            <h3 style="color: #f59e0b; margin: 0; font-size: 2rem; font-weight: bold;">{total_agencies}</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">UN Agencies</p>
        </div>
        ''', unsafe_allow_html=True)

    with stats_col4:
        st.markdown(f'''
        <div class="stat-item" style="border-left: 4px solid #dc2626;">
            <h3 style="color: #dc2626; margin: 0; font-size: 1.5rem; font-weight: bold;">{format_currency(total_funding)}</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">Total Funding</p>
        </div>
        ''', unsafe_allow_html=True)

    with stats_col5:
        st.markdown(f'''
        <div class="stat-item" style="border-left: 4px solid #7c3aed;">
            <h3 style="color: #7c3aed; margin: 0; font-size: 2rem; font-weight: bold;">{total_regions}</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">Regions</p>
        </div>
        ''', unsafe_allow_html=True)

    with stats_col6:
        st.markdown(f'''
        <div class="stat-item" style="border-left: 4px solid #059669;">
            <h3 style="color: #059669; margin: 0; font-size: 2rem; font-weight: bold;">{total_themes}</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">Themes</p>
        </div>
        ''', unsafe_allow_html=True)

    # ---------- Core Platform Capabilities ----------
    st.markdown('''
    <div class="capabilities-section">
        <h2 style="color: #009edb; margin: 0 0 2rem 0; font-weight: 700; text-align: center; font-size: 2.5rem;">🚀 Platform Capabilities</h2>
    </div>
    ''', unsafe_allow_html=True)

    cap_col1, cap_col2 = st.columns([1, 1], gap="large")

    with cap_col1:
        st.markdown('''
        <div class="feature-card">
            <h3 style="color: #009edb; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 600;">
                🏠 <strong>Interactive Dashboard</strong>
            </h3>
            <ul style="color: #64748b; margin: 0; padding-left: 1.5rem; line-height: 1.8;">
                <li><strong>Global funding maps</strong> with real-time data visualization</li>
                <li><strong>Multi-year trend analysis</strong> across 2020-2026</li>
                <li><strong>Country-specific insights</strong> with interactive filtering</li>
                <li><strong>Regional comparison tools</strong> for strategic planning</li>
                <li><strong>UN agency collaboration patterns</strong></li>
            </ul>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(0,158,219,0.1); border-radius: 8px;">
                <p style="color: #009edb; margin: 0; font-weight: 500;">
                    🎯 Navigate to: <strong>DASHBOARD</strong> page
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with cap_col2:
        st.markdown('''
        <div class="feature-card">
            <h3 style="color: #7c3aed; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 600;">
                🎯 <strong>AI Strategic Models</strong>
            </h3>
            <ul style="color: #64748b; margin: 0; padding-left: 1.5rem; line-height: 1.8;">
                <li><strong>XGBoost multi-label predictions</strong> for SDG goals & agencies</li>
                <li><strong>Azure OpenAI GPT-4o integration</strong> for strategic insights</li>
                <li><strong>Interactive country mapping</strong> with model predictions</li>
                <li><strong>Historical context analysis</strong> from past collaborations</li>
                <li><strong>Real-time strategic recommendations</strong></li>
            </ul>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(124,58,237,0.1); border-radius: 8px;">
                <p style="color: #7c3aed; margin: 0; font-weight: 500;">
                    🎯 Navigate to: <strong>MODELS</strong> page
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    cap_col3, cap_col4 = st.columns([1, 1], gap="large")

    with cap_col3:
        st.markdown('''
        <div class="feature-card">
            <h3 style="color: #dc2626; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 600;">
                📊 <strong>Advanced Analytics</strong>
            </h3>
            <ul style="color: #64748b; margin: 0; padding-left: 1.5rem; line-height: 1.8;">
                <li><strong>Funding prediction models</strong> with RandomForest (R²: 0.41-0.79)</li>
                <li><strong>Anomaly detection</strong> using LocalOutlierFactor</li>
                <li><strong>Agency performance clustering</strong> with KMeans</li>
                <li><strong>Multi-year comparative analysis</strong></li>
                <li><strong>Interactive filtering & exploration</strong></li>
            </ul>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(220,38,38,0.1); border-radius: 8px;">
                <p style="color: #dc2626; margin: 0; font-weight: 500;">
                    🎯 Navigate to: <strong>ANALYSIS</strong> page
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with cap_col4:
        st.markdown('''
        <div class="feature-card">
            <h3 style="color: #059669; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 600;">
                🤖 <strong>AI Financial Chatbot</strong>
            </h3>
            <ul style="color: #64748b; margin: 0; padding-left: 1.5rem; line-height: 1.8;">
                <li><strong>Natural language queries</strong> on financial data</li>
                <li><strong>Real-time data filtering</strong> by region & theme</li>
                <li><strong>GPT-4o powered responses</strong> with context analysis</li>
                <li><strong>Interactive chat interface</strong> with quick actions</li>
                <li><strong>Professional financial insights</strong></li>
            </ul>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(5,150,105,0.1); border-radius: 8px;">
                <p style="color: #059669; margin: 0; font-weight: 500;">
                    🎯 Navigate to: <strong>CHATBOT</strong> page
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # ---------- Data Science & Modeling Showcase ----------
    st.markdown("---")
    st.markdown('''
    <div class="model-showcase">
        <h2 style="color: #7c3aed; margin: 0 0 2rem 0; font-weight: 700; text-align: center; font-size: 2.5rem;">
            🤖 Data Science & Machine Learning Pipeline
        </h2>
    </div>
    ''', unsafe_allow_html=True)

    model_col1, model_col2 = st.columns([1, 1], gap="large")

    with model_col1:
        st.markdown("### 🔬 Machine Learning Models")
        
        st.markdown("#### 1. Predictive Analytics")
        st.markdown("""
        - **RandomForest Regressor** for funding predictions
        - **Multi-target regression** (Required, Available, Expenditure)
        - **Feature engineering:** funding ratios, temporal patterns
        - **Performance:** R² 0.41-0.79 across targets
        """)
        
        st.markdown("#### 2. Anomaly Detection")
        st.markdown("""
        - **LocalOutlierFactor** for unusual funding patterns
        - **39 engineered features** from raw financial data
        - **Contamination rate:** 5% (balanced detection)
        - **Silhouette score:** 0.1227 (good separation)
        """)
        
        st.markdown("#### 3. Performance Clustering")
        st.markdown("""
        - **KMeans clustering** (k=4) for agency performance
        - **Categories:** Top/Low/Execution Gap/Moderate
        - **Features:** efficiency ratios, resource scale
        - **Silhouette score:** 0.34 (good clusters)
        """)
        
        st.markdown("#### 4. Strategic Predictions")
        st.markdown("""
        - **XGBoost Multi-Label** for SDG goals & agencies
        - **SDG Model:** F1: 0.522, Hamming Loss: 0.200
        - **Agency Model:** F1: 0.491, Hamming Loss: 0.042
        - **Features:** Country, Theme, Priority Code
        """)

    with model_col2:
        st.markdown("### 🧠 AI Integration & Technology Stack")
        
        st.markdown("#### Azure OpenAI Integration")
        st.markdown("""
        - **GPT-4o deployment** for strategic analysis
        - **Context-aware prompts** with financial data
        - **5-section analysis framework**
        - **Real-time insights** with fallback mechanisms
        """)
        
        st.markdown("#### Data Processing Pipeline")
        st.markdown("""
        - **Pandas & NumPy** for data manipulation
        - **Scikit-learn** for ML model implementation
        - **XGBoost** for gradient boosting models
        - **Feature engineering** with ratio calculations
        """)
        
        st.markdown("#### Visualization & Interface")
        st.markdown("""
        - **Streamlit** for interactive web application
        - **Plotly** for dynamic charts and maps
        - **Custom CSS** for UN-themed design
        - **Real-time filtering** and data updates
        """)
        
        st.markdown("#### Data Sources")
        st.markdown("""
        - **Financial_Cleaned.csv:** 2.2M+ records
        - **Model outputs:** Predictions, anomalies, performance
        - **Real-time processing** with caching optimization
        - **Multi-year coverage:** 2020-2026
        """)

    # ---------- User Workflow Guide ----------
    st.markdown("---")
    st.markdown('''
    <h2 style="color: #009edb; margin: 2rem 0 2rem 0; font-weight: 700; text-align: center; font-size: 2.5rem;">
        🗺️ Platform Navigation Guide
    </h2>
    ''', unsafe_allow_html=True)

    workflow_col1, workflow_col2 = st.columns([1, 1], gap="large")

    with workflow_col1:
        st.markdown("### 📋 Recommended User Journey")
        
        st.markdown("""
        **1. Start with Dashboard Overview**
        
        Get familiar with global funding patterns, key metrics, and regional distributions.
        
        **2. Explore Advanced Analytics**
        
        Dive into predictive models, anomaly detection, and performance clustering.
        
        **3. Generate Strategic Predictions**
        
        Use AI models for SDG goal and UN agency recommendations with GPT-4o insights.
        
        **4. Query with AI Chatbot**
        
        Ask natural language questions and get intelligent financial analysis.
        """)

    with workflow_col2:
        st.markdown("### 🎯 Key Use Cases")
        
        st.markdown("#### For Strategic Planners")
        st.markdown("""
        - Identify funding gaps and resource allocation opportunities
        - Predict future SDG collaboration patterns
        - Analyze agency performance and efficiency
        """)
        
        st.markdown("#### For Financial Analysts")
        st.markdown("""
        - Monitor funding trends and anomalies
        - Compare regional and thematic investments
        - Generate detailed financial reports
        """)
        
        st.markdown("#### For Decision Makers")
        st.markdown("""
        - Get AI-powered strategic recommendations
        - Understand collaboration opportunities
        - Access real-time data insights
        """)

    # ---------- Quick Access Navigation ----------
    st.markdown("---")
    st.markdown("## 🚀 Quick Access")

    st.markdown("### Navigate to any section of the platform:")

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4, gap="medium")

    with nav_col1:
        if st.button("🏠 **DASHBOARD**", use_container_width=True, help="Interactive maps and financial overview", type="primary"):
            st.switch_page("pages/main_page.py")

    with nav_col2:
        if st.button("📊 **ANALYSIS**", use_container_width=True, help="Advanced analytics and ML models", type="primary"):
            st.switch_page("pages/prediction.py")

    with nav_col3:
        if st.button("🎯 **MODELS**", use_container_width=True, help="AI strategic predictions with GPT-4o", type="primary"):
            st.switch_page("pages/model.py")

    with nav_col4:
        if st.button("🤖 **CHATBOT**", use_container_width=True, help="AI financial intelligence assistant", type="primary"):
            st.switch_page("pages/bot.py")

    # ---------- Technical Specifications ----------
    st.markdown("---")
    st.markdown("## ⚙️ Technical Specifications")

    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4, gap="medium")

    with tech_col1:
        st.markdown("#### 🔧 Core Technologies")
        st.markdown("""
        - **Python 3.8+** with Streamlit framework
        - **Pandas & NumPy** for data processing
        - **Scikit-learn & XGBoost** for ML
        - **Plotly** for interactive visualizations
        - **Azure OpenAI** for GPT-4o integration
        """)

    with tech_col2:
        st.markdown("#### 📊 Data Processing")
        st.markdown("""
        - **2.2M+** financial records processed
        - **Real-time filtering** and aggregation
        - **Caching optimization** for performance
        - **Multi-year analysis** (2020-2026)
        - **Dynamic visualization** updates
        """)

    with tech_col3:
        st.markdown("#### 🤖 AI Capabilities")
        st.markdown("""
        - **4 ML models** for different analyses
        - **Natural language processing** with GPT-4o
        - **Context-aware prompting** system
        - **Real-time inference** capabilities
        - **Fallback mechanisms** for reliability
        """)

    with tech_col4:
        st.markdown("#### 🎨 User Experience")
        st.markdown("""
        - **Responsive design** for all devices
        - **UN-themed styling** with custom CSS
        - **Interactive elements** and animations
        - **Professional data presentation**
        - **Intuitive navigation** structure
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("## 📋 Data Sources & AI Prompts")
    
    # Data Source Section
    st.markdown("""
    <div class="prompt-info-card">
        <h3 style="color: #009edb; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 600;">
            🌐 <strong>Primary Data Source</strong>
        </h3>
        <p style="color: #64748b; margin: 0; line-height: 1.8; font-size: 1.1rem;">
            This platform utilizes data from the official UN Information Portal's Data Explorer:
        </p>
        <div style="margin: 1.5rem 0; padding: 1rem; background: rgba(0,158,219,0.1); border-radius: 8px; border-left: 4px solid #009edb;">
            <p style="color: #009edb; margin: 0; font-weight: 600; font-size: 1.1rem;">
                🔗 <strong>Source:</strong> <a href="https://uninfo.org/data-explorer/cooperation-framework/activity-report" target="_blank" style="color: #009edb; text-decoration: none;">
                UN Info Data Explorer - Cooperation Framework Activity Report
                </a>
            </p>
        </div>
        <p style="color: #64748b; margin: 1rem 0 0 0; line-height: 1.8;">
            The data includes comprehensive information on UN joint work plans, funding requirements, 
            agency collaborations, and SDG alignments across multiple countries and regions from 2020-2026.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Prompts Section
    st.markdown("## 🤖 AI Prompt System")
    
    st.markdown("""
    <div class="prompt-info-card">
        <h3 style="color: #7c3aed; margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 600;">
            📁 <strong>Specialized Prompt Modules</strong>
        </h3>
        <p style="color: #64748b; margin: 0 0 1.5rem 0; line-height: 1.8;">
            Each page and feature utilizes specialized AI prompts optimized for different analysis purposes:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    prompt_modules = [
        {
            "file": "dashboard.py",
            "title": "Dashboard Analytics",
            "description": "O1 model prompts for comprehensive financial analysis and overview insights",
            "page": "Main Dashboard Page",
            "color": "#009edb"
        },
        {
            "file": "funding_prediction.py", 
            "title": "Funding Predictions",
            "description": "O1 model prompts for analyzing funding prediction models and trends",
            "page": "Analysis Page - Funding Predictions",
            "color": "#dc2626"
        },
        {
            "file": "anomaly_detection.py",
            "title": "Anomaly Detection",
            "description": "O1 model prompts for identifying and analyzing unusual funding patterns",
            "page": "Analysis Page - Anomaly Detection",
            "color": "#f59e0b"
        },
        {
            "file": "agency_performance.py",
            "title": "Agency Performance",
            "description": "O1 model prompts for agency performance clustering and efficiency analysis",
            "page": "Analysis Page - Agency Performance",
            "color": "#22c55e"
        },
        {
            "file": "models.py",
            "title": "Strategic Insights",
            "description": "GPT-4o prompts for strategic insights based on ML predictions and contextual analysis",
            "page": "Models Page - Strategic Predictions",
            "color": "#7c3aed"
        },
        {
            "file": "chatbot.py",
            "title": "Conversational AI",
            "description": "GPT-4o prompts for natural language financial data analysis and interactive queries",
            "page": "Chatbot Page - AI Assistant",
            "color": "#059669"
        }
    ]
    
    for i, module in enumerate(prompt_modules):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"""
            <div style="background: {module['color']}; color: white; padding: 1rem; border-radius: 10px; text-align: center; height: 100%;">
                <h4 style="margin: 0; font-size: 1.1rem;">📄 {module['file']}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.8); border: 1px solid {module['color']}; border-radius: 10px; padding: 1rem; height: 100%;">
                <h4 style="color: {module['color']}; margin: 0 0 0.5rem 0; font-size: 1.2rem;">{module['title']}</h4>
                <p style="color: #64748b; margin: 0 0 0.5rem 0; line-height: 1.6;">{module['description']}</p>
                <p style="color: {module['color']}; margin: 0; font-weight: 600; font-size: 0.9rem;">
                    🎯 Used in: {module['page']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(prompt_modules) - 1:
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Prompt Engineering Details
    st.markdown("---")
    st.markdown("### 🔧 Prompt Engineering Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        #### **O1 Model Integration**
        - **Complex reasoning** for financial data analysis
        - **Multi-step analysis** workflows
        - **Statistical interpretation** capabilities
        - **Pattern recognition** in large datasets
        """)
        
        st.markdown("""
        #### **Context-Aware Processing**
        - **Dynamic data filtering** based on user selections
        - **Real-time data integration** with prompts
        - **Historical context** incorporation
        - **Multi-dimensional analysis** support
        """)
    
    with features_col2:
        st.markdown("""
        #### **GPT-4o Integration**
        - **Natural language** understanding and generation
        - **Strategic insights** generation
        - **Interactive dialogue** capabilities
        - **Professional report** formatting
        """)
        
        st.markdown("""
        #### **Specialized Outputs**
        - **5-section analysis** framework
        - **Executive summaries** for decision makers
        - **Technical details** for analysts
        - **Actionable recommendations**
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("## 👥 Project Acknowledgments")
    
    st.markdown("""
    <div class="acknowledgement-card">
        <h3 style="color: #009edb; margin: 0 0 2rem 0; font-size: 2.2rem; font-weight: 700; text-align: center;">
            🇺🇳 UN JointWork Plans Intelligence Platform
        </h3>
        
        <div style="text-align: center; margin: 2rem 0;">
            <p style="color: #64748b; margin: 0; line-height: 1.8; font-size: 1.2rem;">
                Developed under the <strong style="color: #009edb;">United Nations Development Coordination Office</strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Developer Section
    st.markdown("""
    <div class="acknowledgement-card">
        <h4 style="color: #7c3aed; margin: 0 0 1rem 0; font-size: 1.6rem; font-weight: 600;">
            👨‍💻 <strong>Lead Developer</strong>
        </h4>
        <div style="background: rgba(124,58,237,0.1); border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="width: 60px; height: 60px; background: #7c3aed; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem; font-weight: bold;">
                    ZZ
                </div>
                <div>
                    <h5 style="color: #7c3aed; margin: 0; font-size: 1.3rem; font-weight: 600;">Zichen Zhao</h5>
                    <p style="color: #64748b; margin: 0.3rem 0 0 0; font-size: 1rem;">Developer Intern</p>
                    <p style="color: #7c3aed; margin: 0.3rem 0 0 0; font-size: 0.9rem; font-weight: 500;">📧 zichen.zhao@un.org</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Guidance Team Section
    st.markdown("""
    <div class="acknowledgement-card">
        <h4 style="color: #059669; margin: 0 0 1rem 0; font-size: 1.6rem; font-weight: 600;">
            🎯 <strong>Project Guidance & Supervision</strong>
        </h4>
        <p style="color: #64748b; margin: 0 0 1.5rem 0; line-height: 1.6;">
            This project was developed under the expert guidance and supervision of:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Supervisors
    supervisors = [
        {
            "name": "Kirit Patel",
            "email": "patelk@un.org",
            "role": "Project Supervisor",
            "color": "#009edb",
            "initials": "KP"
        },
        {
            "name": "Tala Chammas",
            "email": "tala.chammas@un.org", 
            "role": "Technical Guidance",
            "color": "#dc2626",
            "initials": "TC"
        },
        {
            "name": "Hanna Stenback-Koehler",
            "email": "hanna.stenbackakoehler@un.org",
            "role": "Strategic Oversight",
            "color": "#f59e0b",
            "initials": "HSK"
        },
        {
            "name": "Muhammad Ahmad",
            "email": "muhammad.ahmad2@un.org",
            "role": "Data & Analytics Guidance",
            "color": "#22c55e", 
            "initials": "MA"
        }
    ]
    
    supervisor_cols = st.columns(2)
    
    for i, supervisor in enumerate(supervisors):
        with supervisor_cols[i % 2]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.8); border: 2px solid {supervisor['color']}; border-radius: 15px; padding: 1.5rem; margin: 0.5rem 0;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="width: 50px; height: 50px; background: {supervisor['color']}; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.1rem; font-weight: bold;">
                        {supervisor['initials']}
                    </div>
                    <div>
                        <h5 style="color: {supervisor['color']}; margin: 0; font-size: 1.1rem; font-weight: 600;">{supervisor['name']}</h5>
                        <p style="color: #64748b; margin: 0.2rem 0; font-size: 0.9rem;">{supervisor['role']}</p>
                        <p style="color: {supervisor['color']}; margin: 0.2rem 0 0 0; font-size: 0.8rem; font-weight: 500;">📧 {supervisor['email']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Organization Section
    st.markdown("---")
    st.markdown("""
    <div class="acknowledgement-card">
        <h4 style="color: #009edb; margin: 0 0 1rem 0; font-size: 1.6rem; font-weight: 600;">
            🏢 <strong>United Nations Development Coordination Office</strong>
        </h4>
        <p style="color: #64748b; margin: 0 0 1rem 0; line-height: 1.8; font-size: 1.1rem;">
            The UN Development Coordination Office (DCO) supports the Resident Coordinator system, 
            which leads United Nations development action at the country level towards achieving the 
            Sustainable Development Goals.
        </p>
        <div style="background: rgba(0,158,219,0.1); border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <p style="color: #009edb; margin: 0; font-weight: 600; font-size: 1rem; text-align: center;">
                🌍 <strong>Mission:</strong> Promoting peace, dignity and equality on a healthy planet through coordinated action
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology & Innovation Section
    st.markdown("""
    <div class="acknowledgement-card">
        <h4 style="color: #7c3aed; margin: 0 0 1rem 0; font-size: 1.6rem; font-weight: 600;">
            🚀 <strong>Innovation & Technology</strong>
        </h4>
        <p style="color: #64748b; margin: 0 0 1rem 0; line-height: 1.8;">
            This platform represents the UN's commitment to leveraging cutting-edge artificial intelligence 
            and data science technologies to enhance global development coordination and strategic planning.
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
            <div style="background: rgba(124,58,237,0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                <h6 style="color: #7c3aed; margin: 0 0 0.5rem 0;">🤖 AI Integration</h6>
                <p style="color: #64748b; margin: 0; font-size: 0.9rem;">GPT-4o & O1 Models</p>
            </div>
            <div style="background: rgba(220,38,38,0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                <h6 style="color: #dc2626; margin: 0 0 0.5rem 0;">📊 Data Science</h6>
                <p style="color: #64748b; margin: 0; font-size: 0.9rem;">ML & Analytics</p>
            </div>
            <div style="background: rgba(5,150,105,0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                <h6 style="color: #059669; margin: 0 0 0.5rem 0;">🌐 Global Impact</h6>
                <p style="color: #64748b; margin: 0; font-size: 0.9rem;">SDG Achievement</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
        <p style='font-size: 1.3rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;'>
            🇺🇳 <strong>United Nations JointWork Plans Intelligence Platform</strong>
        </p>
        <p style='margin: 1rem 0; color: #475569; font-size: 1.1rem; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.6;'>
            Empowering global development through advanced data science, machine learning, and AI-powered strategic insights. 
            Supporting the UN mission to advance sustainable development goals worldwide.
        </p>
        <div style='margin: 2rem 0; padding: 1.5rem; background: rgba(255,255,255,0.7); border-radius: 15px; display: inline-block;'>
            <p style='color: #009edb; margin: 0; font-weight: 600; font-size: 1rem;'>
                🚀 <strong>Ready to explore?</strong> Navigate to any page above to begin your analysis journey.
            </p>
        </div>
        <p style='font-size: 0.9rem; color: #64748b; margin-top: 2rem; padding-top: 1.5rem; border-top: 2px solid rgba(0,158,219,0.2);'>
            © 2025 United Nations Development Coordination Office | Advanced Analytics Platform<br>
            Promoting peace, dignity and equality on a healthy planet through data-driven insights
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
