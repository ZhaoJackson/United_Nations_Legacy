import sys
import os

# Ensure project root is in Python path for imports
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.commonconst import *
from src.dynamic_analysis import DynamicDataProcessor
from src.prompt.dashboard import get_dashboard_insights

# Initialize dynamic data processor
@st.cache_resource
def get_dynamic_processor():
    return DynamicDataProcessor()

# ---------- Helper Functions ----------
def get_iso_alpha(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

def format_number(n):
    """Format numbers with appropriate units for better readability"""
    if pd.isna(n) or n == 0:
        return "0"
    elif abs(n) >= 1e12:
        return f"${n/1e12:.1f}T"
    elif abs(n) >= 1e9:
        return f"${n/1e9:.1f}B"
    elif abs(n) >= 1e6:
        return f"${n/1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"${n/1e3:.1f}K"
    else:
        return f"${n:.0f}"

def format_number_short(n):
    """Short format for display"""
    if pd.isna(n) or n == 0:
        return "0"
    elif abs(n) >= 1e9:
        return f"{n/1e9:.1f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.0f}"

def format_number_table(n):
    """Table format with units for better readability"""
    if pd.isna(n) or n == 0:
        return "$0"
    elif abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"${n/1e3:.0f}K"
    else:
        return f"${n:.0f}"

# ---------- Enhanced Custom Styles ----------
with open(STYLE_CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add enhanced custom CSS for UN-themed layout
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
    
    .bottom-dots {
        position: fixed;
        bottom: 10px;
        right: 20px;
        z-index: 1000;
    }
    
    .dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
        margin: 0 3px;
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #009edb 0%, #006bb6 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* UN-themed enhancements */
    .un-section {
        background: linear-gradient(135deg, rgba(0,158,219,0.1) 0%, rgba(0,107,182,0.05) 100%);
        border-left: 4px solid #009edb;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.8) 100%);
        border: 1px solid rgba(0,158,219,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,158,219,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,158,219,0.15);
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #009edb 50%, transparent 100%);
        margin: 2rem 0;
        border: none;
    }
    
    /* Enhanced table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,158,219,0.1);
    }
    
    /* Remove top margins for header alignment */
    .main .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Vertical Color Bar ----------
st.markdown('<div class="vertical-color-bar"></div>', unsafe_allow_html=True)

# ---------- Main Content Container ----------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ---------- Top-Aligned UN Theme Header ----------
st.markdown('''
<div class="top-header">
    <h1>🇺🇳 United Nations JointWork Plans Dashboard 🌍</h1>
    <p>Advancing Global Development Goals Through Data-Driven Insights</p>
    <div class="header-features">
        <span>🎯 Sustainable Development • 🤝 Global Partnership • 📊 Evidence-Based Policy</span>
    </div>
</div>
''', unsafe_allow_html=True)

# ---------- Enhanced Sidebar Filters ----------
st.sidebar.markdown('''
<div style="background: linear-gradient(135deg, #009edb 0%, #006bb6 100%); padding: 1.5rem; margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 15px 15px;">
    <h2 style="color: white; text-align: center; margin: 0; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
        🎛️ Dashboard Controls
    </h2>
    <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 0.5rem 0 0 0; font-size: 0.85rem;">
        Configure your analysis parameters
    </p>
</div>
''', unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Year selection
selected_year = st.sidebar.selectbox(
    "📅 Select Year", 
    PLOT_YEAR_RANGE, 
    index=DEFAULT_YEAR_INDEX,
    help="Choose a year to view funding data"
)

# Theme selection
selected_theme = st.sidebar.selectbox(
    "🎯 Select Theme", 
    get_theme_list(),
    help="Choose a thematic area for analysis"
)

# Region selection
all_regions = ["All Regions"] + get_region_list()
selected_region = st.sidebar.selectbox(
    "🌍 Select Region", 
    all_regions,
    help="Filter by region or view all regions"
)

# UN Agencies selection
all_agencies = ["All Agencies"] + get_agencies_list()
selected_agency = st.sidebar.selectbox(
    "🏢 Select UN Agency", 
    all_agencies,
    help="Filter by UN agency or view all agencies"
)

# SDG Goals selection
all_sdg_goals = ["All SDG Goals"] + get_sdg_goals_list()
selected_sdg = st.sidebar.selectbox(
    "🎯 Select SDG Goal", 
    all_sdg_goals,
    help="Filter by SDG goal or view all goals"
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Add UN mission statement to sidebar
st.sidebar.markdown('''
<div class="un-section">
    <h4 style="color: #009edb; margin: 0 0 0.5rem 0; font-weight: 600;">🇺🇳 UN Mission</h4>
    <p style="color: #1e293b; font-size: 0.8rem; margin: 0; line-height: 1.4;">
        Promoting peace, dignity and equality on a healthy planet through sustainable development and international cooperation.
    </p>
</div>
''', unsafe_allow_html=True)

# ---------- Data Filtering ----------
full_df = financial_df.copy()
full_df = full_df[full_df["Country"] != "Grand Total"]
full_df = full_df[full_df["Theme"] == selected_theme].copy()

# Apply region filter
if selected_region != "All Regions":
    full_df = full_df[full_df["Region"] == selected_region].copy()

# Apply UN Agency filter
full_df = filter_by_agency(full_df, selected_agency)

# Apply SDG Goal filter
full_df = filter_by_sdg(full_df, selected_sdg)

full_df["iso_alpha"] = full_df["Country"].apply(get_iso_alpha)

# Calculate aggregated metrics for map hover
required_col = f"{selected_year} Required"
available_col = f"{selected_year} Available"
expenditure_col = f"{selected_year} Expenditure"
gap_col = f"{selected_year} Gap"

# ---------- Enhanced Key Performance Indicators ----------
if required_col in full_df.columns:
    total_required = full_df[required_col].sum()
    total_available = full_df[available_col].sum() if available_col in full_df.columns else 0
    total_expenditure = full_df[expenditure_col].sum() if expenditure_col in full_df.columns else 0
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3, gap="large")
    
    with kpi_col1:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #dc2626;">
            <h3 style="color: #dc2626; margin: 0; font-size: 1.3rem; font-weight: 600;">💰 Total Required</h3>
            <p style="color: #dc2626; font-size: 2.2rem; font-weight: bold; margin: 0.5rem 0;">{format_number_short(total_required)}</p>
            <small style="color: #64748b; font-size: 0.8rem;">Global funding needs</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #009edb;">
            <h3 style="color: #009edb; margin: 0; font-size: 1.3rem; font-weight: 600;">💵 Total Available</h3>
            <p style="color: #009edb; font-size: 2.2rem; font-weight: bold; margin: 0.5rem 0;">{format_number_short(total_available)}</p>
            <small style="color: #64748b; font-size: 0.8rem;">Accessible resources</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with kpi_col3:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #22c55e;">
            <h3 style="color: #22c55e; margin: 0; font-size: 1.3rem; font-weight: 600;">💸 Total Expenditure</h3>
            <p style="color: #22c55e; font-size: 2.2rem; font-weight: bold; margin: 0.5rem 0;">{format_number_short(total_expenditure)}</p>
            <small style="color: #64748b; font-size: 0.8rem;">Already disbursed</small>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Resized Layout Columns (Bigger Map, Smaller Sides) ----------
col = st.columns((1.5, 6, 1.5), gap='medium')

# ---------- Enhanced World Map (Bigger with Bottom Color Bar) ----------
with col[1]:
    st.markdown('''
    <div class="un-section">
        <h3 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">🌍 Global Funding Distribution</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    if required_col in full_df.columns:
        # Prepare map data with aggregated information
        map_df = full_df.groupby(['Country', 'iso_alpha']).agg({
            required_col: 'sum',
            available_col: 'sum' if available_col in full_df.columns else lambda x: 0,
            expenditure_col: 'sum' if expenditure_col in full_df.columns else lambda x: 0
        }).reset_index()
        
        map_df = map_df.dropna(subset=['iso_alpha'])
        map_df[f'{selected_year}_Gap'] = map_df[required_col] - map_df[available_col]
        
        fig = px.choropleth(
            map_df,
            locations="iso_alpha",
            color=required_col,
            hover_name="Country",
            hover_data={
                required_col: ":,.0f",
                available_col: ":,.0f", 
                expenditure_col: ":,.0f",
                "iso_alpha": False
            },
            color_continuous_scale=UN_BLUE_GRADIENT,
            title=f"Required Funding Distribution - {selected_theme} ({selected_year})"
        )
        
        fig.update_layout(
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                showframe=False,
                showcoastlines=True
            ),
            margin=dict(l=0, r=0, t=50, b=80),
            coloraxis_colorbar=dict(
                title="Required Funding (USD)",
                tickprefix="$",
                tickformat=".2s",
                ticks="outside",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.15,
                yanchor="top",
                len=0.8,
                thickness=15
            ),
            font=dict(size=10),
            height=540,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {selected_year}")

# ---------- Compact Top 10 Countries Table ----------
with col[2]:
    st.markdown('''
    <div class="un-section">
        <h4 style="color: #009edb; margin: 0 0 0.8rem 0; font-weight: 600;">🏆 Top 10 Countries</h4>
    </div>
    ''', unsafe_allow_html=True)
    
    if required_col in full_df.columns:
        df_top = full_df.groupby('Country')[required_col].sum().reset_index()
        df_top = df_top.sort_values(by=required_col, ascending=False).head(10)
        
        # Format the values with units
        df_top['Formatted Amount'] = df_top[required_col].apply(format_number_table)
        df_top['Rank'] = range(1, len(df_top) + 1)
        
        # Create clean table
        display_df = df_top[['Rank', 'Country', 'Formatted Amount']].copy()

        st.dataframe(
            display_df,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Country": st.column_config.TextColumn("Country"),
                "Formatted Amount": st.column_config.TextColumn(f"Required ({selected_year})")
            },
            hide_index=True,
            use_container_width=True,
            height=350
        )
    else:
        st.warning("No data available for ranking.")

# ---------- Compact Funding Changes ----------
with col[0]:
    st.markdown('''
    <div class="un-section">
        <h4 style="color: #009edb; margin: 0 0 0.8rem 0; font-weight: 600;">📈 Funding Changes</h4>
    </div>
    ''', unsafe_allow_html=True)
    
    prev_year = selected_year - 1
    col_prev = f"{prev_year} Required"
    
    if required_col in full_df.columns and col_prev in full_df.columns:
        # Calculate changes by country
        change_df = full_df.groupby('Country').agg({
            required_col: 'sum',
            col_prev: 'sum'
        }).reset_index()
        
        change_df["change"] = change_df[required_col] - change_df[col_prev]
        change_df["change_pct"] = (change_df["change"] / change_df[col_prev] * 100).fillna(0)
        change_df = change_df.dropna().sort_values("change", ascending=False)

        if len(change_df) > 0:
            # Biggest increase
            top_increase = change_df.iloc[0]
            st.markdown("**🔥 Top Increase:**")
            st.metric(
                label=top_increase["Country"], 
                value=format_number_short(top_increase[required_col]), 
                delta=format_number_short(top_increase["change"]),
                help=f"Increased by {top_increase['change_pct']:.1f}% from {prev_year}"
            )

            # Biggest decrease
            if len(change_df) > 1:
                bottom_decrease = change_df.iloc[-1]
                st.markdown("**📉 Top Decrease:**")
                st.metric(
                    label=bottom_decrease["Country"], 
                    value=format_number_short(bottom_decrease[required_col]), 
                    delta=format_number_short(bottom_decrease["change"]),
                    help=f"Changed by {bottom_decrease['change_pct']:.1f}% from {prev_year}"
                )
    else:
        st.info("Year-over-year comparison not available.")
    
    # Enhanced About Section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown(f"""
        **🔍 Data Overview:**
        - **Source**: UN Joint Work Plan Financial Data
        - **Theme**: {selected_theme}
        - **Region**: {selected_region}
        - **Agency**: {selected_agency}
        - **SDG**: {selected_sdg}
        - **Year**: {selected_year}
        
        **📊 Key Metrics:**
        - **Required**: Total funding needed
        - **Available**: Currently accessible funds  
        - **Expenditure**: Already disbursed amounts
        - **Gap**: Difference between required and available
        
        **🎯 Purpose:**
        Supporting UN mission to advance sustainable development goals through transparent funding tracking and analysis.
        """)

# ---------- Enhanced Country Analysis ----------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
coll = st.columns((2, 4), gap='large')

# ---------- Enhanced Funding Breakdown Pie Chart ----------
with coll[0]:
    st.markdown('''
    <div class="un-section">
        <h3 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">📊 Total Funding Breakdown</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    available_countries = sorted(full_df["Country"].unique())
    selected_country = st.selectbox(
        "🏳️ Select a country", 
        available_countries,
        help="Choose a country for detailed analysis"
    )
    
    if required_col in full_df.columns:
        total_required = full_df[required_col].sum()
        total_available = full_df[available_col].sum() if available_col in full_df.columns else 0
        total_expenditure = full_df[expenditure_col].sum() if expenditure_col in full_df.columns else 0
        
        # Create pie chart data
        pie_data = pd.DataFrame({
            "Category": ["Required", "Available", "Expenditure"],
            "Value": [total_required, total_available, total_expenditure]
        })

        # Remove zero values
        pie_data = pie_data[pie_data["Value"] > 0]
        
        if not pie_data.empty:
            fig_pie = px.pie(
                pie_data,
                names="Category",
                values="Value",
                color="Category",
                color_discrete_map={
                    "Required": "#dc2626",
                    "Available": "#009edb", 
                    "Expenditure": "#22c55e"
                },
                title=f"Global Totals ({selected_year})"
            )
            
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                height=450,
                margin=dict(t=50, b=80, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available for pie chart.")
    else:
        st.info("No funding data available.")

# ---------- Enhanced Multi-Year Analysis Chart (Bigger & Aligned) ----------
with coll[1]:
    st.markdown('''
    <div class="un-section">
        <h3 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">📈 Multi-Year Funding Analysis</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    if selected_country:
        selected_row = full_df[full_df["Country"] == selected_country]
        
        if not selected_row.empty:
            # Prepare trend data for all available years
            trend_years = PLOT_YEAR_RANGE
            trend_data = []
            
            country_summary = selected_row.groupby('Country').agg({
                **{f"{year} Required": 'sum' for year in trend_years if f"{year} Required" in selected_row.columns},
                **{f"{year} Available": 'sum' for year in trend_years if f"{year} Available" in selected_row.columns},
                **{f"{year} Expenditure": 'sum' for year in trend_years if f"{year} Expenditure" in selected_row.columns}
            }).iloc[0]
            
            for year in trend_years:
                required_col_year = f"{year} Required"
                available_col_year = f"{year} Available"
                expenditure_col_year = f"{year} Expenditure"
                
                if all(col in country_summary.index for col in [required_col_year, available_col_year, expenditure_col_year]):
                    required_val = country_summary[required_col_year]
                    available_val = country_summary[available_col_year]
                    expenditure_val = country_summary[expenditure_col_year]
                    gap_val = required_val - available_val
                    
                    trend_data.append({
                        "Year": year,
                        "Required": required_val,
                        "Available": available_val,
                        "Expenditure": expenditure_val,
                        "Funding Gap": gap_val
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                
                # Create enhanced combined bar and line chart
                fig = go.Figure()
                
                # Add bar charts with enhanced styling
                fig.add_trace(go.Bar(
                    x=trend_df["Year"],
                    y=trend_df["Required"],
                    name='Required',
                    marker_color='#dc2626',
                    marker_line=dict(width=1, color='#b91c1c'),
                    yaxis='y'
                ))
                
                fig.add_trace(go.Bar(
                    x=trend_df["Year"],
                    y=trend_df["Available"],
                    name='Available',
                    marker_color='#009edb',
                    marker_line=dict(width=1, color='#0284c7'),
                    yaxis='y'
                ))
                
                fig.add_trace(go.Bar(
                    x=trend_df["Year"],
                    y=trend_df["Expenditure"],
                    name='Expenditure',
                    marker_color='#22c55e',
                    marker_line=dict(width=1, color='#16a34a'),
                    yaxis='y'
                ))
                
                # Add enhanced line for Funding Gap
                fig.add_trace(go.Scatter(
                    x=trend_df["Year"],
                    y=trend_df["Funding Gap"],
                    mode='lines+markers',
                    name='Funding Gap',
                    line=dict(color='#f59e0b', width=5),
                    marker=dict(size=12, symbol='diamond', line=dict(width=2, color='#d97706')),
                    yaxis='y2'
                ))
                
            fig.update_layout(
                        title=f"{selected_country} - Comprehensive Funding Analysis",
                        xaxis_title="Year",
                        yaxis=dict(
                            title="Funding Amount (USD)",
                            tickformat="$.2s",
                            side="left",
                            gridcolor="rgba(0,158,219,0.1)"
                        ),
                        yaxis2=dict(
                            title="Funding Gap (USD)",
                            tickformat="$.2s",
                            overlaying="y",
                            side="right",
                            gridcolor="rgba(245,158,11,0.1)"
                        ),
                        barmode='group',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="rgba(0,158,219,0.2)",
                            borderwidth=1
                        ),
                        margin=dict(t=100, b=20),
                        height=450,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(248,250,252,0.5)"
                    )
                    
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 No data available for selected country with current filters.")
    else:
        st.info("🏳️ Select a country to view funding analysis.")
# ---------- Enhanced Bottom Summary Statistics ----------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('''
<div class="un-section">
    <h3 style="color: #009edb; margin: 0 0 1.5rem 0; font-weight: 600; text-align: center;">📊 Summary Statistics</h3>
</div>
''', unsafe_allow_html=True)

if required_col in full_df.columns:
    total_required = full_df[required_col].sum()
    total_available = full_df[available_col].sum() if available_col in full_df.columns else 0
    funding_gap = total_required - total_available
    coverage_ratio = (total_available / total_required * 100) if total_required > 0 else 0
    
    # Show number of countries
    num_countries = len(full_df['Country'].unique())
    num_agencies = len(full_df['Agencies'].dropna().unique())
    
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5, gap="large")
    
    with stats_col1:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #6366f1;">
            <h4 style="color: #6366f1; margin: 0;">🌍 Total Countries</h4>
            <p style="color: #6366f1; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{num_countries}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #059669;">
            <h4 style="color: #059669; margin: 0;">📈 Global Coverage</h4>
            <p style="color: #059669; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{coverage_ratio:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #dc2626;">
            <h4 style="color: #dc2626; margin: 0;">💸 Total Gap</h4>
            <p style="color: #dc2626; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{format_number_short(funding_gap)}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid #0891b2;">
            <h4 style="color: #0891b2; margin: 0;">🏢 UN Agencies</h4>
            <p style="color: #0891b2; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{num_agencies}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with stats_col5:
        if selected_region != "All Regions":
            region_countries = len(full_df[full_df['Region'] == selected_region]['Country'].unique()) if selected_region != "All Regions" else num_countries
            st.markdown(f'''
            <div class="metric-card" style="border-left: 4px solid #7c3aed;">
                <h4 style="color: #7c3aed; margin: 0;">🗺️ Region Countries</h4>
                <p style="color: #7c3aed; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{region_countries}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            theme_projects = len(full_df)
            st.markdown(f'''
            <div class="metric-card" style="border-left: 4px solid #7c3aed;">
                <h4 style="color: #7c3aed; margin: 0;">📋 Total Projects</h4>
                <p style="color: #7c3aed; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{theme_projects}</p>
            </div>
            ''', unsafe_allow_html=True)

# ---------- Close Main Content Container ----------
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Enhanced Bottom Colored Dots ----------
st.markdown('''
<div class="bottom-dots">
    <span class="dot" style="background-color: #ff6b6b; box-shadow: 0 2px 4px rgba(255,107,107,0.3);"></span>
    <span class="dot" style="background-color: #ffa500; box-shadow: 0 2px 4px rgba(255,165,0,0.3);"></span>
    <span class="dot" style="background-color: #ffeb3b; box-shadow: 0 2px 4px rgba(255,235,59,0.3);"></span>
    <span class="dot" style="background-color: #4caf50; box-shadow: 0 2px 4px rgba(76,175,80,0.3);"></span>
    <span class="dot" style="background-color: #2196f3; box-shadow: 0 2px 4px rgba(33,150,243,0.3);"></span>
    <span class="dot" style="background-color: #3f51b5; box-shadow: 0 2px 4px rgba(63,81,181,0.3);"></span>
    <span class="dot" style="background-color: #9c27b0; box-shadow: 0 2px 4px rgba(156,39,176,0.3);"></span>
    <span class="dot" style="background-color: #e91e63; box-shadow: 0 2px 4px rgba(233,30,99,0.3);"></span>
    <span class="dot" style="background-color: #795548; box-shadow: 0 2px 4px rgba(121,85,72,0.3);"></span>
    <span class="dot" style="background-color: #607d8b; box-shadow: 0 2px 4px rgba(96,125,139,0.3);"></span>
    <span class="dot" style="background-color: #ff9800; box-shadow: 0 2px 4px rgba(255,152,0,0.3);"></span>
    <span class="dot" style="background-color: #009688; box-shadow: 0 2px 4px rgba(0,150,136,0.3);"></span>
    <span class="dot" style="background-color: #8bc34a; box-shadow: 0 2px 4px rgba(139,195,74,0.3);"></span>
    <span class="dot" style="background-color: #cddc39; box-shadow: 0 2px 4px rgba(205,220,57,0.3);"></span>
    <span class="dot" style="background-color: #ffeb3b; box-shadow: 0 2px 4px rgba(255,235,59,0.3);"></span>
    <span class="dot" style="background-color: #ffc107; box-shadow: 0 2px 4px rgba(255,193,7,0.3);"></span>
    <span class="dot" style="background-color: #ff5722; box-shadow: 0 2px 4px rgba(255,87,34,0.3);"></span>
</div>
''', unsafe_allow_html=True)



# ---------- AI Strategic Intelligence Section ----------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Full-width container for AI analysis
st.markdown("## AI Strategic Intelligence")
st.markdown("Get data-driven insights and strategic recommendations based on your current filter selections.")

if st.button("Generate AI Analysis", use_container_width=True, type="primary"):
    with st.spinner("Analyzing data..."):
        # Prepare current filter context
        current_filters = {
            'region': selected_region,
            'theme': selected_theme,
            'year': str(selected_year)
        }
        
        # Prepare data context
        filtered_data = financial_df.copy()
        if selected_region != "All Regions":
            filtered_data = filtered_data[filtered_data['Region'] == selected_region]
        if selected_theme != "All Themes":
            filtered_data = filtered_data[filtered_data['Theme'] == selected_theme]
        
        # Calculate summary metrics
        total_required = filtered_data['Total required resources'].sum() if 'Total required resources' in filtered_data.columns else 0
        total_available = filtered_data['Total available resources'].sum() if 'Total available resources' in filtered_data.columns else 0
        funding_gap = total_required - total_available
        coverage_ratio = (total_available / total_required * 100) if total_required > 0 else 0
        
        # Prepare concise data context for O1
        data_context = {
            'financial_summary': f"Projects: {len(filtered_data)}, Required: ${total_required:,.0f}, Available: ${total_available:,.0f}, Gap: ${funding_gap:,.0f} ({coverage_ratio:.1f}% coverage)"
        }
        
        # Get O1 insights
        insights = get_dashboard_insights("financial", current_filters, data_context)
        
        # Display results in full-width format without icons
        st.markdown("### Strategic Intelligence Report")
        
        # Process text for clean display
        clean_insights = insights.replace('*⚡', 'Analysis completed in').replace('*⏱️', 'Processing time:')
        clean_insights = clean_insights.replace('**', '__')  # Convert ** to __ for markdown bold
        clean_insights = clean_insights.replace('\n', '<br>').replace('• ', '&nbsp;&nbsp;• ')
        
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 2rem; border-radius: 8px; margin: 1rem 0;">
            <div style="color: #334155; line-height: 1.8; font-size: 1rem; max-width: none;">
                {clean_insights}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- Enhanced Footer ----------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 2rem; background: linear-gradient(135deg, rgba(0,158,219,0.05) 0%, rgba(0,107,182,0.02) 100%); border-radius: 15px; margin: 1rem 0;'>
        <p style='font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;'>
            <strong>United Nations Joint Work Plan Dashboard</strong>
        </p>
        <p style='margin: 0.5rem 0; color: #475569;'>
            Advancing Sustainable Development Goals | Data-driven insights for global development
        </p>
        <p style='font-size: 0.85rem; color: #64748b; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,158,219,0.1);'>
            © 2025 United Nations Development Coordination Office | Promoting peace, dignity and equality on a healthy planet
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)