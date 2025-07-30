from src.commonconst import *

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
    <h1>🤖 UN Advanced Analytics & Model Insights</h1>
    <p>Machine Learning-Driven Analysis for Strategic Decision Making</p>
    <div class="header-features">
        <span>🔮 Predictive Analytics • 🚨 Anomaly Detection • 📊 Performance Clustering</span>
    </div>
</div>
''', unsafe_allow_html=True)

# ---------- Enhanced Sidebar Filters ----------
st.sidebar.markdown('''
<div style="background: linear-gradient(135deg, #009edb 0%, #006bb6 100%); padding: 1.5rem; margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 15px 15px;">
    <h2 style="color: white; text-align: center; margin: 0; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
        🔬 Interactive Controls
    </h2>
    <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 0.5rem 0 0 0; font-size: 0.85rem;">
        Configure analysis parameters across all models
    </p>
</div>
''', unsafe_allow_html=True)

# Enhanced unified sidebar filters for all tabs
prediction_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
selected_year = st.sidebar.selectbox(
    "📅 Select Year", 
    prediction_years,
    index=len(prediction_years)-1,
    help="Choose a year for analysis"
)

selected_theme = st.sidebar.selectbox(
    "🎯 Select Theme", 
    get_theme_list(),
    help="Choose a thematic area for analysis"
)

selected_region = st.sidebar.selectbox(
    "🌍 Select Region", 
    ["All Regions"] + get_region_list(),
    help="Filter by region or view all regions"
)

selected_agency = st.sidebar.selectbox(
    "🏢 Select UN Agency", 
    ["All Agencies"] + get_agencies_list(),
    help="Filter by UN agency or view all agencies"
)

selected_sdg = st.sidebar.selectbox(
    "🎯 Select SDG Goal", 
    ["All SDG Goals"] + get_sdg_goals_list(),
    help="Filter by SDG goal or view all goals"
)

selected_performance = st.sidebar.selectbox(
    "🏆 Performance Level", 
    ["All Performance Levels"] + get_performance_labels(),
    help="Filter by agency performance level"
)

# Add analysis mode selector
analysis_mode = st.sidebar.radio(
    "🔍 Analysis Mode",
    ["Overview", "Detailed Analysis", "Comparative View"],
    help="Choose analysis depth"
)

st.sidebar.markdown('---')

# Add model information
st.sidebar.markdown('''
<div class="un-section">
    <h4 style="color: #009edb; margin: 0 0 0.5rem 0; font-weight: 600;">🤖 Model Info</h4>
    <p style="color: #1e293b; font-size: 0.75rem; margin: 0; line-height: 1.4;">
        <strong>Prediction:</strong> RandomForest (R²: 0.41-0.79)<br>
        <strong>Anomaly:</strong> LocalOutlierFactor (Score: 0.12)<br>
        <strong>Performance:</strong> KMeans (k=4, Score: 0.34)
    </p>
</div>
''', unsafe_allow_html=True)

# ---------- Create Tabs for Different Analyses ----------
tab1, tab2, tab3 = st.tabs(["🔮 **Funding Predictions**", "🚨 **Anomaly Detection**", "📊 **Agency Performance**"])

# ---------- TAB 1: ENHANCED FUNDING PREDICTIONS ----------
with tab1:
    st.markdown('''
    <div class="un-section">
        <h2 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">💰 Advanced Funding Predictions</h2>
        <p style="color: #64748b; margin: 0; line-height: 1.6;">
            RandomForest model predictions with feature engineering including funding ratios and temporal patterns.
            Interactive analysis across years 2020-2026 with real-time filtering.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Filter prediction data based on sidebar selections
    prediction_filtered = funding_prediction_df.copy()
    prediction_filtered = prediction_filtered[prediction_filtered["Theme"] == selected_theme]
    
    if selected_region != "All Regions":
        prediction_filtered = prediction_filtered[prediction_filtered["Region"] == selected_region]
    
    if selected_agency != "All Agencies":
        prediction_filtered = filter_by_agency(prediction_filtered, selected_agency)
    
    if selected_sdg != "All SDG Goals":
        prediction_filtered = filter_by_sdg(prediction_filtered, selected_sdg)
    
    if not prediction_filtered.empty:
        # Create dynamic year columns based on selected year
        required_col = f"{selected_year} Required"
        available_col = f"{selected_year} Available"
        expenditure_col = f"{selected_year} Expenditure"
        
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            # Dynamic Key Metrics based on selected year
            if required_col in prediction_filtered.columns:
                total_required = prediction_filtered[required_col].sum()
                total_available = prediction_filtered[available_col].sum() if available_col in prediction_filtered.columns else 0
                total_expenditure = prediction_filtered[expenditure_col].sum() if expenditure_col in prediction_filtered.columns else 0
                funding_gap = total_required - total_available
                coverage_ratio = (total_available / total_required * 100) if total_required > 0 else 0
                
                st.markdown(f"### 📊 {selected_year} Funding Analysis")
                
                st.markdown(f'''
                <div class="metric-card" style="border-left: 4px solid #dc2626;">
                    <h4 style="color: #dc2626; margin: 0;">💰 Required Funding</h4>
                    <p style="color: #dc2626; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">{format_currency(total_required)}</p>
                    <small style="color: #64748b;">Total needs for {selected_year}</small>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="metric-card" style="border-left: 4px solid #009edb;">
                    <h4 style="color: #009edb; margin: 0;">💵 Available Funding</h4>
                    <p style="color: #009edb; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">{format_currency(total_available)}</p>
                    <small style="color: #64748b;">Coverage: {coverage_ratio:.1f}%</small>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                    <h4 style="color: #f59e0b; margin: 0;">📊 Funding Gap</h4>
                    <p style="color: #f59e0b; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">{format_currency(funding_gap)}</p>
                    <small style="color: #64748b;">Shortfall amount</small>
                </div>
                ''', unsafe_allow_html=True)
            
            # Prediction Insights
            if selected_year == 2026:
                st.markdown(f'''
                <div class="insight-box prediction-insight">
                    <h4 style="color: #d97706; margin: 0 0 0.5rem 0;">🔮 Prediction Insights</h4>
                    <p style="color: #92400e; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                        • 2026 predictions based on 2020-2025 trends<br>
                        • Model considers funding ratios & temporal patterns<br>
                        • Higher accuracy for expenditure predictions (R²: 0.79)
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            

        
        with col2:
            if analysis_mode == "Overview":
                # Year-over-Year Trend Analysis
                st.markdown("### 📈 Multi-Year Funding Trends")
                
                yearly_data = []
                years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
                
                for year in years:
                    req_col = f"{year} Required"
                    avail_col = f"{year} Available"
                    exp_col = f"{year} Expenditure"
                    
                    if req_col in prediction_filtered.columns:
                        yearly_data.append({
                            "Year": year,
                            "Required": prediction_filtered[req_col].sum(),
                            "Available": prediction_filtered[avail_col].sum() if avail_col in prediction_filtered.columns else 0,
                            "Expenditure": prediction_filtered[exp_col].sum() if exp_col in prediction_filtered.columns else 0,
                            "Type": "Predicted" if year == 2026 else "Historical"
                        })
                
                if yearly_data:
                    trend_df = pd.DataFrame(yearly_data)
                    
                    fig = go.Figure()
                    
                    # Historical data
                    historical_data = trend_df[trend_df["Type"] == "Historical"]
                    predicted_data = trend_df[trend_df["Type"] == "Predicted"]
                    
                    # Add historical lines
                    fig.add_trace(go.Scatter(
                        x=historical_data["Year"],
                        y=historical_data["Required"],
                        mode='lines+markers',
                        name='Required (Historical)',
                        line=dict(color='#dc2626', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=historical_data["Year"],
                        y=historical_data["Available"],
                        mode='lines+markers',
                        name='Available (Historical)',
                        line=dict(color='#009edb', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add predicted data
                    if not predicted_data.empty:
                        # Connect last historical point to prediction
                        last_historical = historical_data.iloc[-1] if not historical_data.empty else None
                        if last_historical is not None:
                            fig.add_trace(go.Scatter(
                                x=[last_historical["Year"], predicted_data["Year"].iloc[0]],
                                y=[last_historical["Required"], predicted_data["Required"].iloc[0]],
                                mode='lines',
                                name='Required (Predicted)',
                                line=dict(color='#dc2626', width=3, dash='dash'),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=[last_historical["Year"], predicted_data["Year"].iloc[0]],
                                y=[last_historical["Available"], predicted_data["Available"].iloc[0]],
                                mode='lines',
                                name='Available (Predicted)',
                                line=dict(color='#009edb', width=3, dash='dash'),
                                showlegend=False
                            ))
                        
                        fig.add_trace(go.Scatter(
                            x=predicted_data["Year"],
                            y=predicted_data["Required"],
                            mode='markers',
                            name='2026 Prediction (Required)',
                            marker=dict(size=12, color='#dc2626', symbol='diamond')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=predicted_data["Year"],
                            y=predicted_data["Available"],
                            mode='markers',
                            name='2026 Prediction (Available)',
                            marker=dict(size=12, color='#009edb', symbol='diamond')
                        ))
                    
                    fig.update_layout(
                        title=f"Funding Trends: {selected_theme} ({selected_region})",
                        xaxis_title="Year",
                        yaxis_title="Funding Amount (USD)",
                        yaxis=dict(tickformat="$.2s"),
                        height=450,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Compact Top Countries Bar Chart for selected year
                if required_col in prediction_filtered.columns:
                    st.markdown(f"### 🏆 Top Countries - {selected_year}")
                    
                    # Create compact bar chart for top countries
                    top_countries_data = prediction_filtered.groupby('Country').agg({
                        required_col: 'sum',
                        available_col: 'sum' if available_col in prediction_filtered.columns else lambda x: 0,
                        expenditure_col: 'sum' if expenditure_col in prediction_filtered.columns else lambda x: 0
                    }).reset_index()
                    
                    # Sort by required and take top 8
                    top_countries_data = top_countries_data.sort_values(required_col, ascending=False).head(8)
                    
                    if not top_countries_data.empty:
                        fig_top = go.Figure()
                        
                        fig_top.add_trace(go.Bar(
                            name='Required',
                            x=top_countries_data['Country'],
                            y=top_countries_data[required_col],
                            marker_color='#dc2626',
                            text=top_countries_data[required_col].apply(lambda x: format_currency(x)),
                            textposition='outside'
                        ))
                        
                        fig_top.add_trace(go.Bar(
                            name='Available',
                            x=top_countries_data['Country'],
                            y=top_countries_data[available_col],
                            marker_color='#009edb'
                        ))
                        
                        fig_top.add_trace(go.Bar(
                            name='Expenditure',
                            x=top_countries_data['Country'],
                            y=top_countries_data[expenditure_col],
                            marker_color='#22c55e'
                        ))
                        
                        fig_top.update_layout(
                            title=f"Top 8 Countries by Funding - {selected_year}",
                            xaxis_title="Country",
                            yaxis_title="Funding Amount (USD)",
                            yaxis=dict(tickformat="$.2s"),
                            barmode='group',
                            height=450,
                            xaxis={'tickangle': 45},
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            margin=dict(t=80, b=60)
                        )
                        
                        st.plotly_chart(fig_top, use_container_width=True)
                
            elif analysis_mode == "Detailed Analysis":
                # Regional Comparison for selected year
                st.markdown(f"### 🌍 Regional Analysis - {selected_year}")
                
                if required_col in prediction_filtered.columns:
                    # Remove region filter for this analysis
                    regional_data = funding_prediction_df.copy()
                    regional_data = regional_data[regional_data["Theme"] == selected_theme]
                    
                    if selected_agency != "All Agencies":
                        regional_data = filter_by_agency(regional_data, selected_agency)
                    
                    if selected_sdg != "All SDG Goals":
                        regional_data = filter_by_sdg(regional_data, selected_sdg)
                    
                    regional_summary = regional_data.groupby('Region').agg({
                        required_col: 'sum',
                        available_col: 'sum' if available_col in regional_data.columns else lambda x: 0,
                        expenditure_col: 'sum' if expenditure_col in regional_data.columns else lambda x: 0
                    }).reset_index()
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Bar(
                        name='Required',
                        x=regional_summary['Region'],
                        y=regional_summary[required_col],
                        marker_color='#dc2626',
                        yaxis='y'
                    ))
                    
                    fig2.add_trace(go.Bar(
                        name='Available',
                        x=regional_summary['Region'],
                        y=regional_summary[available_col],
                        marker_color='#009edb',
                        yaxis='y'
                    ))
                    
                    # Add funding gap line
                    gap_values = regional_summary[required_col] - regional_summary[available_col]
                    fig2.add_trace(go.Scatter(
                        x=regional_summary['Region'],
                        y=gap_values,
                        mode='lines+markers',
                        name='Funding Gap',
                        line=dict(color='#f59e0b', width=4),
                        marker=dict(size=10),
                        yaxis='y2'
                    ))
                    
                    fig2.update_layout(
                        title=f"Regional Funding Distribution - {selected_year}",
                        xaxis_title="Region",
                        yaxis=dict(title="Funding Amount (USD)", tickformat="$.2s", side="left"),
                        yaxis2=dict(title="Funding Gap (USD)", tickformat="$.2s", overlaying="y", side="right"),
                        barmode='group',
                        height=450,
                        xaxis={'tickangle': 45}
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
            elif analysis_mode == "Comparative View":
                # Funding Efficiency Analysis
                st.markdown("### ⚡ Funding Efficiency Analysis")
                
                if required_col in prediction_filtered.columns:
                    efficiency_data = prediction_filtered.copy()
                    efficiency_data['Funding_Efficiency'] = (
                        efficiency_data[available_col] / efficiency_data[required_col]
                    ).fillna(0)
                    efficiency_data['Expenditure_Rate'] = (
                        efficiency_data[expenditure_col] / efficiency_data[available_col]
                    ).fillna(0)
                    
                    # Remove outliers for better visualization
                    efficiency_data = efficiency_data[
                        (efficiency_data['Funding_Efficiency'] <= 2) & 
                        (efficiency_data['Expenditure_Rate'] <= 2)
                    ]
                    
                    fig3 = px.scatter(
                        efficiency_data,
                        x='Funding_Efficiency',
                        y='Expenditure_Rate',
                        size='Total required resources',
                        color='Region',
                        hover_name='Country',
                        hover_data=['SP_Label'],
                        title=f"Funding Efficiency vs Expenditure Rate - {selected_year}",
                        labels={
                            'Funding_Efficiency': 'Funding Efficiency (Available/Required)',
                            'Expenditure_Rate': 'Expenditure Rate (Spent/Available)'
                        }
                    )
                    
                    # Add reference lines
                    fig3.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="100% Expenditure Rate")
                    fig3.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="100% Funding Coverage")
                    
                    fig3.update_layout(height=450)
                    st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("📊 No prediction data available for the selected filters. Try adjusting your selection.")

# ---------- TAB 2: ENHANCED ANOMALY DETECTION ----------  
with tab2:
    st.markdown('''
    <div class="un-section">
        <h2 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">🚨 Interactive Anomaly Detection</h2>
        <p style="color: #64748b; margin: 0; line-height: 1.6;">
            LocalOutlierFactor model identifying unusual strategic priorities based on funding patterns, ratios, and temporal features.
            Interactive exploration of anomalies across countries, themes, and agencies.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Filter anomaly data based on sidebar selections
    anomaly_filtered = anomaly_detection_df.copy()
    anomaly_filtered = anomaly_filtered[anomaly_filtered["Theme"] == selected_theme]
    
    if selected_region != "All Regions":
        anomaly_filtered = anomaly_filtered[anomaly_filtered["Region"] == selected_region]
    
    if selected_agency != "All Agencies":
        anomaly_filtered = filter_by_agency(anomaly_filtered, selected_agency)
    
    if selected_sdg != "All SDG Goals":
        anomaly_filtered = filter_by_sdg(anomaly_filtered, selected_sdg)
    
    if not anomaly_filtered.empty:
        # Define anomalous and normal data at the beginning
        anomalous_data = anomaly_filtered[anomaly_filtered["SP_Anomaly_Flag"] == "Yes"]
        normal_data = anomaly_filtered[anomaly_filtered["SP_Anomaly_Flag"] == "No"]
        
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            # Anomaly Summary
            total_sps = len(anomaly_filtered)
            anomalous_sps = len(anomalous_data)
            normal_sps = len(normal_data)
            anomaly_rate = (anomalous_sps / total_sps * 100) if total_sps > 0 else 0
            
            st.markdown("### 🔍 Anomaly Detection Summary")
            
            st.markdown(f'''
            <div class="metric-card" style="border-left: 4px solid #6366f1;">
                <h4 style="color: #6366f1; margin: 0;">📋 Total Strategic Priorities</h4>
                <p style="color: #6366f1; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">{total_sps}</p>
                <small style="color: #64748b;">Under current filters</small>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="metric-card" style="border-left: 4px solid #dc2626;">
                <h4 style="color: #dc2626; margin: 0;">🚨 Anomalous SPs</h4>
                <p style="color: #dc2626; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">{anomalous_sps}</p>
                <small style="color: #64748b;">Detection rate: {anomaly_rate:.1f}%</small>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="metric-card" style="border-left: 4px solid #22c55e;">
                <h4 style="color: #22c55e; margin: 0;">✅ Normal SPs</h4>
                <p style="color: #22c55e; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">{normal_sps}</p>
                <small style="color: #64748b;">Standard patterns</small>
            </div>
            ''', unsafe_allow_html=True)
            
            # Anomaly detection insights with methodology
            st.markdown(f'''
            <div class="insight-box anomaly-insight">
                <h4 style="color: #dc2626; margin: 0 0 0.5rem 0;">🔍 Detection Methodology</h4>
                <p style="color: #7f1d1d; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                    <strong>LocalOutlierFactor Algorithm:</strong><br>
                    • Analyzes funding ratios (2020-2025 data)<br>
                    • Considers temporal patterns & aggregated totals<br>
                    • Identifies outliers based on local density<br>
                    • 5% contamination rate (balanced detection)<br>
                    • Silhouette score: 0.1227 (good separation)
                </p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Compact Countries with Anomalies
            if not anomalous_data.empty:
                st.markdown("### 🏴 Countries with Anomalies")
                country_anomalies = anomalous_data.groupby('Country').size().sort_values(ascending=False).head(10)
                
                countries_html = ""
                for country, count in country_anomalies.items():
                    total_country_sps = len(anomaly_filtered[anomaly_filtered["Country"] == country])
                    percentage = (count / total_country_sps * 100) if total_country_sps > 0 else 0
                    countries_html += f'<span class="compact-country"><strong>{country}</strong><br>{count} ({percentage:.0f}%)</span>'
                
                st.markdown(f'<div style="margin: 1rem 0;">{countries_html}</div>', unsafe_allow_html=True)
        
        with col2:
            if analysis_mode == "Overview":
                # Anomaly Distribution Visualization
                st.markdown("### 📊 Anomaly Distribution Analysis")
                
                # Add methodology explanation
                st.markdown(f'''
                <div class="insight-box" style="background: linear-gradient(135deg, #f0f9ff, #dbeafe); border-left: 4px solid #3b82f6;">
                    <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">📈 How Anomalies Are Determined</h4>
                    <p style="color: #1e3a8a; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                        The model creates features from raw funding data (Required, Available, Expenditure per year), 
                        calculates ratios (Exp/Req, Exp/Avail, Avail/Req), and applies LocalOutlierFactor to identify 
                        strategic priorities with unusual funding patterns compared to their local neighborhood.
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Create comprehensive anomaly analysis
                anomaly_by_region = anomaly_filtered.groupby(['Region', 'SP_Anomaly_Flag']).size().unstack(fill_value=0)
                
                if 'Yes' in anomaly_by_region.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Normal',
                        x=anomaly_by_region.index,
                        y=anomaly_by_region.get('No', 0),
                        marker_color='#22c55e'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Anomalous',
                        x=anomaly_by_region.index,
                        y=anomaly_by_region.get('Yes', 0),
                        marker_color='#dc2626'
                    ))
                    
                    fig.update_layout(
                        title="Anomaly Distribution by Region",
                        xaxis_title="Region",
                        yaxis_title="Number of Strategic Priorities",
                        barmode='stack',
                        height=400,
                        xaxis={'tickangle': 45}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly characteristics with year information  
                if not anomalous_data.empty:
                    st.markdown("### 📈 Anomaly Characteristics by Year")
                    
                    # Show year-wise anomaly patterns
                    year_cols = ['2020 Required', '2021 Required', '2022 Required', '2023 Required', '2024 Required', '2025 Required']
                    available_year_cols = [col for col in year_cols if col in anomalous_data.columns]
                    
                    if available_year_cols and not normal_data.empty:
                        year_comparison = []
                        for year_col in available_year_cols:
                            year = year_col.split()[0]
                            anomalous_avg = anomalous_data[year_col].mean()
                            normal_avg = normal_data[year_col].mean()
                            year_comparison.append({
                                'Year': year,
                                'Anomalous_Avg': anomalous_avg,
                                'Normal_Avg': normal_avg,
                                'Category': 'Required'
                            })
                        
                        if year_comparison:
                            comparison_df = pd.DataFrame(year_comparison)
                            
                            fig2 = go.Figure()
                            
                            fig2.add_trace(go.Scatter(
                                x=comparison_df['Year'],
                                y=comparison_df['Anomalous_Avg'],
                                mode='lines+markers',
                                name='Anomalous SPs (Avg Required)',
                                line=dict(color='#dc2626', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig2.add_trace(go.Scatter(
                                x=comparison_df['Year'],
                                y=comparison_df['Normal_Avg'],
                                mode='lines+markers',
                                name='Normal SPs (Avg Required)',
                                line=dict(color='#22c55e', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig2.update_layout(
                                title="Average Required Funding: Anomalous vs Normal Strategic Priorities Over Time",
                                xaxis_title="Year",
                                yaxis_title="Average Required Funding (USD)",
                                yaxis=dict(tickformat="$.2s"),
                                height=350
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                        
            elif analysis_mode == "Detailed Analysis":
                # Detailed anomaly analysis by theme and agency
                st.markdown("### 🔬 Detailed Anomaly Analysis")
                
                st.markdown(f'''
                <div class="insight-box" style="background: linear-gradient(135deg, #fef3c7, #fde68a); border-left: 4px solid #f59e0b;">
                    <h4 style="color: #d97706; margin: 0 0 0.5rem 0;">🎯 Feature Engineering Details</h4>
                    <p style="color: #92400e; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                        Model uses raw financial columns (18 features), aggregated totals (3 features), 
                        and engineered ratio features (18 ratios) for each year 2020-2025. All features 
                        are standardized before applying LocalOutlierFactor with n_neighbors=20.
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                if not anomalous_data.empty:
                    # Agency-wise anomaly distribution
                    agency_anomalies = {}
                    for _, row in anomalous_data.iterrows():
                        agencies = str(row['Agencies']).split(';') if pd.notna(row['Agencies']) else ['Unknown']
                        for agency in agencies:
                            agency = agency.strip()
                            if agency and len(agency) > 3:
                                agency_anomalies[agency] = agency_anomalies.get(agency, 0) + 1
                    
                    if agency_anomalies:
                        top_agency_anomalies = dict(sorted(agency_anomalies.items(), key=lambda x: x[1], reverse=True)[:10])
                        
                        fig3 = go.Figure([go.Bar(
                            x=list(top_agency_anomalies.values()),
                            y=list(top_agency_anomalies.keys()),
                            orientation='h',
                            marker_color='#dc2626'
                        )])
                        
                        fig3.update_layout(
                            title="Top 10 UN Agencies with Anomalous Strategic Priorities",
                            xaxis_title="Number of Anomalies",
                            yaxis_title="UN Agency",
                            height=400
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                
                # Timeline of anomalies with year context
                st.markdown("### 📅 Anomaly Timeline Analysis")
                
                year_cols = [col for col in anomalous_data.columns if any(year in col for year in ['2020', '2021', '2022', '2023', '2024', '2025'])]
                required_year_cols = [col for col in year_cols if 'Required' in col]
                
                if required_year_cols:
                    timeline_data = []
                    for year_col in required_year_cols:
                        year = year_col.split()[0]
                        total_anomalous = (anomalous_data[year_col] > 0).sum()
                        timeline_data.append({'Year': int(year), 'Anomalies': total_anomalous})
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    fig4 = px.line(
                        timeline_df,
                        x='Year',
                        y='Anomalies',
                        title="Anomalous Strategic Priorities Over Time",
                        markers=True,
                        line_shape='spline'
                    )
                    
                    fig4.update_traces(line_color='#dc2626', marker_size=10)
                    fig4.update_layout(height=350)
                    st.plotly_chart(fig4, use_container_width=True)
                    
            elif analysis_mode == "Comparative View":
                # Comparative analysis of anomalous vs normal patterns
                st.markdown("### ⚖️ Comparative Pattern Analysis")
                
                st.markdown(f'''
                <div class="insight-box" style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-left: 4px solid #22c55e;">
                    <h4 style="color: #166534; margin: 0 0 0.5rem 0;">🔍 Pattern Recognition Logic</h4>
                    <p style="color: #15803d; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                        The chart below shows funding efficiency ratios where anomalous strategic priorities 
                        exhibit unusual patterns in Available/Required vs Expenditure/Available ratios 
                        compared to normal strategic priorities in their local neighborhood.
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                if not anomalous_data.empty and not normal_data.empty:
                    # Calculate funding ratios for comparison
                    anomalous_data_copy = anomalous_data.copy()
                    normal_data_copy = normal_data.copy()
                    
                    # Calculate ratios
                    for data, label in [(anomalous_data_copy, 'Anomalous'), (normal_data_copy, 'Normal')]:
                        data['Available_Required_Ratio'] = data['Total available resources'] / (data['Total required resources'] + 1)
                        data['Expenditure_Available_Ratio'] = data['Total expenditure resources'] / (data['Total available resources'] + 1)
                        data['Category'] = label
                    
                    # Combine data for comparison
                    comparison_df = pd.concat([
                        anomalous_data_copy[['Available_Required_Ratio', 'Expenditure_Available_Ratio', 'Category', 'Country']],
                        normal_data_copy[['Available_Required_Ratio', 'Expenditure_Available_Ratio', 'Category', 'Country']]
                    ])
                    
                    # Remove extreme outliers
                    comparison_df = comparison_df[
                        (comparison_df['Available_Required_Ratio'] <= 3) & 
                        (comparison_df['Expenditure_Available_Ratio'] <= 3)
                    ]
                    
                    fig5 = px.scatter(
                        comparison_df,
                        x='Available_Required_Ratio',
                        y='Expenditure_Available_Ratio',
                        color='Category',
                        hover_name='Country',
                        title="Funding Pattern Comparison: Anomalous vs Normal",
                        labels={
                            'Available_Required_Ratio': 'Funding Coverage (Available/Required)',
                            'Expenditure_Available_Ratio': 'Utilization Rate (Expenditure/Available)'
                        },
                        color_discrete_map={'Anomalous': '#dc2626', 'Normal': '#22c55e'}
                    )
                    
                    fig5.update_layout(height=450)
                    st.plotly_chart(fig5, use_container_width=True)
        
        # Enhanced Detailed anomaly table with specified features
        if not anomalous_data.empty:
            st.markdown("### 📋 Detailed Anomaly Records")
            
            display_cols = ['Country', 'Region', 'Theme', 'Plan name', 'Sub-Output', 'Agencies', 'SDG Targets', 'SDG Goals', 'Total required resources', 'Total available resources', 'Total expenditure resources']
            
            # Filter available columns
            available_display_cols = [col for col in display_cols if col in anomalous_data.columns]
            
            display_data = anomalous_data[available_display_cols].copy()
            
            # Format currency columns
            for col in ['Total required resources', 'Total available resources', 'Total expenditure resources']:
                if col in display_data.columns:
                    display_data[col] = display_data[col].apply(format_currency)
            
            st.dataframe(
                display_data.head(15),
                column_config={
                    "Country": "Country",
                    "Region": "Region",
                    "Theme": "Theme",
                    "Plan name": "Plan Name",
                    "Sub-Output": "Sub-Output",
                    "Agencies": "UN Agencies",
                    "SDG Targets": "SDG Targets",
                    "SDG Goals": "SDG Goals",
                    "Total required resources": "Required",
                    "Total available resources": "Available", 
                    "Total expenditure resources": "Expenditure"
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("📊 No anomaly data available for the selected filters. Try adjusting your selection.")

# ---------- TAB 3: ENHANCED AGENCY PERFORMANCE ----------
with tab3:
    st.markdown('''
    <div class="un-section">
        <h2 style="color: #009edb; margin: 0 0 1rem 0; font-weight: 600;">🏢 Interactive Agency Performance Analysis</h2>
        <p style="color: #64748b; margin: 0; line-height: 1.6;">
            KMeans clustering analysis revealing four distinct performance categories based on financial efficiency metrics.
            Explore agency performance patterns across themes, regions, and funding scales.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Filter performance data based on sidebar selections
    performance_filtered = un_agency_performance_df.copy()
    performance_filtered = performance_filtered[performance_filtered["Theme"] == selected_theme]
    
    if selected_region != "All Regions":
        performance_filtered = performance_filtered[performance_filtered["Region"] == selected_region]
    
    if selected_agency != "All Agencies":
        performance_filtered = filter_by_agency(performance_filtered, selected_agency)
    
    if selected_sdg != "All SDG Goals":
        performance_filtered = filter_by_sdg(performance_filtered, selected_sdg)
    
    if selected_performance != "All Performance Levels":
        performance_filtered = performance_filtered[performance_filtered["Performance_Label"] == selected_performance]
    
    if not performance_filtered.empty:
        # Horizontal Performance Metrics
        st.markdown("### 📈 Average Performance Metrics")
        
        avg_metrics = performance_filtered.groupby('Performance_Label').agg({
            'Total required resources': 'mean',
            'Avail_per_Req': 'mean',
            'Exp_per_Req': 'mean'
        }).round(3)
        
        # Create horizontal layout using columns
        metric_cols = st.columns(4, gap="medium")
        
        # Color coding for performance levels
        colors = {
            'Top Performer': '#22c55e',
            'Low Performer': '#ef4444',
            'Execution Gap': '#f59e0b',
            'Moderate Performer': '#3b82f6'
        }
        
        for i, label in enumerate(avg_metrics.index):
            metrics = avg_metrics.loc[label]
            
            with metric_cols[i]:
                st.markdown(f'''
                <div class="horizontal-metric-item" style="border-left: 4px solid {colors.get(label, '#64748b')};">
                    <h4 style="color: {colors.get(label, '#64748b')}; margin: 0 0 0.5rem 0; font-size: 1rem;">{label}</h4>
                    <div style="font-size: 0.85rem; color: #64748b; line-height: 1.5;">
                        <strong>Scale:</strong> {format_currency(metrics['Total required resources'])}<br>
                        <strong>Coverage:</strong> {metrics['Avail_per_Req']:.2f}<br>
                        <strong>Efficiency:</strong> {metrics['Exp_per_Req']:.2f}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("---")  # Add a separator
        
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            # Performance Distribution
            st.markdown("### 🎯 Performance Distribution")
            
            performance_counts = performance_filtered['Performance_Label'].value_counts()
            total_count = len(performance_filtered)
            
            for label, count in performance_counts.items():
                percentage = (count / total_count * 100) if total_count > 0 else 0
                class_name = label.lower().replace(" ", "-")
                
                # Performance category descriptions
                descriptions = {
                    "Top Performer": "High efficiency & utilization",
                    "Low Performer": "Large scale, lower efficiency", 
                    "Execution Gap": "Good funding, low spending",
                    "Moderate Performer": "Balanced performance"
                }
                
                st.markdown(f'''
                <div class="performance-card {class_name}">
                    <h4 style="margin: 0; font-size: 1.1rem;">{label}</h4>
                    <p style="margin: 0.3rem 0; font-size: 1.3rem; font-weight: bold;">{count} SPs ({percentage:.1f}%)</p>
                    <small style="opacity: 0.9;">{descriptions.get(label, "")}</small>
                </div>
                ''', unsafe_allow_html=True)
            
            # Performance Insights
            st.markdown(f'''
            <div class="insight-box performance-insight">
                <h4 style="color: #3b82f6; margin: 0 0 0.5rem 0;">📊 Clustering Insights</h4>
                <p style="color: #1e40af; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                    • Model uses funding ratios & resource scale<br>
                    • 4 distinct performance categories identified<br>
                    • Silhouette score: 0.34 (good separation)
                </p>
            </div>
            ''', unsafe_allow_html=True)
            

        
        with col2:
            if analysis_mode == "Overview":
                # Enhanced Performance Clustering Visualization
                st.markdown("### 📊 Performance Clustering Map")
                
                fig = px.scatter(
                    performance_filtered,
                    x='Avail_per_Req',
                    y='Exp_per_Req',
                    color='Performance_Label',
                    size='Total required resources',
                    hover_name='Country',
                    hover_data=['SP_Label', 'Total required resources'],
                    title="Agency Performance Clusters - Financial Efficiency Analysis",
                    labels={
                        'Avail_per_Req': 'Funding Coverage Ratio (Available/Required)',
                        'Exp_per_Req': 'Resource Utilization Ratio (Expenditure/Required)'
                    },
                    color_discrete_map={
                        'Top Performer': '#22c55e',
                        'Low Performer': '#ef4444', 
                        'Execution Gap': '#f59e0b',
                        'Moderate Performer': '#3b82f6'
                    }
                )
                
                # Add quadrant lines for easier interpretation
                fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
                fig.add_vline(x=1.0, line_dash="dot", line_color="gray", opacity=0.5)
                
                fig.update_layout(
                    height=450,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance trend over time
                st.markdown("### 📈 Performance Trends")
                
                # Calculate efficiency trends
                year_columns = [col for col in performance_filtered.columns if any(year in col for year in ['2020', '2021', '2022', '2023', '2024', '2025'])]
                required_cols = [col for col in year_columns if 'Required' in col]
                
                if len(required_cols) >= 3:
                    trend_data = []
                    for year_col in required_cols:
                        year = year_col.split()[0]
                        avail_col = f"{year} Available"
                        
                        if avail_col in performance_filtered.columns:
                            yearly_performance = performance_filtered.groupby('Performance_Label').apply(
                                lambda x: (x[avail_col].sum() / x[year_col].sum()) if x[year_col].sum() > 0 else 0,
                                include_groups=False
                            ).reset_index()
                            yearly_performance.columns = ['Performance_Label', 'Coverage_Ratio']
                            yearly_performance['Year'] = int(year)
                            trend_data.append(yearly_performance)
                    
                    if trend_data:
                        trend_df = pd.concat(trend_data, ignore_index=True)
                        
                        fig2 = px.line(
                            trend_df,
                            x='Year',
                            y='Coverage_Ratio',
                            color='Performance_Label',
                            title="Funding Coverage Trends by Performance Category",
                            markers=True,
                            color_discrete_map={
                                'Top Performer': '#22c55e',
                                'Low Performer': '#ef4444', 
                                'Execution Gap': '#f59e0b',
                                'Moderate Performer': '#3b82f6'
                            }
                        )
                        
                        fig2.update_layout(height=350, yaxis_title="Average Coverage Ratio")
                        st.plotly_chart(fig2, use_container_width=True)
                        
            elif analysis_mode == "Detailed Analysis":
                # Country Performance Breakdown
                st.markdown("### 🌍 Country Performance Analysis")
                
                country_performance = performance_filtered.groupby(['Country', 'Performance_Label']).size().unstack(fill_value=0)
                
                if not country_performance.empty:
                    # Show top 12 countries by total strategic priorities
                    country_totals = country_performance.sum(axis=1).sort_values(ascending=False).head(12)
                    top_countries_performance = country_performance.loc[country_totals.index]
                    
                    fig3 = go.Figure()
                    
                    colors = {
                        'Top Performer': '#22c55e',
                        'Low Performer': '#ef4444',
                        'Execution Gap': '#f59e0b', 
                        'Moderate Performer': '#3b82f6'
                    }
                    
                    for performance_level in top_countries_performance.columns:
                        fig3.add_trace(go.Bar(
                            name=performance_level,
                            x=top_countries_performance.index,
                            y=top_countries_performance[performance_level],
                            marker_color=colors.get(performance_level, '#64748b')
                        ))
                    
                    fig3.update_layout(
                        title="Performance Distribution by Country (Top 12)",
                        xaxis_title="Country",
                        yaxis_title="Number of Strategic Priorities",
                        barmode='stack',
                        height=450,
                        xaxis={'tickangle': 45},
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Agency Effectiveness Analysis
                st.markdown("### 🏢 Agency Effectiveness Analysis")
                
                # Calculate agency-level performance
                agency_performance = {}
                for _, row in performance_filtered.iterrows():
                    agencies = str(row['Agencies']).split(';') if pd.notna(row['Agencies']) else ['Unknown']
                    for agency in agencies:
                        agency = agency.strip()
                        if agency and len(agency) > 3:
                            if agency not in agency_performance:
                                agency_performance[agency] = {'count': 0, 'total_efficiency': 0}
                            agency_performance[agency]['count'] += 1
                            agency_performance[agency]['total_efficiency'] += row['Exp_per_Req']
                
                # Calculate average efficiency for each agency
                agency_data = []
                for agency, data in agency_performance.items():
                    if data['count'] >= 3:  # Only agencies with 3+ strategic priorities
                        avg_efficiency = data['total_efficiency'] / data['count']
                        agency_data.append({
                            'Agency': agency[:50] + '...' if len(agency) > 50 else agency,
                            'Count': data['count'],
                            'Avg_Efficiency': avg_efficiency
                        })
                
                if agency_data:
                    agency_df = pd.DataFrame(agency_data).sort_values('Avg_Efficiency', ascending=True).tail(10)
                    
                    fig4 = go.Figure([go.Bar(
                        x=agency_df['Avg_Efficiency'],
                        y=agency_df['Agency'],
                        orientation='h',
                        marker_color='#3b82f6',
                        text=agency_df['Count'],
                        textposition='auto'
                    )])
                    
                    fig4.update_layout(
                        title="Top 10 Most Efficient UN Agencies",
                        xaxis_title="Average Resource Utilization Ratio",
                        yaxis_title="UN Agency",
                        height=400
                    )
                    
                    st.plotly_chart(fig4, use_container_width=True)
                    
            elif analysis_mode == "Comparative View":
                # Resource Scale vs Performance Analysis
                st.markdown("### 💰 Resource Scale vs Performance Analysis")
                
                # Create resource scale categories
                performance_copy = performance_filtered.copy()
                performance_copy['Resource_Scale'] = pd.cut(
                    performance_copy['Total required resources'],
                    bins=4,
                    labels=['Small Scale', 'Medium Scale', 'Large Scale', 'Mega Scale']
                )
                
                scale_performance = performance_copy.groupby(['Resource_Scale', 'Performance_Label'], observed=True).size().unstack(fill_value=0)
                
                if not scale_performance.empty:
                    fig5 = px.bar(
                        scale_performance.reset_index(),
                        x='Resource_Scale',
                        y=['Top Performer', 'Moderate Performer', 'Execution Gap', 'Low Performer'],
                        title="Performance Distribution by Resource Scale",
                        color_discrete_map={
                            'Top Performer': '#22c55e',
                            'Moderate Performer': '#3b82f6',
                            'Execution Gap': '#f59e0b',
                            'Low Performer': '#ef4444'
                        }
                    )
                    
                    fig5.update_layout(
                        height=400,
                        xaxis_title="Resource Scale Category",
                        yaxis_title="Number of Strategic Priorities",
                        barmode='stack'
                    )
                    
                    st.plotly_chart(fig5, use_container_width=True)
                
                # Performance efficiency box plot
                st.markdown("### 📦 Performance Efficiency Distribution")
                
                fig6 = px.box(
                    performance_filtered,
                    x='Performance_Label',
                    y='Exp_per_Req',
                    title="Resource Utilization Efficiency by Performance Category",
                    color='Performance_Label',
                    color_discrete_map={
                        'Top Performer': '#22c55e',
                        'Low Performer': '#ef4444', 
                        'Execution Gap': '#f59e0b',
                        'Moderate Performer': '#3b82f6'
                    }
                )
                
                fig6.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig6, use_container_width=True)
        
        # Performance summary table
        st.markdown("### 📊 Performance Summary Table")
        
        summary_stats = performance_filtered.groupby('Performance_Label').agg({
            'Total required resources': ['count', 'mean', 'sum'],
            'Avail_per_Req': 'mean',
            'Exp_per_Req': 'mean'
        }).round(3)
        
        summary_stats.columns = ['Count', 'Avg Required', 'Total Required', 'Avg Coverage', 'Avg Efficiency']
        summary_stats['Avg Required'] = summary_stats['Avg Required'].apply(format_currency)
        summary_stats['Total Required'] = summary_stats['Total Required'].apply(format_currency)
        
        st.dataframe(summary_stats, use_container_width=True)
        
    else:
        st.info("📊 No performance data available for the selected filters. Try adjusting your selection.")

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
            🤖 <strong>UN Advanced Analytics Platform</strong>
        </p>
        <p style='margin: 0.5rem 0; color: #475569;'>
            Interactive Machine Learning Insights | Real-time Analysis | Strategic Decision Support
        </p>
        <p style='font-size: 0.85rem; color: #64748b; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,158,219,0.1);'>
            Models: RandomForest (Prediction) • LocalOutlierFactor (Anomaly) • KMeans (Clustering) | Enhanced with Interactive Filtering
        </p>
    </div>
    """, 
    unsafe_allow_html=True
) 