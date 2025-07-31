import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from openai import AzureOpenAI

def get_azure_openai_client(model_type="o1"):
    """Get Azure OpenAI client from commonconst with fallback handling"""
    from src.commonconst import client_4o, client_o1
    
    if model_type == "4o":
        if client_4o is None:
            st.warning("⚠️ GPT-4o client not available. Please configure Azure OpenAI credentials.")
            return None
        return client_4o
    elif model_type == "o1":
        if client_o1 is None:
            st.warning("⚠️ O1 client not available. Please configure Azure OpenAI credentials.")
            return None
        return client_o1
    else:
        st.error(f"Unknown model type: {model_type}")
        return None

def create_dashboard_financial_prompt(
    region: str,
    theme: str,
    year: str,
    funding_summary: str,
    top_countries: Dict[str, float],
    coverage_metrics: Dict[str, float]
) -> str:
    """Create O1 prompt specifically for main dashboard financial analysis"""
    
    top_countries_text = ", ".join([f"{country}: ${amount:,.0f}" for country, amount in list(top_countries.items())[:3]])
    
    return f"""
Analyze UN financial dashboard data for strategic decision-making:

SCOPE: {region} | {theme} | {year}
FINANCIAL DATA: {funding_summary}
TOP FUNDING NEEDS: {top_countries_text}
COVERAGE METRICS: {coverage_metrics}

Provide 4 focused sections (50 words each):

**FINANCIAL INSIGHTS**
• Key funding patterns and resource distribution
• Critical gaps requiring immediate attention
• Coverage efficiency analysis

**RISK FACTORS**
• Financial sustainability concerns
• Resource allocation vulnerabilities
• Implementation risks

**STRATEGIC OPPORTUNITIES**
• Resource optimization possibilities
• Partnership expansion areas
• Efficiency improvements

**PRIORITY ACTIONS**
• Immediate funding decisions needed
• Strategic resource reallocation
• Partnership development steps

Be data-specific and actionable for UN financial planning.
"""

def create_dashboard_regional_prompt(
    selected_region: str,
    regional_comparison: Dict[str, Any],
    cross_theme_analysis: Dict[str, Any]
) -> str:
    """Create O1 prompt for regional comparative analysis"""
    
    return f"""
Analyze regional UN development patterns:

FOCUS REGION: {selected_region}
REGIONAL COMPARISON: {regional_comparison}
CROSS-THEME PATTERNS: {cross_theme_analysis}

Provide 4 strategic sections (50 words each):

**REGIONAL PROFILE**
• {selected_region} development characteristics
• Unique funding patterns and challenges
• Regional coordination effectiveness

**COMPARATIVE ADVANTAGE**
• How {selected_region} performs vs other regions
• Relative strengths and challenges
• Resource efficiency comparisons

**COLLABORATION PATTERNS**
• Inter-regional cooperation opportunities
• Best practice sharing potential
• Resource pooling possibilities

**REGIONAL STRATEGY**
• Specific recommendations for {selected_region}
• Regional partnership priorities
• Development focus areas

Focus on regional development strategy and cross-regional insights.
"""

def create_dashboard_yearly_prompt(
    year_data: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    projection_context: Dict[str, Any]
) -> str:
    """Create O1 prompt for yearly trend analysis"""
    
    return f"""
Analyze UN financial trends and yearly performance:

YEAR ANALYSIS: {year_data}
TREND PATTERNS: {trend_analysis}
PROJECTIONS: {projection_context}

Provide 4 temporal sections (50 words each):

**YEAR PERFORMANCE**
• Current year funding achievements
• Resource allocation effectiveness
• Implementation progress indicators

**TREND ANALYSIS**
• Multi-year pattern identification
• Growth/decline trajectories
• Cyclical funding behaviors

**FUTURE OUTLOOK**
• Projection implications for planning
• Emerging funding trends
• Resource requirement forecasts

**TEMPORAL STRATEGY**
• Year-over-year improvement opportunities
• Long-term resource planning needs
• Timeline optimization recommendations

Focus on temporal patterns and future planning implications.
"""

def call_o1_api(prompt: str) -> str:
    """Call Azure OpenAI O1 model for dashboard analysis"""
    import time
    
    try:
        from src.commonconst import DEPLOYMENT_O1
        
        client = get_azure_openai_client()
        
        if client is None or DEPLOYMENT_O1 is None:
            return "❌ Azure OpenAI O1 service not available. AI features are currently disabled due to missing credentials."
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=DEPLOYMENT_O1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        elapsed_time = time.time() - start_time
        result = response.choices[0].message.content
        
        if elapsed_time > 7:
            result += f"\n\n*Processing time: {elapsed_time:.1f}s - Consider adjusting filters for faster analysis*"
        else:
            result += f"\n\n*Analysis completed in {elapsed_time:.1f}s*"
        
        return result
        
    except Exception as e:
        error_msg = f"Error calling O1 API: {str(e)}"
        st.error(error_msg)
        return f"❌ {error_msg}"

def get_dashboard_insights(
    analysis_type: str,
    filter_context: Dict[str, str],
    data_context: Dict[str, Any]
) -> str:
    """Get O1-powered insights for main dashboard with different analysis types"""
    
    if analysis_type == "financial":
        prompt = create_dashboard_financial_prompt(
            filter_context.get('region', 'All Regions'),
            filter_context.get('theme', 'All Themes'),
            filter_context.get('year', 'Current'),
            data_context.get('funding_summary', ''),
            data_context.get('top_countries', {}),
            data_context.get('coverage_metrics', {})
        )
    elif analysis_type == "regional":
        prompt = create_dashboard_regional_prompt(
            filter_context.get('region', 'All Regions'),
            data_context.get('regional_comparison', {}),
            data_context.get('cross_theme_analysis', {})
        )
    elif analysis_type == "temporal":
        prompt = create_dashboard_yearly_prompt(
            data_context.get('year_data', {}),
            data_context.get('trend_analysis', {}),
            data_context.get('projection_context', {})
        )
    else:
        # Default financial analysis
        prompt = create_dashboard_financial_prompt(
            filter_context.get('region', 'All Regions'),
            filter_context.get('theme', 'All Themes'),
            filter_context.get('year', 'Current'),
            data_context.get('financial_summary', ''),
            data_context.get('top_countries', {}),
            data_context.get('coverage_metrics', {})
        )
    
    return call_o1_api(prompt)
