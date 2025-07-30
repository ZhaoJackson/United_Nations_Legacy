import streamlit as st
import pandas as pd
from typing import Dict, List, Any

def get_azure_openai_client(model_type="o1"):
    """Get Azure OpenAI client from commonconst with safe fallback"""
    try:
        # Import here to avoid circular imports
        from ..commonconst import client_o1, DEPLOYMENT_O1
        if client_o1 is None:
            return None
        return client_o1
    except Exception as e:
        st.warning(f"Azure OpenAI O1 client not available: {e}")
        return None

def create_funding_prediction_prompt(
    theme: str,
    region: str,
    year: str,
    prediction_summary: Dict[str, Any],
    model_performance: Dict[str, float],
    trend_data: Dict[str, Any]
) -> str:
    """Create O1 prompt for funding prediction analysis"""
    
    return f"""
Analyze UN funding prediction models and forecasting accuracy:

PREDICTION SCOPE: {theme} | {region} | {year}
PREDICTION DATA: {prediction_summary}
MODEL PERFORMANCE: RandomForest R²: {model_performance.get('r2_score', 'N/A')}, RMSE: {model_performance.get('rmse', 'N/A')}
TREND ANALYSIS: {trend_data}

Provide 4 prediction-focused sections (50 words each):

**MODEL ACCURACY**
• RandomForest prediction reliability assessment
• Confidence intervals and uncertainty factors
• Historical prediction vs actual variance

**FUNDING FORECASTS**
• 2026 funding requirement projections
• Expected resource availability trends
• Critical funding gap predictions

**PREDICTIVE RISKS**
• Model uncertainty factors to monitor
• Economic and political impact variables
• Data quality and completeness issues

**FORECASTING STRATEGY**
• Resource planning recommendations
• Early warning indicators to track
• Model improvement opportunities

Focus on predictive intelligence for financial planning.
"""

def create_funding_trend_prompt(
    multi_year_data: Dict[str, Any],
    regional_trends: Dict[str, Any],
    thematic_patterns: Dict[str, Any]
) -> str:
    """Create O1 prompt for funding trend analysis"""
    
    return f"""
Analyze multi-year UN funding trends and patterns:

TEMPORAL DATA: {multi_year_data}
REGIONAL TRENDS: {regional_trends}
THEMATIC PATTERNS: {thematic_patterns}

Provide 4 trend-focused sections (50 words each):

**TEMPORAL PATTERNS**
• Multi-year funding evolution (2020-2026)
• Cyclical patterns and seasonal variations
• Growth/decline trajectory analysis

**REGIONAL DYNAMICS**
• Regional funding trend comparisons
• Cross-regional resource flow patterns
• Regional development priority shifts

**THEMATIC EVOLUTION**
• Theme-specific funding trend analysis
• Emerging priority areas identification
• Resource allocation pattern changes

**TREND IMPLICATIONS**
• Future funding requirement forecasts
• Strategic resource planning insights
• Policy and priority shift indicators

Focus on trend intelligence for strategic planning.
"""

def create_funding_efficiency_prompt(
    efficiency_metrics: Dict[str, Any],
    coverage_analysis: Dict[str, Any],
    utilization_data: Dict[str, Any]
) -> str:
    """Create O1 prompt for funding efficiency analysis"""
    
    return f"""
Analyze UN funding efficiency and resource optimization:

EFFICIENCY METRICS: {efficiency_metrics}
COVERAGE ANALYSIS: {coverage_analysis}
UTILIZATION DATA: {utilization_data}

Provide 4 efficiency-focused sections (50 words each):

**EFFICIENCY ASSESSMENT**
• Resource allocation effectiveness measures
• Cost-per-outcome efficiency ratios
• Implementation speed and quality metrics

**OPTIMIZATION OPPORTUNITIES**
• Resource reallocation potential areas
• Process improvement recommendations
• Technology and automation possibilities

**COVERAGE GAPS**
• Underserved population identification
• Geographic coverage analysis
• Service delivery gap assessment

**EFFICIENCY STRATEGY**
• Resource optimization roadmap
• Performance monitoring frameworks
• Efficiency benchmark establishment

Focus on operational excellence and resource optimization.
"""

def call_o1_api(prompt: str) -> str:
    """Call Azure OpenAI O1 model for funding prediction analysis"""
    import time
    
    try:
        client = get_azure_openai_client()
        
        if client is None:
            return "🤖 Azure OpenAI O1 service is currently unavailable. AI prediction analysis features require proper configuration."
        
        # Import deployment name from commonconst
        from ..commonconst import DEPLOYMENT_O1
        
        if DEPLOYMENT_O1 is None:
            return "🤖 Azure OpenAI deployment configuration missing. Please check system settings."
        
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
            result += f"\n\n*⏱️ Processing time: {elapsed_time:.1f}s - Consider adjusting filters for faster analysis*"
        else:
            result += f"\n\n*⚡ Analysis completed in {elapsed_time:.1f}s*"
        
        return result
        
    except Exception as e:
        error_msg = f"AI prediction analysis temporarily unavailable: {str(e)}"
        return f"🤖 {error_msg}. Please try again later or contact system administrator."

def get_funding_prediction_insights(
    analysis_type: str,
    filter_context: Dict[str, str],
    data_context: Dict[str, Any]
) -> str:
    """Get O1-powered insights for funding prediction tab"""
    
    if analysis_type == "prediction":
        prompt = create_funding_prediction_prompt(
            filter_context.get('theme', 'All Themes'),
            filter_context.get('region', 'All Regions'),
            filter_context.get('year', '2026'),
            data_context.get('prediction_summary', {}),
            data_context.get('model_performance', {}),
            data_context.get('trend_data', {})
        )
    elif analysis_type == "trends":
        prompt = create_funding_trend_prompt(
            data_context.get('multi_year_data', {}),
            data_context.get('regional_trends', {}),
            data_context.get('thematic_patterns', {})
        )
    elif analysis_type == "efficiency":
        prompt = create_funding_efficiency_prompt(
            data_context.get('efficiency_metrics', {}),
            data_context.get('coverage_analysis', {}),
            data_context.get('utilization_data', {})
        )
    else:
        # Default prediction analysis
        prompt = create_funding_prediction_prompt(
            filter_context.get('theme', 'All Themes'),
            filter_context.get('region', 'All Regions'),
            filter_context.get('year', '2026'),
            data_context.get('prediction_summary', {}),
            data_context.get('model_performance', {}),
            data_context.get('trend_data', {})
        )
    
    return call_o1_api(prompt)
