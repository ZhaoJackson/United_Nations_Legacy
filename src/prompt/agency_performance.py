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

def create_agency_performance_prompt(
    theme: str,
    region: str,
    performance_summary: Dict[str, Any],
    clustering_metrics: Dict[str, float],
    efficiency_data: Dict[str, Any]
) -> str:
    """Create O1 prompt for agency performance analysis"""
    
    return f"""
Analyze UN agency performance clustering and efficiency patterns:

PERFORMANCE SCOPE: {theme} | {region}
PERFORMANCE SUMMARY: {performance_summary}
CLUSTERING METRICS: KMeans Silhouette Score: {clustering_metrics.get('silhouette_score', 'N/A')}, Clusters: {clustering_metrics.get('n_clusters', '4')}
EFFICIENCY DATA: {efficiency_data}

Provide 4 performance-focused sections (50 words each):

**PERFORMANCE CLUSTERS**
• KMeans clustering effectiveness analysis
• Agency performance category identification
• Top/Low/Moderate/Execution Gap performers

**EFFICIENCY ASSESSMENT**
• Resource utilization ratio analysis
• Financial efficiency comparative metrics
• Implementation effectiveness evaluation

**COLLABORATION PATTERNS**
• Inter-agency coordination effectiveness
• Partnership performance indicators
• Multi-agency project success rates

**PERFORMANCE OPTIMIZATION**
• Improvement opportunity identification
• Best practice sharing recommendations
• Performance enhancement strategies

Focus on operational excellence and institutional effectiveness.
"""

def create_agency_comparison_prompt(
    agency_rankings: Dict[str, Any],
    performance_trends: Dict[str, Any],
    institutional_metrics: Dict[str, Any]
) -> str:
    """Create O1 prompt for agency comparative analysis"""
    
    return f"""
Compare UN agency performance across institutional effectiveness dimensions:

AGENCY RANKINGS: {agency_rankings}
PERFORMANCE TRENDS: {performance_trends}
INSTITUTIONAL METRICS: {institutional_metrics}

Provide 4 comparison-focused sections (50 words each):

**INSTITUTIONAL EXCELLENCE**
• Top-performing agency identification
• Excellence factor analysis
• Institutional capacity assessment

**PERFORMANCE DIFFERENTIALS**
• Agency efficiency gap analysis
• Resource allocation effectiveness comparison
• Implementation speed variations

**TREND ANALYSIS**
• Performance trajectory evaluation
• Improvement/decline pattern identification
• Consistency and reliability metrics

**INSTITUTIONAL STRATEGY**
• Agency-specific improvement recommendations
• Institutional capacity building priorities
• Performance alignment opportunities

Focus on institutional development and comparative excellence.
"""

def create_agency_collaboration_prompt(
    collaboration_data: Dict[str, Any],
    partnership_metrics: Dict[str, Any],
    coordination_effectiveness: Dict[str, Any]
) -> str:
    """Create O1 prompt for agency collaboration analysis"""
    
    return f"""
Analyze UN agency collaboration patterns and partnership effectiveness:

COLLABORATION DATA: {collaboration_data}
PARTNERSHIP METRICS: {partnership_metrics}
COORDINATION EFFECTIVENESS: {coordination_effectiveness}

Provide 4 collaboration-focused sections (50 words each):

**COLLABORATION NETWORKS**
• Inter-agency partnership mapping
• Collaboration frequency and intensity analysis
• Network effect identification

**PARTNERSHIP EFFECTIVENESS**
• Joint project success rate evaluation
• Resource sharing efficiency assessment
• Coordination overhead analysis

**SYNERGY OPPORTUNITIES**
• Unexploited collaboration potential
• Complementary capability identification
• Resource optimization through partnerships

**COLLABORATION STRATEGY**
• Partnership enhancement recommendations
• Coordination mechanism improvements
• Network optimization strategies

Focus on collaborative effectiveness and partnership optimization.
"""

def call_o1_api(prompt: str) -> str:
    """Call Azure OpenAI O1 model for agency performance analysis"""
    import time
    
    try:
        client = get_azure_openai_client()
        
        if client is None:
            return "🤖 Azure OpenAI O1 service is currently unavailable. AI agency performance features require proper configuration."
        
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
        error_msg = f"AI agency performance analysis temporarily unavailable: {str(e)}"
        return f"🤖 {error_msg}. Please try again later or contact system administrator."

def get_agency_performance_insights(
    analysis_type: str,
    filter_context: Dict[str, str],
    data_context: Dict[str, Any]
) -> str:
    """Get O1-powered insights for agency performance tab"""
    
    if analysis_type == "performance":
        prompt = create_agency_performance_prompt(
            filter_context.get('theme', 'All Themes'),
            filter_context.get('region', 'All Regions'),
            data_context.get('performance_summary', {}),
            data_context.get('clustering_metrics', {}),
            data_context.get('efficiency_data', {})
        )
    elif analysis_type == "comparison":
        prompt = create_agency_comparison_prompt(
            data_context.get('agency_rankings', {}),
            data_context.get('performance_trends', {}),
            data_context.get('institutional_metrics', {})
        )
    elif analysis_type == "collaboration":
        prompt = create_agency_collaboration_prompt(
            data_context.get('collaboration_data', {}),
            data_context.get('partnership_metrics', {}),
            data_context.get('coordination_effectiveness', {})
        )
    else:
        # Default performance analysis
        prompt = create_agency_performance_prompt(
            filter_context.get('theme', 'All Themes'),
            filter_context.get('region', 'All Regions'),
            data_context.get('performance_summary', {}),
            data_context.get('clustering_metrics', {}),
            data_context.get('efficiency_data', {})
        )
    
    return call_o1_api(prompt)
