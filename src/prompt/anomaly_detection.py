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

def create_anomaly_detection_prompt(
    theme: str,
    region: str,
    anomaly_summary: Dict[str, Any],
    detection_metrics: Dict[str, float],
    anomalous_patterns: Dict[str, Any]
) -> str:
    """Create O1 prompt for anomaly detection analysis"""
    
    return f"""
Analyze UN financial anomaly detection results and irregular patterns:

ANOMALY SCOPE: {theme} | {region}
DETECTION SUMMARY: {anomaly_summary}
MODEL METRICS: LocalOutlierFactor Silhouette Score: {detection_metrics.get('silhouette_score', 'N/A')}, Contamination Rate: {detection_metrics.get('contamination_rate', '5%')}
PATTERN ANALYSIS: {anomalous_patterns}

Provide 4 anomaly-focused sections (50 words each):

**ANOMALY ASSESSMENT**
• LocalOutlierFactor detection accuracy
• Unusual funding pattern identification
• Strategic priority outlier analysis

**RISK INDICATORS**
• Financial irregularity warning signs
• Potential fraud or misallocation risks
• Resource distribution anomalies

**PATTERN ANALYSIS**
• Systematic vs random anomaly classification
• Geographic and thematic clustering patterns
• Temporal anomaly trend identification

**ANOMALY RESPONSE**
• Investigation priority recommendations
• Risk mitigation strategies
• Process improvement opportunities

Focus on financial integrity and risk management insights.
"""

def create_anomaly_investigation_prompt(
    anomalous_projects: List[Dict[str, Any]],
    agency_anomalies: Dict[str, Any],
    temporal_anomalies: Dict[str, Any]
) -> str:
    """Create O1 prompt for detailed anomaly investigation"""
    
    return f"""
Investigate detailed UN financial anomalies and irregularities:

ANOMALOUS PROJECTS: {anomalous_projects}
AGENCY PATTERNS: {agency_anomalies}
TEMPORAL PATTERNS: {temporal_anomalies}

Provide 4 investigation-focused sections (50 words each):

**INVESTIGATION PRIORITIES**
• High-risk anomaly cases requiring immediate review
• Systematic pattern investigations needed
• Resource allocation discrepancy analysis

**AGENCY CONCERNS**
• UN agencies with unusual patterns
• Inter-agency coordination anomalies
• Institutional capacity irregularities

**TEMPORAL ANOMALIES**
• Time-based funding pattern irregularities
• Seasonal and cyclical anomaly identification
• Historical deviation trend analysis

**CORRECTIVE ACTIONS**
• Immediate investigation protocols
• System improvement recommendations
• Preventive measure implementation

Focus on detailed investigation guidance and corrective strategies.
"""

def create_anomaly_prevention_prompt(
    prevention_metrics: Dict[str, Any],
    control_systems: Dict[str, Any],
    monitoring_frameworks: Dict[str, Any]
) -> str:
    """Create O1 prompt for anomaly prevention and control systems"""
    
    return f"""
Analyze UN financial anomaly prevention and control mechanisms:

PREVENTION METRICS: {prevention_metrics}
CONTROL SYSTEMS: {control_systems}
MONITORING FRAMEWORKS: {monitoring_frameworks}

Provide 4 prevention-focused sections (50 words each):

**PREVENTION SYSTEMS**
• Early warning system effectiveness
• Proactive monitoring mechanism assessment
• Risk identification capability analysis

**CONTROL MECHANISMS**
• Financial oversight framework evaluation
• Quality assurance process effectiveness
• Compliance monitoring system review

**DETECTION ENHANCEMENT**
• Model improvement opportunities
• Feature engineering recommendations
• Detection algorithm optimization

**PREVENTION STRATEGY**
• Comprehensive anomaly prevention roadmap
• Institutional capacity building needs
• Technology enhancement requirements

Focus on proactive risk management and system strengthening.
"""

def call_o1_api(prompt: str) -> str:
    """Call Azure OpenAI O1 model for anomaly detection analysis"""
    import time
    
    try:
        client = get_azure_openai_client()
        
        if client is None:
            return "❌ Unable to connect to Azure OpenAI O1 service. Please check your configuration."
        
        start_time = time.time()
        
        from src.commonconst import DEPLOYMENT_O1
        
        if client is None or DEPLOYMENT_O1 is None:
            return "❌ Azure OpenAI O1 service not available. AI features are currently disabled due to missing credentials."
        
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

def get_anomaly_detection_insights(
    analysis_type: str,
    filter_context: Dict[str, str],
    data_context: Dict[str, Any]
) -> str:
    """Get O1-powered insights for anomaly detection tab"""
    
    if analysis_type == "detection":
        prompt = create_anomaly_detection_prompt(
            filter_context.get('theme', 'All Themes'),
            filter_context.get('region', 'All Regions'),
            data_context.get('anomaly_summary', {}),
            data_context.get('detection_metrics', {}),
            data_context.get('anomalous_patterns', {})
        )
    elif analysis_type == "investigation":
        prompt = create_anomaly_investigation_prompt(
            data_context.get('anomalous_projects', []),
            data_context.get('agency_anomalies', {}),
            data_context.get('temporal_anomalies', {})
        )
    elif analysis_type == "prevention":
        prompt = create_anomaly_prevention_prompt(
            data_context.get('prevention_metrics', {}),
            data_context.get('control_systems', {}),
            data_context.get('monitoring_frameworks', {})
        )
    else:
        # Default detection analysis
        prompt = create_anomaly_detection_prompt(
            filter_context.get('theme', 'All Themes'),
            filter_context.get('region', 'All Regions'),
            data_context.get('anomaly_summary', {}),
            data_context.get('detection_metrics', {}),
            data_context.get('anomalous_patterns', {})
        )
    
    return call_o1_api(prompt)
