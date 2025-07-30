import streamlit as st
import pandas as pd
from typing import Dict, List, Any

def get_azure_openai_client(model_type="4o"):
    """Get Azure OpenAI client from commonconst with safe fallback"""
    try:
        # Import here to avoid circular imports
        from ..commonconst import client_4o, DEPLOYMENT_4O
        if client_4o is None:
            return None
        return client_4o
    except Exception as e:
        st.warning(f"Azure OpenAI GPT-4o client not available: {e}")
        return None

def create_model_prediction_prompt(
    country: str,
    theme: str, 
    strategic_priority_code: float,
    predicted_sdgs: List[str],
    predicted_agencies: List[str],
    historical_context: Dict[str, Any]
) -> str:
    """Create GPT-4o prompt for analyzing ML model predictions"""
    
    # Extract historical data summaries
    funding_summary = ""
    if 'funding_data' in historical_context and not historical_context['funding_data'].empty:
        funding_df = historical_context['funding_data']
        total_required = funding_df['Total required resources'].sum()
        funding_summary = f"Total required resources: ${total_required:,.2f}"
        
        # Add year-by-year breakdown if available
        year_columns = [col for col in funding_df.columns if any(year in col for year in ['2020', '2021', '2022', '2023', '2024', '2025', '2026'])]
        if year_columns:
            funding_summary += "\nFunding trends by year:\n"
            for col in year_columns:
                if 'Required' in col:
                    year = col.split()[0]
                    amount = funding_df[col].sum()
                    funding_summary += f"- {year}: ${amount:,.2f} required\n"
    
    anomaly_info = ""
    if 'is_anomalous' in historical_context:
        anomaly_status = "anomalous" if historical_context['is_anomalous'] else "normal"
        anomaly_info = f"Anomaly status: {anomaly_status} funding patterns"
    
    performance_info = ""
    if 'performance_label' in historical_context:
        performance_info = f"Performance classification: {historical_context['performance_label']}"
    
    historical_collabs = ""
    if 'historical_sdg_goals' in historical_context:
        historical_collabs += f"Historical SDG Goals: {', '.join(historical_context['historical_sdg_goals'])}\n"
    if 'historical_agencies' in historical_context:
        historical_collabs += f"Historical UN Agencies: {', '.join(historical_context['historical_agencies'][:10])}"  # Limit to first 10
    
    prompt = f"""
You are a UN Strategic Policy Analyst with expertise in international development, SDG implementation, and UN agency coordination. 

Analyze the following AI model predictions and historical data to provide strategic insights:

**CONTEXT:**
- Country: {country}
- Theme: {theme}
- Strategic Priority Code: {strategic_priority_code}

**AI MODEL PREDICTIONS:**
- Predicted SDG Goals: {', '.join(predicted_sdgs) if predicted_sdgs else 'None predicted'}
- Predicted UN Agencies: {', '.join(predicted_agencies) if predicted_agencies else 'None predicted'}

**HISTORICAL CONTEXT:**
{funding_summary}
{anomaly_info}
{performance_info}
{historical_collabs}

**ANALYSIS REQUEST:**
Please provide a comprehensive strategic analysis with the following sections:

1. **PREDICTION VALIDATION** (2-3 sentences)
   - Assess how well the AI predictions align with historical patterns
   - Identify any surprising or noteworthy prediction differences

2. **STRATEGIC IMPLICATIONS** (3-4 sentences)
   - What do these predictions suggest about {country}'s development priorities in {theme}?
   - How do the predicted SDG goals interconnect and what synergies exist?
   - What does the agency selection indicate about implementation approach?

3. **RISK ASSESSMENT** (2-3 sentences)
   - Based on the anomaly status and performance history, what are key risks?
   - Are there potential coordination challenges between predicted agencies?

4. **ACTIONABLE RECOMMENDATIONS** (3-4 bullet points)
   - Specific next steps for {country} in the {theme} sector
   - Key partnerships to prioritize among predicted agencies
   - Areas requiring additional attention or resources

5. **FUNDING INSIGHTS** (2-3 sentences)
   - Analysis of funding patterns and resource requirements
   - Recommendations for resource mobilization strategy

Keep your analysis concise, actionable, and grounded in UN development best practices. Focus on practical insights that can inform strategic decision-making.
"""
    
    return prompt

def create_comparative_model_prompt(
    multiple_predictions: List[Dict[str, Any]]
) -> str:
    """Create GPT-4o prompt for comparing multiple country/theme combinations"""
    
    comparison_data = ""
    for i, pred in enumerate(multiple_predictions, 1):
        comparison_data += f"""
**Scenario {i}:**
- Country: {pred['country']}
- Theme: {pred['theme']}
- Priority Code: {pred['strategic_priority_code']}
- Predicted SDGs: {', '.join(pred['predicted_sdgs'])}
- Predicted Agencies: {', '.join(pred['predicted_agencies'])}
- Performance: {pred.get('performance_label', 'Unknown')}
- Anomaly Status: {'Anomalous' if pred.get('is_anomalous', False) else 'Normal'}

"""
    
    prompt = f"""
You are a UN Strategic Policy Analyst conducting a comparative analysis of multiple development scenarios.

**SCENARIOS TO COMPARE:**
{comparison_data}

**COMPARATIVE ANALYSIS REQUEST:**

1. **PATTERN ANALYSIS** (3-4 sentences)
   - What common patterns emerge across these scenarios?
   - Which SDG goals appear most frequently and why?
   - Are there regional or thematic trends?

2. **AGENCY COORDINATION** (2-3 sentences)
   - Which UN agencies appear across multiple scenarios?
   - What does this suggest about core implementation capacity?

3. **RESOURCE ALLOCATION INSIGHTS** (3-4 sentences)
   - How should resources be prioritized across these scenarios?
   - Which scenarios present the highest risk/reward profiles?
   - Are there opportunities for cross-scenario synergies?

4. **STRATEGIC RECOMMENDATIONS** (4-5 bullet points)
   - Priority ranking for implementation
   - Key coordination mechanisms needed
   - Resource sharing opportunities
   - Risk mitigation strategies

Focus on actionable insights for UN strategic planning and resource allocation.
"""
    
    return prompt

def create_model_validation_prompt(
    model_performance: Dict[str, Any],
    prediction_accuracy: Dict[str, Any],
    feature_importance: Dict[str, Any]
) -> str:
    """Create GPT-4o prompt for model validation and improvement analysis"""
    
    return f"""
You are a UN Data Science and Strategic Analysis expert evaluating ML model performance for UN development planning.

**MODEL PERFORMANCE DATA:**
{model_performance}

**PREDICTION ACCURACY METRICS:**
{prediction_accuracy}

**FEATURE IMPORTANCE ANALYSIS:**
{feature_importance}

**MODEL VALIDATION REQUEST:**

1. **MODEL RELIABILITY** (3-4 sentences)
   - Assess the overall model performance and reliability
   - Identify strengths and limitations of current predictions
   - Evaluate confidence levels for strategic decision-making

2. **PREDICTION INSIGHTS** (3-4 sentences)
   - What patterns do the models reveal about UN collaboration?
   - Which features drive the most accurate predictions?
   - How do country, theme, and priority code influence outcomes?

3. **MODEL LIMITATIONS** (2-3 sentences)
   - What are the key uncertainties and blind spots?
   - Where should human judgment supplement model predictions?

4. **IMPROVEMENT RECOMMENDATIONS** (3-4 bullet points)
   - Data quality enhancements needed
   - Additional features that could improve accuracy
   - Model refinement opportunities
   - Validation and testing improvements

Focus on practical insights for model enhancement and strategic application.
"""

def format_gpt4o_response(response_text: str) -> str:
    """Format the GPT-4o response for better display in Streamlit"""
    formatted_response = response_text.replace("**", "**")
    formatted_response = formatted_response.replace("1. **", "\n### 1. **")
    formatted_response = formatted_response.replace("2. **", "\n### 2. **")
    formatted_response = formatted_response.replace("3. **", "\n### 3. **")
    formatted_response = formatted_response.replace("4. **", "\n### 4. **")
    formatted_response = formatted_response.replace("5. **", "\n### 5. **")
    
    return formatted_response

def call_gpt4o_api(prompt: str) -> str:
    """Call Azure OpenAI GPT-4o API for strategic analysis"""
    try:
        client = get_azure_openai_client()
        if client is None:
            return "🤖 Azure OpenAI GPT-4o service is currently unavailable. AI strategic analysis features require proper configuration."
        
        # Import deployment name from commonconst
        from ..commonconst import DEPLOYMENT_4O
        
        if DEPLOYMENT_4O is None:
            return "🤖 Azure OpenAI GPT-4o deployment configuration missing. Please check system settings."
        
        response = client.chat.completions.create(
            model=DEPLOYMENT_4O,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior UN Strategic Policy Analyst with extensive experience in international development, SDG implementation, and UN agency coordination. Provide concise, actionable strategic insights based on AI model predictions and historical data."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.3,
            top_p=0.9
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"AI strategic analysis temporarily unavailable: {str(e)}"
        return f"🤖 {error_msg}. Please try again later or contact system administrator.\n\n### Fallback Analysis Available\n\nThe system detected a temporary issue with AI services. Strategic insights can still be generated using local analysis tools."

def get_strategic_insights(
    country: str,
    theme: str,
    strategic_priority_code: float,
    predicted_sdgs: List[str],
    predicted_agencies: List[str],
    historical_context: Dict[str, Any]
) -> str:
    """Main function to get strategic insights using Azure OpenAI GPT-4o"""
    try:
        prompt = create_model_prediction_prompt(
            country, theme, strategic_priority_code,
            predicted_sdgs, predicted_agencies, historical_context
        )
        
        # Call Azure OpenAI GPT-4o API
        response = call_gpt4o_api(prompt)
        
        # Format the response
        formatted_response = format_gpt4o_response(response)
        
        return formatted_response
        
    except Exception as e:
        st.error(f"Error generating strategic insights: {str(e)}")
        return f"Error generating strategic insights: {str(e)}"

def get_comparative_insights(
    multiple_predictions: List[Dict[str, Any]]
) -> str:
    """Get comparative analysis for multiple predictions"""
    try:
        prompt = create_comparative_model_prompt(multiple_predictions)
        response = call_gpt4o_api(prompt)
        formatted_response = format_gpt4o_response(response)
        return formatted_response
    except Exception as e:
        st.error(f"Error generating comparative insights: {str(e)}")
        return f"Error generating comparative insights: {str(e)}"

def get_model_validation_insights(
    model_performance: Dict[str, Any],
    prediction_accuracy: Dict[str, Any],
    feature_importance: Dict[str, Any]
) -> str:
    """Get model validation and improvement insights"""
    try:
        prompt = create_model_validation_prompt(model_performance, prediction_accuracy, feature_importance)
        response = call_gpt4o_api(prompt)
        formatted_response = format_gpt4o_response(response)
        return formatted_response
    except Exception as e:
        st.error(f"Error generating model validation insights: {str(e)}")
        return f"Error generating model validation insights: {str(e)}"
