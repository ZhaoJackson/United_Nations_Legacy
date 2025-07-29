import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import json
from openai import AzureOpenAI

def get_azure_openai_client():
    """Initialize Azure OpenAI client with secrets from streamlit"""
    try:
        client = AzureOpenAI(
            api_key=st.secrets["AZURE_OPENAI_4O_API_KEY"],
            api_version=st.secrets["AZURE_OPENAI_4O_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_OPENAI_4O_ENDPOINT"]
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI client: {e}")
        return None

def create_gpt4o_analysis_prompt(
    country: str,
    theme: str, 
    strategic_priority_code: float,
    predicted_sdgs: List[str],
    predicted_agencies: List[str],
    historical_context: Dict[str, Any]
) -> str:
    """
    Create a comprehensive prompt for GPT-4o to analyze UN model predictions
    and provide strategic insights.
    """
    
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

def create_comparative_analysis_prompt(
    multiple_predictions: List[Dict[str, Any]]
) -> str:
    """
    Create a prompt for comparing multiple country/theme combinations
    """
    
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

def format_gpt4o_response(response_text: str) -> str:
    """
    Format the GPT-4o response for better display in Streamlit
    """
    # Add some basic formatting improvements
    formatted_response = response_text.replace("**", "**")
    formatted_response = formatted_response.replace("1. **", "\n### 1. **")
    formatted_response = formatted_response.replace("2. **", "\n### 2. **")
    formatted_response = formatted_response.replace("3. **", "\n### 3. **")
    formatted_response = formatted_response.replace("4. **", "\n### 4. **")
    formatted_response = formatted_response.replace("5. **", "\n### 5. **")
    
    return formatted_response

def call_gpt4o_api(prompt: str) -> str:
    """
    Call Azure OpenAI GPT-4o API for strategic analysis
    """
    try:
        client = get_azure_openai_client()
        if client is None:
            return "Error: Could not initialize Azure OpenAI client. Please check your API configuration."
        
        response = client.chat.completions.create(
            model=st.secrets["AZURE_OPENAI_4O_DEPLOYMENT"],
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
        st.error(f"Error calling Azure OpenAI API: {e}")
        return f"""
### Error Generating Strategic Insights

We encountered an issue connecting to the Azure OpenAI service. Here's a placeholder analysis:

### 1. **PREDICTION VALIDATION**
The AI predictions appear consistent with typical UN collaboration patterns for this country-theme combination. The predicted SDG goals show alignment with regional development priorities.

### 2. **STRATEGIC IMPLICATIONS** 
These predictions suggest a multi-sectoral approach focusing on sustainable development outcomes. The agency recommendations indicate a need for coordinated implementation across technical and operational domains.

### 3. **RISK ASSESSMENT**
Based on historical patterns, coordination challenges may arise between multiple agencies. Resource mobilization should be prioritized to ensure implementation success.

### 4. **ACTIONABLE RECOMMENDATIONS**
• Establish inter-agency coordination mechanism
• Develop integrated programming approach
• Prioritize resource mobilization efforts  
• Implement enhanced monitoring systems

### 5. **FUNDING INSIGHTS**
Historical funding patterns suggest consistent resource requirements. A diversified funding strategy is recommended to ensure sustainable implementation.

*Note: This is a fallback response. Please check the Azure OpenAI API configuration.*
"""

def get_strategic_insights(
    country: str,
    theme: str,
    strategic_priority_code: float,
    predicted_sdgs: List[str],
    predicted_agencies: List[str],
    historical_context: Dict[str, Any]
) -> str:
    """
    Main function to get strategic insights using Azure OpenAI GPT-4o
    """
    try:
        prompt = create_gpt4o_analysis_prompt(
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

# ============================================================================
# CHATBOT FUNCTIONALITY FOR FINANCIAL DATA ANALYSIS
# ============================================================================

def create_financial_data_summary(df: pd.DataFrame, region: str, theme: str) -> str:
    """Create a concise summary of the filtered financial data"""
    if df.empty:
        return "No data available for the current filters."
    
    # Basic statistics
    total_projects = len(df)
    total_required = df['Total required resources'].sum()
    total_available = df['Total available resources'].sum()
    funding_gap = total_required - total_available
    coverage_ratio = (total_available / total_required * 100) if total_required > 0 else 0
    
    # Top countries by funding need
    top_countries = df.groupby('Country')['Total required resources'].sum().sort_values(ascending=False).head(3)
    
    # Top SDG goals
    sdg_goals = df['SDG Goals'].dropna().str.split(';').explode().str.strip()
    top_sdgs = sdg_goals.value_counts().head(3)
    
    # Top agencies
    agencies = df['Agencies'].dropna().str.split(';').explode().str.strip()
    top_agencies = agencies.value_counts().head(3)
    
    summary = f"""
**CURRENT DATA SUMMARY** (Filters: {region} | {theme})
- Total Projects: {total_projects:,}
- Required Resources: ${total_required:,.0f}
- Available Resources: ${total_available:,.0f}
- Funding Gap: ${funding_gap:,.0f} ({coverage_ratio:.1f}% coverage)

**TOP COUNTRIES BY FUNDING NEED:**
{chr(10).join([f"• {country}: ${amount:,.0f}" for country, amount in top_countries.items()])}

**TOP SDG GOALS:**
{chr(10).join([f"• {goal}: {count} projects" for goal, count in top_sdgs.items()])}

**TOP UN AGENCIES:**
{chr(10).join([f"• {agency}: {count} projects" for agency, count in top_agencies.items()])}
"""
    return summary

def create_chatbot_system_prompt() -> str:
    """Create the system prompt for the UN Financial Intelligence Chatbot"""
    return """
You are the UN Financial Intelligence Assistant, an expert AI analyst specializing in UN financial data, development projects, and strategic resource allocation. You have deep knowledge of:

- UN System operations and agency collaboration
- Sustainable Development Goals (SDG) implementation
- International development financing mechanisms
- Regional development challenges and opportunities
- Strategic priority frameworks and planning

**YOUR CAPABILITIES:**
- Analyze UN financial data patterns and trends
- Identify funding gaps and resource allocation opportunities
- Provide strategic recommendations for development planning
- Explain complex financial relationships in clear terms
- Compare regional and thematic development priorities

**COMMUNICATION STYLE:**
- Professional yet accessible language
- Use specific data points and evidence
- Provide actionable insights and recommendations
- Structure responses clearly with headers when appropriate
- Include relevant emojis for visual clarity (📊💰🌍🎯)

**RESPONSE FORMAT:**
- Start with a brief direct answer
- Provide supporting data analysis
- Include strategic implications
- End with actionable recommendations when relevant

Always base your responses on the provided data context and maintain focus on UN development objectives and best practices.
"""

def create_chatbot_analysis_prompt(
    user_question: str,
    data_summary: str,
    region_filter: str,
    theme_filter: str,
    raw_data_sample: str = ""
) -> str:
    """Create a comprehensive prompt for chatbot financial data analysis"""
    
    prompt = f"""
**USER QUESTION:** {user_question}

**CURRENT DATA CONTEXT:**
{data_summary}

**ACTIVE FILTERS:**
- Region: {region_filter}
- Theme: {theme_filter}

**SAMPLE DATA REFERENCE:**
{raw_data_sample}

**ANALYSIS REQUEST:**
Please analyze the user's question in the context of the UN financial data provided. Focus on:

1. **DIRECT ANSWER** - Address the specific question asked
2. **DATA INSIGHTS** - Highlight key patterns, trends, or findings from the data
3. **STRATEGIC IMPLICATIONS** - What do these findings mean for UN operations and development goals?
4. **ACTIONABLE RECOMMENDATIONS** - Specific next steps or strategic recommendations

**FORMATTING GUIDELINES:**
- Use clear headers with emojis
- Include specific numbers and percentages where relevant
- Highlight key insights in **bold**
- Keep responses concise but comprehensive (300-500 words optimal)
- End with practical next steps when appropriate

Remember: You are analyzing real UN financial data to provide strategic intelligence for development planning and resource allocation decisions.
"""
    
    return prompt

def analyze_data_for_question(df: pd.DataFrame, question: str) -> str:
    """Analyze the data based on the user's question and return key insights"""
    if df.empty:
        return "No data available for analysis with current filters."
    
    question_lower = question.lower()
    insights = []
    
    # Funding gap analysis
    if any(word in question_lower for word in ['gap', 'shortage', 'need', 'deficit']):
        funding_gaps = df.groupby('Country').agg({
            'Total required resources': 'sum',
            'Total available resources': 'sum'
        })
        funding_gaps['Gap'] = funding_gaps['Total required resources'] - funding_gaps['Total available resources']
        top_gaps = funding_gaps[funding_gaps['Gap'] > 0].sort_values('Gap', ascending=False).head(5)
        
        insights.append("**TOP FUNDING GAPS BY COUNTRY:**")
        for country, row in top_gaps.iterrows():
            gap_pct = (row['Gap'] / row['Total required resources'] * 100)
            insights.append(f"• {country}: ${row['Gap']:,.0f} gap ({gap_pct:.1f}% of need)")
    
    # Country analysis
    if any(word in question_lower for word in ['country', 'countries', 'nation']):
        country_stats = df.groupby('Country').agg({
            'Total required resources': 'sum',
            'Total available resources': 'sum'
        }).sort_values('Total required resources', ascending=False).head(5)
        
        insights.append("**TOP COUNTRIES BY RESOURCE REQUIREMENTS:**")
        for country, row in country_stats.iterrows():
            coverage = (row['Total available resources'] / row['Total required resources'] * 100)
            insights.append(f"• {country}: ${row['Total required resources']:,.0f} ({coverage:.1f}% funded)")
    
    # Agency analysis
    if any(word in question_lower for word in ['agency', 'agencies', 'organization']):
        agencies = df['Agencies'].dropna().str.split(';').explode().str.strip()
        agency_counts = agencies.value_counts().head(5)
        
        insights.append("**TOP UN AGENCIES BY PROJECT COUNT:**")
        for agency, count in agency_counts.items():
            insights.append(f"• {agency}: {count} projects")
    
    # SDG analysis
    if any(word in question_lower for word in ['sdg', 'sustainable', 'goals', 'priorities']):
        sdgs = df['SDG Goals'].dropna().str.split(';').explode().str.strip()
        sdg_counts = sdgs.value_counts().head(5)
        
        insights.append("**TOP SDG PRIORITIES:**")
        for sdg, count in sdg_counts.items():
            insights.append(f"• {sdg}: {count} projects")
    
    # Time trends
    if any(word in question_lower for word in ['trend', 'time', 'year', 'over time']):
        year_columns = [col for col in df.columns if any(year in col for year in ['2020', '2021', '2022', '2023', '2024'])]
        required_cols = [col for col in year_columns if 'Required' in col]
        
        if required_cols:
            insights.append("**FUNDING TRENDS BY YEAR:**")
            for col in required_cols:
                year = col.split()[0]
                total = df[col].sum()
                insights.append(f"• {year}: ${total:,.0f} required")
    
    return "\n".join(insights) if insights else "Analysis performed but no specific patterns found for this question."

def get_chatbot_response(
    user_question: str,
    filtered_df: pd.DataFrame,
    region_filter: str,
    theme_filter: str
) -> str:
    """
    Main function to get chatbot response for financial data queries
    """
    try:
        # Create data summary
        data_summary = create_financial_data_summary(filtered_df, region_filter, theme_filter)
        
        # Analyze data for specific insights related to the question
        data_analysis = analyze_data_for_question(filtered_df, user_question)
        
        # Get a sample of raw data for context (first few rows)
        if not filtered_df.empty:
            sample_data = filtered_df[['Country', 'Theme', 'Total required resources', 'Total available resources']].head(3).to_string()
        else:
            sample_data = "No data available"
        
        # Create the analysis prompt
        analysis_prompt = create_chatbot_analysis_prompt(
            user_question,
            f"{data_summary}\n\n**SPECIFIC DATA ANALYSIS:**\n{data_analysis}",
            region_filter,
            theme_filter,
            sample_data
        )
        
        # Get system prompt
        system_prompt = create_chatbot_system_prompt()
        
        # Call Azure OpenAI API
        try:
            client = get_azure_openai_client()
            if client is None:
                return "❌ **Error:** Could not connect to Azure OpenAI service. Please check the configuration."
            
            response = client.chat.completions.create(
                model=st.secrets["AZURE_OPENAI_4O_DEPLOYMENT"],
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.3,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as api_error:
            # Fallback response with data analysis
            return f"""
### 📊 **Data Analysis Response**

**Question:** {user_question}

### 🔍 **Key Findings**
{data_analysis if data_analysis != "Analysis performed but no specific patterns found for this question." else "Based on your current filters, here's what I found in the data:"}

### 📈 **Data Summary**
{data_summary}

### 💡 **Recommendations**
- Consider adjusting filters to explore different regions or themes
- Focus on countries with the largest funding gaps for priority action
- Leverage agencies with strong collaboration patterns for efficiency

*Note: This is a data-driven response. Full AI analysis temporarily unavailable.*
"""
        
    except Exception as e:
        return f"""
### ❌ **Error Processing Request**

I encountered an issue analyzing your question: {str(e)}

### 📊 **Available Data Summary**
{create_financial_data_summary(filtered_df, region_filter, theme_filter) if not filtered_df.empty else "No data available with current filters."}

**Suggestion:** Try rephrasing your question or adjusting the region/theme filters.
"""
