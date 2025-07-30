import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from openai import AzureOpenAI

def get_azure_openai_client(model_type="4o"):
    """Initialize Azure OpenAI client for GPT-4o model"""
    try:
        client = AzureOpenAI(
            api_key=st.secrets["AZURE_OPENAI_4O_API_KEY"],
            api_version=st.secrets["AZURE_OPENAI_4O_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_OPENAI_4O_ENDPOINT"]
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI GPT-4o client: {e}")
        return None

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
    """Create the system prompt for the UN Financial Intelligence Chatbot with dynamic theme awareness"""
    try:
        from src.commonconst import get_theme_list, get_region_list
        
        # Get current available themes and regions dynamically
        available_themes = get_theme_list()
        available_regions = get_region_list()
        
        themes_text = ", ".join(available_themes[:10]) if available_themes else "various UN focus areas"  # Limit to first 10 for readability
        if len(available_themes) > 10:
            themes_text += f" (and {len(available_themes) - 10} other themes)"
            
        regions_text = ", ".join(available_regions) if available_regions else "multiple global regions"
        
    except ImportError:
        # Fallback if import fails
        themes_text = "education, governance, environment, gender equality, youth development, and other UN focus areas"
        regions_text = "Africa, Asia Pacific, Arab States, Europe & Central Asia, Latin America"
        available_themes = []
        available_regions = []
    
    return f"""
You are the UN Financial Intelligence Assistant, an expert AI analyst specializing in UN financial data, development projects, and strategic resource allocation. You have deep knowledge of:

- UN System operations and agency collaboration
- Sustainable Development Goals (SDG) implementation
- International development financing mechanisms
- Regional development challenges and opportunities
- Strategic priority frameworks and planning

**CURRENT DATA SCOPE:**
- Available Regions: {regions_text}
- Available Themes: {themes_text}
- Data Coverage: {len(available_regions)} regions, {len(available_themes)} thematic areas
- Time Period: 2020-2026 with projections

**YOUR CAPABILITIES:**
- Analyze UN financial data patterns and trends across all available themes and regions
- Identify funding gaps and resource allocation opportunities
- Provide strategic recommendations for development planning
- Explain complex financial relationships in clear terms
- Compare regional and thematic development priorities
- Adapt analysis to the specific themes and regions in the current dataset

**COMMUNICATION STYLE:**
- Professional yet accessible language
- Use specific data points and evidence from available themes/regions
- Provide actionable insights and recommendations
- Structure responses clearly with headers when appropriate
- Include relevant emojis for visual clarity (📊💰🌍🎯)
- Reference specific themes and regions from the current data scope

**RESPONSE FORMAT:**
- Start with a brief direct answer
- Provide supporting data analysis with specific theme/region references
- Include strategic implications
- End with actionable recommendations when relevant

Always base your responses on the provided data context and maintain focus on UN development objectives and best practices. Adapt your analysis to the specific themes and regions available in the current dataset.
"""

def create_chatbot_analysis_prompt(
    user_question: str,
    data_summary: str,
    region_filter: str,
    theme_filter: str,
    raw_data_sample: str = ""
) -> str:
    """Create a comprehensive prompt for chatbot financial data analysis with dynamic theme support"""
    
    try:
        from src.commonconst import get_theme_list, get_region_list, get_available_themes_for_region
        
        # Get current data scope
        all_themes = get_theme_list()
        all_regions = get_region_list()
        
        # Provide context about available data
        data_scope_context = f"""
**AVAILABLE DATA SCOPE:**
- Total Themes: {len(all_themes)} ({', '.join(all_themes[:5])}{'...' if len(all_themes) > 5 else ''})
- Total Regions: {len(all_regions)} ({', '.join(all_regions)})
- Current Filter - Region: {region_filter if region_filter != 'All Regions' else 'All Available Regions'}
- Current Filter - Theme: {theme_filter if theme_filter != 'All Themes' else 'All Available Themes'}
"""
        
        # Add region-specific theme availability if region is selected
        if region_filter and region_filter != 'All Regions':
            available_themes_for_region = get_available_themes_for_region(region_filter)
            if available_themes_for_region:
                data_scope_context += f"- Available Themes for {region_filter}: {', '.join(available_themes_for_region)}\n"
        
    except ImportError:
        data_scope_context = ""
    
    prompt = f"""
**USER QUESTION:** {user_question}

{data_scope_context}

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

def create_chatbot_trend_prompt(
    user_question: str,
    trend_data: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    comparative_metrics: Dict[str, Any]
) -> str:
    """Create specialized prompt for trend analysis questions"""
    
    return f"""
**USER TREND QUESTION:** {user_question}

**TREND DATA ANALYSIS:**
{trend_data}

**TEMPORAL PATTERNS:**
{temporal_analysis}

**COMPARATIVE METRICS:**
{comparative_metrics}

**TREND ANALYSIS REQUEST:**
Provide comprehensive trend analysis focusing on:

1. **TREND IDENTIFICATION** 📈
   - Key patterns in the data over time
   - Growth/decline trajectories
   - Seasonal or cyclical patterns

2. **DRIVING FACTORS** 🔍
   - What factors are driving these trends?
   - External influences (economic, political, social)
   - Internal UN system changes

3. **COMPARATIVE INSIGHTS** ⚖️
   - How do different regions/themes compare?
   - Performance benchmarking
   - Best and worst performers

4. **FUTURE IMPLICATIONS** 🔮
   - What do current trends suggest for future planning?
   - Early warning indicators
   - Strategic recommendations for trend optimization

Focus on actionable trend intelligence for strategic planning and resource allocation.
"""

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
    """Main function to get chatbot response for financial data queries"""
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
        
        # Determine if this is a trend question for specialized handling
        is_trend_question = any(word in user_question.lower() for word in ['trend', 'time', 'year', 'over time', 'evolution', 'change'])
        
        if is_trend_question:
            # Create trend-specific analysis
            trend_data = {"data_analysis": data_analysis, "sample": sample_data}
            temporal_analysis = {"time_scope": "2020-2026", "filtered_scope": f"{region_filter} | {theme_filter}"}
            comparative_metrics = {"total_projects": len(filtered_df), "coverage": f"{data_summary}"}
            
            analysis_prompt = create_chatbot_trend_prompt(
                user_question,
                trend_data,
                temporal_analysis,
                comparative_metrics
            )
        else:
            # Create the standard analysis prompt
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
