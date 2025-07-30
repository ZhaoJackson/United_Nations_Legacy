"""
UN Financial Intelligence Dashboard - Specialized Prompt Modules

This package contains specialized prompt modules for different AI analysis purposes:
- dashboard.py: O1 model prompts for main dashboard financial analysis
- funding_prediction.py: O1 model prompts for funding prediction analysis  
- anomaly_detection.py: O1 model prompts for anomaly detection analysis
- agency_performance.py: O1 model prompts for agency performance analysis
- models.py: GPT-4o model prompts for strategic insights based on ML predictions
- chatbot.py: GPT-4o model prompts for conversational financial data analysis

Each module is optimized for its specific use case with tailored prompts and data interactions.
"""

# Main function imports for backwards compatibility and easy access
from .dashboard import get_dashboard_insights
from .funding_prediction import get_funding_prediction_insights
from .anomaly_detection import get_anomaly_detection_insights
from .agency_performance import get_agency_performance_insights
from .models import get_strategic_insights, get_comparative_insights, get_model_validation_insights
from .chatbot import get_chatbot_response

__version__ = "1.0.0"
__author__ = "UN Financial Intelligence Team"

# Export main functions
__all__ = [
    'get_dashboard_insights',
    'get_funding_prediction_insights', 
    'get_anomaly_detection_insights',
    'get_agency_performance_insights',
    'get_strategic_insights',
    'get_comparative_insights',
    'get_model_validation_insights',
    'get_chatbot_response'
] 