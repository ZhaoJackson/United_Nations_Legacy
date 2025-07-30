"""
Fallback Data Module for UN Financial Intelligence Dashboard

This module provides sample/demo data when actual data files are not available,
ensuring the application can still run in demonstration mode.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_sample_financial_data():
    """Create sample financial data for demonstration purposes"""
    
    # Sample countries and regions
    countries = ['Afghanistan', 'Bangladesh', 'Ethiopia', 'Kenya', 'Nigeria', 'Pakistan', 'Somalia', 'South Sudan', 'Sudan', 'Yemen']
    regions = ['Africa', 'Asia', 'Middle East']
    themes = ['Health', 'Education', 'Infrastructure', 'Agriculture', 'Water & Sanitation', 'Climate Action']
    agencies = ['UNICEF', 'WHO', 'WFP', 'UNDP', 'UNHCR', 'UNESCO', 'FAO']
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    
    # Create sample data
    sample_data = []
    for _ in range(100):  # Create 100 sample records
        record = {
            'Country': np.random.choice(countries),
            'Region': np.random.choice(regions),
            'Theme': np.random.choice(themes),
            'Year': np.random.choice(years),
            'Agencies': ';'.join(np.random.choice(agencies, size=np.random.randint(1, 4), replace=False)),
            'Total required resources': np.random.uniform(1000000, 50000000),
            'Available resources': np.random.uniform(500000, 30000000),
            'Expenditure': np.random.uniform(400000, 25000000),
            'Strategic Priority Code': f"SP{np.random.randint(1, 6)}",
            'SDG Goals': f"SDG {np.random.randint(1, 17)}"
        }
        sample_data.append(record)
    
    return pd.DataFrame(sample_data)

def create_sample_sdg_data():
    """Create sample SDG goals data"""
    
    sdg_goals = [f"SDG {i}" for i in range(1, 18)]
    countries = ['Afghanistan', 'Bangladesh', 'Ethiopia', 'Kenya', 'Nigeria']
    
    sample_data = []
    for _ in range(50):
        record = {
            'Country': np.random.choice(countries),
            'SDG Goals': np.random.choice(sdg_goals),
            'Theme': np.random.choice(['Health', 'Education', 'Infrastructure']),
            'Strategic Priority Code': f"SP{np.random.randint(1, 6)}"
        }
        sample_data.append(record)
    
    return pd.DataFrame(sample_data)

def create_sample_agency_data():
    """Create sample UN agencies data"""
    
    agencies = ['UNICEF', 'WHO', 'WFP', 'UNDP', 'UNHCR']
    countries = ['Afghanistan', 'Bangladesh', 'Ethiopia', 'Kenya', 'Nigeria']
    
    sample_data = []
    for _ in range(50):
        record = {
            'Country': np.random.choice(countries),
            'Agencies': np.random.choice(agencies),
            'Theme': np.random.choice(['Health', 'Education', 'Infrastructure']),
            'Strategic Priority Code': f"SP{np.random.randint(1, 6)}"
        }
        sample_data.append(record)
    
    return pd.DataFrame(sample_data)

def get_fallback_data(data_type='financial'):
    """Get fallback data based on type"""
    
    if data_type == 'financial':
        return create_sample_financial_data()
    elif data_type == 'sdg':
        return create_sample_sdg_data()
    elif data_type == 'agency':
        return create_sample_agency_data()
    else:
        return pd.DataFrame()

def create_demo_message():
    """Create a demo mode message"""
    return """
    📋 **Demo Mode Active**
    
    This application is running with sample data for demonstration purposes.
    The actual UN JointWork Plans data is not available in this environment.
    
    All functionality is preserved, but results are based on representative sample data.
    """

if __name__ == "__main__":
    # Test the fallback data creation
    print("Creating sample financial data...")
    financial_df = create_sample_financial_data()
    print(f"Created {len(financial_df)} financial records")
    
    print("\nCreating sample SDG data...")
    sdg_df = create_sample_sdg_data()
    print(f"Created {len(sdg_df)} SDG records")
    
    print("\nCreating sample agency data...")
    agency_df = create_sample_agency_data()
    print(f"Created {len(agency_df)} agency records") 