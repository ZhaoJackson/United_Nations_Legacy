"""
Setup configuration for UN Financial Intelligence Dashboard
"""

from setuptools import setup, find_packages

setup(
    name="united_nations_legacy",
    version="1.0.0",
    description="UN Financial Intelligence Dashboard",
    author="UN Financial Intelligence Team",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.47.0,<2.0.0",
        "openai>=1.35.0,<2.0.0",
        "pandas>=2.0.0,<2.3.0",
        "numpy>=1.24.0,<1.27.0",
        "plotly>=5.15.0,<6.0.0",
        "scikit-learn>=1.3.0,<1.6.0",
        "xgboost>=1.7.0,<3.0.0",
        "joblib>=1.3.0,<2.0.0",
        "pycountry>=22.0.0,<25.0.0",
    ],
    package_data={
        'pages': ['style/*.css'],
        'src': ['outputs/**/*'],
    },
) 