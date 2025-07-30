# Data Files

This repository contains analysis code for UN Financial Intelligence Dashboard. Due to file size constraints, the following data files are excluded from version control but are required for the application to function:

## Required Local Data Files

### Notebooks Directory (`src/notebooks/`)
- `Financial.csv` (~10.4 MB) - Main financial dataset with funding, expenditure, and gap analysis
- `SDG_Goals.csv` (~2.3 MB) - Sustainable Development Goals mapping and analysis data
- `UN_Agencies.csv` (~2.5 MB) - UN agency collaboration and performance data

### Data Directory (`src/data/`)
The `src/data/` directory contains regional Excel files organized by geographic regions:
- `Africa/` - 11 thematic Excel files (crime, digital, education, environment, food, gender, governance, poverty, water, work, youth)
- `Arab States/` - 11 thematic Excel files
- `Asia Pacific/` - 11 thematic Excel files  
- `Europe and Central Asia/` - 11 thematic Excel files
- `Latin America and the Caribbean/` - 11 thematic Excel files

## Generated Output Files

The application also uses processed data files in:
- `src/outputs/data_output/` - Cleaned datasets
- `src/outputs/model_output/` - ML model outputs and predictions

## Setup Instructions

1. Ensure all required data files are present in their respective directories
2. Run the data processing notebooks in `src/notebooks/` if needed
3. The Streamlit app (`app.py`) will automatically load the processed data files

## File Size Summary
- Total excluded data: ~55+ files, >15 MB
- Reason for exclusion: GitHub file size limits and data privacy considerations 