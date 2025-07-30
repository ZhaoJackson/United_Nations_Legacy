"""
Dynamic Analysis Module for UN Financial Intelligence Dashboard

This module provides utilities to automatically adapt analysis and modeling
to new themes and regions without manual code changes. Updated to work with
the new modular prompt structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import importlib
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicDataProcessor:
    """Handles dynamic data processing for new themes and regions"""
    
    def __init__(self, data_dir: str = "src/data", output_dir: str = "src/outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_output_dir = self.output_dir / "data_output"
        self.model_output_dir = self.output_dir / "model_output"
        
        # Ensure output directories exist
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_data_structure(self) -> Dict[str, Any]:
        """Discover all available themes and regions from the file system"""
        structure = {
            'regions': {},
            'themes': set(),
            'total_files': 0,
            'missing_files': []
        }
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return structure
        
        for region_dir in self.data_dir.iterdir():
            if region_dir.is_dir() and not region_dir.name.startswith('.'):
                region_name = region_dir.name
                structure['regions'][region_name] = []
                
                for theme_file in region_dir.iterdir():
                    if theme_file.suffix == '.xlsx' and not theme_file.name.startswith('.'):
                        theme_name = theme_file.stem
                        structure['regions'][region_name].append(theme_name)
                        structure['themes'].add(theme_name)
                        structure['total_files'] += 1
        
        structure['themes'] = sorted(list(structure['themes']))
        logger.info(f"Discovered {len(structure['regions'])} regions and {len(structure['themes'])} themes")
        return structure
    
    def load_theme_data(self, theme: str, regions: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data for a specific theme across all or specified regions"""
        if regions is None:
            regions = [d.name for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        combined_data = []
        
        for region in regions:
            theme_file = self.data_dir / region / f"{theme}.xlsx"
            if theme_file.exists():
                try:
                    df = pd.read_excel(theme_file)
                    df['Region'] = region
                    df['Theme'] = theme
                    combined_data.append(df)
                    logger.debug(f"Loaded {theme} data for {region}")
                except Exception as e:
                    logger.error(f"Error loading {theme_file}: {e}")
            else:
                logger.warning(f"Missing file: {theme_file}")
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined {theme} data: {len(result)} rows across {len(combined_data)} regions")
            return result
        else:
            logger.warning(f"No data found for theme: {theme}")
            return pd.DataFrame()
    
    def process_all_themes(self) -> Dict[str, pd.DataFrame]:
        """Process all available themes and return combined datasets"""
        structure = self.discover_data_structure()
        processed_data = {}
        
        for theme in structure['themes']:
            logger.info(f"Processing theme: {theme}")
            theme_data = self.load_theme_data(theme)
            if not theme_data.empty:
                processed_data[theme] = theme_data
        
        return processed_data
    
    def create_unified_dataset(self, save_to_file: bool = True) -> pd.DataFrame:
        """Create a unified dataset from all themes and regions"""
        all_theme_data = self.process_all_themes()
        
        if not all_theme_data:
            logger.warning("No theme data found")
            return pd.DataFrame()
        
        # Combine all theme data
        unified_data = pd.concat(all_theme_data.values(), ignore_index=True)
        
        # Standardize columns
        unified_data = self._standardize_columns(unified_data)
        
        if save_to_file:
            output_file = self.data_output_dir / "unified_data.csv"
            unified_data.to_csv(output_file, index=False)
            logger.info(f"Saved unified dataset to {output_file}")
        
        return unified_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across all themes"""
        # Common column mappings
        column_mappings = {
            'country': 'Country',
            'region': 'Region',
            'theme': 'Theme',
            'year': 'Year',
            'required': 'Required',
            'available': 'Available',
            'expenditure': 'Expenditure',
            'agency': 'Agency',
            'agencies': 'Agencies',
            'sdg': 'SDG',
            'sdg_goals': 'SDG Goals',
            'strategic_priority': 'Strategic Priority',
            'strategic_priority_code': 'Strategic Priority Code'
        }
        
        # Rename columns
        df_columns = df.columns.str.lower()
        new_columns = {}
        
        for old_col, new_col in column_mappings.items():
            matches = [col for col in df.columns if old_col in col.lower()]
            if matches:
                new_columns[matches[0]] = new_col
        
        df = df.rename(columns=new_columns)
        return df

class DynamicModelManager:
    """Manages dynamic model training and updating for new themes"""
    
    def __init__(self, output_dir: str = "src/outputs/model_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def needs_retraining(self, theme_list: List[str]) -> bool:
        """Check if models need retraining based on available themes"""
        # Check if model files exist
        sdg_model_file = self.output_dir / "SDG_model.pkl"
        agency_model_file = self.output_dir / "Agency_model.pkl"
        
        if not (sdg_model_file.exists() and agency_model_file.exists()):
            return True
        
        # Check if theme list has changed
        theme_cache_file = self.output_dir / "trained_themes.txt"
        if theme_cache_file.exists():
            with open(theme_cache_file, 'r') as f:
                cached_themes = set(f.read().strip().split('\n'))
            
            current_themes = set(theme_list)
            if cached_themes != current_themes:
                logger.info(f"Theme changes detected. Cached: {cached_themes}, Current: {current_themes}")
                return True
        else:
            return True
        
        return False
    
    def update_theme_cache(self, theme_list: List[str]):
        """Update the cache of trained themes"""
        theme_cache_file = self.output_dir / "trained_themes.txt"
        with open(theme_cache_file, 'w') as f:
            f.write('\n'.join(sorted(theme_list)))
        logger.info(f"Updated theme cache with {len(theme_list)} themes")

class DynamicPromptUpdater:
    """Updates prompt modules to handle new themes dynamically"""
    
    def __init__(self):
        self.prompt_modules = {
            'dashboard': 'src.prompt.dashboard',
            'funding_prediction': 'src.prompt.funding_prediction', 
            'anomaly_detection': 'src.prompt.anomaly_detection',
            'agency_performance': 'src.prompt.agency_performance',
            'models': 'src.prompt.models',
            'chatbot': 'src.prompt.chatbot'
        }
    
    def check_prompt_module_compatibility(self, module_name: str) -> Dict[str, Any]:
        """Check if a prompt module can handle dynamic themes"""
        try:
            module = importlib.import_module(self.prompt_modules[module_name])
            
            # Check for required functions
            required_functions = [
                'get_azure_openai_client',
                'call_o1_api' if module_name != 'models' and module_name != 'chatbot' else 'call_gpt4o_api'
            ]
            
            missing_functions = []
            for func in required_functions:
                if not hasattr(module, func):
                    missing_functions.append(func)
            
            # Check for theme-aware prompt functions
            theme_aware_functions = [func for func in dir(module) if 'prompt' in func.lower()]
            
            return {
                'module': module_name,
                'compatible': len(missing_functions) == 0,
                'missing_functions': missing_functions,
                'theme_aware_functions': theme_aware_functions,
                'module_loaded': True
            }
            
        except ImportError as e:
            return {
                'module': module_name,
                'compatible': False,
                'error': str(e),
                'module_loaded': False
            }
    
    def validate_all_prompt_modules(self) -> Dict[str, Dict[str, Any]]:
        """Validate all prompt modules for theme compatibility"""
        results = {}
        for module_name in self.prompt_modules.keys():
            results[module_name] = self.check_prompt_module_compatibility(module_name)
        return results

class DynamicDashboardUpdater:
    """Updates dashboard components to handle new themes dynamically"""
    
    def __init__(self):
        self.pages_dir = Path("pages")
        self.page_modules = {
            'main_page.py': 'src.prompt.dashboard',
            'prediction.py': ['src.prompt.funding_prediction', 'src.prompt.anomaly_detection', 'src.prompt.agency_performance'],
            'model.py': 'src.prompt.models',
            'bot.py': 'src.prompt.chatbot',
            'overview.py': None  # No prompt module
        }
    
    def get_required_dashboard_functions(self) -> Dict[str, List[str]]:
        """Get list of functions that need to be theme-aware in each page"""
        return {
            'overview.py': ['generate_overview_metrics', 'create_summary_charts'],
            'main_page.py': ['get_dashboard_insights', 'create_theme_filters', 'generate_visualizations'],
            'prediction.py': ['get_funding_prediction_insights', 'get_anomaly_detection_insights', 'get_agency_performance_insights'],
            'model.py': ['get_strategic_insights', 'get_prediction_inputs', 'display_model_results'],
            'bot.py': ['get_chatbot_response', 'load_financial_data', 'process_user_query']
        }
    
    def check_theme_compatibility(self, page_file: str) -> bool:
        """Check if a page file uses dynamic theme loading"""
        page_path = self.pages_dir / page_file
        if not page_path.exists():
            return False
        
        with open(page_path, 'r') as f:
            content = f.read()
        
        # Check for dynamic theme functions
        dynamic_indicators = [
            'get_theme_list()',
            'get_region_list()',
            'discover_available_themes',
            'refresh_data()',
            'DynamicDataProcessor',
            'src.prompt.'  # New modular prompt structure
        ]
        
        return any(indicator in content for indicator in dynamic_indicators)
    
    def check_prompt_integration(self, page_file: str) -> Dict[str, Any]:
        """Check if a page properly integrates with the new prompt modules"""
        page_path = self.pages_dir / page_file
        if not page_path.exists():
            return {'integrated': False, 'error': 'Page file not found'}
        
        with open(page_path, 'r') as f:
            content = f.read()
        
        integration_status = {
            'integrated': False,
            'prompt_modules_used': [],
            'old_imports': [],
            'issues': []
        }
        
        # Check for new modular imports
        expected_modules = self.page_modules.get(page_file, [])
        if expected_modules:
            if isinstance(expected_modules, str):
                expected_modules = [expected_modules]
            
            for module in expected_modules:
                if module in content:
                    integration_status['prompt_modules_used'].append(module)
        
        # Check for old imports that should be updated
        old_import_patterns = [
            'from src.prompt import',
            'src.prompt.get_o1_dashboard_insights',
            'src.prompt.get_strategic_insights',
            'src.prompt.get_chatbot_response'
        ]
        
        for pattern in old_import_patterns:
            if pattern in content:
                integration_status['old_imports'].append(pattern)
        
        # Determine integration status
        has_expected_modules = len(integration_status['prompt_modules_used']) > 0
        has_old_imports = len(integration_status['old_imports']) > 0
        
        if has_expected_modules and not has_old_imports:
            integration_status['integrated'] = True
        elif has_old_imports:
            integration_status['issues'].append('Contains old prompt imports that should be updated')
        elif not has_expected_modules and expected_modules:
            integration_status['issues'].append('Missing expected prompt module imports')
        
        return integration_status

def auto_update_analysis():
    """Main function to automatically update analysis for new themes"""
    import time
    start_time = time.time()
    
    logger.info("Starting automatic analysis update with new prompt structure...")
    
    # Initialize processors
    data_processor = DynamicDataProcessor()
    model_manager = DynamicModelManager()
    prompt_updater = DynamicPromptUpdater()
    dashboard_updater = DynamicDashboardUpdater()
    
    # Discover current data structure
    structure = data_processor.discover_data_structure()
    themes = structure['themes']
    regions = list(structure['regions'].keys())
    
    logger.info(f"Found {len(themes)} themes and {len(regions)} regions")
    
    # Check if models need retraining
    if model_manager.needs_retraining(themes):
        logger.info("Models need retraining due to theme changes")
        logger.info("Please run the modeling notebooks to retrain with new themes")
    
    # Validate prompt modules
    prompt_validation = prompt_updater.validate_all_prompt_modules()
    logger.info("Prompt Module Validation Results:")
    for module, status in prompt_validation.items():
        status_icon = "✓" if status['compatible'] else "✗"
        logger.info(f"  {module}: {status_icon} {'Compatible' if status['compatible'] else 'Issues found'}")
    
    # Create unified dataset
    unified_data = data_processor.create_unified_dataset()
    
    # Check dashboard compatibility and prompt integration
    page_files = ['overview.py', 'main_page.py', 'prediction.py', 'model.py', 'bot.py']
    dashboard_status = {}
    
    for page_file in page_files:
        theme_compatible = dashboard_updater.check_theme_compatibility(page_file)
        prompt_integration = dashboard_updater.check_prompt_integration(page_file)
        
        dashboard_status[page_file] = {
            'theme_compatible': theme_compatible,
            'prompt_integration': prompt_integration
        }
        
        # Log results
        theme_icon = "✓" if theme_compatible else "✗"
        prompt_icon = "✓" if prompt_integration['integrated'] else "✗"
        logger.info(f"{page_file}:")
        logger.info(f"  Theme compatibility: {theme_icon}")
        logger.info(f"  Prompt integration: {prompt_icon}")
        
        if prompt_integration['issues']:
            for issue in prompt_integration['issues']:
                logger.warning(f"    Issue: {issue}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis update completed in {elapsed_time:.2f}s")
    
    return {
        'themes_found': themes,
        'regions_found': regions,
        'unified_data_shape': unified_data.shape,
        'models_need_retraining': model_manager.needs_retraining(themes),
        'prompt_modules_status': prompt_validation,
        'dashboard_status': dashboard_status,
        'processing_time': elapsed_time
    }

def get_performance_metrics():
    """Get performance metrics for AI insight optimization"""
    return {
        'recommended_max_thinking_time': 7.0,  # seconds
        'optimal_response_length': 400,  # words
        'structured_sections': 4,  # number of sections
        'section_max_words': 100  # words per section
    }

def validate_system_readiness():
    """Validate that the entire system is ready for dynamic theme adaptation"""
    logger.info("Validating system readiness for dynamic theme adaptation...")
    
    # Run auto update analysis
    results = auto_update_analysis()
    
    # Check overall readiness
    readiness_status = {
        'data_structure': len(results['themes_found']) > 0 and len(results['regions_found']) > 0,
        'prompt_modules': all(status['compatible'] for status in results['prompt_modules_status'].values()),
        'dashboard_integration': all(
            status['theme_compatible'] and status['prompt_integration']['integrated'] 
            for status in results['dashboard_status'].values()
        ),
        'models_ready': not results['models_need_retraining']
    }
    
    overall_ready = all(readiness_status.values())
    
    logger.info(f"System Readiness Assessment:")
    logger.info(f"  Data Structure: {'✓' if readiness_status['data_structure'] else '✗'}")
    logger.info(f"  Prompt Modules: {'✓' if readiness_status['prompt_modules'] else '✗'}")
    logger.info(f"  Dashboard Integration: {'✓' if readiness_status['dashboard_integration'] else '✗'}")
    logger.info(f"  Models Ready: {'✓' if readiness_status['models_ready'] else '✗'}")
    logger.info(f"  Overall Ready: {'✓' if overall_ready else '✗'}")
    
    return {
        'overall_ready': overall_ready,
        'readiness_status': readiness_status,
        'detailed_results': results
    }

if __name__ == "__main__":
    # Run system validation
    validation_results = validate_system_readiness()
    print(f"System Validation Results: {validation_results}") 