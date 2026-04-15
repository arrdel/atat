"""Global Dataset Configuration Manager

Provides unified interface to dataset configuration across training, evaluation, and validation.
Supports debug, validation, and production stages with seamless switching.

USAGE:
    from atat.utils import get_dataset_manager
    
    # Get config for a phase
    manager = get_dataset_manager()
    config = manager.get_phase_config('A1_IMPORTANCE_ESTIMATOR_ABLATION')
    
    # Or get specific dataset
    config = manager.get_config('openwebtext', variant='full')
    
    # Or use preset stages
    manager = get_dataset_manager(preset='debug')
    config = manager.get_config('openwebtext')
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset configuration object"""
    name: str
    source: str
    variant: str
    num_tokens: int
    samples: Optional[int]
    cache_file: str
    description: str


class DatasetConfigManager:
    """Global dataset configuration manager"""
    
    def __init__(self, config_file: Optional[Path] = None, preset: str = 'production'):
        """
        Initialize manager
        
        Args:
            config_file: Path to datasets.yaml (auto-detect if None)
            preset: Stage preset (debug, validation, production)
        """
        if config_file is None:
            # Auto-detect config file location
            this_dir = Path(__file__).parent
            config_path = this_dir.parent / "configs" / "datasets.yaml"
        else:
            config_path = config_file
        
        self.config_file = config_path
        self.preset = preset
        self._load_config()
    
    def _load_config(self):
        """Load YAML configuration"""
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_phase_config(self, phase_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific research phase
        
        Args:
            phase_id: Phase identifier (e.g., 'A1_IMPORTANCE_ESTIMATOR_ABLATION')
        
        Returns:
            Phase configuration dictionary
        """
        phase_configs = self.config.get('phase_configurations', {})
        if phase_id not in phase_configs:
            raise ValueError(f"Phase '{phase_id}' not found in configuration")
        
        return phase_configs[phase_id]
    
    def get_config(self, dataset_name: str, variant: Optional[str] = None) -> DatasetConfig:
        """
        Get configuration for a specific dataset variant
        
        Args:
            dataset_name: Dataset name (e.g., 'openwebtext')
            variant: Size variant ('small', 'medium', 'full'). Uses preset default if None.
        
        Returns:
            DatasetConfig object
        """
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        dataset_info = self.config['datasets'][dataset_name]
        
        # Determine variant
        if variant is None:
            variant = self.preset if self.preset != 'validation' else 'medium'
            if self.preset == 'debug':
                variant = 'small'
            elif self.preset == 'validation':
                variant = 'medium'
            else:  # production
                variant = 'full'
        
        if variant not in dataset_info['variants']:
            raise ValueError(f"Variant '{variant}' not found for dataset '{dataset_name}'")
        
        variant_info = dataset_info['variants'][variant]
        
        return DatasetConfig(
            name=dataset_info['name'],
            source=dataset_info['source'],
            variant=variant,
            num_tokens=variant_info['num_tokens'],
            samples=variant_info['samples'],
            cache_file=variant_info['cache_file'],
            description=variant_info['description']
        )
    
    def get_stage(self, stage: str) -> Dict[str, Any]:
        """Get stage configuration (debug, validation, production)"""
        if stage not in self.config['stages']:
            raise ValueError(f"Stage '{stage}' not found")
        
        return self.config['stages'][stage]
    
    def list_datasets(self) -> list:
        """List all available datasets"""
        return list(self.config['datasets'].keys())
    
    def list_phases(self) -> list:
        """List all available research phases"""
        return list(self.config.get('phase_configurations', {}).keys())


# Global singleton instance
_manager = None


def get_dataset_manager(config_file: Optional[Path] = None, preset: str = 'production') -> DatasetConfigManager:
    """
    Get or create global dataset manager instance
    
    Args:
        config_file: Path to datasets.yaml config
        preset: Stage preset (debug, validation, production)
    
    Returns:
        DatasetConfigManager instance
    """
    global _manager
    if _manager is None:
        _manager = DatasetConfigManager(config_file=config_file, preset=preset)
    return _manager


def reset_dataset_manager():
    """Reset global manager instance (for testing)"""
    global _manager
    _manager = None
