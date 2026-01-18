"""Configuration file parser for Schrödinger Bridge simulations."""

from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Parse configuration files for Schrödinger Bridge simulations."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        File format:
            # Comments
            key = value
        
        Supported types: bool, int, float, str
        
        Parameters
        ----------
        config_path : str - path to config file
        
        Returns
        -------
        config : dict - parsed configuration
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = {}
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty/comment lines
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                # Parse key = value
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove inline comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Parse type
                config[key] = ConfigManager._parse_value(value)
        
        return config
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string to appropriate Python type."""
        # Boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Numeric
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fill defaults for configuration.
        
        Parameters
        ----------
        config : dict - raw configuration
        
        Returns
        -------
        config : dict - validated configuration with defaults
        """
        defaults = {
            'scenario_name': 'Schrödinger Bridge',
            'source_type': 'circle',
            'target_type': 'circle',
            'n_particles': 1000,
            'epsilon': 0.05,
            'max_iter': 2000,
            'tol': 1e-9,
            'n_cores': 0,
            'output_dir': 'outputs',
            'save_netcdf': True,
            'save_csv': True,
            'save_animation': True,
            'save_diagnostics': True,
            'n_frames': 120,
            'fps': 30,
            'dpi': 150,
            'colormap': 'viridis',
            'marker_size': 8,
            'alpha': 0.8,
            'seed': 42
        }
        
        # Fill defaults
        for key, default in defaults.items():
            if key not in config:
                config[key] = default
        
        return config
