"""
Configuration Manager for Paper2Code

Handles loading, validation, and management of configuration settings.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .defaults import DEFAULT_CONFIG


class ConfigManager:
    """
    Manages configuration for Paper2Code application.
    
    Supports YAML configuration files, environment variables, and programmatic updates.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self._config = DEFAULT_CONFIG.copy()
        self._config_path = None
        self._secrets = {}
        
        # Load configuration files
        self._load_config_files(config_path)
        
        # Load environment variables
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_config()
    
    def _load_config_files(self, config_path: Optional[Union[str, Path]] = None):
        """Load configuration from YAML files"""
        # Determine config file path
        if config_path:
            config_file = Path(config_path)
        else:
            # Look for config files in standard locations
            possible_paths = [
                Path("paper2code/config/config.yaml"),
                Path("config/config.yaml"),
                Path("config.yaml"),
                Path.home() / ".paper2code" / "config.yaml",
            ]
            
            config_file = None
            for path in possible_paths:
                if path.exists():
                    config_file = path
                    break
        
        # Load main config file
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self._merge_config(self._config, file_config)
                self._config_path = config_file
                print(f"✅ Loaded configuration from: {config_file}")
            except Exception as e:
                print(f"⚠️ Error loading config file {config_file}: {e}")
        
        # Load secrets file
        secrets_paths = [
            Path("paper2code/config/secrets.yaml"),
            Path("config/secrets.yaml"), 
            Path("secrets.yaml"),
            Path.home() / ".paper2code" / "secrets.yaml",
        ]
        
        for secrets_path in secrets_paths:
            if secrets_path.exists():
                try:
                    with open(secrets_path, 'r', encoding='utf-8') as f:
                        secrets = yaml.safe_load(f)
                        if secrets:
                            self._secrets = secrets
                    print(f"✅ Loaded secrets from: {secrets_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Error loading secrets file {secrets_path}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'PAPER2CODE_OUTPUT_DIR': 'output.base_directory',
            'PAPER2CODE_LOG_LEVEL': 'logging.level',
            'PAPER2CODE_CACHE_DIR': 'cache.directory',
            'PAPER2CODE_MAX_TOKENS': 'llm.max_tokens',
            'PAPER2CODE_FAST_MODE': 'processing.fast_mode',
            'OPENAI_API_KEY': 'openai.api_key',
            'ANTHROPIC_API_KEY': 'anthropic.api_key',
            'BRAVE_API_KEY': 'brave_search.api_key',
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert boolean strings
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                
                # Convert numeric strings
                elif value.isdigit():
                    value = int(value)
                
                self._set_nested_value(self._config, config_path, value)
                print(f"✅ Set {config_path} from environment variable {env_var}")
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value using dot notation"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation"""
        keys = path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Check required paths exist
        output_dir = Path(self.get('output.base_directory', './output'))
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created output directory: {output_dir}")
            except Exception as e:
                print(f"⚠️ Could not create output directory {output_dir}: {e}")
        
        # Validate LLM configuration - allow Ollama without API keys
        has_openai = bool(self.get_secret('openai.api_key'))
        has_anthropic = bool(self.get_secret('anthropic.api_key'))
        preferred_provider = self.get('llm.preferred_provider', 'ollama')
        
        if preferred_provider == 'ollama':
            # Ollama doesn't need API keys, just check if it's running
            pass
        elif not has_openai and not has_anthropic:
            print("⚠️ Warning: No LLM API keys configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        # Validate processing settings
        if self.get('document_segmentation.size_threshold_chars', 50000) < 1000:
            print("⚠️ Warning: Document segmentation threshold is very low")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'output.base_directory')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._get_nested_value(self._config, key, default)
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        Get secret value using dot notation.
        
        Args:
            key: Secret key (e.g., 'openai.api_key')  
            default: Default value if key not found
            
        Returns:
            Secret value
        """
        return self._get_nested_value(self._secrets, key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self._set_nested_value(self._config, key, value)
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._merge_config(self._config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save configuration to YAML file.
        
        Args:
            path: Optional path to save to (defaults to loaded config path)
        """
        if path:
            config_file = Path(path)
        elif self._config_path:
            config_file = self._config_path
        else:
            config_file = Path("config.yaml")
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            print(f"✅ Configuration saved to: {config_file}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with API keys"""
        llm_config = {
            'openai': {
                'api_key': self.get_secret('openai.api_key'),
                'base_url': self.get_secret('openai.base_url', 'https://api.openai.com/v1'),
                'default_model': self.get('llm.openai.default_model', 'gpt-4'),
            },
            'anthropic': {
                'api_key': self.get_secret('anthropic.api_key'),
                'default_model': self.get('llm.anthropic.default_model', 'claude-3-5-sonnet-20241022'),
            },
            'preferred_provider': self.get('llm.preferred_provider', 'anthropic'),
            'max_tokens': self.get('llm.max_tokens', 8192),
            'temperature': self.get('llm.temperature', 0.3),
        }
        
        return llm_config
    
    def get_mcp_servers_config(self) -> Dict[str, Any]:
        """Get MCP servers configuration"""
        return self.get('mcp.servers', {})
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"Paper2CodeConfig(config_path={self._config_path})"
    
    def __repr__(self) -> str:
        return self.__str__()
