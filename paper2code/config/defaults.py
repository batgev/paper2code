"""
Default configuration for Paper2Code
"""

DEFAULT_CONFIG = {
    # Processing configuration
    "processing": {
        "mode": "comprehensive",  # "comprehensive" or "fast"
        "enable_repository_search": True,
        "enable_code_indexing": True,
        "parallel_agents": True,
        "max_concurrent_agents": 3,
    },
    
    # Document segmentation
    "document_segmentation": {
        "enabled": True,
        "size_threshold_chars": 50000,
        "overlap_chars": 1000,
        "max_segments": 10,
        "min_segment_chars": 5000,
    },
    
    # LLM configuration
    "llm": {
        "preferred_provider": "ollama",  # "openai", "anthropic", or "ollama"
        "enabled": True,
        "max_tokens": 8192,
        "temperature": 0.3,
        "openai": {
            "default_model": "gpt-4",
            "backup_model": "gpt-3.5-turbo",
        },
        "anthropic": {
            "default_model": "claude-3-5-sonnet-20241022",
            "backup_model": "claude-3-haiku-20240307",
        },
        "ollama": {
            "default_model": "deepseek-r1:8b",
            "base_url": "http://127.0.0.1:11434"
        }
    },
    
    # Output configuration
    "output": {
        "base_directory": "./output",
        "create_readme": True,
        "include_analysis": True,
        "include_references": True,
        "code_format": "python",
        "generate_tests": True,
        "generate_docs": True,
    },
    
    # MCP Servers configuration
    "mcp": {
        "servers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "description": "File system operations",
            },
            "brave": {
                "command": "npx", 
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "description": "Web search via Brave Search API",
                "env": {
                    "BRAVE_API_KEY": ""  # Set in secrets
                }
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "description": "Web content retrieval",
            },
        }
    },
    
    # Search configuration  
    "search": {
        "default_server": "brave",
        "max_results": 10,
        "timeout_seconds": 30,
    },
    
    # Cache configuration
    "cache": {
        "enabled": True,
        "directory": "./cache",
        "ttl_seconds": 3600,  # 1 hour
        "max_size_mb": 1000,
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "./logs/paper2code.log",
        "console": True,
        "max_bytes": 10485760,  # 10MB
        "backup_count": 5,
    },
    
    # Prompts configuration
    "prompts": {
        "use_segmentation": True,
        "custom_prompts_dir": "./prompts/custom",
        "temperature": 0.3,
        "max_retries": 3,
    },
    
    # Performance configuration
    "performance": {
        "async_processing": True,
        "batch_size": 10,
        "timeout_seconds": 300,  # 5 minutes per operation
        "memory_limit_mb": 2048,
    },
    
    # Security configuration
    "security": {
        "sandbox_code_execution": True,
        "allowed_file_types": [".pdf", ".docx", ".doc", ".txt", ".md", ".html"],
        "max_file_size_mb": 50,
        "validate_urls": True,
    },
    
    # Development configuration  
    "development": {
        "debug_mode": False,
        "verbose_logging": False,
        "save_intermediate_files": False,
        "enable_profiling": False,
    }
}
