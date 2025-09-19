"""
Research Genie - Complete Paper-to-Code Implementation System

Transform research papers into working code automatically using complete AI agent system.
"""

__version__ = "1.0.0"
__author__ = "Research Genie Team"
__license__ = "MIT"

# Import complete main components
from .processor import Paper2CodeProcessor, ProcessingResult
from .config.manager import ConfigManager
from .utils.file_processor import FileProcessor
from .workflows.orchestrator import WorkflowOrchestrator

# Import complete agent system
from .agents import (
    DocumentAnalysisAgent,
    CodePlanningAgent, 
    CodeGeneratorAgent,
    RepositoryFinderAgent
)

# Import utilities
from .utils.logger import get_logger, setup_logging

# Optional interface imports (with fallback)
try:
    from .interfaces import (
        EnhancedCLIInterface, 
        WebInterface, 
        NotebookInterface,
        show_quick_start,
        launch_interface_selector
    )
    _interfaces_available = True
except ImportError:
    _interfaces_available = False

__all__ = [
    "Paper2CodeProcessor",
    "ConfigManager",
    "FileProcessor", 
    "WorkflowOrchestrator",
    "__version__",
    "__author__",
    "__license__",
]

# Add interfaces to __all__ if available
if _interfaces_available:
    __all__.extend([
        "EnhancedCLIInterface",
        "WebInterface", 
        "NotebookInterface",
        "show_quick_start",
        "launch_interface_selector"
    ])
