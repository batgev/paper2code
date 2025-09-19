# Tools Directory

This directory is reserved for MCP (Model Context Protocol) server tools and custom tool implementations.

## Purpose
- Store MCP server configurations
- Implement custom tools for the AI agents
- Extend functionality with specialized tools

## Current Configuration
The system is configured to use MCP servers as defined in `config/defaults.py`:
- **Filesystem Server**: File system operations
- **Brave Search Server**: Web search capabilities
- **Fetch Server**: Web content retrieval

## Usage
Custom tools can be added here by creating Python modules:
```python
from paper2code.tools.base import BaseMCPServer

class CustomToolServer(BaseMCPServer):
    def setup_tools(self):
        # Define custom tools
        pass
```

## Future Implementation
This directory will contain:
- Custom MCP server implementations
- Tool wrappers for external services
- Specialized processing tools

## Note
The MCP servers are currently configured but not actively used in the implementation.
The agents use direct API calls instead of MCP protocol.

