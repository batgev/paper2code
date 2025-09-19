"""
FastAPI Backend for Paper2Code
"""

from .server import app, start_server
from .routes import router

__all__ = ["app", "start_server", "router"]

