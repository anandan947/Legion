"""Legion: A provider-agnostic framework for building AI agent systems"""

__version__ = "0.1.0"

# Core interfaces
from .agents.decorators import agent
from .interface.decorators import tool
from .groups.decorators import chain, leader, team

# Core agent system

# Provider management

# Error types

__all__ = ["agent", "tool", "chain", "leader", "team"]