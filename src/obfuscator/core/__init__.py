"""Core obfuscation configuration module.

This module provides the core configuration data model, profile
management functionality, and dependency graph analysis for the obfuscator.

Classes:
    ObfuscationConfig: Configuration data model
    ProfileManager: Profile save/load/validation manager
    DependencyNode: Represents a file in the dependency graph
    DependencyEdge: Represents a dependency relationship
    DependencyGraph: Main graph container for dependencies
    DependencyAnalyzer: Analyzes file dependencies and builds graphs
    CircularDependencyError: Exception for circular dependencies
    DependencyResolutionError: Exception for unresolved imports
"""

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.dependency_graph import (
    CircularDependencyError,
    DependencyAnalyzer,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    DependencyResolutionError,
)
from obfuscator.core.profile_manager import ProfileManager

__all__ = [
    "ObfuscationConfig",
    "ProfileManager",
    "DependencyNode",
    "DependencyEdge",
    "DependencyGraph",
    "DependencyAnalyzer",
    "CircularDependencyError",
    "DependencyResolutionError",
]

