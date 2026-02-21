"""Core obfuscation configuration module.

This module provides the core configuration data model, profile
management functionality, dependency graph analysis, checkpoint management,
worker processing, and plugin system for the obfuscator.

Classes:
    ObfuscationConfig: Configuration data model
    ProfileManager: Profile save/load/validation manager
    DependencyNode: Represents a file in the dependency graph
    DependencyEdge: Represents a dependency relationship
    DependencyGraph: Main graph container for dependencies
    DependencyAnalyzer: Analyzes file dependencies and builds graphs
    CircularDependencyError: Exception for circular dependencies
    DependencyResolutionError: Exception for unresolved imports
    CheckpointManager: Manages saving and restoring obfuscation state checkpoints
    WorkerTask: Task payload for worker process
    WorkerResult: Result container from worker process
    WorkerProcess: Worker process handler for file batch processing
    ObfuscatorPlugin: Abstract base class for obfuscation plugins
    PluginMetadata: Metadata for an obfuscation plugin
    PluginContext: Read-only context passed to plugin calls
    PluginManager: Manages plugin discovery and execution
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
from obfuscator.core.exceptions import UnsupportedFeatureWarning
from obfuscator.core.profile_manager import ProfileManager
from obfuscator.core.worker import WorkerTask, WorkerResult, WorkerProcess, process_file_batch, setup_worker_logging
from obfuscator.core.checkpoint_manager import CheckpointManager
from obfuscator.core.plugin_interface import ObfuscatorPlugin, PluginMetadata, PluginContext
from obfuscator.core.plugin_manager import PluginManager

__all__ = [
    "ObfuscationConfig",
    "ProfileManager",
    "DependencyNode",
    "DependencyEdge",
    "DependencyGraph",
    "DependencyAnalyzer",
    "CircularDependencyError",
    "DependencyResolutionError",
    "UnsupportedFeatureWarning",
    "WorkerTask",
    "WorkerResult",
    "WorkerProcess",
    "process_file_batch",
    "setup_worker_logging",
    "CheckpointManager",
    "ObfuscatorPlugin",
    "PluginMetadata",
    "PluginContext",
    "PluginManager",
]
