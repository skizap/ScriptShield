"""Plugin interface for ScriptShield obfuscator.

Defines the base classes and data structures for creating custom obfuscation plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from obfuscator.core.config import ObfuscationConfig
    from obfuscator.core.symbol_table import GlobalSymbolTable


@dataclass
class PluginMetadata:
    """Metadata for an obfuscation plugin.
    
    Mirrors the plugin.json schema exactly.
    """
    name: str
    version: str
    author: str
    description: str
    supported_languages: list[str]
    priority: int
    requires_runtime: bool


@dataclass
class PluginContext:
    """Read-only context passed to every plugin call.
    
    Contains configuration and symbol table information needed for
    plugin transformations.
    """
    config: ObfuscationConfig
    symbol_table: GlobalSymbolTable | None
    language: str


class ObfuscatorPlugin(ABC):
    """Abstract base class for obfuscation plugins.
    
    All custom plugins must inherit from this class and implement
    the transform method. Plugins may optionally override generate_runtime
    if they need to inject runtime code.
    """
    
    metadata: PluginMetadata
    """Class-level or instance attribute; must be set by every concrete plugin."""
    
    @abstractmethod
    def transform(self, ast_node: Any, context: PluginContext) -> Any:
        """Transform an AST node.
        
        Args:
            ast_node: The AST node to transform
            context: Plugin context containing config and symbol table
            
        Returns:
            TransformResult containing the transformed AST and metadata
        """
        pass
    
    def generate_runtime(self, context: PluginContext) -> str:
        """Generate runtime code for this plugin.
        
        Default implementation returns an empty string. Plugins that
        need to inject runtime code should override this method.
        
        Args:
            context: Plugin context containing config
            
        Returns:
            Runtime code as a string, or empty string if no runtime needed
        """
        return ""
