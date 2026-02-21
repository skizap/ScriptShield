"""Plugin manager for ScriptShield obfuscator.

Handles discovery, loading, and execution of custom obfuscation plugins.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from obfuscator.processors.ast_transformer import TransformResult
from obfuscator.utils.logger import get_logger

from .plugin_interface import ObfuscatorPlugin, PluginContext, PluginMetadata

logger = get_logger("obfuscator.core.plugin_manager")


class PluginManager:
    """Manages obfuscation plugin discovery and execution.
    
    Handles loading plugins from a directory, validating their metadata,
    and executing them in the correct order based on priority.
    """
    
    def __init__(self, plugin_dir: Path | None = None) -> None:
        """Initialize the plugin manager.
        
        Args:
            plugin_dir: Directory containing plugin subdirectories.
                        Defaults to ~/.obfuscator/plugins if not provided.
        """
        if plugin_dir is None:
            plugin_dir = Path.home() / ".obfuscator" / "plugins"
        
        self._plugin_dir = Path(plugin_dir)
        self._plugins: list[ObfuscatorPlugin] = self.load_plugins(self._plugin_dir)
    
    def load_plugins(self, plugin_dir: Path) -> list[ObfuscatorPlugin]:
        """Load all plugins from the specified directory.
        
        Each plugin must be in its own subdirectory containing:
        - plugin.json: Plugin metadata
        - plugin.py: Plugin implementation
        
        Args:
            plugin_dir: Directory containing plugin subdirectories
            
        Returns:
            List of loaded plugin instances, sorted by priority
        """
        plugins: list[ObfuscatorPlugin] = []
        
        if not plugin_dir.exists():
            logger.debug(f"Plugin directory does not exist: {plugin_dir}")
            return plugins
        
        for subdir in plugin_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Read plugin.json
            json_path = subdir / "plugin.json"
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Plugin {subdir.name}: missing plugin.json")
                continue
            except json.JSONDecodeError as exc:
                logger.warning(f"Plugin {subdir.name}: invalid JSON in plugin.json: {exc}")
                continue
            
            # Validate metadata
            try:
                metadata = self._validate_metadata(data)
            except ValueError as exc:
                logger.warning(f"Plugin {subdir.name}: invalid metadata: {exc}")
                continue
            
            # Locate plugin.py
            plugin_py_path = subdir / "plugin.py"
            if not plugin_py_path.exists():
                logger.warning(f"Plugin {subdir.name}: missing plugin.py")
                continue
            
            # Load the module
            module_name = f"plugin_{subdir.name}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, plugin_py_path)
                if spec is None or spec.loader is None:
                    logger.warning(f"Plugin {subdir.name}: failed to create module spec")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                logger.warning(f"Plugin {subdir.name}: failed to load module: {exc}")
                continue
            
            # Find the ObfuscatorPlugin subclass
            plugin_class: type[ObfuscatorPlugin] | None = None
            for obj in vars(module).values():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, ObfuscatorPlugin)
                    and obj is not ObfuscatorPlugin
                ):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                logger.warning(f"Plugin {subdir.name}: no ObfuscatorPlugin subclass found")
                continue
            
            # Instantiate the plugin
            try:
                plugin_instance = plugin_class()
            except Exception as exc:
                logger.warning(f"Plugin {subdir.name}: failed to instantiate: {exc}")
                continue
            
            # Assign validated metadata to the instance
            try:
                plugin_instance.metadata = metadata
            except Exception as exc:
                logger.warning(f"Plugin {subdir.name}: failed to assign metadata: {exc}")
                continue
            
            plugins.append(plugin_instance)
        
        # Filter out plugins without metadata attribute before sorting
        valid_plugins = [p for p in plugins if hasattr(p, "metadata")]
        if len(valid_plugins) != len(plugins):
            skipped_count = len(plugins) - len(valid_plugins)
            logger.warning(f"Skipped {skipped_count} plugin(s) missing metadata attribute")
        
        # Sort by priority (lower = earlier)
        valid_plugins.sort(key=lambda p: p.metadata.priority)
        
        logger.info(f"Successfully loaded {len(valid_plugins)} plugin(s)")
        return valid_plugins
    
    def _validate_metadata(self, data: dict) -> PluginMetadata:
        """Validate plugin metadata from JSON.
        
        Args:
            data: Dictionary loaded from plugin.json
            
        Returns:
            Validated PluginMetadata instance
            
        Raises:
            ValueError: If metadata is invalid or missing required fields
        """
        required_keys = [
            "name", "version", "author", "description",
            "supported_languages", "priority", "requires_runtime"
        ]
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required field: {key}")
        
        # Validate supported_languages
        supported_languages = data["supported_languages"]
        if not isinstance(supported_languages, list) or len(supported_languages) == 0:
            raise ValueError("supported_languages must be a non-empty list")
        
        valid_languages = {"python", "lua"}
        for lang in supported_languages:
            if lang not in valid_languages:
                raise ValueError(f"Invalid language '{lang}'; must be one of {valid_languages}")
        
        # Validate priority is int
        priority = data["priority"]
        if not isinstance(priority, int):
            raise ValueError(f"priority must be an integer, got {type(priority).__name__}")
        
        # Validate requires_runtime is bool
        requires_runtime = data["requires_runtime"]
        if not isinstance(requires_runtime, bool):
            raise ValueError(f"requires_runtime must be a boolean, got {type(requires_runtime).__name__}")
        
        return PluginMetadata(**data)
    
    def execute_plugin(
        self, 
        plugin: ObfuscatorPlugin, 
        ast_node: Any, 
        context: PluginContext
    ) -> TransformResult:
        """Execute a plugin's transform method with error handling.
        
        Args:
            plugin: Plugin instance to execute
            ast_node: AST node to transform
            context: Plugin context
            
        Returns:
            TransformResult from the plugin, or a fallback result on error
        """
        try:
            result = plugin.transform(ast_node, context)
            return result
        except Exception as exc:
            logger.warning(
                f"Plugin {plugin.metadata.name} raised exception during transform: {exc}"
            )
            # Return fallback: pass through original AST unmodified
            return TransformResult(
                ast_node=ast_node,
                success=True,
                transformation_count=0,
                errors=[]
            )
    
    def get_plugins_for_language(self, language: str) -> list[ObfuscatorPlugin]:
        """Get plugins that support the specified language.
        
        Args:
            language: Language to filter by ("python" or "lua")
            
        Returns:
            List of plugins supporting the language, sorted by priority
        """
        return [p for p in self._plugins if language in p.metadata.supported_languages]
