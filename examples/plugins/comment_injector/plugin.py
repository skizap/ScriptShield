import ast
from typing import Any

from obfuscator.core.plugin_interface import ObfuscatorPlugin, PluginMetadata, PluginContext
from obfuscator.processors.ast_transformer import TransformResult


class CommentInjectorPlugin(ObfuscatorPlugin):
    """Injects a harmless comment/docstring at the top of the AST."""
    
    metadata = PluginMetadata(
        name="comment_injector",
        version="1.0.0",
        author="ScriptShield",
        description="Injects a harmless comment/docstring at the top of the AST",
        supported_languages=["python", "lua"],
        priority=100,
        requires_runtime=False
    )

    def transform(self, ast_node: Any, context: PluginContext) -> TransformResult:
        if context.language == "python":
            if isinstance(ast_node, ast.Module):
                comment_node = ast.Expr(value=ast.Constant(value="ScriptShield: comment_injector"))
                ast.fix_missing_locations(comment_node)
                ast_node.body.insert(0, comment_node)
                
                return TransformResult(
                    ast_node=ast_node,
                    success=True,
                    transformation_count=1,
                    errors=[]
                )
                
        elif context.language == "lua":
            try:
                import luaparser.astnodes as lua_nodes
            except ImportError:
                return TransformResult(
                    ast_node=ast_node,
                    success=True,
                    transformation_count=0,
                    errors=[]
                )
                
            comment_node = lua_nodes.Comment(s="ScriptShield: comment_injector")
            
            # Some versions of luaparser use `body.body`, others `body.list`
            if hasattr(ast_node, 'body'):
                if hasattr(ast_node.body, 'list'):
                    ast_node.body.list.insert(0, comment_node)
                elif hasattr(ast_node.body, 'body'):
                    ast_node.body.body.insert(0, comment_node)
            
            return TransformResult(
                ast_node=ast_node,
                success=True,
                transformation_count=1,
                errors=[]
            )
            
        return TransformResult(
            ast_node=ast_node,
            success=True,
            transformation_count=0,
            errors=[]
        )
