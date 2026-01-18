"""Comprehensive tests for the dependency graph module.

This test suite covers:
- DependencyNode creation and validation
- DependencyEdge creation and self-loop prevention
- DependencyGraph construction and queries
- DependencyAnalyzer file analysis
- Topological sorting
- Cycle detection
- Error handling and edge cases
"""

from pathlib import Path
from typing import Any

import pytest

from obfuscator.core.dependency_graph import (
    CircularDependencyError,
    DependencyAnalyzer,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    DependencyResolutionError,
)
from obfuscator.processors.symbol_extractor import (
    ImportInfo,
    SymbolTable,
    FunctionInfo,
)
from obfuscator.processors.lua_symbol_extractor import (
    LuaImportInfo,
    LuaSymbolTable,
    LuaFunctionInfo,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create temporary project structure with multiple files."""
    # Create project directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "tests").mkdir()
    
    # Create Python files
    (tmp_path / "main.py").write_text(
        "from src import app\nimport os\n",
        encoding="utf-8"
    )
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text(
        "from .utils import helper\n",
        encoding="utf-8"
    )
    (tmp_path / "src" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "helper.py").write_text(
        "def helper(): pass\n",
        encoding="utf-8"
    )
    
    return tmp_path


@pytest.fixture
def sample_python_files() -> dict[str, str]:
    """Return dict of sample Python code with various import patterns."""
    return {
        "main.py": "import os\nfrom utils import helper\n",
        "utils.py": "import sys\n",
        "circular_a.py": "from circular_b import func_b\n",
        "circular_b.py": "from circular_a import func_a\n",
        "relative.py": "from . import sibling\nfrom ..parent import module\n",
    }


@pytest.fixture
def sample_lua_files() -> dict[str, str]:
    """Return dict of sample Lua code with require() patterns."""
    return {
        "main.lua": 'local utils = require("utils")\nlocal http = require("http")\n',
        "utils.lua": 'local json = require("json")\n',
        "relative.lua": 'local helper = require("./helper")\n',
        "circular_a.lua": 'local b = require("circular_b")\n',
        "circular_b.lua": 'local a = require("circular_a")\n',
    }


@pytest.fixture
def dependency_analyzer(tmp_path: Path) -> DependencyAnalyzer:
    """Return configured DependencyAnalyzer instance."""
    return DependencyAnalyzer(tmp_path)


# ============================================================================
# Test Helpers
# ============================================================================


def create_test_file(path: Path, content: str, language: str) -> Path:
    """Helper to create test files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def assert_topological_order(order: list[Path], graph: DependencyGraph) -> None:
    """Verify that the order is a valid topological order."""
    position = {path: i for i, path in enumerate(order)}
    
    for path in order:
        for dep in graph.get_dependencies(path):
            if dep in position:
                assert position[dep] < position[path], (
                    f"Dependency {dep} should come before {path} in topological order"
                )


def create_symbol_table(
    imports: list[str],
    exports: list[str],
    file_path: Path,
    language: str,
    import_details: list[dict[str, Any]] | None = None
) -> SymbolTable | LuaSymbolTable:
    """Helper to create mock symbol tables."""
    if language == "python":
        import_infos = []
        if import_details:
            for detail in import_details:
                import_infos.append(ImportInfo(
                    module_name=detail.get("module", ""),
                    imported_names=detail.get("names", []),
                    alias=detail.get("alias"),
                    is_from_import=detail.get("is_from", False),
                    line_number=detail.get("line", 0),
                    level=detail.get("level", 0)
                ))
        else:
            for i, imp in enumerate(imports):
                import_infos.append(ImportInfo(
                    module_name=imp,
                    imported_names=[],
                    alias=None,
                    is_from_import=False,
                    line_number=i + 1,
                    level=0
                ))
        
        functions = [FunctionInfo(name=name, scope="global") for name in exports]
        return SymbolTable(
            imports=import_infos,
            functions=functions,
            classes=[],
            variables=[],
            file_path=file_path
        )
    else:
        import_infos = []
        if import_details:
            for detail in import_details:
                import_infos.append(LuaImportInfo(
                    module_path=detail.get("module", ""),
                    alias=detail.get("alias"),
                    line_number=detail.get("line", 0),
                    is_relative=detail.get("is_relative", False)
                ))
        else:
            for i, imp in enumerate(imports):
                import_infos.append(LuaImportInfo(
                    module_path=imp,
                    alias=imp.split(".")[-1],
                    line_number=i + 1,
                    is_relative=imp.startswith(".")
                ))
        
        functions = [LuaFunctionInfo(name=name, scope="global") for name in exports]
        return LuaSymbolTable(
            imports=import_infos,
            functions=functions,
            variables=[],
            file_path=file_path,
            roblox_api_usage=[]
        )


# ============================================================================
# Test Class: TestDependencyNode
# ============================================================================


class TestDependencyNode:
    """Test DependencyNode creation and validation."""

    def test_node_creation(self, tmp_path: Path) -> None:
        """Verify node initialization with all fields."""
        file_path = tmp_path / "test.py"
        file_path.touch()

        node = DependencyNode(
            file_path=file_path,
            language="python",
            imports=["os", "sys"],
            exports=["main", "Config"]
        )

        assert node.file_path == file_path.resolve()
        assert node.language == "python"
        assert node.imports == ["os", "sys"]
        assert node.exports == ["main", "Config"]
        assert node.is_processed is False
        assert node.metadata == {}

    def test_node_validation_resolves_path(self, tmp_path: Path) -> None:
        """Test that path is resolved to absolute in __post_init__."""
        file_path = tmp_path / "subdir" / "test.py"
        file_path.parent.mkdir()
        file_path.touch()

        relative = Path("subdir/test.py")
        # Create from workspace
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            node = DependencyNode(file_path=relative, language="python")
            assert node.file_path.is_absolute()
        finally:
            os.chdir(old_cwd)

    def test_node_validation_invalid_language(self, tmp_path: Path) -> None:
        """Test that invalid language raises ValueError."""
        file_path = tmp_path / "test.py"
        file_path.touch()

        with pytest.raises(ValueError, match="Invalid language"):
            DependencyNode(file_path=file_path, language="javascript")

    def test_node_equality(self, tmp_path: Path) -> None:
        """Test __eq__() and __hash__() methods."""
        file1 = tmp_path / "test.py"
        file2 = tmp_path / "other.py"
        file1.touch()
        file2.touch()

        node1 = DependencyNode(file_path=file1, language="python")
        node2 = DependencyNode(file_path=file1, language="python", imports=["os"])
        node3 = DependencyNode(file_path=file2, language="python")

        # Same path = equal
        assert node1 == node2
        assert hash(node1) == hash(node2)

        # Different path = not equal
        assert node1 != node3

        # Can be used in sets
        node_set = {node1, node2, node3}
        assert len(node_set) == 2

    def test_node_with_metadata(self, tmp_path: Path) -> None:
        """Verify metadata storage."""
        file_path = tmp_path / "test.py"
        file_path.touch()

        metadata = {"line_count": 100, "has_errors": False}
        node = DependencyNode(
            file_path=file_path,
            language="python",
            metadata=metadata
        )

        assert node.metadata == metadata
        assert node.metadata["line_count"] == 100


# ============================================================================
# Test Class: TestDependencyEdge
# ============================================================================


class TestDependencyEdge:
    """Test DependencyEdge creation and validation."""

    def test_edge_creation(self, tmp_path: Path) -> None:
        """Test edge initialization with all fields."""
        from_file = tmp_path / "main.py"
        to_file = tmp_path / "utils.py"
        from_file.touch()
        to_file.touch()

        edge = DependencyEdge(
            from_node=from_file,
            to_node=to_file,
            import_type="relative",
            imported_symbols=["helper", "Config"],
            line_number=5
        )

        assert edge.from_node == from_file.resolve()
        assert edge.to_node == to_file.resolve()
        assert edge.import_type == "relative"
        assert edge.imported_symbols == ["helper", "Config"]
        assert edge.line_number == 5

    def test_edge_prevents_self_loop(self, tmp_path: Path) -> None:
        """Test that self-loops raise ValueError."""
        file_path = tmp_path / "test.py"
        file_path.touch()

        with pytest.raises(ValueError, match="Self-loop detected"):
            DependencyEdge(from_node=file_path, to_node=file_path)

    def test_edge_invalid_import_type(self, tmp_path: Path) -> None:
        """Test that invalid import_type raises ValueError."""
        from_file = tmp_path / "main.py"
        to_file = tmp_path / "utils.py"
        from_file.touch()
        to_file.touch()

        with pytest.raises(ValueError, match="Invalid import_type"):
            DependencyEdge(
                from_node=from_file,
                to_node=to_file,
                import_type="dynamic"
            )

    def test_edge_default_values(self, tmp_path: Path) -> None:
        """Test default field values."""
        from_file = tmp_path / "main.py"
        to_file = tmp_path / "utils.py"
        from_file.touch()
        to_file.touch()

        edge = DependencyEdge(from_node=from_file, to_node=to_file)

        assert edge.import_type == "absolute"
        assert edge.imported_symbols == []
        assert edge.line_number == 0


# ============================================================================
# Test Class: TestDependencyGraph
# ============================================================================


class TestDependencyGraph:
    """Test DependencyGraph construction and queries."""

    def test_add_node(self, tmp_path: Path) -> None:
        """Test node addition and retrieval."""
        graph = DependencyGraph()
        file_path = tmp_path / "test.py"
        file_path.touch()

        node = graph.add_node(
            file_path=file_path,
            language="python",
            imports=["os"],
            exports=["main"]
        )

        assert node.file_path == file_path.resolve()
        assert file_path.resolve() in graph.nodes
        assert graph.get_node(file_path) == node

    def test_add_edge(self, tmp_path: Path) -> None:
        """Test edge creation and adjacency list updates."""
        graph = DependencyGraph()
        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        main_file.touch()
        utils_file.touch()

        graph.add_node(main_file, "python", ["utils"], ["main"])
        graph.add_node(utils_file, "python", [], ["helper"])

        graph.add_edge(
            from_path=main_file,
            to_path=utils_file,
            import_type="relative",
            symbols=["helper"],
            line_num=1
        )

        assert len(graph.edges) == 1
        assert utils_file.resolve() in graph.adjacency_list[main_file.resolve()]

    def test_get_dependencies(self, tmp_path: Path) -> None:
        """Verify dependency lookup."""
        graph = DependencyGraph()
        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        config_file = tmp_path / "config.py"
        main_file.touch()
        utils_file.touch()
        config_file.touch()

        graph.add_node(main_file, "python", [], [])
        graph.add_node(utils_file, "python", [], [])
        graph.add_node(config_file, "python", [], [])

        graph.add_edge(main_file, utils_file, "absolute", [], 1)
        graph.add_edge(main_file, config_file, "absolute", [], 2)

        deps = graph.get_dependencies(main_file)
        assert utils_file.resolve() in deps
        assert config_file.resolve() in deps
        assert len(deps) == 2

    def test_get_dependents(self, tmp_path: Path) -> None:
        """Verify reverse dependency lookup."""
        graph = DependencyGraph()
        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        tests_file = tmp_path / "tests.py"
        main_file.touch()
        utils_file.touch()
        tests_file.touch()

        graph.add_node(main_file, "python", [], [])
        graph.add_node(utils_file, "python", [], [])
        graph.add_node(tests_file, "python", [], [])

        graph.add_edge(main_file, utils_file, "absolute", [], 1)
        graph.add_edge(tests_file, utils_file, "absolute", [], 1)

        dependents = graph.get_dependents(utils_file)
        assert main_file.resolve() in dependents
        assert tests_file.resolve() in dependents
        assert len(dependents) == 2

    def test_has_path(self, tmp_path: Path) -> None:
        """Test path existence checking."""
        graph = DependencyGraph()
        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"
        a_file.touch()
        b_file.touch()
        c_file.touch()

        graph.add_node(a_file, "python", [], [])
        graph.add_node(b_file, "python", [], [])
        graph.add_node(c_file, "python", [], [])

        # A -> B -> C
        graph.add_edge(a_file, b_file, "absolute", [], 1)
        graph.add_edge(b_file, c_file, "absolute", [], 1)

        assert graph.has_path(a_file, c_file)
        assert graph.has_path(a_file, b_file)
        assert graph.has_path(b_file, c_file)
        assert not graph.has_path(c_file, a_file)

    def test_incremental_construction(self, tmp_path: Path) -> None:
        """Add nodes/edges one at a time."""
        graph = DependencyGraph()

        # Add nodes incrementally
        for i in range(5):
            file_path = tmp_path / f"file{i}.py"
            file_path.touch()
            graph.add_node(file_path, "python", [], [])

        assert len(graph.nodes) == 5

        # Add edges incrementally
        files = list(graph.nodes.keys())
        for i in range(len(files) - 1):
            graph.add_edge(files[i], files[i + 1], "absolute", [], i + 1)

        assert len(graph.edges) == 4

    def test_graph_serialization(self, tmp_path: Path) -> None:
        """Test to_dict() method."""
        graph = DependencyGraph()
        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        main_file.touch()
        utils_file.touch()

        graph.add_node(main_file, "python", ["utils"], ["main"])
        graph.add_node(utils_file, "python", [], ["helper"])
        graph.add_edge(main_file, utils_file, "relative", ["helper"], 5)

        data = graph.to_dict()

        assert data["node_count"] == 2
        assert data["edge_count"] == 1
        assert str(main_file.resolve()) in data["nodes"]
        assert len(data["edges"]) == 1
        assert data["edges"][0]["line_number"] == 5


# ============================================================================
# Test Class: TestDependencyAnalyzer
# ============================================================================


class TestDependencyAnalyzer:
    """Test DependencyAnalyzer file analysis."""

    def test_analyze_python_file(self, tmp_path: Path) -> None:
        """Test Python import analysis."""
        analyzer = DependencyAnalyzer(tmp_path)

        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        main_file.touch()
        utils_file.touch()

        symbols = create_symbol_table(
            imports=["utils", "os"],
            exports=["main"],
            file_path=main_file,
            language="python"
        )

        node = analyzer.analyze_file(main_file, symbols)

        assert node.language == "python"
        assert "utils" in node.imports
        assert "os" in node.imports
        assert "main" in node.exports

    def test_analyze_lua_file(self, tmp_path: Path) -> None:
        """Test Lua require() analysis."""
        analyzer = DependencyAnalyzer(tmp_path)

        main_file = tmp_path / "main.lua"
        utils_file = tmp_path / "utils.lua"
        main_file.touch()
        utils_file.touch()

        symbols = create_symbol_table(
            imports=["utils", "http"],
            exports=["init"],
            file_path=main_file,
            language="lua"
        )

        node = analyzer.analyze_file(main_file, symbols)

        assert node.language == "lua"
        assert "utils" in node.imports
        assert "http" in node.imports

    def test_resolve_absolute_imports(self, tmp_path: Path) -> None:
        """Test absolute import resolution."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create file structure
        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        main_file.touch()
        utils_file.write_text("def helper(): pass", encoding="utf-8")

        symbols = create_symbol_table(
            imports=["utils"],
            exports=["main"],
            file_path=main_file,
            language="python",
            import_details=[{
                "module": "utils",
                "names": [],
                "line": 1,
                "level": 0
            }]
        )

        analyzer.analyze_file(main_file, symbols)

        # Check that edge was created
        assert len(analyzer.graph.edges) == 1
        assert analyzer.graph.edges[0].to_node == utils_file.resolve()

    def test_resolve_relative_imports(self, tmp_path: Path) -> None:
        """Test relative import resolution."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create package structure
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()

        init_file = pkg_dir / "__init__.py"
        module_file = pkg_dir / "module.py"
        sibling_file = pkg_dir / "sibling.py"
        init_file.touch()
        module_file.touch()
        sibling_file.write_text("def func(): pass", encoding="utf-8")

        symbols = create_symbol_table(
            imports=["sibling"],
            exports=["test"],
            file_path=module_file,
            language="python",
            import_details=[{
                "module": "sibling",
                "names": ["func"],
                "is_from": True,
                "line": 1,
                "level": 1  # relative import
            }]
        )

        analyzer.analyze_file(module_file, symbols)

        # Should resolve to sibling.py
        deps = analyzer.graph.get_dependencies(module_file)
        assert sibling_file.resolve() in deps

    def test_build_graph_incremental(self, tmp_path: Path) -> None:
        """Test incremental graph building with multiple files."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create files
        files_data = []
        for i in range(5):
            file_path = tmp_path / f"file{i}.py"
            file_path.touch()

            # Each file imports the next one
            imports = [f"file{i+1}"] if i < 4 else []
            symbols = create_symbol_table(
                imports=imports,
                exports=[f"func{i}"],
                file_path=file_path,
                language="python"
            )
            files_data.append((file_path, symbols))

        graph = analyzer.build_graph_incremental(files_data)

        assert len(graph.nodes) == 5

    def test_mixed_python_lua_project(self, tmp_path: Path) -> None:
        """Test analyzing projects with both languages."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create Python files
        py_file = tmp_path / "main.py"
        py_file.touch()
        py_symbols = create_symbol_table(
            imports=["os"],
            exports=["main"],
            file_path=py_file,
            language="python"
        )

        # Create Lua files
        lua_file = tmp_path / "script.lua"
        lua_file.touch()
        lua_symbols = create_symbol_table(
            imports=["http"],
            exports=["init"],
            file_path=lua_file,
            language="lua"
        )

        analyzer.analyze_file(py_file, py_symbols)
        analyzer.analyze_file(lua_file, lua_symbols)

        assert len(analyzer.graph.nodes) == 2
        assert analyzer.graph.get_node(py_file).language == "python"
        assert analyzer.graph.get_node(lua_file).language == "lua"


# ============================================================================
# Test Class: TestTopologicalSort
# ============================================================================


class TestTopologicalSort:
    """Test topological sorting."""

    def test_simple_linear_dependencies(self, tmp_path: Path) -> None:
        """A -> B -> C should return [C, B, A]."""
        analyzer = DependencyAnalyzer(tmp_path)

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"
        a_file.touch()
        b_file.write_text("def b(): pass", encoding="utf-8")
        c_file.write_text("def c(): pass", encoding="utf-8")

        # A imports B, B imports C
        a_symbols = create_symbol_table(["b"], ["a"], a_file, "python",
            [{"module": "b", "line": 1, "level": 0}])
        b_symbols = create_symbol_table(["c"], ["b"], b_file, "python",
            [{"module": "c", "line": 1, "level": 0}])
        c_symbols = create_symbol_table([], ["c"], c_file, "python")

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)
        analyzer.analyze_file(c_file, c_symbols)

        order = analyzer.get_processing_order()

        # C should come before B, B before A
        assert_topological_order(order, analyzer.graph)

        c_idx = order.index(c_file.resolve())
        b_idx = order.index(b_file.resolve())
        a_idx = order.index(a_file.resolve())

        assert c_idx < b_idx < a_idx

    def test_parallel_dependencies(self, tmp_path: Path) -> None:
        """A -> C, B -> C should have C first."""
        analyzer = DependencyAnalyzer(tmp_path)

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"
        a_file.touch()
        b_file.touch()
        c_file.write_text("def c(): pass", encoding="utf-8")

        a_symbols = create_symbol_table(["c"], ["a"], a_file, "python",
            [{"module": "c", "line": 1, "level": 0}])
        b_symbols = create_symbol_table(["c"], ["b"], b_file, "python",
            [{"module": "c", "line": 1, "level": 0}])
        c_symbols = create_symbol_table([], ["c"], c_file, "python")

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)
        analyzer.analyze_file(c_file, c_symbols)

        order = analyzer.get_processing_order()

        # C should come before both A and B
        c_idx = order.index(c_file.resolve())
        a_idx = order.index(a_file.resolve())
        b_idx = order.index(b_file.resolve())

        assert c_idx < a_idx
        assert c_idx < b_idx

    def test_no_dependencies(self, tmp_path: Path) -> None:
        """Independent files should all be processable."""
        analyzer = DependencyAnalyzer(tmp_path)

        files = []
        for i in range(5):
            file_path = tmp_path / f"file{i}.py"
            file_path.touch()
            symbols = create_symbol_table([], [f"func{i}"], file_path, "python")
            analyzer.analyze_file(file_path, symbols)
            files.append(file_path.resolve())

        order = analyzer.get_processing_order()

        assert len(order) == 5
        assert set(order) == set(files)

    def test_complex_dag(self, tmp_path: Path) -> None:
        """Test with diamond dependencies and multiple levels."""
        analyzer = DependencyAnalyzer(tmp_path)

        #     A
        #    / \
        #   B   C
        #    \ /
        #     D

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"
        d_file = tmp_path / "d.py"

        for f in [a_file, b_file, c_file]:
            f.touch()
        d_file.write_text("def d(): pass", encoding="utf-8")

        a_symbols = create_symbol_table(["b", "c"], ["a"], a_file, "python",
            [{"module": "b", "line": 1, "level": 0},
             {"module": "c", "line": 2, "level": 0}])
        b_symbols = create_symbol_table(["d"], ["b"], b_file, "python",
            [{"module": "d", "line": 1, "level": 0}])
        c_symbols = create_symbol_table(["d"], ["c"], c_file, "python",
            [{"module": "d", "line": 1, "level": 0}])
        d_symbols = create_symbol_table([], ["d"], d_file, "python")

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)
        analyzer.analyze_file(c_file, c_symbols)
        analyzer.analyze_file(d_file, d_symbols)

        order = analyzer.get_processing_order()

        # D should come first, then B and C, then A
        d_idx = order.index(d_file.resolve())
        b_idx = order.index(b_file.resolve())
        c_idx = order.index(c_file.resolve())
        a_idx = order.index(a_file.resolve())

        assert d_idx < b_idx
        assert d_idx < c_idx
        assert b_idx < a_idx
        assert c_idx < a_idx

    def test_empty_graph(self, tmp_path: Path) -> None:
        """Handle empty graph gracefully."""
        analyzer = DependencyAnalyzer(tmp_path)

        order = analyzer.get_processing_order()

        assert order == []


# ============================================================================
# Test Class: TestCycleDetection
# ============================================================================


class TestCycleDetection:
    """Test cycle detection."""

    def test_simple_cycle(self, tmp_path: Path) -> None:
        """A -> B -> A."""
        analyzer = DependencyAnalyzer(tmp_path)

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        a_file.write_text("from b import func_b", encoding="utf-8")
        b_file.write_text("from a import func_a", encoding="utf-8")

        a_symbols = create_symbol_table(["b"], ["func_a"], a_file, "python",
            [{"module": "b", "line": 1, "level": 0}])
        b_symbols = create_symbol_table(["a"], ["func_b"], b_file, "python",
            [{"module": "a", "line": 1, "level": 0}])

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)

        cycles = analyzer.detect_cycles()

        assert len(cycles) >= 1

    def test_three_node_cycle(self, tmp_path: Path) -> None:
        """A -> B -> C -> A."""
        analyzer = DependencyAnalyzer(tmp_path)

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"
        a_file.write_text("import b", encoding="utf-8")
        b_file.write_text("import c", encoding="utf-8")
        c_file.write_text("import a", encoding="utf-8")

        a_symbols = create_symbol_table(["b"], ["a"], a_file, "python",
            [{"module": "b", "line": 1, "level": 0}])
        b_symbols = create_symbol_table(["c"], ["b"], b_file, "python",
            [{"module": "c", "line": 1, "level": 0}])
        c_symbols = create_symbol_table(["a"], ["c"], c_file, "python",
            [{"module": "a", "line": 1, "level": 0}])

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)
        analyzer.analyze_file(c_file, c_symbols)

        cycles = analyzer.detect_cycles()

        assert len(cycles) >= 1

    def test_self_loop_prevented(self, tmp_path: Path) -> None:
        """Self-loop should be prevented by DependencyEdge validation."""
        file_path = tmp_path / "test.py"
        file_path.touch()

        with pytest.raises(ValueError, match="Self-loop"):
            DependencyEdge(from_node=file_path, to_node=file_path)

    def test_no_cycles(self, tmp_path: Path) -> None:
        """Return empty list for acyclic graph."""
        analyzer = DependencyAnalyzer(tmp_path)

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"
        a_file.touch()
        b_file.write_text("def b(): pass", encoding="utf-8")
        c_file.write_text("def c(): pass", encoding="utf-8")

        a_symbols = create_symbol_table(["b"], ["a"], a_file, "python",
            [{"module": "b", "line": 1, "level": 0}])
        b_symbols = create_symbol_table(["c"], ["b"], b_file, "python",
            [{"module": "c", "line": 1, "level": 0}])
        c_symbols = create_symbol_table([], ["c"], c_file, "python")

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)
        analyzer.analyze_file(c_file, c_symbols)

        cycles = analyzer.detect_cycles()

        assert cycles == []

    def test_cycle_raises_error_on_sort(self, tmp_path: Path) -> None:
        """Verify CircularDependencyError is raised during topological sort."""
        analyzer = DependencyAnalyzer(tmp_path)

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        a_file.write_text("import b", encoding="utf-8")
        b_file.write_text("import a", encoding="utf-8")

        a_symbols = create_symbol_table(["b"], ["a"], a_file, "python",
            [{"module": "b", "line": 1, "level": 0}])
        b_symbols = create_symbol_table(["a"], ["b"], b_file, "python",
            [{"module": "a", "line": 1, "level": 0}])

        analyzer.analyze_file(a_file, a_symbols)
        analyzer.analyze_file(b_file, b_symbols)

        with pytest.raises(CircularDependencyError) as exc_info:
            analyzer.get_processing_order()

        assert len(exc_info.value.cycle) >= 2


# ============================================================================
# Test Class: TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_circular_dependency_error(self, tmp_path: Path) -> None:
        """Verify exception is raised with proper message."""
        cycle = [
            tmp_path / "a.py",
            tmp_path / "b.py",
            tmp_path / "a.py"
        ]
        line_info = {
            (cycle[0], cycle[1]): 10,
            (cycle[1], cycle[0]): 5
        }

        error = CircularDependencyError(cycle, line_info)

        assert "Circular dependency detected" in str(error)
        assert error.cycle == cycle

    def test_dependency_resolution_error(self, tmp_path: Path) -> None:
        """Verify DependencyResolutionError formatting."""
        file_path = tmp_path / "main.py"

        error = DependencyResolutionError(
            file_path=file_path,
            import_statement="missing_module",
            details="Module not found in project"
        )

        assert "Cannot resolve import" in str(error)
        assert "missing_module" in str(error)
        assert str(file_path) in str(error)

    def test_unresolved_import_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Check that warnings are logged for missing imports."""
        import logging
        caplog.set_level(logging.WARNING)

        analyzer = DependencyAnalyzer(tmp_path)

        main_file = tmp_path / "main.py"
        main_file.touch()

        # Import a non-existent module
        symbols = create_symbol_table(
            imports=["nonexistent_module"],
            exports=["main"],
            file_path=main_file,
            language="python"
        )

        analyzer.analyze_file(main_file, symbols)

        # Should log a warning about unresolved import
        assert any("Could not resolve import" in record.message
                   for record in caplog.records)

    def test_invalid_file_path_node(self, tmp_path: Path) -> None:
        """Handle non-existent files gracefully."""
        analyzer = DependencyAnalyzer(tmp_path)

        nonexistent = tmp_path / "nonexistent.py"

        # Should still create a node even if file doesn't exist
        symbols = create_symbol_table(
            imports=[],
            exports=["test"],
            file_path=nonexistent,
            language="python"
        )

        node = analyzer.analyze_file(nonexistent, symbols)

        assert node is not None
        assert node.file_path == nonexistent.resolve()

    def test_large_project_performance(self, tmp_path: Path) -> None:
        """Test with 1000+ files to verify memory efficiency."""
        analyzer = DependencyAnalyzer(tmp_path)

        num_files = 1000
        files_data = []

        # Create a chain of dependencies: file0 <- file1 <- ... <- file999
        for i in range(num_files):
            file_path = tmp_path / f"file{i}.py"
            file_path.touch()

            # Each file (except first) imports the previous one
            if i > 0:
                symbols = create_symbol_table(
                    imports=[f"file{i-1}"],
                    exports=[f"func{i}"],
                    file_path=file_path,
                    language="python",
                    import_details=[{
                        "module": f"file{i-1}",
                        "line": 1,
                        "level": 0
                    }]
                )
            else:
                symbols = create_symbol_table(
                    imports=[],
                    exports=["func0"],
                    file_path=file_path,
                    language="python"
                )

            files_data.append((file_path, symbols))

        # Build graph incrementally
        graph = analyzer.build_graph_incremental(files_data)

        assert len(graph.nodes) == num_files

        # Detect cycles (should be none)
        cycles = analyzer.detect_cycles()
        assert cycles == []

        # Get processing order
        order = analyzer.get_processing_order()
        assert len(order) == num_files

        # Verify file0 comes first (it has no dependencies)
        first_file = tmp_path / "file0.py"
        assert order[0] == first_file.resolve()

    def test_empty_symbol_table(self, tmp_path: Path) -> None:
        """Test analyzing file with empty symbol table."""
        analyzer = DependencyAnalyzer(tmp_path)

        file_path = tmp_path / "empty.py"
        file_path.touch()

        symbols = create_symbol_table(
            imports=[],
            exports=[],
            file_path=file_path,
            language="python"
        )

        node = analyzer.analyze_file(file_path, symbols)

        assert node is not None
        assert node.imports == []
        assert node.exports == []

    def test_duplicate_imports(self, tmp_path: Path) -> None:
        """Test handling of duplicate imports in same file."""
        analyzer = DependencyAnalyzer(tmp_path)

        main_file = tmp_path / "main.py"
        utils_file = tmp_path / "utils.py"
        main_file.touch()
        utils_file.write_text("def helper(): pass", encoding="utf-8")

        # Same module imported twice
        symbols = create_symbol_table(
            imports=["utils", "utils"],
            exports=["main"],
            file_path=main_file,
            language="python",
            import_details=[
                {"module": "utils", "line": 1, "level": 0},
                {"module": "utils", "line": 2, "level": 0, "names": ["helper"]}
            ]
        )

        analyzer.analyze_file(main_file, symbols)

        # Should have two edges (both imports are tracked)
        assert len(analyzer.graph.edges) == 2


# ============================================================================
# Test Class: TestPythonRelativeImports
# ============================================================================


class TestPythonRelativeImports:
    """Test Python relative import resolution including empty module names and multi-level imports."""

    def test_from_dot_import_sibling(self, tmp_path: Path) -> None:
        """Test 'from . import sibling' pattern with empty module name."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create package structure
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()

        init_file = pkg_dir / "__init__.py"
        module_file = pkg_dir / "module.py"
        sibling_file = pkg_dir / "sibling.py"
        init_file.touch()
        module_file.touch()
        sibling_file.write_text("def func(): pass", encoding="utf-8")

        # 'from . import sibling' - module_name is empty, level=1
        symbols = create_symbol_table(
            imports=[""],  # module_name is empty for 'from . import sibling'
            exports=["test"],
            file_path=module_file,
            language="python",
            import_details=[{
                "module": "",  # Empty module name
                "names": ["sibling"],  # The imported names
                "is_from": True,
                "line": 1,
                "level": 1  # Single dot
            }]
        )

        analyzer.analyze_file(module_file, symbols)

        # Should resolve to sibling.py
        deps = analyzer.graph.get_dependencies(module_file)
        assert sibling_file.resolve() in deps

    def test_from_dotdot_import_module(self, tmp_path: Path) -> None:
        """Test 'from .. import module' pattern with level=2."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create nested package structure
        # pkg/
        #   __init__.py
        #   parent_module.py
        #   subpkg/
        #     __init__.py
        #     deep_module.py
        pkg_dir = tmp_path / "pkg"
        subpkg_dir = pkg_dir / "subpkg"
        pkg_dir.mkdir()
        subpkg_dir.mkdir()

        (pkg_dir / "__init__.py").touch()
        (subpkg_dir / "__init__.py").touch()
        parent_module = pkg_dir / "parent_module.py"
        deep_module = subpkg_dir / "deep_module.py"
        parent_module.write_text("def parent_func(): pass", encoding="utf-8")
        deep_module.touch()

        # 'from .. import parent_module' - level=2, empty module name
        symbols = create_symbol_table(
            imports=[""],
            exports=["test"],
            file_path=deep_module,
            language="python",
            import_details=[{
                "module": "",  # Empty for 'from .. import name'
                "names": ["parent_module"],
                "is_from": True,
                "line": 1,
                "level": 2  # Double dot
            }]
        )

        analyzer.analyze_file(deep_module, symbols)

        # Should resolve to parent_module.py (one level up from subpkg)
        deps = analyzer.graph.get_dependencies(deep_module)
        assert parent_module.resolve() in deps

    def test_from_dotdot_package_import_submodule(self, tmp_path: Path) -> None:
        """Test 'from ..sibling_pkg import util' pattern."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create package structure:
        # root/
        #   pkg_a/
        #     __init__.py
        #     subpkg/
        #       __init__.py
        #       module.py
        #   pkg_b/
        #     __init__.py
        #     util.py
        root_dir = tmp_path / "root"
        pkg_a = root_dir / "pkg_a"
        subpkg = pkg_a / "subpkg"
        pkg_b = root_dir / "pkg_b"

        for d in [root_dir, pkg_a, subpkg, pkg_b]:
            d.mkdir(parents=True, exist_ok=True)

        (pkg_a / "__init__.py").touch()
        (subpkg / "__init__.py").touch()
        (pkg_b / "__init__.py").touch()

        module_file = subpkg / "module.py"
        util_file = pkg_b / "util.py"
        module_file.touch()
        util_file.write_text("def util_func(): pass", encoding="utf-8")

        # Use root as the analyzer root
        analyzer = DependencyAnalyzer(root_dir)

        # 'from ...pkg_b import util' - level=3, module=pkg_b, names=[util]
        symbols = create_symbol_table(
            imports=["pkg_b"],
            exports=["test"],
            file_path=module_file,
            language="python",
            import_details=[{
                "module": "pkg_b",
                "names": ["util"],
                "is_from": True,
                "line": 1,
                "level": 3  # ... goes up to root
            }]
        )

        analyzer.analyze_file(module_file, symbols)

        # Should resolve to pkg_b/util.py
        deps = analyzer.graph.get_dependencies(module_file)
        assert util_file.resolve() in deps


# ============================================================================
# Test Class: TestLuaRelativeRequire
# ============================================================================


class TestLuaRelativeRequire:
    """Test Lua relative require() resolution with path prefixes."""

    def test_require_dot_slash_helper(self, tmp_path: Path) -> None:
        """Test require('./helper') resolves to helper.lua."""
        analyzer = DependencyAnalyzer(tmp_path)

        main_file = tmp_path / "main.lua"
        helper_file = tmp_path / "helper.lua"
        main_file.touch()
        helper_file.write_text("return {}", encoding="utf-8")

        # require("./helper")
        symbols = create_symbol_table(
            imports=["./helper"],
            exports=["init"],
            file_path=main_file,
            language="lua",
            import_details=[{
                "module": "./helper",
                "alias": "helper",
                "line": 1,
                "is_relative": True
            }]
        )

        analyzer.analyze_file(main_file, symbols)

        # Should resolve to helper.lua
        deps = analyzer.graph.get_dependencies(main_file)
        assert helper_file.resolve() in deps

    def test_require_leading_dot_helper(self, tmp_path: Path) -> None:
        """Test require('.helper') resolves to helper.lua."""
        analyzer = DependencyAnalyzer(tmp_path)

        main_file = tmp_path / "main.lua"
        helper_file = tmp_path / "helper.lua"
        main_file.touch()
        helper_file.write_text("return {}", encoding="utf-8")

        # require(".helper") - leading dot as relative marker
        symbols = create_symbol_table(
            imports=[".helper"],
            exports=["init"],
            file_path=main_file,
            language="lua",
            import_details=[{
                "module": ".helper",
                "alias": "helper",
                "line": 1,
                "is_relative": True
            }]
        )

        analyzer.analyze_file(main_file, symbols)

        # Should resolve to helper.lua
        deps = analyzer.graph.get_dependencies(main_file)
        assert helper_file.resolve() in deps

    def test_relative_require_not_converted_to_absolute(self, tmp_path: Path) -> None:
        """Ensure relative require paths don't become absolute paths."""
        analyzer = DependencyAnalyzer(tmp_path)

        # Create subdirectory with files
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        main_file = subdir / "main.lua"
        helper_file = subdir / "helper.lua"
        main_file.touch()
        helper_file.write_text("return {}", encoding="utf-8")

        # require("./helper") from within subdir
        symbols = create_symbol_table(
            imports=["./helper"],
            exports=["init"],
            file_path=main_file,
            language="lua",
            import_details=[{
                "module": "./helper",
                "alias": "helper",
                "line": 1,
                "is_relative": True
            }]
        )

        analyzer.analyze_file(main_file, symbols)

        # Should resolve to subdir/helper.lua, not /helper.lua
        deps = analyzer.graph.get_dependencies(main_file)
        assert helper_file.resolve() in deps
        # Make sure it didn't accidentally try to find a file at root
        root_helper = tmp_path / "helper.lua"
        assert root_helper.resolve() not in deps
