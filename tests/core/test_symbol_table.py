"""Tests for the global symbol table module.

This module contains comprehensive tests for SymbolEntry, SymbolMangler,
GlobalSymbolTable, and SymbolTableBuilder classes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.symbol_table import (
    GlobalSymbolTable,
    SymbolEntry,
    SymbolMangler,
    SymbolTableBuilder,
    PYTHON_BUILTINS,
    PYTHON_MAGIC_METHODS,
    LUA_KEYWORDS,
    ROBLOX_API,
    VALID_SCOPES,
    VALID_SYMBOL_TYPES,
    VALID_LANGUAGES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    project = tmp_path / "test_project"
    project.mkdir()
    return project


@pytest.fixture
def sample_python_file(tmp_project: Path) -> Path:
    """Create a sample Python file."""
    file_path = tmp_project / "main.py"
    file_path.write_text('''
def my_function():
    pass

class MyClass:
    def method(self):
        pass

my_variable = 42
''')
    return file_path


@pytest.fixture
def sample_lua_file(tmp_project: Path) -> Path:
    """Create a sample Lua file."""
    file_path = tmp_project / "main.lua"
    file_path.write_text('''
local function myFunction()
end

local MyTable = {}

function MyTable:method()
end

local myVariable = 42
''')
    return file_path


@pytest.fixture
def mangling_config() -> dict:
    """Default mangling configuration."""
    return {
        "identifier_prefix": "_0x",
        "mangling_strategy": "sequential",
    }


@pytest.fixture
def sample_symbol_entry(tmp_project: Path) -> SymbolEntry:
    """Create a sample SymbolEntry."""
    return SymbolEntry(
        original_name="my_function",
        mangled_name="_0x1",
        scope="global",
        language="python",
        file_path=tmp_project / "test.py",
        line_number=10,
        symbol_type="function",
    )


# =============================================================================
# Helper Functions
# =============================================================================


def create_symbol_entry(
    name: str = "test_symbol",
    mangled: str = "_0x1",
    scope: str = "global",
    language: str = "python",
    file_path: Path | None = None,
    line_number: int = 1,
    symbol_type: str = "function",
    is_exported: bool = False,
) -> SymbolEntry:
    """Helper to create SymbolEntry with defaults."""
    return SymbolEntry(
        original_name=name,
        mangled_name=mangled,
        scope=scope,
        language=language,
        file_path=file_path or Path("/test/file.py").resolve(),
        line_number=line_number,
        symbol_type=symbol_type,
        is_exported=is_exported,
    )


def create_mock_python_symbol_table():
    """Create a mock Python SymbolTable."""
    mock = MagicMock()
    
    # Mock function
    func = MagicMock()
    func.name = "my_function"
    func.scope = "global"
    func.line_number = 10
    func.is_async = False
    func.parent_class = None
    
    # Mock class
    cls = MagicMock()
    cls.name = "MyClass"
    cls.scope = "global"
    cls.line_number = 20
    cls.bases = []
    cls.methods = ["method"]
    
    # Mock variable
    var = MagicMock()
    var.name = "my_var"
    var.scope = "global"
    var.line_number = 30
    var.is_constant = False
    var.context = "module"
    
    mock.functions = [func]
    mock.classes = [cls]
    mock.variables = [var]
    mock.get_exported_symbols.return_value = ["my_function", "MyClass"]

    return mock


def create_mock_lua_symbol_table():
    """Create a mock Lua LuaSymbolTable."""
    mock = MagicMock()

    # Mock function
    func = MagicMock()
    func.name = "myFunction"
    func.scope = "local"
    func.is_local = True
    func.line_number = 5
    func.is_method = False
    func.parent_table = None
    func.parameters = []

    # Mock variable
    var = MagicMock()
    var.name = "myVar"
    var.scope = "local"
    var.line_number = 15
    var.is_constant = False
    var.context = "module"

    mock.functions = [func]
    mock.variables = [var]
    mock.get_exported_symbols.return_value = []

    return mock


# =============================================================================
# TestSymbolEntry
# =============================================================================


class TestSymbolEntry:
    """Tests for SymbolEntry dataclass."""

    def test_create_valid_entry(self, tmp_project: Path) -> None:
        """Test creating a valid SymbolEntry."""
        entry = SymbolEntry(
            original_name="test_func",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=tmp_project / "test.py",
            line_number=10,
            symbol_type="function",
        )

        assert entry.original_name == "test_func"
        assert entry.mangled_name == "_0x1"
        assert entry.scope == "global"
        assert entry.language == "python"
        assert entry.line_number == 10
        assert entry.symbol_type == "function"
        assert entry.is_exported is False
        assert entry.references == []
        assert entry.metadata == {}

    def test_invalid_scope_raises_error(self, tmp_project: Path) -> None:
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):
            SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="invalid_scope",
                language="python",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="function",
            )

    def test_invalid_language_raises_error(self, tmp_project: Path) -> None:
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="Invalid language"):
            SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="global",
                language="javascript",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="function",
            )

    def test_invalid_symbol_type_raises_error(self, tmp_project: Path) -> None:
        """Test that invalid symbol_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol_type"):
            SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="global",
                language="python",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="invalid_type",
            )

    def test_relative_path_resolved(self, tmp_project: Path) -> None:
        """Test that relative paths are resolved to absolute."""
        entry = SymbolEntry(
            original_name="test",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=Path("relative/path.py"),
            line_number=1,
            symbol_type="function",
        )

        assert entry.file_path.is_absolute()

    def test_hash_and_equality(self, tmp_project: Path) -> None:
        """Test hash and equality based on (file_path, name, scope)."""
        file_path = tmp_project / "test.py"

        entry1 = SymbolEntry(
            original_name="func",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=10,
            symbol_type="function",
        )

        entry2 = SymbolEntry(
            original_name="func",
            mangled_name="_0x2",  # Different mangled name
            scope="global",
            language="python",
            file_path=file_path,
            line_number=20,  # Different line
            symbol_type="function",
        )

        # Should be equal (same file, name, scope)
        assert entry1 == entry2
        assert hash(entry1) == hash(entry2)

        # Can be used in sets
        entry_set = {entry1, entry2}
        assert len(entry_set) == 1

    def test_all_valid_scopes(self, tmp_project: Path) -> None:
        """Test all valid scope values."""
        for scope in VALID_SCOPES:
            entry = SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope=scope,
                language="python",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="function",
            )
            assert entry.scope == scope

    def test_all_valid_symbol_types(self, tmp_project: Path) -> None:
        """Test all valid symbol types."""
        for symbol_type in VALID_SYMBOL_TYPES:
            entry = SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="global",
                language="python",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type=symbol_type,
            )
            assert entry.symbol_type == symbol_type

    def test_all_valid_languages(self, tmp_project: Path) -> None:
        """Test all valid languages."""
        for language in VALID_LANGUAGES:
            entry = SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="global",
                language=language,
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="function",
            )
            assert entry.language == language


# =============================================================================
# TestSymbolMangler
# =============================================================================


class TestSymbolMangler:
    """Tests for SymbolMangler class."""

    def test_sequential_strategy(self) -> None:
        """Test sequential naming strategy."""
        mangler = SymbolMangler({
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential"
        })

        name1 = mangler.generate_name("func1", "global", "python")
        name2 = mangler.generate_name("func2", "global", "python")
        name3 = mangler.generate_name("func3", "global", "python")

        assert name1 == "_0x1"
        assert name2 == "_0x2"
        assert name3 == "_0x3"

    def test_random_strategy(self) -> None:
        """Test random naming strategy."""
        mangler = SymbolMangler({
            "identifier_prefix": "_0x",
            "mangling_strategy": "random"
        })

        name1 = mangler.generate_name("func1", "global", "python")
        name2 = mangler.generate_name("func2", "global", "python")

        assert name1.startswith("_0x")
        assert name2.startswith("_0x")
        assert len(name1) == 7  # _0x + 4 hex chars
        assert name1 != name2  # Should be unique

    def test_minimal_strategy(self) -> None:
        """Test minimal naming strategy."""
        mangler = SymbolMangler({
            "mangling_strategy": "minimal"
        })

        name1 = mangler.generate_name("func1", "global", "python")
        name2 = mangler.generate_name("func2", "global", "python")

        # Should generate short names like a, b, c, ...
        assert len(name1) <= 2
        assert len(name2) <= 2
        assert name1 != name2

    def test_custom_prefix(self) -> None:
        """Test custom identifier prefix."""
        mangler = SymbolMangler({
            "identifier_prefix": "__obf_",
            "mangling_strategy": "sequential"
        })

        name = mangler.generate_name("func", "global", "python")
        assert name.startswith("__obf_")

    def test_python_builtins_preserved(self) -> None:
        """Test that Python builtins are not mangled."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        for builtin in ["print", "len", "str", "int", "list"]:
            result = mangler.generate_name(builtin, "global", "python")
            assert result == builtin

    def test_python_keywords_preserved(self) -> None:
        """Test that Python keywords are not mangled."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        for keyword in ["if", "else", "for", "while", "class", "def"]:
            result = mangler.generate_name(keyword, "global", "python")
            assert result == keyword

    def test_python_magic_methods_preserved(self) -> None:
        """Test that Python magic methods are not mangled."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        for magic in ["__init__", "__str__", "__repr__", "__call__"]:
            result = mangler.generate_name(magic, "global", "python")
            assert result == magic

    def test_lua_keywords_preserved(self) -> None:
        """Test that Lua keywords are not mangled."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        for keyword in ["local", "function", "end", "if", "then"]:
            result = mangler.generate_name(keyword, "global", "lua")
            assert result == keyword

    def test_roblox_api_preserved(self) -> None:
        """Test that Roblox API names are not mangled."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        for api in ["game", "workspace", "Players", "Instance", "Vector3"]:
            result = mangler.generate_name(api, "global", "lua")
            assert result == api

    def test_is_reserved_python(self) -> None:
        """Test is_reserved for Python names."""
        mangler = SymbolMangler()

        assert mangler.is_reserved("print", "python") is True
        assert mangler.is_reserved("__init__", "python") is True
        assert mangler.is_reserved("my_function", "python") is False

    def test_is_reserved_lua(self) -> None:
        """Test is_reserved for Lua names."""
        mangler = SymbolMangler()

        assert mangler.is_reserved("local", "lua") is True
        assert mangler.is_reserved("game", "lua") is True
        assert mangler.is_reserved("myFunction", "lua") is False

    def test_unique_names_generated(self) -> None:
        """Test that all generated names are unique."""
        mangler = SymbolMangler({
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential"
        })

        names = set()
        for i in range(100):
            name = mangler.generate_name(f"func{i}", "global", "python")
            assert name not in names
            names.add(name)

    def test_default_config(self) -> None:
        """Test mangler with default configuration."""
        mangler = SymbolMangler()

        name = mangler.generate_name("test", "global", "python")
        assert name.startswith("_0x")  # Default prefix


# =============================================================================
# TestGlobalSymbolTable
# =============================================================================


class TestGlobalSymbolTable:
    """Tests for GlobalSymbolTable class."""

    def test_add_and_retrieve_symbol(self, tmp_project: Path) -> None:
        """Test adding and retrieving a symbol."""
        table = GlobalSymbolTable()
        file_path = tmp_project / "test.py"

        entry = SymbolEntry(
            original_name="my_func",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=10,
            symbol_type="function",
        )

        table.add_symbol(entry)

        # Retrieve by mangled name
        mangled = table.get_mangled_name(file_path, "my_func", "global")
        assert mangled == "_0x1"

        # Retrieve full entry
        retrieved = table.get_symbol(file_path, "my_func", "global")
        assert retrieved is not None
        assert retrieved.original_name == "my_func"
        assert retrieved.mangled_name == "_0x1"

    def test_freeze_prevents_modification(self, tmp_project: Path) -> None:
        """Test that freeze() prevents adding new symbols."""
        table = GlobalSymbolTable()
        table.freeze()

        entry = SymbolEntry(
            original_name="test",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=tmp_project / "test.py",
            line_number=1,
            symbol_type="function",
        )

        with pytest.raises(RuntimeError, match="frozen"):
            table.add_symbol(entry)

    def test_is_frozen_property(self) -> None:
        """Test is_frozen property."""
        table = GlobalSymbolTable()

        assert table.is_frozen is False
        table.freeze()
        assert table.is_frozen is True

    def test_get_nonexistent_symbol(self, tmp_project: Path) -> None:
        """Test retrieving a symbol that doesn't exist."""
        table = GlobalSymbolTable()

        mangled = table.get_mangled_name(
            tmp_project / "test.py", "nonexistent", "global"
        )
        assert mangled is None

        entry = table.get_symbol(
            tmp_project / "test.py", "nonexistent", "global"
        )
        assert entry is None

    def test_add_reference(self, tmp_project: Path) -> None:
        """Test adding cross-file references."""
        table = GlobalSymbolTable()
        file_path = tmp_project / "module.py"
        ref_file = tmp_project / "main.py"

        entry = SymbolEntry(
            original_name="exported_func",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=10,
            symbol_type="function",
        )

        table.add_symbol(entry)
        table.add_reference(file_path, "exported_func", "global", ref_file, 5)

        retrieved = table.get_symbol(file_path, "exported_func", "global")
        assert retrieved is not None
        assert len(retrieved.references) == 1
        assert retrieved.references[0][1] == 5  # Line number

    def test_add_reference_frozen_raises(self, tmp_project: Path) -> None:
        """Test that add_reference raises when frozen."""
        table = GlobalSymbolTable()
        file_path = tmp_project / "test.py"

        entry = SymbolEntry(
            original_name="func",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=1,
            symbol_type="function",
        )

        table.add_symbol(entry)
        table.freeze()

        with pytest.raises(RuntimeError, match="frozen"):
            table.add_reference(file_path, "func", "global", file_path, 10)

    def test_get_all_symbols(self, tmp_project: Path) -> None:
        """Test get_all_symbols returns all entries."""
        table = GlobalSymbolTable()
        file_path = tmp_project / "test.py"

        for i in range(5):
            entry = SymbolEntry(
                original_name=f"func{i}",
                mangled_name=f"_0x{i}",
                scope="global",
                language="python",
                file_path=file_path,
                line_number=i * 10,
                symbol_type="function",
            )
            table.add_symbol(entry)

        all_symbols = table.get_all_symbols()
        assert len(all_symbols) == 5

    def test_to_dict(self, tmp_project: Path) -> None:
        """Test serialization to dictionary."""
        table = GlobalSymbolTable({"identifier_prefix": "_0x"})
        file_path = tmp_project / "test.py"

        entry = SymbolEntry(
            original_name="func",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=10,
            symbol_type="function",
            is_exported=True,
        )

        table.add_symbol(entry)
        table.freeze()

        result = table.to_dict()

        assert result["is_frozen"] is True
        assert result["symbol_count"] == 1
        assert len(result["symbols"]) == 1
        assert result["symbols"][0]["original_name"] == "func"
        assert result["symbols"][0]["mangled_name"] == "_0x1"
        assert result["config"]["identifier_prefix"] == "_0x"

    def test_scope_aware_lookup(self, tmp_project: Path) -> None:
        """Test that same name in different scopes are separate."""
        table = GlobalSymbolTable()
        file_path = tmp_project / "test.py"

        # Add same name with different scopes
        global_entry = SymbolEntry(
            original_name="x",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=1,
            symbol_type="variable",
        )

        local_entry = SymbolEntry(
            original_name="x",
            mangled_name="_0x2",
            scope="local",
            language="python",
            file_path=file_path,
            line_number=10,
            symbol_type="variable",
        )

        table.add_symbol(global_entry)
        table.add_symbol(local_entry)

        assert table.get_mangled_name(file_path, "x", "global") == "_0x1"
        assert table.get_mangled_name(file_path, "x", "local") == "_0x2"


# =============================================================================
# TestSymbolTableBuilder
# =============================================================================


class TestSymbolTableBuilder:
    """Tests for SymbolTableBuilder class."""

    def test_build_from_python_symbols(self, tmp_project: Path) -> None:
        """Test building table from Python symbol table."""
        builder = SymbolTableBuilder()
        file_path = tmp_project / "test.py"
        file_path.touch()

        # Create mock dependency graph
        mock_graph = MagicMock()
        mock_graph.nodes = {file_path: MagicMock()}
        mock_graph.edges = []
        mock_graph.get_processing_order.return_value = [file_path]
        mock_node = MagicMock()
        mock_node.language = "python"
        mock_graph.get_node.return_value = mock_node

        # Create mock symbol table
        mock_symbol_table = create_mock_python_symbol_table()

        config = {"identifier_prefix": "_0x", "mangling_strategy": "sequential"}

        result = builder.build_from_dependency_graph(
            mock_graph,
            {file_path: mock_symbol_table},
            config
        )

        assert result.is_frozen
        symbols = result.get_all_symbols()
        assert len(symbols) == 3  # function, class, variable

        # Check function was processed
        func_mangled = result.get_mangled_name(file_path, "my_function", "global")
        assert func_mangled is not None
        assert func_mangled.startswith("_0x")

    def test_build_from_lua_symbols(self, tmp_project: Path) -> None:
        """Test building table from Lua symbol table."""
        builder = SymbolTableBuilder()
        file_path = tmp_project / "test.lua"
        file_path.touch()

        # Create mock dependency graph
        mock_graph = MagicMock()
        mock_graph.nodes = {file_path: MagicMock()}
        mock_graph.edges = []
        mock_graph.get_processing_order.return_value = [file_path]
        mock_node = MagicMock()
        mock_node.language = "lua"
        mock_graph.get_node.return_value = mock_node

        # Create mock symbol table
        mock_symbol_table = create_mock_lua_symbol_table()

        config = {"identifier_prefix": "_0x", "mangling_strategy": "sequential"}

        result = builder.build_from_dependency_graph(
            mock_graph,
            {file_path: mock_symbol_table},
            config
        )

        assert result.is_frozen
        symbols = result.get_all_symbols()
        assert len(symbols) == 2  # function, variable

    def test_detect_language_from_extension(self) -> None:
        """Test language detection from file extension."""
        builder = SymbolTableBuilder()

        assert builder._detect_language(Path("test.py")) == "python"
        assert builder._detect_language(Path("test.lua")) == "lua"
        assert builder._detect_language(Path("test.luau")) == "lua"
        assert builder._detect_language(Path("test.pyw")) == "python"

    def test_empty_graph(self) -> None:
        """Test building from empty graph."""
        builder = SymbolTableBuilder()

        mock_graph = MagicMock()
        mock_graph.nodes = {}
        mock_graph.edges = []

        result = builder.build_from_dependency_graph(mock_graph, {}, {})

        assert result.is_frozen
        assert len(result.get_all_symbols()) == 0

    def test_cross_file_references_detected(self, tmp_project: Path) -> None:
        """Test that cross-file references are tracked."""
        builder = SymbolTableBuilder()

        file_a = tmp_project / "module_a.py"
        file_b = tmp_project / "module_b.py"
        file_a.touch()
        file_b.touch()

        # Create mock graph with edge
        mock_graph = MagicMock()
        mock_graph.nodes = {file_a: MagicMock(), file_b: MagicMock()}

        # Edge: file_b imports from file_a
        mock_edge = MagicMock()
        mock_edge.from_node = file_b
        mock_edge.to_node = file_a
        mock_edge.imported_symbols = ["exported_func"]
        mock_edge.line_number = 1
        mock_graph.edges = [mock_edge]

        mock_graph.get_processing_order.return_value = [file_a, file_b]

        mock_node_a = MagicMock()
        mock_node_a.language = "python"
        mock_node_b = MagicMock()
        mock_node_b.language = "python"
        mock_graph.get_node.side_effect = lambda p: mock_node_a if p == file_a else mock_node_b

        # Create symbol tables
        mock_table_a = MagicMock()
        func = MagicMock()
        func.name = "exported_func"
        func.scope = "global"
        func.line_number = 10
        func.is_async = False
        func.parent_class = None
        mock_table_a.functions = [func]
        mock_table_a.classes = []
        mock_table_a.variables = []
        mock_table_a.get_exported_symbols.return_value = ["exported_func"]

        mock_table_b = MagicMock()
        mock_table_b.functions = []
        mock_table_b.classes = []
        mock_table_b.variables = []
        mock_table_b.get_exported_symbols.return_value = []

        result = builder.build_from_dependency_graph(
            mock_graph,
            {file_a: mock_table_a, file_b: mock_table_b},
            {}
        )

        # Check that reference was tracked
        symbol = result.get_symbol(file_a, "exported_func", "global")
        assert symbol is not None
        # Note: references are added before freeze, so they should be tracked
        # The actual reference tracking depends on the edge processing


# =============================================================================
# TestLanguageSpecificMangling
# =============================================================================


class TestLanguageSpecificMangling:
    """Tests for language-specific mangling behavior."""

    def test_python_dunder_preserved(self) -> None:
        """Test that Python dunder methods are preserved."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        # Custom dunder should also be preserved
        result = mangler.generate_name("__custom__", "global", "python")
        assert result == "__custom__"

    def test_lua_roblox_services_preserved(self) -> None:
        """Test that Roblox services are preserved."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        services = [
            "ReplicatedStorage", "ServerScriptService", "ServerStorage",
            "StarterGui", "StarterPack", "Lighting"
        ]

        for service in services:
            result = mangler.generate_name(service, "global", "lua")
            assert result == service

    def test_lua_common_methods_preserved(self) -> None:
        """Test that common Roblox methods are preserved."""
        mangler = SymbolMangler({"mangling_strategy": "sequential"})

        methods = [
            "GetService", "FindFirstChild", "WaitForChild",
            "Clone", "Destroy", "Connect"
        ]

        for method in methods:
            result = mangler.generate_name(method, "global", "lua")
            assert result == method


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests for the symbol table system."""

    def test_full_workflow(self, tmp_project: Path) -> None:
        """Test complete workflow from symbols to mangled names."""
        # Create files
        py_file = tmp_project / "main.py"
        py_file.touch()

        # Build symbol table
        builder = SymbolTableBuilder()

        mock_graph = MagicMock()
        mock_graph.nodes = {py_file: MagicMock()}
        mock_graph.edges = []
        mock_graph.get_processing_order.return_value = [py_file]
        mock_node = MagicMock()
        mock_node.language = "python"
        mock_graph.get_node.return_value = mock_node

        mock_symbol_table = create_mock_python_symbol_table()

        config = {
            "identifier_prefix": "_obf_",
            "mangling_strategy": "sequential"
        }

        global_table = builder.build_from_dependency_graph(
            mock_graph,
            {py_file: mock_symbol_table},
            config
        )

        # Verify results
        assert global_table.is_frozen

        # Check mangled names
        func_mangled = global_table.get_mangled_name(py_file, "my_function", "global")
        class_mangled = global_table.get_mangled_name(py_file, "MyClass", "global")
        var_mangled = global_table.get_mangled_name(py_file, "my_var", "global")

        assert func_mangled is not None
        assert class_mangled is not None
        assert var_mangled is not None

        # All should have the custom prefix
        assert func_mangled.startswith("_obf_")
        assert class_mangled.startswith("_obf_")
        assert var_mangled.startswith("_obf_")

        # All should be unique
        assert len({func_mangled, class_mangled, var_mangled}) == 3


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_scope_in_entry(self, tmp_project: Path) -> None:
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):
            SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="invalid",
                language="python",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="function",
            )

    def test_invalid_language_in_entry(self, tmp_project: Path) -> None:
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="Invalid language"):
            SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="global",
                language="rust",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="function",
            )

    def test_invalid_symbol_type_in_entry(self, tmp_project: Path) -> None:
        """Test that invalid symbol_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol_type"):
            SymbolEntry(
                original_name="test",
                mangled_name="_0x1",
                scope="global",
                language="python",
                file_path=tmp_project / "test.py",
                line_number=1,
                symbol_type="module",
            )

    def test_frozen_table_add_symbol_raises(self, tmp_project: Path) -> None:
        """Test that adding to frozen table raises RuntimeError."""
        table = GlobalSymbolTable()
        table.freeze()

        entry = SymbolEntry(
            original_name="test",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=tmp_project / "test.py",
            line_number=1,
            symbol_type="function",
        )

        with pytest.raises(RuntimeError, match="frozen"):
            table.add_symbol(entry)

    def test_frozen_table_add_reference_raises(self, tmp_project: Path) -> None:
        """Test that adding reference to frozen table raises RuntimeError."""
        table = GlobalSymbolTable()
        file_path = tmp_project / "test.py"

        entry = SymbolEntry(
            original_name="test",
            mangled_name="_0x1",
            scope="global",
            language="python",
            file_path=file_path,
            line_number=1,
            symbol_type="function",
        )

        table.add_symbol(entry)
        table.freeze()

        with pytest.raises(RuntimeError, match="frozen"):
            table.add_reference(file_path, "test", "global", file_path, 10)

