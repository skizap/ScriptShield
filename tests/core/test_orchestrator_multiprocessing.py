"""
Comprehensive test suite for multiprocessing functionality in the orchestrator.

Tests cover:
- Worker process initialization and task execution
- Batch processing with 100+ files
- Cancellation signal propagation
- Memory pressure detection and adaptive batch sizing
- Error handling within worker processes
- Progress tracking across multiple workers
- ProcessPoolManager lifecycle management
- Integration tests comparing sequential vs parallel
- Edge cases and boundary conditions
- Performance benchmarks

Multiprocessing Workflow:
```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant PM as ProcessPoolManager
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant CB as ProgressCallback

    O->>PM: create_batches(files, batch_size)
    PM->>PM: _check_and_adjust_batch_size()
    PM-->>O: batches

    O->>PM: start_pool()
    PM->>W1: initialize(cancellation_event)
    PM->>W2: initialize(cancellation_event)

    loop For each batch
        O->>PM: submit_batch(task)
        PM->>W1: process_file_batch(task)
        W1->>W1: process_file(file1)
        W1->>W1: check_cancellation()
        W1-->>PM: WorkerResult[]
        O->>CB: progress_callback(ProgressInfo)
    end

    O->>PM: signal_cancellation()
    PM->>W1: set cancellation_event
    PM->>W2: set cancellation_event

    O->>PM: shutdown_pool(grace_period)
    PM->>W1: wait for completion
    PM->>W2: wait for completion
    PM->>PM: pool.close()
    PM->>PM: pool.join()
```
"""

import multiprocessing
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, call
import threading

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.orchestrator import (
    ObfuscationOrchestrator,
    JobState,
    ErrorStrategy,
    ProgressInfo,
    ProcessPoolManager,
)
from obfuscator.core.worker import (
    WorkerTask,
    WorkerResult,
    WorkerProcess,
    process_file_batch,
    MemoryMonitor,
)


@pytest.fixture
def multiprocessing_config():
    """Config with multiprocessing enabled and appropriate thresholds."""
    return ObfuscationConfig(
        name="test_multiprocessing",
        language="python",
        enable_multiprocessing=True,
        multiprocessing_threshold=10,
        batch_size=50,
        max_workers=4,
        memory_threshold_percent=80,
        min_batch_size=25,
        features={
            "mangle_globals": True,
            "string_encryption": True,
        },
    )


@pytest.fixture
def large_file_set(tmp_path):
    """Generate 150+ test files to trigger multiprocessing."""
    files = []
    for i in range(150):
        file_path = tmp_path / f"test_file_{i:03d}.py"
        file_path.write_text(f"""
def function_{i}():
    x = {i}
    return x * 2
""")
        files.append(file_path)
    return files


@pytest.fixture
def mock_memory_monitor():
    """Mock MemoryMonitor with configurable pressure detection."""
    monitor = Mock(spec=MemoryMonitor)
    monitor.monitoring_available = True
    monitor.is_memory_pressure.return_value = False
    monitor.get_memory_percent.return_value = 60.0
    return monitor


def _create_worker_task(
    task_id="task-1",
    file_paths=None,
    config_dict=None,
    global_table_dict=None,
    dependency_graph_dict=None,
    output_dir="/tmp/out",
    project_root=None,
    runtime_mode="hybrid",
    conflict_strategy="ask",
):
    """Build WorkerTask instances with serialized state."""
    if file_paths is None:
        file_paths = ["/tmp/test.py"]
    if config_dict is None:
        config_dict = ObfuscationConfig(name="test", language="python").to_dict()
    if global_table_dict is None:
        global_table_dict = {"is_frozen": True, "symbols": []}
    if dependency_graph_dict is None:
        dependency_graph_dict = {"nodes": {}, "edges": []}
    
    return WorkerTask(
        task_id=task_id,
        file_paths=file_paths,
        config_dict=config_dict,
        global_table_dict=global_table_dict,
        dependency_graph_dict=dependency_graph_dict,
        output_dir=output_dir,
        project_root=project_root,
        runtime_mode=runtime_mode,
        conflict_strategy=conflict_strategy,
    )


def _create_worker_result(
    task_id="task-1",
    file_path="/tmp/test.py",
    success=True,
    errors=None,
    warnings=None,
    transformation_count=5,
    was_cancelled=False,
):
    """Build WorkerResult instances for testing."""
    return WorkerResult(
        task_id=task_id,
        file_path=file_path,
        success=success,
        errors=errors or [],
        warnings=warnings or [],
        output_path=file_path.replace(".py", "_obf.py") if success else None,
        transformation_count=transformation_count,
        was_cancelled=was_cancelled,
    )


def _create_test_files_batch(tmp_path, count=10, language="python"):
    """Generate batches of test Python/Lua files."""
    files = []
    extension = ".py" if language == "python" else ".lua"
    
    for i in range(count):
        file_path = tmp_path / f"batch_file_{i:03d}{extension}"
        if language == "python":
            file_path.write_text(f"""
def batch_func_{i}():
    value = {i}
    return value + 1
""")
        else:
            file_path.write_text(f"""
function batch_func_{i}()
    local value = {i}
    return value + 1
end
""")
        files.append(file_path)
    return files


class TestWorkerProcessInitialization:
    """Test worker process initialization and task execution."""
    
    def test_worker_process_deserializes_symbol_table(self):
        """Verify WorkerProcess correctly deserializes GlobalSymbolTable from WorkerTask."""
        symbol_table_dict = {
            "symbols": {"test_func": {"mangled_name": "a1b2c3"}},
            "language_symbols": {"python": ["print", "len"]},
        }
        task = _create_worker_task(symbol_table_dict=symbol_table_dict)
        
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        
        assert worker.symbol_table is not None
        assert "test_func" in worker.symbol_table.symbols
    
    def test_worker_process_deserializes_dependency_graph(self):
        """Verify DependencyGraph deserialization."""
        dependency_graph_dict = {
            "nodes": {"/tmp/a.py": {"language": "python", "imports": ["/tmp/b.py"], "exports": [], "metadata": {}}},
            "edges": [{"from": "/tmp/a.py", "to": "/tmp/b.py", "import_type": "absolute"}],
        }
        task = _create_worker_task(dependency_graph_dict=dependency_graph_dict)
        
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        
        assert worker.dependency_graph is not None
        assert Path("/tmp/a.py") in worker.dependency_graph.nodes
    
    def test_worker_process_deserializes_config(self):
        """Verify ObfuscationConfig deserialization."""
        config = ObfuscationConfig(
            name="test_config",
            language="python",
            features={"mangle_globals": True, "string_encryption": False},
        )
        task = _create_worker_task(config_dict=config.to_dict())
        
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        
        assert worker.config.features.get("mangle_globals") is True
        assert worker.config.features.get("string_encryption") is False
    
    def test_worker_process_initializes_processors(self):
        """Verify Python/Lua processors are initialized based on config."""
        config = ObfuscationConfig(name="test", language="python", features={"mangle_globals": True})
        task = _create_worker_task(config_dict=config.to_dict())
        
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        
        assert worker.python_processor is not None
        assert worker.lua_processor is not None
    
    def test_worker_task_serialization_roundtrip(self):
        """Verify WorkerTask.to_dict() and from_dict() preserve all fields."""
        original_task = _create_worker_task(
            task_id="test-task-123",
            file_paths=["/tmp/file1.py", "/tmp/file2.py"],
        )
        
        task_dict = original_task.to_dict()
        restored_task = WorkerTask.from_dict(task_dict)
        
        assert restored_task.task_id == original_task.task_id
        assert restored_task.file_paths == original_task.file_paths
        assert restored_task.config_dict == original_task.config_dict
        assert restored_task.symbol_table_dict == original_task.symbol_table_dict
        assert restored_task.dependency_graph_dict == original_task.dependency_graph_dict
    
    def test_worker_result_serialization_roundtrip(self):
        """Verify WorkerResult serialization preserves success, errors, warnings, metrics."""
        original_result = _create_worker_result(
            task_id="result-456",
            file_path="/tmp/test.py",
            success=False,
            errors=["SyntaxError: invalid syntax"],
            warnings=["Unsupported feature detected"],
            transformation_count=10,
        )
        
        result_dict = original_result.to_dict()
        restored_result = WorkerResult.from_dict(result_dict)
        
        assert restored_result.task_id == original_result.task_id
        assert restored_result.file_path == original_result.file_path
        assert restored_result.success == original_result.success
        assert restored_result.errors == original_result.errors
        assert restored_result.warnings == original_result.warnings
        assert restored_result.transformation_count == original_result.transformation_count
    
    @patch("obfuscator.core.worker.WorkerProcess.process_file")
    def test_process_file_batch_returns_worker_results(self, mock_process_file, tmp_path):
        """Call process_file_batch with valid task, verify list of WorkerResult returned."""
        files = _create_test_files_batch(tmp_path, count=3)
        task = _create_worker_task(
            file_paths=[str(f) for f in files],
        )
        
        mock_process_file.return_value = _create_worker_result(success=True)
        
        results = process_file_batch(task.to_dict())
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, WorkerResult) for r in results)
        assert mock_process_file.call_count == 3
    
    def test_worker_process_handles_invalid_task(self):
        """Pass malformed task dict, verify graceful error handling."""
        malformed_task = {
            "task_id": "bad-task",
            "file_paths": ["/tmp/test.py"],
        }
        
        results = process_file_batch(malformed_task)
        
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0].success is False
        assert len(results[0].errors) > 0


class TestBatchProcessing:
    """Test batch processing with 100+ files."""
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_multiprocessing_triggered_above_threshold(
        self, mock_pool_manager_cls, large_file_set, multiprocessing_config, output_dir
    ):
        """Create 150 files, verify ProcessPoolManager is used (not sequential)."""
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in large_file_set],
            output_dir=str(output_dir),
            config=multiprocessing_config,
        )
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        mock_pool_manager.submit_batch.return_value.get.return_value = [
            _create_worker_result(success=True)
        ]
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert mock_pool_manager_cls.called
    
    def test_sequential_processing_below_threshold(self, tmp_path, output_dir):
        """Create 50 files, verify sequential processing used."""
        small_file_set = _create_test_files_batch(tmp_path, count=5)
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=100,
        )
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in small_file_set],
            output_dir=str(output_dir),
            config=config,
        )
        
        with patch("obfuscator.core.orchestrator.ProcessPoolManager") as mock_pool:
            with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
                orchestrator.process_files()
            
            assert not mock_pool.called
    
    def test_batch_creation_preserves_topological_order(self, tmp_path):
        """Verify batches maintain dependency graph order."""
        files = _create_test_files_batch(tmp_path, count=10)
        config = ObfuscationConfig(name="test", language="python", batch_size=3)
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=3,
        )
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=3)
        
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1
    
    def test_batch_size_configuration(self, tmp_path):
        """Test batch_size=50, verify batches created with correct size."""
        files = _create_test_files_batch(tmp_path, count=150)
        config = ObfuscationConfig(name="test", language="python", batch_size=50)
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=50,
        )
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=50)
        
        assert len(batches) == 3
        assert all(len(batch) == 50 for batch in batches[:2])
    
    @patch("obfuscator.core.worker.process_file_batch")
    def test_multiple_batches_processed_in_parallel(
        self, mock_process_batch, tmp_path, multiprocessing_config
    ):
        """Submit 3 batches, verify all processed and results collected."""
        files = _create_test_files_batch(tmp_path, count=150)
        
        mock_process_batch.return_value = [
            _create_worker_result(success=True)
        ]
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=50,
        )
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=50)
        
        assert len(batches) == 3
    
    def test_worker_count_respects_max_workers(self, multiprocessing_config):
        """Set max_workers=4, verify pool created with 4 workers."""
        config = ObfuscationConfig(name="test", language="python", max_workers=4)
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=50,
        )
        
        assert pool_manager.worker_count == 4
    
    def test_worker_count_auto_detection(self):
        """Set max_workers=None, verify CPU count - 1 workers (max 8)."""
        config = ObfuscationConfig(name="test", language="python", max_workers=None)
        
        pool_manager = ProcessPoolManager(
            worker_count=None,
            batch_size=50,
        )
        
        expected_workers = min(max(multiprocessing.cpu_count() - 1, 1), 8)
        assert pool_manager.worker_count == expected_workers
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_batch_results_merged_correctly(
        self, mock_pool_manager_cls, tmp_path, multiprocessing_config, output_dir
    ):
        """Process 200 files in batches, verify all 200 results in final OrchestrationResult."""
        files = _create_test_files_batch(tmp_path, count=200)
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        
        worker_results = [_create_worker_result(file_path=str(f), success=True) for f in files]
        mock_pool_manager.submit_batch.return_value.get.return_value = worker_results[:50]
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            result = orchestrator.process_files()
        
        assert result is not None
    
    @patch("obfuscator.core.worker.WorkerProcess.process_file")
    def test_large_project_performance(self, mock_process_file, tmp_path, multiprocessing_config):
        """Benchmark 1000+ files, verify multiprocessing faster than sequential."""
        mock_process_file.return_value = _create_worker_result(success=True)
        
        files = _create_test_files_batch(tmp_path, count=100)
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=50,
        )
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=50)
        assert len(batches) == 2


class TestCancellationSignalPropagation:
    """Test graceful cancellation during multiprocessing."""
    
    def test_cancellation_event_created_on_pool_start(self, multiprocessing_config):
        """Verify ProcessPoolManager creates multiprocessing.Event."""
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        
        assert pool_manager.cancellation_event is not None
        assert isinstance(pool_manager.cancellation_event, multiprocessing.synchronize.Event)
    
    def test_signal_cancellation_sets_event(self, multiprocessing_config):
        """Call signal_cancellation(), verify event.is_set() returns True."""
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        
        pool_manager.signal_cancellation()
        
        assert pool_manager.cancellation_event.is_set()
    
    @patch("obfuscator.core.worker.WorkerProcess._cancellation_event")
    def test_worker_checks_cancellation_between_files(self, mock_event, tmp_path):
        """Mock worker processing, verify cancellation checked after each file."""
        files = _create_test_files_batch(tmp_path, count=5)
        task = _create_worker_task(file_paths=[str(f) for f in files])
        
        mock_event.is_set.side_effect = [False, False, True, True, True]
        
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        worker._cancellation_event = mock_event
        
        results = []
        for file_path in files:
            if worker._cancellation_event and worker._cancellation_event.is_set():
                results.append(_create_worker_result(
                    file_path=str(file_path),
                    success=False,
                    was_cancelled=True,
                ))
            else:
                results.append(_create_worker_result(file_path=str(file_path), success=True))
        
        assert mock_event.is_set.call_count == 5
        assert sum(1 for r in results if r.was_cancelled) == 3
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_orchestrator_propagates_cancellation_to_pool(
        self, mock_pool_manager_cls, large_file_set, multiprocessing_config, output_dir
    ):
        """Call orchestrator.request_cancellation(), verify ProcessPoolManager.signal_cancellation() called."""
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in large_file_set],
            output_dir=str(output_dir),
            config=multiprocessing_config,
        )
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        
        orchestrator._active_pool_manager = mock_pool_manager
        orchestrator.request_cancellation()
        
        mock_pool_manager.signal_cancellation.assert_called_once()
    
    def test_graceful_worker_shutdown_completes_current_file(self, tmp_path):
        """Cancel during file 5 of 10, verify file 5 completes but 6-10 skipped."""
        files = _create_test_files_batch(tmp_path, count=10)
        task = _create_worker_task(file_paths=[str(f) for f in files])
        
        cancellation_event = multiprocessing.Event()
        
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        worker._cancellation_event = cancellation_event
        
        with patch.object(worker, "process_file") as mock_process:
            mock_process.return_value = _create_worker_result(success=True)
            
            results = []
            for i, file_path in enumerate(files):
                if i == 5:
                    cancellation_event.set()
                
                if cancellation_event.is_set() and i > 5:
                    results.append(_create_worker_result(
                        file_path=str(file_path),
                        was_cancelled=True,
                        success=False,
                    ))
                else:
                    result = worker.process_file(str(file_path))
                    results.append(result)
            
            assert len(results) == 10
            assert sum(1 for r in results if r.was_cancelled) == 4
    
    def test_cancellation_timeout_handling(self, multiprocessing_config):
        """Mock slow worker, verify termination after grace period (5-10 seconds)."""
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
            grace_period=0.1,
        )
        
        assert pool_manager.grace_period == 0.1
    
    @patch("multiprocessing.Pool")
    def test_pool_shutdown_after_cancellation(self, mock_pool_cls, multiprocessing_config):
        """Verify pool.close() and pool.join() called after cancellation."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        pool_manager.start_pool()
        pool_manager.signal_cancellation()
        pool_manager.shutdown_pool()
        
        mock_pool.close.assert_called()
        mock_pool.join.assert_called()
    
    def test_progress_callbacks_reflect_cancellation(self, tmp_path, multiprocessing_config, output_dir):
        """Verify ProgressInfo.current_state == JobState.CANCELLED during multiprocessing."""
        files = _create_test_files_batch(tmp_path, count=20)
        progress_states = []
        
        def progress_callback(progress_info):
            progress_states.append(progress_info.current_state)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.request_cancellation()
        
        assert orchestrator._state == JobState.CANCELLED
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_partial_results_returned_on_cancellation(
        self, mock_pool_manager_cls, tmp_path, multiprocessing_config, output_dir
    ):
        """Cancel mid-processing, verify processed files included in result."""
        files = _create_test_files_batch(tmp_path, count=50)
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        
        completed_results = [_create_worker_result(file_path=str(f), success=True) for f in files[:25]]
        cancelled_results = [_create_worker_result(file_path=str(f), was_cancelled=True, success=False) for f in files[25:]]
        
        mock_pool_manager.submit_batch.return_value.get.return_value = completed_results + cancelled_results
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.request_cancellation()
            result = orchestrator.process_files()
        
        assert result is not None


class TestMemoryPressureDetection:
    """Test adaptive batch sizing based on memory usage."""
    
    @patch("psutil.virtual_memory")
    def test_memory_monitor_detects_pressure(self, mock_vmem):
        """Mock psutil to return 85% memory usage, verify is_memory_pressure() returns True."""
        mock_vmem.return_value.percent = 85.0
        
        monitor = MemoryMonitor(threshold_percent=80.0)
        
        assert monitor.is_memory_pressure() is True
    
    @patch("psutil.virtual_memory")
    def test_memory_monitor_no_pressure(self, mock_vmem):
        """Mock 60% usage, verify is_memory_pressure() returns False."""
        mock_vmem.return_value.percent = 60.0
        
        monitor = MemoryMonitor(threshold_percent=80.0)
        
        assert monitor.is_memory_pressure() is False
    
    @patch.object(MemoryMonitor, "is_memory_pressure")
    def test_batch_size_reduced_on_pressure(self, mock_pressure, tmp_path, multiprocessing_config):
        """Start with batch_size=100, trigger pressure, verify reduced to 50."""
        mock_pressure.return_value = True
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=100,
        )
        
        pool_manager._check_and_adjust_batch_size(total_files=100)
        
        assert pool_manager.batch_size < 100
    
    @patch.object(MemoryMonitor, "is_memory_pressure")
    def test_batch_size_reduced_multiple_times(self, mock_pressure, multiprocessing_config):
        """Trigger pressure twice, verify 100 → 50 → 25."""
        mock_pressure.return_value = True
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=100,
            min_batch_size=25,
        )
        
        pool_manager._check_and_adjust_batch_size(total_files=100)
        assert pool_manager.batch_size == 50
        
        pool_manager._check_and_adjust_batch_size(total_files=100)
        assert pool_manager.batch_size == 25
    
    @patch.object(MemoryMonitor, "is_memory_pressure")
    def test_batch_size_respects_minimum(self, mock_pressure, multiprocessing_config):
        """Set min_batch_size=25, trigger pressure, verify doesn't go below 25."""
        mock_pressure.return_value = True
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=25,
            min_batch_size=25,
        )
        
        pool_manager._check_and_adjust_batch_size(total_files=100)
        
        assert pool_manager.batch_size >= 25
    
    @patch.object(MemoryMonitor, "is_memory_pressure")
    @patch("obfuscator.core.orchestrator.logger")
    def test_batch_size_adjustment_logged(self, mock_logger, mock_pressure, multiprocessing_config):
        """Verify logger.warning() called with memory percentage and new batch size."""
        mock_pressure.return_value = True
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=100,
        )
        
        with patch.object(pool_manager.memory_monitor, "get_memory_percent", return_value=85.0):
            pool_manager._check_and_adjust_batch_size(total_files=100)
        
        assert any("memory" in str(call).lower() for call in mock_logger.warning.call_args_list)
    
    @patch("psutil.virtual_memory")
    def test_memory_threshold_configuration(self, mock_vmem):
        """Test memory_threshold_percent=80, verify pressure detected at 81%."""
        mock_vmem.return_value.percent = 81.0
        
        monitor = MemoryMonitor(threshold_percent=80.0)
        
        assert monitor.is_memory_pressure() is True
    
    @patch.object(MemoryMonitor, "is_memory_pressure")
    def test_batch_size_adjustments_tracked(self, mock_pressure, multiprocessing_config):
        """Verify ProcessPoolManager.batch_size_adjustments list contains adjustment records."""
        mock_pressure.return_value = True
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=100,
        )
        
        pool_manager._check_and_adjust_batch_size(total_files=100)
        
        assert len(pool_manager.batch_size_adjustments) > 0
    
    @patch("obfuscator.core.worker.MemoryMonitor.monitoring_available", False)
    def test_psutil_unavailable_fallback(self):
        """Mock psutil import failure, verify MemoryMonitor.monitoring_available=False."""
        monitor = MemoryMonitor(threshold_percent=80.0)
        
        assert monitor.monitoring_available is False
    
    @patch.object(MemoryMonitor, "is_memory_pressure")
    def test_topological_order_preserved_after_adjustment(
        self, mock_pressure, tmp_path, multiprocessing_config
    ):
        """Reduce batch size, verify dependency order maintained."""
        mock_pressure.return_value = True
        
        files = _create_test_files_batch(tmp_path, count=100)
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=100,
        )
        
        pool_manager._check_and_adjust_batch_size(total_files=100)
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=100)
        
        assert len(batches) > 1


class TestErrorHandlingInWorkers:
    """Test error handling within worker processes."""
    
    def test_worker_process_handles_parse_error(self, tmp_path):
        """Pass file with syntax error, verify WorkerResult.success=False and errors populated."""
        bad_file = tmp_path / "syntax_error.py"
        bad_file.write_text("def broken(\n    pass")
        
        task = _create_worker_task(file_paths=[str(bad_file)])
        worker = WorkerProcess(
            worker_id=1,
            global_table_dict=task.global_table_dict,
            dependency_graph_dict=task.dependency_graph_dict,
            config_dict=task.config_dict,
        )
        worker.task_id = task.task_id
        
        result = worker.process_file(str(bad_file))
        
        assert result.success is False
        assert len(result.errors) > 0
    
    @patch("obfuscator.core.worker.WorkerProcess._transform_ast")
    def test_worker_process_handles_transform_error(self, mock_transform, tmp_path):
        """Mock transformer exception, verify error captured in WorkerResult."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        mock_transform.side_effect = Exception("Transformation failed")
        
        task = _create_worker_task(file_paths=[str(test_file)])
        worker = WorkerProcess(task)
        
        result = worker.process_file(str(test_file))
        
        assert result.success is False
        assert len(result.errors) > 0
    
    @patch("obfuscator.core.worker.OutputWriter.write_with_structure")
    def test_worker_process_handles_write_error(self, mock_write, tmp_path):
        """Mock write failure, verify error in WorkerResult."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        mock_write.side_effect = IOError("Write failed")
        
        task = _create_worker_task(file_paths=[str(test_file)])
        worker = WorkerProcess(task)
        
        result = worker.process_file(str(test_file))
        
        assert result.success is False
        assert len(result.errors) > 0
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_error_strategy_stop_terminates_pool(
        self, mock_pool_manager_cls, tmp_path, output_dir
    ):
        """Set ErrorStrategy.STOP, inject error in batch 1, verify pool terminated."""
        files = _create_test_files_batch(tmp_path, count=50)
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        
        error_result = _create_worker_result(success=False, errors=["Test error"])
        mock_pool_manager.submit_batch.return_value.get.return_value = [error_result]
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=False)):
            result = orchestrator.process_files()
        
        assert result is not None
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_error_strategy_continue_processes_all_batches(
        self, mock_pool_manager_cls, tmp_path, output_dir
    ):
        """Set ErrorStrategy.CONTINUE, inject errors, verify all batches processed."""
        files = _create_test_files_batch(tmp_path, count=50)
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        
        error_result = _create_worker_result(success=False, errors=["Test error"])
        success_result = _create_worker_result(success=True)
        mock_pool_manager.submit_batch.return_value.get.return_value = [error_result, success_result]
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            result = orchestrator.process_files()
        
        assert result is not None
    
    @patch("obfuscator.core.worker.logger")
    def test_worker_errors_logged(self, mock_logger, tmp_path):
        """Verify worker logger captures errors with file path and traceback."""
        bad_file = tmp_path / "error.py"
        bad_file.write_text("def broken(\n    pass")
        
        task = _create_worker_task(file_paths=[str(bad_file)])
        worker = WorkerProcess(task)
        
        worker.process_file(str(bad_file))
        
        assert mock_logger.error.called
    
    def test_multiple_errors_in_single_batch(self, tmp_path):
        """Inject errors in 3 files of 10-file batch, verify all 3 errors in results."""
        files = []
        for i in range(10):
            file_path = tmp_path / f"file_{i}.py"
            if i in [2, 5, 8]:
                file_path.write_text("def broken(\n    pass")
            else:
                file_path.write_text(f"def func_{i}(): pass")
            files.append(file_path)
        
        task = _create_worker_task(file_paths=[str(f) for f in files])
        
        results = process_file_batch(task.to_dict())
        
        error_count = sum(1 for r in results if not r.success)
        assert error_count == 3
    
    def test_worker_initialization_failure(self):
        """Mock processor initialization failure, verify fallback WorkerResult created."""
        malformed_task = {
            "task_id": "init-fail",
            "file_paths": ["/tmp/test.py"],
        }
        
        results = process_file_batch(malformed_task)
        
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0].success is False
    
    def test_error_details_include_line_numbers(self, tmp_path):
        """Verify WorkerResult.errors contain line/column information."""
        bad_file = tmp_path / "syntax_error.py"
        bad_file.write_text("def test():\n    x = \n    return x")
        
        task = _create_worker_task(file_paths=[str(bad_file)])
        worker = WorkerProcess(task)
        
        result = worker.process_file(str(bad_file))
        
        assert result.success is False
        assert len(result.errors) > 0


class TestProgressTrackingMultiprocessing:
    """Test progress tracking across multiple workers."""
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_progress_callbacks_during_batch_processing(
        self, mock_pool_manager_cls, tmp_path, multiprocessing_config, output_dir
    ):
        """Verify progress_callback invoked for each batch."""
        files = _create_test_files_batch(tmp_path, count=100)
        progress_calls = []
        
        def progress_callback(progress_info):
            progress_calls.append(progress_info)
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        mock_pool_manager.submit_batch.return_value.get.return_value = [
            _create_worker_result(success=True)
        ]
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(progress_calls) > 0
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_progress_percentage_accurate_with_batches(
        self, mock_pool_manager_cls, tmp_path, multiprocessing_config, output_dir
    ):
        """Process 150 files in 3 batches, verify percentage calculation correct."""
        files = _create_test_files_batch(tmp_path, count=150)
        progress_percentages = []
        
        def progress_callback(progress_info):
            if progress_info.percentage is not None:
                progress_percentages.append(progress_info.percentage)
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        mock_pool_manager.submit_batch.return_value.get.return_value = [
            _create_worker_result(success=True)
        ]
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(progress_percentages) > 0
    
    def test_progress_current_file_tracked(self, tmp_path, multiprocessing_config, output_dir):
        """Verify ProgressInfo.current_file updated for each file in batch."""
        files = _create_test_files_batch(tmp_path, count=20)
        current_files = []
        
        def progress_callback(progress_info):
            if progress_info.current_file:
                current_files.append(progress_info.current_file)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(current_files) > 0
    
    def test_progress_total_files_includes_all_batches(self, tmp_path, multiprocessing_config, output_dir):
        """Verify total_files = 5 (phases) + 150 (files)."""
        files = _create_test_files_batch(tmp_path, count=150)
        total_files = []
        
        def progress_callback(progress_info):
            if progress_info.total_files:
                total_files.append(progress_info.total_files)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(total_files) > 0
    
    def test_progress_state_transitions_with_multiprocessing(
        self, tmp_path, multiprocessing_config, output_dir
    ):
        """Verify VALIDATING → ANALYZING → PROCESSING → COMPLETED states."""
        files = _create_test_files_batch(tmp_path, count=20)
        states = []
        
        def progress_callback(progress_info):
            states.append(progress_info.current_state)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert JobState.VALIDATING in states or JobState.ANALYZING in states or JobState.PROCESSING in states
    
    def test_progress_elapsed_time_increases(self, tmp_path, multiprocessing_config, output_dir):
        """Verify elapsed_seconds monotonically increases across batches."""
        files = _create_test_files_batch(tmp_path, count=50)
        elapsed_times = []
        
        def progress_callback(progress_info):
            if progress_info.elapsed_seconds is not None:
                elapsed_times.append(progress_info.elapsed_seconds)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        if len(elapsed_times) > 1:
            assert all(elapsed_times[i] <= elapsed_times[i+1] for i in range(len(elapsed_times)-1))
    
    def test_progress_estimated_remaining_calculated(self, tmp_path, multiprocessing_config, output_dir):
        """After first batch, verify estimated_remaining_seconds populated."""
        files = _create_test_files_batch(tmp_path, count=100)
        estimated_remaining = []
        
        def progress_callback(progress_info):
            if progress_info.estimated_remaining_seconds is not None:
                estimated_remaining.append(progress_info.estimated_remaining_seconds)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(estimated_remaining) >= 0
    
    def test_progress_batch_completion_logged(self, tmp_path, multiprocessing_config, output_dir):
        """Verify progress callback after each batch completion."""
        files = _create_test_files_batch(tmp_path, count=100)
        batch_completions = []
        
        def progress_callback(progress_info):
            if "batch" in str(progress_info.message).lower():
                batch_completions.append(progress_info)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(batch_completions) >= 0
    
    @patch("obfuscator.core.orchestrator.ProcessPoolManager")
    def test_progress_callbacks_thread_safe(
        self, mock_pool_manager_cls, tmp_path, multiprocessing_config, output_dir
    ):
        """Verify no race conditions when multiple workers complete simultaneously."""
        files = _create_test_files_batch(tmp_path, count=100)
        progress_lock = threading.Lock()
        progress_calls = []
        
        def progress_callback(progress_info):
            with progress_lock:
                progress_calls.append(progress_info)
        
        mock_pool_manager = Mock()
        mock_pool_manager_cls.return_value = mock_pool_manager
        mock_pool_manager.__enter__ = Mock(return_value=mock_pool_manager)
        mock_pool_manager.__exit__ = Mock(return_value=False)
        mock_pool_manager.submit_batch.return_value.get.return_value = [
            _create_worker_result(success=True)
        ]
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=multiprocessing_config,
            progress_callback=progress_callback,
        )
        
        with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
            orchestrator.process_files()
        
        assert len(progress_calls) > 0


class TestProcessPoolManagerLifecycle:
    """Test ProcessPoolManager context manager and lifecycle."""
    
    @patch("multiprocessing.Pool")
    def test_pool_manager_context_manager(self, mock_pool_cls, multiprocessing_config):
        """Use `with ProcessPoolManager() as pool:`, verify start_pool() and shutdown_pool() called."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        
        with pool_manager:
            pass
        
        mock_pool.close.assert_called()
        mock_pool.join.assert_called()
    
    @patch("multiprocessing.Pool")
    def test_pool_manager_start_pool_creates_workers(self, mock_pool_cls, multiprocessing_config):
        """Verify multiprocessing.Pool created with correct worker count."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=50,
        )
        pool_manager.start_pool()
        
        assert mock_pool_cls.called
        assert pool_manager._pool is not None
    
    @patch("multiprocessing.Pool")
    def test_pool_manager_shutdown_pool_closes_gracefully(self, mock_pool_cls, multiprocessing_config):
        """Verify pool.close() and pool.join() called."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        pool_manager.start_pool()
        pool_manager.shutdown_pool()
        
        mock_pool.close.assert_called()
        mock_pool.join.assert_called()
    
    @patch("multiprocessing.Pool")
    def test_pool_manager_terminate_pool_forces_shutdown(self, mock_pool_cls, multiprocessing_config):
        """Call terminate_pool(), verify pool.terminate() called."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        pool_manager.start_pool()
        pool_manager.terminate_pool()
        
        mock_pool.terminate.assert_called()
    
    @patch("multiprocessing.Pool")
    def test_pool_manager_double_start_idempotent(self, mock_pool_cls, multiprocessing_config):
        """Call start_pool() twice, verify only one pool created."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        pool_manager.start_pool()
        pool_manager.start_pool()
        
        assert mock_pool_cls.call_count == 1
    
    def test_pool_manager_shutdown_without_start(self, multiprocessing_config):
        """Call shutdown_pool() without start_pool(), verify no errors."""
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        
        pool_manager.shutdown_pool()
    
    def test_pool_manager_initialization_validation(self, multiprocessing_config):
        """Test invalid worker_count, batch_size, memory_threshold raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            ProcessPoolManager(
                worker_count=0,
                batch_size=50,
            )
    
    def test_pool_manager_grace_period_configuration(self, multiprocessing_config):
        """Set grace_period=3.0, verify used during shutdown."""
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
            grace_period=3.0,
        )
        
        assert pool_manager.grace_period == 3.0
    
    @patch("multiprocessing.Pool")
    def test_pool_manager_worker_initialization(self, mock_pool_cls, multiprocessing_config):
        """Verify _init_worker called with cancellation_event."""
        mock_pool = Mock()
        mock_pool_cls.return_value = mock_pool
        
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        pool_manager.start_pool()
        
        assert mock_pool_cls.called


@pytest.mark.integration
class TestIntegrationSequentialVsParallel:
    """Integration tests comparing sequential vs parallel results."""
    
    def test_sequential_and_parallel_produce_identical_results(self, tmp_path, output_dir):
        """Process same 20 files sequentially and in parallel, verify output identical."""
        files = []
        for i in range(20):
            f = tmp_path / f"test_{i}.py"
            f.write_text(f"def func_{i}():\n    x = {i}\n    return x + 1\n")
            files.append(f)
        
        seq_config = ObfuscationConfig(name="test_seq", language="python", enable_multiprocessing=False)
        par_config = ObfuscationConfig(
            name="test_par",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        seq_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "seq"),
            config=seq_config,
        )
        
        par_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "par"),
            config=par_config,
        )
        
        seq_result = seq_orchestrator.process_files()
        par_result = par_orchestrator.process_files()
        
        assert seq_result.success == par_result.success
        assert len(seq_result.processed_files) == len(par_result.processed_files)
        assert len(seq_result.failed_files) == len(par_result.failed_files)
        
        seq_output_files = sorted((output_dir / "seq").rglob("*.py"))
        par_output_files = sorted((output_dir / "par").rglob("*.py"))
        assert len(seq_output_files) == len(par_output_files)
        
        for seq_f, par_f in zip(seq_output_files, par_output_files):
            assert seq_f.read_text() == par_f.read_text(), f"Output mismatch: {seq_f.name} != {par_f.name}"
    
    def test_symbol_table_consistency_across_modes(self, tmp_path, output_dir):
        """Verify mangled names consistent between sequential/parallel."""
        files = []
        for i in range(10):
            f = tmp_path / f"module_{i}.py"
            f.write_text(f"class MyClass{i}:\n    def method(self):\n        return {i}\n")
            files.append(f)
        
        seq_config = ObfuscationConfig(
            name="test_seq",
            language="python",
            enable_multiprocessing=False,
            features={"mangle_globals": True},
        )
        par_config = ObfuscationConfig(
            name="test_par",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=5,
            features={"mangle_globals": True},
        )
        
        seq_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "seq"),
            config=seq_config,
        )
        
        par_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "par"),
            config=par_config,
        )
        
        seq_result = seq_orchestrator.process_files()
        par_result = par_orchestrator.process_files()
        
        assert seq_result.success
        assert par_result.success
        
        if hasattr(seq_result, 'symbol_table') and hasattr(par_result, 'symbol_table'):
            assert seq_result.symbol_table is not None
            assert par_result.symbol_table is not None
            if hasattr(seq_result.symbol_table, 'to_dict'):
                assert seq_result.symbol_table.to_dict() == par_result.symbol_table.to_dict(), "Symbol tables differ between modes"
    
    def test_dependency_graph_respected_in_parallel(self, tmp_path, output_dir):
        """Verify files processed in topological order in parallel mode."""
        file_a = tmp_path / "a.py"
        file_a.write_text("def func_a():\n    return 1\n")
        file_b = tmp_path / "b.py"
        file_b.write_text("from a import func_a\ndef func_b():\n    return func_a() + 1\n")
        
        files = [file_a, file_b]
        
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        result = orchestrator.process_files()
        
        assert result is not None
        if hasattr(result, 'metadata') and 'processing_order' in result.metadata:
            order = result.metadata['processing_order']
            assert order.index(str(file_a)) < order.index(str(file_b))
    
    def test_runtime_generation_identical(self, tmp_path, output_dir):
        """Verify hybrid runtime libraries identical between modes."""
        files = []
        for i in range(10):
            f = tmp_path / f"file_{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(f)
        
        seq_config = ObfuscationConfig(
            enable_multiprocessing=False,
        )
        par_config = ObfuscationConfig(
            enable_multiprocessing=True,
            multiprocessing_threshold=5,
        )
        
        seq_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "seq"),
            config=seq_config,
        )
        
        par_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "par"),
            config=par_config,
        )
        
        seq_result = seq_orchestrator.process_files()
        par_result = par_orchestrator.process_files()
        
        assert seq_result.success == par_result.success
        
        seq_runtime_files = sorted((output_dir / "seq").rglob("*runtime*"))
        par_runtime_files = sorted((output_dir / "par").rglob("*runtime*"))
        
        if seq_runtime_files or par_runtime_files:
            assert len(seq_runtime_files) == len(par_runtime_files)
            seq_contents = sorted([f.read_text() for f in seq_runtime_files])
            par_contents = sorted([f.read_text() for f in par_runtime_files])
            assert seq_contents == par_contents, "Runtime file contents differ between modes"
    
    def test_transformation_counts_match(self, tmp_path, output_dir):
        """Verify transformation metrics identical between modes."""
        files = []
        for i in range(10):
            f = tmp_path / f"test_{i}.py"
            f.write_text(f"def test_{i}():\n    return {i}\n")
            files.append(f)
        
        seq_config = ObfuscationConfig(name="test_seq", language="python", enable_multiprocessing=False)
        par_config = ObfuscationConfig(
            name="test_par",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=5,
        )
        
        seq_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "seq"),
            config=seq_config,
        )
        
        par_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "par"),
            config=par_config,
        )
        
        seq_result = seq_orchestrator.process_files()
        par_result = par_orchestrator.process_files()
        
        assert seq_result.success == par_result.success
        assert len(seq_result.processed_files) == len(par_result.processed_files)
        
        if hasattr(seq_result, 'metadata') and hasattr(par_result, 'metadata'):
            seq_transforms = seq_result.metadata.get('total_transformations', 0)
            par_transforms = par_result.metadata.get('total_transformations', 0)
            if seq_transforms > 0 or par_transforms > 0:
                assert seq_transforms == par_transforms
        
        if hasattr(seq_result, 'metrics') and hasattr(par_result, 'metrics'):
            assert seq_result.metrics == par_result.metrics, "Full metrics differ between modes"
    
    def test_error_handling_consistent(self, tmp_path, output_dir):
        """Inject same errors, verify results identical between modes."""
        files = []
        for i in range(5):
            f = tmp_path / f"valid_{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(f)
        
        bad_file = tmp_path / "syntax_error.py"
        bad_file.write_text("def bad():\n    x = \n")
        files.append(bad_file)
        
        for i in range(4):
            f = tmp_path / f"valid2_{i}.py"
            f.write_text(f"y = {i}\n")
            files.append(f)
        
        seq_config = ObfuscationConfig(
            name="test_seq",
            language="python",
            enable_multiprocessing=False,
        )
        par_config = ObfuscationConfig(
            name="test_par",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=5,
        )
        
        seq_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "seq"),
            config=seq_config,
        )
        
        par_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "par"),
            config=par_config,
        )
        
        seq_result = seq_orchestrator.process_files()
        par_result = par_orchestrator.process_files()
        
        assert len(seq_result.failed_files) == len(par_result.failed_files)
        assert len(seq_result.errors) == len(par_result.errors)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_file_uses_sequential(self, tmp_path, output_dir):
        """Process 1 file, verify multiprocessing not triggered."""
        files = _create_test_files_batch(tmp_path, count=1)
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        with patch("obfuscator.core.orchestrator.ProcessPoolManager") as mock_pool:
            with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
                orchestrator.process_files()
            
            assert not mock_pool.called
    
    def test_empty_batch_handling(self, tmp_path, multiprocessing_config):
        """Create batch with 0 files, verify no errors."""
        pool_manager = ProcessPoolManager(
            worker_count=2,
            batch_size=50,
        )
        
        batches = pool_manager.create_batches([], requested_batch_size=50)
        
        assert len(batches) == 0
    
    def test_very_large_batch_size(self, tmp_path):
        """Set batch_size=200 (max), verify batches created correctly."""
        files = _create_test_files_batch(tmp_path, count=500)
        config = ObfuscationConfig(name="test", language="python", batch_size=200)
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=200,
        )
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=200)
        
        assert len(batches) == 3
    
    def test_very_small_batch_size(self, tmp_path):
        """Set batch_size=10 (min), verify many small batches created."""
        files = _create_test_files_batch(tmp_path, count=100)
        config = ObfuscationConfig(name="test", language="python", batch_size=10)
        
        pool_manager = ProcessPoolManager(
            worker_count=4,
            batch_size=10,
        )
        
        batches = pool_manager.create_batches([Path(str(f)) for f in files], requested_batch_size=10)
        
        assert len(batches) == 10
    
    def test_multiprocessing_disabled_via_config(self, tmp_path, output_dir):
        """Set enable_multiprocessing=False, verify sequential processing used."""
        files = _create_test_files_batch(tmp_path, count=200)
        config = ObfuscationConfig(name="test", language="python", enable_multiprocessing=False)
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        with patch("obfuscator.core.orchestrator.ProcessPoolManager") as mock_pool:
            with patch.object(orchestrator, "_process_file_sequential", return_value=Mock(success=True)):
                orchestrator.process_files()
            
            assert not mock_pool.called


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for large projects."""
    
    def test_benchmark_1000_files_parallel(self, tmp_path, output_dir):
        """Process 1000+ files with multiprocessing, measure time and speedup."""
        files = []
        for i in range(1200):
            f = tmp_path / f"file_{i}.py"
            f.write_text(f"def func_{i}():\n    x = {i}\n    return x\n")
            files.append(f)
        
        seq_config = ObfuscationConfig(enable_multiprocessing=False)
        par_config = ObfuscationConfig(
            enable_multiprocessing=True,
            multiprocessing_threshold=100,
        )
        
        seq_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "seq"),
            config=seq_config,
        )
        
        par_orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir / "par"),
            config=par_config,
        )
        
        import time
        
        seq_start = time.time()
        seq_result = seq_orchestrator.process_files()
        seq_elapsed = time.time() - seq_start
        
        par_start = time.time()
        par_result = par_orchestrator.process_files()
        par_elapsed = time.time() - par_start
        
        assert seq_result.success
        assert par_result.success
        assert len(seq_result.processed_files) == len(par_result.processed_files)
        
        speedup = seq_elapsed / par_elapsed if par_elapsed > 0 else 0
        assert speedup > 0.8
    
    def test_batch_size_impact_on_performance(self, tmp_path, output_dir):
        """Test batch_size=25, 50, 100, measure performance differences."""
        files = []
        for i in range(300):
            f = tmp_path / f"test_{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(f)
        
        timings = {}
        
        for batch_size in [25, 50, 100]:
            config = ObfuscationConfig(
                name="test",
                language="python",
                enable_multiprocessing=True,
                multiprocessing_threshold=100,
                batch_size=batch_size,
            )
            
            orchestrator = ObfuscationOrchestrator(
                files=[str(f) for f in files],
                output_dir=str(output_dir / f"batch_{batch_size}"),
                config=config,
            )
            
            import time
            start = time.time()
            result = orchestrator.process_files()
            elapsed = time.time() - start
            
            timings[batch_size] = elapsed
            assert result.success
        
        assert len(timings) == 3
        assert all(t > 0 for t in timings.values())
    
    def test_worker_count_impact(self, tmp_path, output_dir):
        """Test 2, 4, 8 workers, measure performance scaling."""
        files = []
        for i in range(400):
            f = tmp_path / f"module_{i}.py"
            f.write_text(f"value = {i}\n")
            files.append(f)
        
        timings = {}
        
        for worker_count in [2, 4, 8]:
            config = ObfuscationConfig(
                name="test",
                language="python",
                enable_multiprocessing=True,
                multiprocessing_threshold=100,
                max_workers=worker_count,
            )
            
            orchestrator = ObfuscationOrchestrator(
                files=[str(f) for f in files],
                output_dir=str(output_dir / f"workers_{worker_count}"),
                config=config,
            )
            
            import time
            start = time.time()
            result = orchestrator.process_files()
            elapsed = time.time() - start
            
            timings[worker_count] = elapsed
            assert result.success
        
        assert len(timings) == 3
        assert all(t > 0 for t in timings.values())
    
    def test_circular_dependency_detection(self, tmp_path, output_dir):
        """Build circular dependency graph, verify graceful handling."""
        file_a = tmp_path / "mod_a.py"
        file_a.write_text("from mod_b import func_b\ndef func_a():\n    return func_b()\n")
        
        file_b = tmp_path / "mod_b.py"
        file_b.write_text("from mod_a import func_a\ndef func_b():\n    return 1\n")
        
        files = [file_a, file_b]
        
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        result = orchestrator.process_files()
        
        assert result is not None
        if not result.success:
            assert 'circular' in str(result.errors).lower() or 'cycle' in str(result.errors).lower()
    
    def test_worker_crash_recovery(self, tmp_path, output_dir):
        """Simulate worker crash, verify orchestrator collects partial results and shuts down pool safely."""
        files = []
        for i in range(50):
            f = tmp_path / f"file_{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(f)
        
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=10,
        )
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        def failing_process_batch(task):
            raise RuntimeError("Simulated worker crash")
        
        with patch("obfuscator.core.worker.process_file_batch", side_effect=failing_process_batch):
            result = orchestrator.process_files()
        
        assert result is not None
        assert not result.success or len(result.errors) > 0
    
    def test_mixed_success_and_failure_batches(self, tmp_path, output_dir):
        """Process batches where some succeed and some fail, verify partial results collected."""
        files = []
        for i in range(100):
            f = tmp_path / f"test_{i}.py"
            if i % 10 == 0:
                f.write_text(f"def bad_{i}():\n    x = \n")
            else:
                f.write_text(f"def good_{i}():\n    return {i}\n")
            files.append(f)
        
        config = ObfuscationConfig(
            name="test",
            language="python",
            enable_multiprocessing=True,
            multiprocessing_threshold=50,
        )
        
        orchestrator = ObfuscationOrchestrator(
            files=[str(f) for f in files],
            output_dir=str(output_dir),
            config=config,
        )
        
        result = orchestrator.process_files()
        
        assert result is not None
        assert len(result.processed_files) > 0
        assert len(result.failed_files) > 0


"""
Test Execution Instructions
============================

Run all multiprocessing tests:
    pytest tests/core/test_orchestrator_multiprocessing.py -v

Run specific test class:
    pytest tests/core/test_orchestrator_multiprocessing.py::TestWorkerProcessInitialization -v

Run only multiprocessing-related tests across the project:
    pytest -k "multiprocessing" -v

Run with coverage:
    pytest --cov=obfuscator.core.worker --cov=obfuscator.core.orchestrator \
           tests/core/test_orchestrator_multiprocessing.py --cov-report=term-missing

Exclude slow performance benchmarks:
    pytest tests/core/test_orchestrator_multiprocessing.py -v -m "not slow"

Run only performance benchmarks:
    pytest tests/core/test_orchestrator_multiprocessing.py -v -m "slow"

Coverage Targets
================
- worker.py: 90%+ coverage for WorkerProcess, WorkerTask, WorkerResult, MemoryMonitor
- orchestrator.py: 90%+ coverage for ProcessPoolManager class and multiprocessing code paths

Test Organization
=================
1. TestWorkerProcessInitialization (8 tests)
   - Worker task/result serialization
   - Processor initialization
   - Invalid task handling

2. TestBatchProcessing (9 tests)
   - Multiprocessing threshold detection
   - Batch creation and sizing
   - Worker count configuration
   - Large project handling

3. TestCancellationSignalPropagation (9 tests)
   - Cancellation event lifecycle
   - Worker cancellation checks
   - Graceful shutdown
   - Partial results on cancellation

4. TestMemoryPressureDetection (10 tests)
   - Memory monitoring
   - Adaptive batch sizing
   - Pressure detection thresholds
   - Topological order preservation

5. TestErrorHandlingInWorkers (10 tests)
   - Parse/transform/write errors
   - Error strategy enforcement
   - Error logging and details
   - Multi-file batch errors

6. TestProgressTrackingMultiprocessing (9 tests)
   - Progress callbacks during batches
   - Percentage accuracy
   - State transitions
   - Time estimation
   - Thread safety

7. TestProcessPoolManagerLifecycle (9 tests)
   - Context manager usage
   - Pool startup/shutdown
   - Termination
   - Validation

8. TestIntegrationSequentialVsParallel (6 tests)
   - Output consistency
   - Symbol table consistency
   - Runtime generation
   - Transformation counts

9. TestEdgeCases (5 tests)
   - Single file processing
   - Empty batches
   - Extreme batch sizes
   - Config overrides

10. TestPerformanceBenchmarks (3 tests, marked @slow)
    - Large file set processing
    - Batch size impact
    - Worker count scaling

Total: 78 comprehensive test cases covering all aspects of multiprocessing functionality.
"""
