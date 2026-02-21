import hashlib
import json
import uuid
import datetime
from pathlib import Path
from typing import Any

from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import ensure_directory

CHECKPOINT_VERSION = "1.0"
CHECKPOINT_SUFFIX = ".checkpoint.json"

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self._checkpoint_dir = checkpoint_dir
        self._logger = get_logger("obfuscator.core.checkpoint_manager")

    def create_checkpoint(self, job_state: dict[str, Any]) -> Path:
        ensure_directory(self._checkpoint_dir)
        
        session_id = job_state["session_id"]
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        body = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "timestamp": timestamp,
            "session_id": session_id,
            "progress": job_state["progress"],
            "state": {
                "dependency_graph": job_state["dependency_graph"].to_dict(),
                "symbol_table": job_state["global_symbol_table"].to_dict(),
                "processed_file_hashes": job_state["processed_file_hashes"]
            },
            "errors": job_state["errors"]
        }
        
        serialized = json.dumps(body, sort_keys=True)
        checksum = hashlib.sha256(serialized.encode()).hexdigest()
        body["checksum"] = checksum
        
        timestamp_safe = timestamp.replace(":", "-")
        filename = f"{session_id}_{timestamp_safe}{CHECKPOINT_SUFFIX}"
        
        final_path = self._checkpoint_dir / filename
        temp_path = self._checkpoint_dir / f"{filename}.tmp"
        
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2, sort_keys=True)
            
        temp_path.replace(final_path)
        self._logger.info(f"Checkpoint saved: {final_path.name}")
        
        return final_path

    def restore_checkpoint(self, checkpoint_path: Path) -> dict:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse checkpoint {checkpoint_path}: {e}")
            
        if "checksum" not in data:
            raise ValueError(f"Checkpoint missing checksum")
            
        checksum = data.pop("checksum")
        expected_checksum = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        if checksum != expected_checksum:
            raise ValueError(f"Checkpoint integrity check failed: {checkpoint_path}")
            
        data["checksum"] = checksum
        return data

    @staticmethod
    def validate_checkpoint(data: dict[str, Any]) -> bool:
        try:
            if data.get("checkpoint_version") != CHECKPOINT_VERSION:
                return False
                
            required_keys = {"checkpoint_version", "timestamp", "session_id", "progress", "state", "errors", "checksum"}
            if not required_keys.issubset(data.keys()):
                return False
                
            required_state_keys = {"dependency_graph", "symbol_table", "processed_file_hashes"}
            if not required_state_keys.issubset(data["state"].keys()):
                return False
                
            data_without_checksum = data.copy()
            checksum = data_without_checksum.pop("checksum")
            
            expected_checksum = hashlib.sha256(json.dumps(data_without_checksum, sort_keys=True).encode()).hexdigest()
            return checksum == expected_checksum
        except Exception:
            return False

    @staticmethod
    def find_latest_checkpoint(output_dir: Path) -> Path | None:
        checkpoint_dir = output_dir / ".checkpoints"
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return None
            
        logger = get_logger("obfuscator.core.checkpoint_manager")
        latest_file = None
        latest_timestamp = ""
        
        for checkpoint_path in checkpoint_dir.glob(f"*{CHECKPOINT_SUFFIX}"):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                if not CheckpointManager.validate_checkpoint(data):
                    logger.debug(f"Skipping checkpoint with invalid schema or checksum: {checkpoint_path}")
                    continue
                    
                timestamp = data.get("timestamp", "")
                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_file = checkpoint_path
            except Exception as e:
                logger.debug(f"Skipping invalid checkpoint {checkpoint_path}: {e}")
                continue
                
        return latest_file

    def cleanup_checkpoints(self, session_id: str) -> None:
        if not self._checkpoint_dir.exists():
            return
            
        count = 0
        for file in self._checkpoint_dir.glob(f"{session_id}_*{CHECKPOINT_SUFFIX}"):
            file.unlink(missing_ok=True)
            count += 1
            
        if count > 0:
            self._logger.info(f"Cleaned up {count} checkpoint(s) for session {session_id}")
