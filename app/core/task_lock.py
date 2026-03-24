"""
任务锁管理器 - 防止重复请求
"""
import threading
from typing import Dict, Optional
from datetime import datetime


class TaskLockManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, dict] = {}
                    cls._instance._task_locks: Dict[str, threading.Lock] = {}
        return cls._instance

    def is_locked(self, task_name: str) -> bool:
        if task_name not in self._task_locks:
            return False
        task_info = self._tasks.get(task_name, {})
        return task_info.get("running", False)

    def acquire(self, task_name: str, description: str = "") -> bool:
        if task_name not in self._task_locks:
            self._task_locks[task_name] = threading.Lock()

        with self._task_locks[task_name]:
            if self._tasks.get(task_name, {}).get("running", False):
                return False

            self._tasks[task_name] = {
                "running": True,
                "start_time": datetime.now().isoformat(),
                "description": description
            }
            return True

    def release(self, task_name: str) -> bool:
        if task_name not in self._task_locks:
            return False

        with self._task_locks[task_name]:
            if task_name in self._tasks:
                self._tasks[task_name]["running"] = False
                self._tasks[task_name]["end_time"] = datetime.now().isoformat()
                return True
            return False

    def get_task_status(self, task_name: str) -> Optional[dict]:
        return self._tasks.get(task_name)

    def get_all_running_tasks(self) -> Dict[str, dict]:
        return {
            name: info for name, info in self._tasks.items()
            if info.get("running", False)
        }


task_lock_manager = TaskLockManager()
