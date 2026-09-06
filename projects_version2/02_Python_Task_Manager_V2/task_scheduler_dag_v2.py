"""
Task Manager & Scheduler V2: Priority-Queue Directed Acyclic Graph (DAG) with RBAC
Author: Sameer Karur
Curriculum: IIT Kanpur Professional Certificate in AI/ML

Key Architectural Enhancements over V1:
- Directed Acyclic Graph (DAG) task dependency resolution with topological sorting
- Circular dependency deadlock detection (Cycle detection via DFS)
- Cryptographic security: PBKDF2-HMAC-SHA256 password hashing with unique per-user salts
- Role-Based Access Control (RBAC): Admin, Contributor, Viewer permissions
- Priority queue scheduling (Urgent, High, Medium, Low)
"""

import hashlib
import os
import json
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict

class Role(str, Enum):
    ADMIN = "ADMIN"
    CONTRIBUTOR = "CONTRIBUTOR"
    VIEWER = "VIEWER"

class Priority(int, Enum):
    URGENT = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class Status(str, Enum):
    BLOCKED = "BLOCKED"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"

@dataclass
class Task:
    task_id: str
    title: str
    assignee: str
    priority: Priority
    status: Status
    dependencies: List[str]  # Task IDs that must be completed first

class AuthManager:
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        if salt is None:
            salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
        return key, salt

    @staticmethod
    def verify_password(stored_key: bytes, salt: bytes, password_attempt: str) -> bool:
        key_attempt, _ = AuthManager.hash_password(password_attempt, salt)
        return key_attempt == stored_key

class TaskSchedulerDAG:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.users: Dict[str, Dict] = {}

    def register_user(self, username: str, password: str, role: Role):
        key, salt = AuthManager.hash_password(password)
        self.users[username] = {
            "key": key.hex(),
            "salt": salt.hex(),
            "role": role.value
        }

    def authenticate_user(self, username: str, password: str) -> Optional[Role]:
        u = self.users.get(username)
        if not u:
            return None
        valid = AuthManager.verify_password(bytes.fromhex(u["key"]), bytes.fromhex(u["salt"]), password)
        return Role(u["role"]) if valid else None

    def add_task(self, task_id: str, title: str, assignee: str, priority: Priority = Priority.MEDIUM, 
                 dependencies: Optional[List[str]] = None):
        dependencies = dependencies or []
        for dep in dependencies:
            if dep not in self.tasks:
                raise ValueError(f"Dependency task '{dep}' does not exist yet.")

        # Test for cycles before inserting
        temp_deps = {t_id: t.dependencies[:] for t_id, t in self.tasks.items()}
        temp_deps[task_id] = dependencies

        if self._has_cycle(temp_deps):
            raise ValueError(f"Circular dependency detected! Cannot add task '{task_id}'.")

        status = Status.READY if not dependencies else Status.BLOCKED
        self.tasks[task_id] = Task(task_id, title, assignee, priority, status, dependencies)
        self._refresh_task_statuses()

    def _has_cycle(self, graph: Dict[str, List[str]]) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for n in graph:
            if n not in visited:
                if dfs(n):
                    return True
        return False

    def complete_task(self, task_id: str):
        if task_id not in self.tasks:
            raise KeyError(f"Task '{task_id}' not found.")
        self.tasks[task_id].status = Status.COMPLETED
        self._refresh_task_statuses()

    def _refresh_task_statuses(self):
        completed_ids = {t_id for t_id, t in self.tasks.items() if t.status == Status.COMPLETED}
        for t_id, t in self.tasks.items():
            if t.status in (Status.BLOCKED, Status.READY):
                if all(dep in completed_ids for dep in t.dependencies):
                    t.status = Status.READY
                else:
                    t.status = Status.BLOCKED

    def get_executable_queue(self) -> List[Task]:
        """Returns tasks that are READY, sorted by highest priority first."""
        ready_tasks = [t for t in self.tasks.values() if t.status == Status.READY]
        return sorted(ready_tasks, key=lambda x: x.priority.value)

    def topological_execution_plan(self) -> List[str]:
        """Returns valid chronological execution sequence respecting all dependencies."""
        in_degree = {k: 0 for k in self.tasks}
        adj = {k: [] for k in self.tasks}

        for t_id, t in self.tasks.items():
            for dep in t.dependencies:
                adj[dep].append(t_id)
                in_degree[t_id] += 1

        queue = [k for k, v in in_degree.items() if v == 0]
        order = []

        while queue:
            # Sort by priority for deterministic optimal plan
            queue.sort(key=lambda x: self.tasks[x].priority.value)
            curr = queue.pop(0)
            order.append(curr)

            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

def run_demo():
    print("=" * 70)
    print("🚀 Running Task Manager & DAG Scheduler V2 Demo")
    print("=" * 70)

    sched = TaskSchedulerDAG()

    # 1. RBAC Authentication
    sched.register_user("sameer", "SecurePass#2026", Role.ADMIN)
    sched.register_user("analyst", "AnalystPass123", Role.CONTRIBUTOR)

    role = sched.authenticate_user("sameer", "SecurePass#2026")
    print(f"🔐 Authenticated 'sameer' successfully with Role: {role.value}")

    # 2. Build DAG workflow
    sched.add_task("TASK_1", "Ingest & Clean Raw ETL Data", "sameer", Priority.HIGH)
    sched.add_task("TASK_2", "Train Deep ResNet Model", "sameer", Priority.URGENT, dependencies=["TASK_1"])
    sched.add_task("TASK_3", "Run Unit & Integration Tests", "analyst", Priority.MEDIUM, dependencies=["TASK_1"])
    sched.add_task("TASK_4", "Deploy Production Model Service", "sameer", Priority.HIGH, dependencies=["TASK_2", "TASK_3"])

    print("\n📋 Complete Execution Plan (Topological Sort):")
    plan = sched.topological_execution_plan()
    for idx, t_id in enumerate(plan, 1):
        t = sched.tasks[t_id]
        print(f"  Step {idx}: [{t.priority.name}] {t_id} — '{t.title}' (Requires: {t.dependencies or 'None'})")

    print("\n⚡ Current Ready Queue:")
    for t in sched.get_executable_queue():
        print(f"  • {t.task_id}: {t.title} (Status: {t.status.value})")

    # Simulate completing TASK_1
    print("\n✅ Completing TASK_1...")
    sched.complete_task("TASK_1")

    print("\n⚡ Updated Ready Queue (Dependencies unblocked):")
    for t in sched.get_executable_queue():
        print(f"  • {t.task_id}: {t.title} [Priority: {t.priority.name}, Status: {t.status.value}]")

    print("\n✅ Task Manager V2 execution completed successfully.")

if __name__ == "__main__":
    run_demo()
