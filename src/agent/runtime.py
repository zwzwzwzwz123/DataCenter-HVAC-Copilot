from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class AgentTodo:
    id: str
    step_index: int
    route: str
    description: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_index": self.step_index,
            "route": self.route,
            "description": self.description,
            "status": self.status,
        }


@dataclass
class AgentRuntimeTrace:
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    todos: list[AgentTodo] = field(default_factory=list)
    todo_events: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[dict[str, Any]] = field(default_factory=list)
    recoveries: list[dict[str, Any]] = field(default_factory=list)

    def create_todos(self, steps: list[Any]) -> None:
        self.todos = []
        for index, step in enumerate(steps, start=1):
            self._create_todo(step, index)

    def add_todo(self, step: Any) -> AgentTodo:
        return self._create_todo(step, len(self.todos) + 1)

    def mark_todo(self, step_index: int, status: str) -> None:
        todo = self.todos[step_index - 1]
        todo.status = status
        self.todo_events.append(
            {
                "event": f"todo.{status}",
                "todo_id": todo.id,
                "step_index": step_index,
                "route": todo.route,
                "status": status,
                "timestamp": _utc_now(),
            }
        )

    def record_hook(self, event: dict[str, Any]) -> None:
        self.hooks.append({"timestamp": _utc_now(), **event})

    def record_recovery(self, event: dict[str, Any]) -> None:
        self.recoveries.append({"timestamp": _utc_now(), **event})

    def _create_todo(self, step: Any, index: int) -> AgentTodo:
        route = str(getattr(step, "route", "unknown"))
        reason = str(getattr(step, "reason", route))
        todo = AgentTodo(
            id=f"todo_{index:03d}",
            step_index=index,
            route=route,
            description=reason,
        )
        self.todos.append(todo)
        self.todo_events.append(
            {
                "event": "todo.created",
                "todo_id": todo.id,
                "step_index": index,
                "route": route,
                "status": todo.status,
                "timestamp": _utc_now(),
            }
        )
        return todo

    def to_dict(self) -> dict[str, Any]:
        statuses = [todo.status for todo in self.todos]
        hooks = [
            *self.hooks,
            {
                "timestamp": _utc_now(),
                "hook": "RunComplete",
                "status": "blocked" if "blocked" in statuses else "completed",
                "todo_count": len(self.todos),
                "completed_todo_count": statuses.count("completed"),
                "blocked_todo_count": statuses.count("blocked"),
            },
        ]
        return {
            "run_id": self.run_id,
            "todos": [todo.to_dict() for todo in self.todos],
            "todo_events": list(self.todo_events),
            "hooks": hooks,
            "recoveries": list(self.recoveries),
            "summary": {
                "todo_count": len(self.todos),
                "completed_todo_count": statuses.count("completed"),
                "blocked_todo_count": statuses.count("blocked"),
                "tool_call_count": sum(1 for event in hooks if event.get("hook") == "PostToolUse"),
                "recovery_count": len(self.recoveries),
                "approval_count": sum(
                    1
                    for event in hooks
                    if event.get("hook") == "PreToolUse" and event.get("approval", {}).get("required")
                ),
            },
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
