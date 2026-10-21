"""Evaluation report writing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["EvaluationReport", "write_report"]


class EvaluationReport:
    """Collects evaluation results and renders them to disk."""

    def __init__(self, model_name: str = "tinycode") -> None:
        self.model_name = model_name
        self.created_at = time.time()
        self.sections: dict[str, Any] = {}

    def add(self, name: str, payload: Any) -> None:
        """Add a named section to the report."""

        self.sections[name] = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "created_at": self.created_at,
            "sections": self.sections,
        }

    def to_text(self) -> str:
        lines = [f"Evaluation report: {self.model_name}", "=" * 40]
        for name, payload in self.sections.items():
            lines.append(f"\n[{name}]")
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (list, dict)):
                        lines.append(f"  {key}: {json.dumps(value)[:200]}")
                    else:
                        lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {payload}")
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        """Write the report as JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return target


def write_report(report: EvaluationReport, directory: str | Path, name: str = "evaluation.json") -> Path:
    """Write an evaluation report into ``directory``."""

    return report.write(Path(directory) / name)
