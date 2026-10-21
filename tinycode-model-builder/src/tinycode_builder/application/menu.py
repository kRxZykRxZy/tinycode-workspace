"""The interactive menu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["MenuItem", "Menu", "BUILDER_MENU"]


@dataclass
class MenuItem:
    """A single menu entry."""

    key: str
    title: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "description": self.description}


class Menu:
    """A simple numbered menu."""

    def __init__(self, title: str, items: Sequence[MenuItem]) -> None:
        self.title = title
        self.items = list(items)

    def render(self) -> str:
        lines = [self.title, "-" * len(self.title)]
        for index, item in enumerate(self.items, start=1):
            lines.append(f"  {index:>2}. {item.title}")
            if item.description:
                lines.append(f"      {item.description}")
        lines.append(f"   q. Quit")
        return "\n".join(lines)

    def item_for(self, choice: str) -> MenuItem | None:
        """Resolve a user choice (number or key) to a menu item."""

        if not choice:
            return None
        choice = choice.strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(self.items):
                return self.items[index]
            return None
        for item in self.items:
            if item.key == choice:
                return item
        return None


BUILDER_MENU = Menu(
    "TinyCode Model Builder",
    (
        MenuItem("hardware", "Inspect hardware capabilities", "GPU, RAM, disk and Colab detection."),
        MenuItem("configure", "Configure the model", "Parameter target, vocabulary and architecture."),
        MenuItem("data", "Prepare the dataset", "The Stack download, filtering, packing."),
        MenuItem("tokenizer", "Train the tokenizer", "Coding oriented byte level BPE."),
        MenuItem("train", "Train the model", "Real optimizer steps on packed 1024 token sequences."),
        MenuItem("evaluate", "Evaluate the model", "Perplexity and structural coding tasks."),
        MenuItem("export", "Export the artifact", "Versioned manifest for the inference engine."),
        MenuItem("controller", "Train the budget controller", "Optional trainable adaptive budget policy."),
        MenuItem("dashboard", "Start the training dashboard", "Authenticated metrics and controls."),
        MenuItem("diagnostics", "Show environment diagnostics", "Versions, paths and optional packages."),
    ),
)
