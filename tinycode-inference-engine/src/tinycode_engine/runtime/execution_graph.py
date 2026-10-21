"""Description of the execution graph of the model.

The graph is a documentation and diagnostic artefact: it lists the
operators that the engine has to support for a given architecture, which
is exactly what the artifact manifest declares as ``required_operators``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["GraphNode", "ExecutionGraph", "build_execution_graph"]


@dataclass
class GraphNode:
    """A single node of the execution graph."""

    name: str
    operator: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": dict(self.attributes),
        }


@dataclass
class ExecutionGraph:
    """The full execution graph of a model."""

    architecture_id: str = "tinycode-decoder-v1"
    nodes: list[GraphNode] = field(default_factory=list)

    @property
    def operators(self) -> list[str]:
        seen: list[str] = []
        for node in self.nodes:
            if node.operator not in seen:
                seen.append(node.operator)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "node_count": len(self.nodes),
            "operators": self.operators,
            "nodes": [node.to_dict() for node in self.nodes],
        }


def build_execution_graph(config: dict[str, Any]) -> ExecutionGraph:
    """Build the execution graph of a TinyCode configuration."""

    graph = ExecutionGraph(architecture_id=str(config.get("architecture_id", "tinycode-decoder-v1")))
    graph.nodes.append(
        GraphNode("token_embedding", "embedding", outputs=["hidden_0"])
    )
    if config.get("position_encoding", "learned") == "learned":
        graph.nodes.append(GraphNode("position_embedding", "embedding", outputs=["positions"]))
    else:
        graph.nodes.append(GraphNode("position_embedding", "sinusoidal", outputs=["positions"]))

    for layer in range(int(config.get("n_layers", 0))):
        graph.nodes.append(GraphNode(f"layers.{layer}.norm1", "rmsnorm", inputs=["hidden"]))
        graph.nodes.append(
            GraphNode(
                f"layers.{layer}.attention",
                "causal_attention",
                inputs=["norm1"],
                attributes={"n_heads": config.get("n_heads"), "head_dim": None},
            )
        )
        graph.nodes.append(GraphNode(f"layers.{layer}.norm2", "rmsnorm", inputs=["hidden"]))
        graph.nodes.append(
            GraphNode(f"layers.{layer}.feed_forward", "silu", inputs=["norm2"])
        )
    graph.nodes.append(GraphNode("final_norm", "rmsnorm", inputs=["hidden"]))
    graph.nodes.append(GraphNode("lm_head", "linear", inputs=["hidden"], outputs=["logits"]))
    graph.nodes.append(GraphNode("sampler", "sampling", inputs=["logits"], outputs=["token"]))
    return graph
