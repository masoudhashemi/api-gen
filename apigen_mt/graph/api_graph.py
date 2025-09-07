from __future__ import annotations
from typing import Dict
import networkx as nx
from apigen_mt.env.base import Environment


def build_api_graph(env: Environment) -> nx.DiGraph:
    g = nx.DiGraph()
    for name, tool in env.tools.items():
        g.add_node(name, write=tool.write, schema=tool.schema)
    for name, tool in env.tools.items():
        for dep in tool.deps:
            if dep in env.tools:
                g.add_edge(dep, name)
    return g


def graph_summary(g: nx.DiGraph) -> str:
    parts = []
    for n in g.nodes:
        write = g.nodes[n].get("write", False)
        deps = list(g.predecessors(n))
        parts.append(f"{n}({'W' if write else 'R'}) <- {deps}")
    return "; ".join(parts)

