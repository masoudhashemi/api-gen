from __future__ import annotations
import random
from typing import List
import networkx as nx


def propose_skeleton(g: nx.DiGraph, rng: random.Random, *, max_len: int = 3) -> List[str]:
    # Prefer sequences that include at least one write tool
    write_nodes = [n for n, d in g.nodes(data=True) if d.get("write")]
    if not write_nodes:
        return []
    start = rng.choice(write_nodes)
    seq = []
    # Add dependencies (ancestors) in topo order to make a plausible chain
    ancestors = list(nx.ancestors(g, start))
    order = [n for n in nx.topological_sort(g) if n in ancestors]
    seq.extend(order[-(max_len - 1):])  # keep last few ancestors
    seq.append(start)
    # Optionally extend with a successor if available
    succ = list(g.successors(start))
    if succ and len(seq) < max_len:
        seq.append(rng.choice(succ))
    # Remove duplicates while keeping order
    seen = set()
    uniq = []
    for n in seq:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq

