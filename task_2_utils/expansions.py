"""Task-2 helpers around the shared BIC-cherry construction.

Kept here so that ``utils/bic_cherry.py`` stays exactly as its authors wrote it.
"""
import networkx as nx

from utils.bic_cherry import bic_cherry_expansion

__all__ = ["network_depth", "EXPANSIONS", "DEFAULT_EXPANSION", "explaining_network"]


def network_depth(network: nx.DiGraph) -> int:
    """Number of arcs on the longest root-to-leaf path."""
    return nx.dag_longest_path_length(network) if network.number_of_nodes() else 0


#: name -> construction, used by experiments.analyse_instance
EXPANSIONS = {
    "restricted": lambda G: bic_cherry_expansion(G, restricted=True),
    "plain": lambda G: bic_cherry_expansion(G, restricted=False),
}
DEFAULT_EXPANSION = "restricted"


def explaining_network(bmg, expansion: str = DEFAULT_EXPANSION):
    return EXPANSIONS[expansion](bmg)
