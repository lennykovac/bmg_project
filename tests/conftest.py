"""Shared fixtures and helpers for all test modules."""
import random

import numpy as np
import pytest

from tests.helpers import colored_graph, colored_tree
from utils.tree_utils import create_gene_tree_n_leaves


@pytest.fixture
def example_tree():
    """((a1,b1),a2) -- the running example of the visual guide."""
    return colored_tree([("r", "x"), ("r", "a2"), ("x", "a1"), ("x", "b1")],
                        {"a1": "A", "a2": "A", "b1": "B"})


@pytest.fixture
def star_bmg():
    """one gene of species 2, three of species 3 -> the LRT is a star"""
    return colored_graph({2: 2, 5: 3, 6: 3, 7: 3}, [(2, 5), (2, 6), (2, 7), (5, 2), (6, 2), (7, 2)])


@pytest.fixture(scope="session")
def instances():
    """25 AsymmeTree instances, 2-4 species, up to 12 genes (seeded)."""
    random.seed(3)
    np.random.seed(3)
    out = []
    for _ in range(25):
        s = random.randint(2, 4)
        out.append(create_gene_tree_n_leaves(random.randint(s, 12), s))
    return out


@pytest.fixture(scope="session")
def small_instances(instances):
    return [d for d in instances if d.bmg.number_of_nodes() <= 6][:5]
