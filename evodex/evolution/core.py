import copy
import random
import numpy as np

from typing import Optional, TypeVar

from evodex.evolution.tree import Node, get_all_nodes

from .types import EvolvableConfig, Gene, GeneList
from .tree import config_to_tree, tree_to_config

T = TypeVar("T", bound=EvolvableConfig)


def mutate(config: T) -> T:
    """
    Recursively traverses a Pydantic model and applies mutations to fields
    that have `Gene` metadata.
    """
    mutated_config = copy.deepcopy(config)

    for field_name, gene in mutated_config._genes.items():
        current_value = getattr(mutated_config, field_name)

        if isinstance(gene, Gene):
            # A. Handle Allele (numerical parameter) mutation
            noise = np.random.normal(0, gene.mutation_std)
            new_value = current_value + noise

            new_value = np.clip(new_value, gene.min_val, gene.max_val)

            if isinstance(current_value, int):
                new_value = int(round(new_value))
            else:
                new_value = float(new_value)
            setattr(mutated_config, field_name, new_value)

            # B. Handle Chromosome (structural list/tuple) mutation
        elif isinstance(gene, GeneList) and isinstance(current_value, (list, tuple)):
            mutable_list = list(current_value)

            # Add an element
            if np.random.rand() < gene.add_prob and len(mutable_list) < gene.max_len:
                if mutable_list:
                    mutable_list.append(copy.deepcopy(np.random.choice(mutable_list)))

            # Remove an element
            if np.random.rand() < gene.remove_prob and len(mutable_list) > gene.min_len:
                idx_to_remove = np.random.randint(0, len(mutable_list))
                del mutable_list[idx_to_remove]

            setattr(mutated_config, field_name, tuple(mutable_list))

    for field_name in mutated_config.__class__.model_fields:
        # C. Recurse into nested models or lists of models
        current_value = getattr(mutated_config, field_name)
        if isinstance(current_value, EvolvableConfig):
            setattr(mutated_config, field_name, mutate(current_value))
        elif isinstance(current_value, (list, tuple)):
            mutated_list = [
                mutate(item) if isinstance(item, EvolvableConfig) else item
                for item in current_value
            ]
            setattr(mutated_config, field_name, type(current_value)(mutated_list))

    return mutated_config


def _crossover_alleles(child: Node[T], parent_a: Node[T], parent_b: Node[T]) -> Node[T]:
    for field_name, gene in child.data._genes.items():
        if isinstance(gene, Gene):
            val_a = getattr(parent_a.data, field_name)
            val_b = getattr(parent_b.data, field_name)
            alpha = np.random.rand()
            new_val = alpha * val_a + (1 - alpha) * val_b

            if isinstance(val_a, int):
                new_val = int(round(new_val))

            setattr(child.data, field_name, new_val)

    for i, child_node in enumerate(child.children):
        if i < len(parent_a.children) and i < len(parent_b.children):
            corresponding_node_a = parent_a.children[i]
            corresponding_node_b = parent_b.children[i]
            if (
                type(child_node.data)
                == type(corresponding_node_a.data)
                == type(corresponding_node_b.data)
            ):
                _crossover_alleles(
                    child_node, corresponding_node_a, corresponding_node_b
                )

    return child


def _crossover_tree(
    tree_a: Node[T], tree_b: Node[T], role: Optional[str] = None
) -> Node[T]:
    nodes_a = get_all_nodes(tree_a, role)
    nodes_b = get_all_nodes(tree_b, role)

    node_to_replace: Node[T] = random.choice(nodes_a)

    compatible_nodes_b = [
        n for n in nodes_b if type(n.data) == type(node_to_replace.data)
    ]

    if compatible_nodes_b:
        subtree_to_insert: Node[T] = copy.deepcopy(random.choice(compatible_nodes_b))

        if node_to_replace.parent is None:
            return subtree_to_insert

        parent_of_node = node_to_replace.parent
        for i, child in enumerate(parent_of_node.children):
            if child is node_to_replace:
                parent_of_node.children[i] = subtree_to_insert
                subtree_to_insert.parent = parent_of_node
                break

    return tree_a


def crossover(parent_a: T, parent_b: T, role: Optional[str] = None) -> T:
    """Performs one-point subtree crossover o two trees"""
    tree_a = config_to_tree(parent_a)
    tree_b = config_to_tree(parent_b)

    child_tree = _crossover_tree(tree_a, tree_b, role)
    child_tree = _crossover_alleles(child_tree, tree_a, tree_b)

    return tree_to_config(child_tree)
