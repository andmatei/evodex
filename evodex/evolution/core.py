import copy
import random
import numpy as np

from typing import Optional, TypeVar

from evodex.evolution.tree import Node, get_all_nodes

from .types import EvolvableConfig, Gene, GeneList

T = TypeVar("T", bound=EvolvableConfig)


def mutate(config: T) -> T:
    """
    Recursively traverses a Pydantic model and applies mutations to fields
    that have `Gene` metadata.
    """
    mutated_config = copy.deepcopy(config)

    for field_name, field_info in mutated_config.__class__.model_fields.items():
        metadata_dict = field_info.json_schema_extra or {}
        gene: Optional[Gene] = None
        gene_list: Optional[GeneList] = None

        if isinstance(metadata_dict, dict):
            gene_info = metadata_dict.get("gene")
            if isinstance(gene_info, dict):
                gene = Gene(**gene_info)  # type: ignore

            gene_list_info = metadata_dict.get("gene_list")
            if isinstance(gene_list_info, dict):
                gene_list = GeneList(**gene_list_info)  # type: ignore

        current_value = getattr(mutated_config, field_name)

        if gene:
            # A. Handle Allele (numerical parameter) mutation
            noise = np.random.normal(0, gene.mutation_std)
            new_value = current_value + noise

            new_value = np.clip(new_value, gene.min_val, gene.max_val)

            if field_info.annotation is int:
                new_value = int(round(new_value))
            else:
                new_value = float(new_value)
            setattr(mutated_config, field_name, new_value)

            # B. Handle Chromosome (structural list/tuple) mutation
        elif gene_list and isinstance(current_value, (list, tuple)):
            mutable_list = list(current_value)

            # Add an element
            if (
                np.random.rand() < gene_list.add_prob
                and len(mutable_list) < gene_list.max_len
            ):
                if mutable_list:
                    mutable_list.append(copy.deepcopy(np.random.choice(mutable_list)))

            # Remove an element
            if (
                np.random.rand() < gene_list.remove_prob
                and len(mutable_list) > gene_list.min_len
            ):
                idx_to_remove = np.random.randint(0, len(mutable_list))
                del mutable_list[idx_to_remove]

            setattr(mutated_config, field_name, tuple(mutable_list))

        # C. Recurse into nested models or lists of models
        if isinstance(current_value, EvolvableConfig):
            setattr(mutated_config, field_name, mutate(current_value))
        elif isinstance(current_value, (list, tuple)):
            mutated_list = [
                mutate(item) if isinstance(item, EvolvableConfig) else item
                for item in current_value
            ]
            setattr(mutated_config, field_name, type(current_value)(mutated_list))

    return mutated_config


def crossover_tree(
    parent_a: Node[T], parent_b: Node[T], role: Optional[str] = None
) -> Node[T]:
    """Performs one-point subtree crossover o two trees"""
    child_tree = copy.deepcopy(parent_a)
    nodes_a = get_all_nodes(parent_a, role)
    nodes_b = get_all_nodes(parent_b, role)

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

    return child_tree
