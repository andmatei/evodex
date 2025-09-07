import copy

from typing import TypeVar, List, Optional, Any

from .types import GeneList, EvolvableConfig

T = TypeVar("T", bound=EvolvableConfig)


class Node[T]:
    def __init__(self, data: T, field_name: str = ""):
        self.data = data
        self.field_name = field_name
        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None

    def add_child(self, child: "Node"):
        self.children.append(child)
        child.parent = self


def config_to_tree(config: T, field_name: str = "") -> Node[T]:
    root = Node(copy.deepcopy(config), field_name)

    for field_name, gene in config._genes.items():
        if isinstance(gene, GeneList) and hasattr(config, field_name):
            child_list = getattr(config, field_name)

            if gene.structure == GeneList.Structure.PARALLEL:
                # All children are attatched directly to the root
                for child_config in child_list:
                    child_node = config_to_tree(child_config, field_name)
                    root.add_child(child_node)

            elif gene.structure == GeneList.Structure.CHAIN:
                current_parent_node = root
                for child_config in child_list:
                    child_node = config_to_tree(child_config, field_name)
                    current_parent_node.add_child(child_node)
                    current_parent_node = child_node

    for f_name in config.__class__.model_fields:
        if f_name not in config._genes:
            field_value = getattr(config, f_name)
            if isinstance(field_value, EvolvableConfig):
                child_node = config_to_tree(field_value, f_name)
                root.add_child(child_node)

    return root


def tree_to_config(root: Node[T]) -> T:
    config = root.data

    children_by_field: dict[str, List[Node[T]]] = {}
    for child in root.children:
        if child.field_name not in children_by_field:
            children_by_field[child.field_name] = []
        children_by_field[child.field_name].append(child)

    for field_name, children in children_by_field.items():
        gene = config._genes.get(field_name)

        if isinstance(gene, GeneList):
            if gene.structure == GeneList.Structure.PARALLEL:
                child_configs = [tree_to_config(child) for child in children]
                setattr(config, field_name, tuple(child_configs))

            elif gene.structure == GeneList.Structure.CHAIN:
                chain_configs = []
                current_node = children[0] if children else None
                while current_node:
                    next_node = next(
                        (
                            c
                            for c in current_node.children
                            if c.field_name == field_name
                        ),
                        None,
                    )

                    current_node.children.remove(next_node) if next_node else None
                    chain_configs.append(tree_to_config(current_node))
                    # Find the next link in the chain
                    current_node = next_node

                setattr(config, field_name, tuple(chain_configs))
        else:
            # This handles single nested models like 'base'
            setattr(config, field_name, tree_to_config(children[0]))

    return config


def get_all_nodes(root: Node[T], role: Optional[str] = None) -> List[Node[T]]:
    nodes = []
    if role is None or root.data._role == role:
        nodes.append(root)
    for child in root.children:
        nodes.extend(get_all_nodes(child, role))
    return nodes
