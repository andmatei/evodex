import copy

from typing import TypeVar, List, Optional, Any

from .types import GeneList, EvolvableConfig

T = TypeVar("T", bound=EvolvableConfig)


class Node[T]:
    def __init__(self, data: T, field_name: Optional[str] = None):
        self.data = data
        self.field_name = field_name
        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None

    def add_child(self, child: "Node"):
        self.children.append(child)
        child.parent = self


def config_to_tree(config: T, field_name: Optional[str] = None) -> Node[T]:
    root = Node(config, field_name)

    for f_name, f_info in config.__class__.model_fields.items():
        gene_metadata = next(
            (m for m in f_info.metadata if isinstance(m, GeneList)), None
        )

        if gene_metadata and hasattr(config, f_name):
            child_list = getattr(config, f_name)

            if gene_metadata.structure == GeneList.Structure.PARALLEL:
                # All children are attatched directly to the root
                for child_config in child_list:
                    child_node = config_to_tree(child_config, f_name)
                    root.add_child(child_node)

            elif gene_metadata.structure == GeneList.Structure.CHAIN:
                current_parent_node = root
                for child_config in child_list:
                    child_node = config_to_tree(child_config, f_name)
                    current_parent_node.add_child(child_node)
                    current_parent_node = child_node

    return root


def tree_to_config(root: Node[T]) -> T:
    config = copy.deepcopy(root.data)

    children_by_field: dict[str, List[Any]] = {}

    for child in root.children:
        if child.field_name:
            if child.field_name not in children_by_field:
                children_by_field[child.field_name] = []
            child_config = tree_to_config(child)
            children_by_field[child.field_name].append(child_config)

    for field_name, children_list in children_by_field.items():
        setattr(config, field_name, tuple(children_list))

    return config


def get_all_nodes(root: Node[T], role: Optional[str] = None) -> List[Node[T]]:
    nodes = []
    if role is None or root.data.__role == role:
        nodes.append(root)
    for child in root.children:
        nodes.extend(get_all_nodes(child, role))
    return nodes
