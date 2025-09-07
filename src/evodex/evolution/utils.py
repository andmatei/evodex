import copy
import numpy as np
from pydantic import BaseModel
from typing import Any, Iterable, Type, TypeVar

from .types import EvolvableConfig, Gene, GeneList

T = TypeVar("T", bound=EvolvableConfig)


def partial_update(template: dict, updates: dict) -> dict:
    def _partial_update(template: Any, update: Any) -> Any:
        if isinstance(template, dict) and isinstance(update, dict):
            for key, value in update.items():
                if key in template:
                    template[key] = _partial_update(template[key], value)
                else:
                    template[key] = value
            return template

        elif (isinstance(template, list) or isinstance(template, tuple)) and (
            isinstance(update, list) or isinstance(update, tuple)
        ):
            updated_list = []
            for i in range(min(len(template), len(update))):
                updated_list.append(_partial_update(template[i], update[i]))

            if len(update) > len(template):
                item_template = template[0]
                for i in range(len(template), len(update)):
                    updated_list.append(
                        _partial_update(copy.deepcopy(item_template), update[i])
                    )

            return updated_list

        return update if update is not None else template

    result = copy.deepcopy(template)
    return _partial_update(result, updates)


def flatten_genome(config: T) -> np.ndarray:
    """
    Recursively traverses a Pydantic model instance and flattens all
    fields marked with `Gene` or `GeneList` metadata into a single NumPy array.

    Uses padding for variable-length lists based on `max_len` metadata.
    """
    flat_list = []

    for field_name, field_info in config.__class__.model_fields.items():
        gene_metadata = next(
            (m for m in field_info.metadata if isinstance(m, (Gene, GeneList))), None
        )
        current_value = getattr(config, field_name)

        if gene_metadata:
            if isinstance(gene_metadata, Gene):
                flat_list.append(current_value)

            elif isinstance(gene_metadata, GeneList) and isinstance(
                current_value, (list, tuple)
            ):
                max_len = gene_metadata.max_len

                # First, add the actual number of items as a gene
                flat_list.append(len(current_value))

                # Flatten each item in the list
                item_len = 0
                for item in current_value:
                    if isinstance(item, EvolvableConfig):
                        flat_item = flatten_genome(item)
                        if item_len == 0:
                            item_len = len(flat_item)
                        if item_len != len(flat_item):
                            raise ValueError("Inconsistent item lengths found.")
                        flat_list.extend(flat_item)

                # Pad the rest of the list up to max_len
                if len(current_value) > 0:
                    # To pad, we need a template of one item's flattened size
                    num_to_pad = max_len - len(current_value)

                    # Use np.nan as a clear padding value
                    padding = [np.nan] * (num_to_pad * item_len)
                    flat_list.extend(padding)

    return np.array(flat_list, dtype=np.float32)


def unflatten_genome(flat_array: np.ndarray, template: T) -> T:
    """
    Recursively reconstructs a Pydantic model instance from a flat NumPy array.
    """
    # Use a mutable list to easily pop values from the front
    values = list(flat_array)

    def _unflatten_recursive(cls: Type[T]) -> dict:
        data = {}
        for field_name, field_info in cls.model_fields.items():
            gene_metadata = next(
                (m for m in field_info.metadata if isinstance(m, (Gene, GeneList))),
                None,
            )

            if gene_metadata:
                if isinstance(gene_metadata, Gene):
                    data[field_name] = values.pop(0)

                elif isinstance(gene_metadata, GeneList):
                    # First, get the actual number of items
                    num_items = int(round(values.pop(0)))
                    max_len = gene_metadata.max_len

                    sub_items = []

                    # The type of items in the list (e.g., EvolvableFingerConfig)
                    if field_info.annotation is not None and hasattr(
                        field_info.annotation, "__args__"
                    ):
                        item_type = field_info.annotation.__args__[0]
                    else:
                        raise TypeError(
                            f"Cannot determine item type for field '{field_name}' (annotation: {field_info.annotation})"
                        )

                    item_flat_size = 0
                    prev_values_size = len(values)
                    for _ in range(num_items):
                        sub_items.append(_unflatten_recursive(item_type))
                        if item_flat_size == 0:
                            item_flat_size = prev_values_size - len(values)

                    data[field_name] = sub_items

                    # Discard the padded values
                    if num_items > 0:
                        num_to_pad = max_len - num_items

                        for _ in range(num_to_pad * item_flat_size):
                            values.pop(0)

        return data

    # Use the template's class to reconstruct the instance
    template_class = type(template)
    data = _unflatten_recursive(template_class)
    template_data = template.model_dump()
    updated_data = partial_update(template_data, data)

    return template_class(**updated_data)
