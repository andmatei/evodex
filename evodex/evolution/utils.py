import numpy as np
from pydantic import BaseModel
from typing import List, Tuple, Type, TypeVar

# Assuming your model definitions from evolutionary_schema.py are available
from .core import Gene, GeneList

T = TypeVar("T", bound=BaseModel)

def flatten_genome(config: T) -> np.ndarray:
    """
    Recursively traverses a Pydantic model instance and flattens all
    fields marked with `Gene` or `GeneList` metadata into a single NumPy array.

    Uses padding for variable-length lists based on `max_len` metadata.
    """
    flat_list = []
    
    for field_name, field_info in config.__class__.model_fields.items():
        gene_metadata = next((m for m in field_info.metadata if isinstance(m, (Gene, GeneList))), None)
        current_value = getattr(config, field_name)

        if gene_metadata:
            if isinstance(gene_metadata, Gene):
                flat_list.append(current_value)
            
            elif isinstance(gene_metadata, GeneList) and isinstance(current_value, (list, tuple)):
                max_len = gene_metadata.max_len
                
                # First, add the actual number of items as a gene
                flat_list.append(len(current_value))

                # Flatten each item in the list
                for item in current_value:
                    if isinstance(item, BaseModel):
                        flat_list.extend(flatten_genome(item))
                
                # Pad the rest of the list up to max_len
                if len(current_value) > 0:
                    # To pad, we need a template of one item's flattened size
                    template_item_flat_size = len(flatten_genome(current_value[0]))
                    num_to_pad = max_len - len(current_value)
                    
                    # Use np.nan as a clear padding value
                    padding = [np.nan] * (num_to_pad * template_item_flat_size)
                    flat_list.extend(padding)

    return np.array(flat_list, dtype=np.float32)


def unflatten_genome(flat_array: np.ndarray, model_class: Type[T]) -> T:
    """
    Recursively reconstructs a Pydantic model instance from a flat NumPy array.
    """
    # Use a mutable list to easily pop values from the front
    values = list(flat_array)

    def _unflatten_recursive(cls: Type[BaseModel]):
        data = {}
        for field_name, field_info in cls.model_fields.items():
            gene_metadata = next((m for m in field_info.metadata if isinstance(m, (Gene, GeneList))), None)
            
            if gene_metadata:
                if isinstance(gene_metadata, Gene):
                    data[field_name] = values.pop(0)
                
                elif isinstance(gene_metadata, GeneList):
                    # First, get the actual number of items
                    num_items = int(round(values.pop(0)))
                    max_len = gene_metadata.max_len
                    
                    sub_items = []
                    
                    # The type of items in the list (e.g., EvolvableFingerConfig)
                    item_type = field_info.annotation.__args__[0]
                    
                    for _ in range(num_items):
                        sub_items.append(_unflatten_recursive(item_type))
                    
                    data[field_name] = sub_items
                    
                    # Discard the padded values
                    if num_items > 0:
                        template_item_flat_size = len(flatten_genome(sub_items[0]))
                        num_to_pad = max_len - num_items
                        
                        for _ in range(num_to_pad * template_item_flat_size):
                            values.pop(0)
            else:
                # For non-gene fields, use the default value from the model
                if field_info.default is not None:
                    data[field_name] = field_info.default

        return cls(**data)

    return _unflatten_recursive(model_class)
