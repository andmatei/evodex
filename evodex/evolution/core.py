import copy
import numpy as np

from pydantic import BaseModel
from typing import TypeVar

from .types import Gene, GeneList

T = TypeVar("T", bound=BaseModel)


def apply_mutations(config: T) -> T:
    """
    Recursively traverses a Pydantic model and applies mutations to fields
    that have `Gene` metadata.
    """
    mutated_config = copy.deepcopy(config)

    for field_name, field_info in mutated_config.__class__.model_fields.items():
        gene_metadata: Gene | GeneList | None = next(
            (
                m
                for m in field_info.metadata
                if isinstance(m, Gene) or isinstance(m, GeneList)
            ),
            None,
        )
        current_value = getattr(mutated_config, field_name)

        if gene_metadata:
            # A. Handle Allele (numerical parameter) mutation
            if isinstance(gene_metadata, Gene):
                allele = gene_metadata
                noise = np.random.normal(0, allele.mutation_std)
                new_value = current_value + noise
                new_value = np.clip(new_value, allele.min_val, allele.max_val)
                setattr(mutated_config, field_name, new_value)

            # B. Handle Chromosome (structural list/tuple) mutation
            if isinstance(gene_metadata, GeneList) and isinstance(
                current_value, (list, tuple)
            ):
                chromosome = gene_metadata
                mutable_list = list(current_value)

                # Add an element
                if (
                    np.random.rand() < chromosome.add_prob
                    and len(mutable_list) < chromosome.max_len
                ):
                    if mutable_list:
                        mutable_list.append(
                            copy.deepcopy(np.random.choice(mutable_list))
                        )

                # Remove an element
                if (
                    np.random.rand() < chromosome.remove_prob
                    and len(mutable_list) > chromosome.min_len
                ):
                    idx_to_remove = np.random.randint(0, len(mutable_list))
                    del mutable_list[idx_to_remove]

                setattr(mutated_config, field_name, tuple(mutable_list))

        # C. Recurse into nested models or lists of models
        if isinstance(current_value, BaseModel):
            setattr(mutated_config, field_name, apply_mutations(current_value))
        elif isinstance(current_value, (list, tuple)):
            mutated_list = [
                apply_mutations(item) if isinstance(item, BaseModel) else item
                for item in current_value
            ]
            setattr(mutated_config, field_name, type(current_value)(mutated_list))

    return mutated_config


def apply_crossover(parent_a: T, parent_b: T) -> T:
    child_config = copy.deepcopy(parent_a)

    for field_name, field_info in child_config.__class__.model_fields.items():
        gene_metadata: Gene | GeneList | None = next((m for m in field_info.metadata if isinstance(m, (Gene, GeneList))), None)
        current_value = getattr(child_config, field_name)

        if gene_metadata:
            # A. If the field is a GeneList, perform one-point structural crossover
            if isinstance(gene_metadata, GeneList) and isinstance(current_value, (list, tuple)):
                list_a = getattr(parent_a, field_name)
                list_b = getattr(parent_b, field_name)
                
                if len(list_a) > 0 and len(list_b) > 0:
                    crossover_point = np.random.randint(0, min(len(list_a), len(list_b)))
                    new_list = list(list_a[:crossover_point]) + list(list_b[crossover_point:])
                    
                    max_len = gene_metadata.max_len
                    if len(new_list) > max_len:
                        new_list = new_list[:max_len]

                    setattr(child_config, field_name, tuple(new_list))
            
            # B. If the field is a Gene, perform arithmetic crossover (blending)
            elif isinstance(gene_metadata, Gene):
                val_a = getattr(parent_a, field_name)
                val_b = getattr(parent_b, field_name)
                
                alpha = np.random.rand()
                new_val = alpha * val_a + (1 - alpha) * val_b
                
                # THE FIX: Check if the original field was an int and round if so
                if field_info.annotation is int:
                    new_val = int(round(new_val))
                
                setattr(child_config, field_name, new_val)

        # C. Recurse into nested models (that are not genes themselves)
        elif isinstance(current_value, BaseModel):
            nested_child = apply_crossover(getattr(parent_a, field_name), getattr(parent_b, field_name))
            setattr(child_config, field_name, nested_child)
        
        # D. Recurse into lists/tuples of nested models
        elif isinstance(current_value, (list, tuple)):
            list_a = getattr(parent_a, field_name)
            list_b = getattr(parent_b, field_name)
            
            crossed_list = []
            min_len = min(len(list_a), len(list_b))
            for i in range(min_len):
                item_a = list_a[i]
                item_b = list_b[i]

                if isinstance(item_a, BaseModel) and isinstance(item_b, BaseModel):
                    crossed_list.append(apply_crossover(item_a, item_b))
                else:
                    crossed_list.append(item_a) 
            
            if len(list_a) > min_len:
                crossed_list.extend(list_a[min_len:])
            elif len(list_b) > min_len:
                crossed_list.extend(list_b[min_len:])

            setattr(child_config, field_name, type(current_value)(crossed_list))

    return child_config
