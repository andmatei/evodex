import copy
import numpy as np

from pydantic import BaseModel

from .types import Gene

def apply_mutations(config: BaseModel) -> BaseModel:
    """
    Recursively traverses a Pydantic model and applies mutations to fields
    that have `Gene` metadata.
    """
    mutated_config = copy.deepcopy(config)

    for field_name, field_info in mutated_config.model_fields.items():
        gene_metadata: Gene | None = next((m for m in field_info.metadata if isinstance(m, Gene)), None)
        current_value = getattr(mutated_config, field_name)

        if gene_metadata:
            # A. Handle Allele (numerical parameter) mutation
            if gene_metadata.allele:
                allele = gene_metadata.allele
                noise = np.random.normal(0, allele.mutation_std)
                new_value = current_value + noise
                new_value = np.clip(new_value, allele.min_val, allele.max_val)
                setattr(mutated_config, field_name, new_value)

            # B. Handle Chromosome (structural list/tuple) mutation
            if gene_metadata.chromosome:
                chromosome = gene_metadata.chromosome
                mutable_list = list(current_value)
                # Add an element
                if np.random.rand() < chromosome.add_prob and len(mutable_list) < chromosome.max_len:
                    if mutable_list:
                        mutable_list.append(copy.deepcopy(np.random.choice(mutable_list)))
                # Remove an element
                if np.random.rand() < chromosome.remove_prob and len(mutable_list) > chromosome.min_len:
                    idx_to_remove = np.random.randint(0, len(mutable_list))
                    del mutable_list[idx_to_remove]
                
                setattr(mutated_config, field_name, tuple(mutable_list))

        # C. Recurse into nested models or lists of models
        if isinstance(current_value, BaseModel):
            setattr(mutated_config, field_name, apply_mutations(current_value))
        elif isinstance(current_value, (list, tuple)):
            mutated_list = [apply_mutations(item) if isinstance(item, BaseModel) else item for item in current_value]
            setattr(mutated_config, field_name, type(current_value)(mutated_list))
            
    return mutated_config

def apply_crossover(parent_a: BaseModel, parent_b: BaseModel) -> BaseModel:
    """
    Recursively traverses two Pydantic models and performs crossover on fields
    marked as chromosomes.
    """
    child_config = copy.deepcopy(parent_a)

    for field_name, field_info in child_config.model_fields.items():
        gene_metadata: Gene | None = next((m for m in field_info.metadata if isinstance(m, Gene)), None)
        
        # A. If the field is a chromosome, perform crossover
        if gene_metadata and gene_metadata.chromosome:
            list_a = getattr(parent_a, field_name)
            list_b = getattr(parent_b, field_name)
            
            if len(list_a) > 1 and len(list_b) > 1:
                # One-point crossover
                crossover_point = np.random.randint(1, min(len(list_a), len(list_b)))
                new_list = list(list_a[:crossover_point]) + list(list_b[crossover_point:])
                setattr(child_config, field_name, tuple(new_list))

        # B. Recurse into nested models
        elif isinstance(getattr(child_config, field_name), BaseModel):
            nested_child = apply_crossover(
                getattr(parent_a, field_name),
                getattr(parent_b, field_name)
            )
            setattr(child_config, field_name, nested_child)
            
    return child_config