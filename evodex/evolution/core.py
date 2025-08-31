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


def apply_crossover(parent_a: BaseModel, parent_b: BaseModel) -> BaseModel:
    return parent_a
