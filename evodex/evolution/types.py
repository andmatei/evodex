import numpy as np
import copy
from pydantic import BaseModel, Field
from typing import Annotated, Any, Literal, Optional
from enum import Enum


class Gene(BaseModel):
    """Metadata for a single numerical gene."""

    mutation_std: float = Field(
        ..., description="Standard deviation for Gaussian mutation."
    )
    min_val: float = Field(-np.inf, description="Minimum allowed value after mutation.")
    max_val: float = Field(np.inf, description="Maximum allowed value after mutation.")


class GeneList(BaseModel):
    class Structure(str, Enum):
        PARALLEL = "parallel"
        CHAIN = "chain"

    """Metadata for a structural gene (a list of sub-components)."""
    structure: Structure = Field(
        ..., description="The structure type of the gene list."
    )
    min_len: int = Field(1, description="Minimum number of elements in the list.")
    max_len: int = Field(10, description="Maximum number of elements in the list.")
    add_prob: float = Field(
        0.1, description="Probability of adding an element during mutation."
    )
    remove_prob: float = Field(
        0.05, description="Probability of removing an element during mutation."
    )


class EvolvableConfig(BaseModel):
    """Base class for all evolvable configurations."""

    __role: Annotated[
        Optional[str], Field(description="The role of the configuration.")
    ] = "user"


def Allele(
    mutation_std: float,
    min_val: float = -np.inf,
    max_val: float = np.inf,
    **kwargs: Any,
) -> Any:
    gene = Gene(
        mutation_std=mutation_std,
        min_val=min_val,
        max_val=max_val,
    )
    return Field(**kwargs, json_schema_extra={"gene": gene.model_dump()})


def AlleleList(
    structure: GeneList.Structure,
    min_len: int = 1,
    max_len: int = 10,
    add_prob: float = 0.1,
    remove_prob: float = 0.05,
    **kwargs: Any,
) -> Any:
    gene_list = GeneList(
        structure=structure,
        min_len=min_len,
        max_len=max_len,
        add_prob=add_prob,
        remove_prob=remove_prob,
    )
    return Field(**kwargs, json_schema_extra={"gene_list": gene_list.model_dump()})
