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

    _genes: dict[str, Gene | GeneList] = {}
    _role: Optional[str]
