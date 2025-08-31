import numpy as np
import copy
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Any, Annotated, List


class Gene(BaseModel):
    """Metadata for a single numerical gene."""

    mutation_std: float = Field(
        ..., description="Standard deviation for Gaussian mutation."
    )
    min_val: float = Field(-np.inf, description="Minimum allowed value after mutation.")
    max_val: float = Field(np.inf, description="Maximum allowed value after mutation.")


class GeneList(BaseModel):
    """Metadata for a structural gene (a list of sub-components)."""

    min_len: int = Field(1, description="Minimum number of elements in the list.")
    max_len: int = Field(10, description="Maximum number of elements in the list.")
    add_prob: float = Field(
        0.1, description="Probability of adding an element during mutation."
    )
    remove_prob: float = Field(
        0.05, description="Probability of removing an element during mutation."
    )
