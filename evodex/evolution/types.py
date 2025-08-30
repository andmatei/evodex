import numpy as np
import copy
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Any, Annotated, List

class Allele(BaseModel):
    """Metadata for a single numerical gene."""
    mutation_std: float = Field(..., description="Standard deviation for Gaussian mutation.")
    min_val: float = Field(-np.inf, description="Minimum allowed value after mutation.")
    max_val: float = Field(np.inf, description="Maximum allowed value after mutation.")

class Chromosome(BaseModel):
    """Metadata for a structural gene (a list of sub-components)."""
    min_len: int = Field(1, description="Minimum number of elements in the list.")
    max_len: int = Field(10, description="Maximum number of elements in the list.")
    add_prob: float = Field(0.1, description="Probability of adding an element during mutation.")
    remove_prob: float = Field(0.05, description="Probability of removing an element during mutation.")

class Gene(BaseModel):
    """A container for evolutionary metadata attached to a Pydantic field."""
    allele: Optional[Allele] = None
    chromosome: Optional[Chromosome] = None

# --- 2. Annotate the Robot Configuration Models ---
# We use `Annotated` to attach our `Gene` metadata to each evolvable field.

class BaseConfig(BaseModel):
    width: Annotated[float, Field(..., ge=0.0), Gene(allele=Allele(mutation_std=1.0, min_val=10.0, max_val=50.0))]
    height: Annotated[float, Field(..., ge=0.0), Gene(allele=Allele(mutation_std=2.0, min_val=50.0, max_val=150.0))]
    mass: Annotated[float, Field(..., gt=0.0), Gene(allele=Allele(mutation_std=0.5, min_val=5.0, max_val=20.0))]

class SegmentConfig(BaseModel):
    length: Annotated[float, Field(..., ge=0.0), Gene(allele=Allele(mutation_std=2.0, min_val=20.0, max_val=120.0))]
    width: Annotated[float, Field(..., ge=0.0), Gene(allele=Allele(mutation_std=1.0, min_val=5.0, max_val=30.0))]
    mass: Annotated[float, Field(..., ge=0.0), Gene(allele=Allele(mutation_std=0.2, min_val=1.0, max_val=10.0))]
    # Non-evolvable properties have no `Gene` annotation
    motor_stiffness: float
    motor_damping: float
    joint_angle_limit: Tuple[float, float]

class FingerConfig(BaseModel):
    # This list of segments is a chromosome that can grow or shrink
    segments: Annotated[
        Tuple[SegmentConfig, ...], 
        Gene(chromosome=Chromosome(min_len=1, max_len=4, add_prob=0.1, remove_prob=0.1))
    ]
    defaults: Optional[dict] = None # Assuming defaults are not evolved

class RobotConfig(BaseModel):
    base: BaseConfig
    # The list of fingers is the primary chromosome for crossover
    fingers: Annotated[
        Tuple[FingerConfig, ...],
        Gene(chromosome=Chromosome(min_len=1, max_len=5, add_prob=0.1, remove_prob=0.05))
    ]