from typing import Annotated, Tuple

from pydantic import Field

from evodex.simulation.robot.config import (
    RobotConfig,
    BaseConfig,
    FingerConfig,
    SegmentConfig,
)

from .types import EvolvableConfig, Gene, GeneList, Allele, AlleleList


class EvolvableBaseConfig(BaseConfig, EvolvableConfig):
    width: float = Allele(mutation_std=1.0, min_val=10.0, max_val=50.0, ge=0.0)
    height: float = Allele(mutation_std=2.0, min_val=20.0, max_val=150.0, ge=0.0)


class EvolvableSegmentConfig(SegmentConfig, EvolvableConfig):
    length: float = Allele(mutation_std=2.0, min_val=20.0, max_val=120.0, ge=0.0)
    width: float = Allele(mutation_std=1.0, min_val=5.0, max_val=30.0, ge=0.0)


class EvolvableFingerConfig(FingerConfig, EvolvableConfig):
    segments: Tuple[EvolvableSegmentConfig, ...] = AlleleList(
        min_len=1,
        max_len=10,
        structure=GeneList.Structure.CHAIN,
        add_prob=0.1,
        remove_prob=0.1,
    )


class EvolvableRobotConfig(RobotConfig, EvolvableConfig):
    base: EvolvableBaseConfig
    fingers: Tuple[EvolvableFingerConfig, ...] = AlleleList(
        min_len=1,
        max_len=5,
        structure=GeneList.Structure.PARALLEL,
        add_prob=0.1,
        remove_prob=0.05,
    )
