from typing import Annotated, Tuple

from pydantic import Field

from evodex.simulation.robot.config import (
    RobotConfig,
    BaseConfig,
    FingerConfig,
    SegmentConfig,
)

from .types import EvolvableConfig, Gene, GeneList


class EvolvableBaseConfig(BaseConfig, EvolvableConfig):
    _genes = {
        "width": Gene(mutation_std=1.0, min_val=10.0, max_val=50.0),
        "height": Gene(mutation_std=2.0, min_val=20.0, max_val=150.0),
    }


class EvolvableSegmentConfig(SegmentConfig, EvolvableConfig):
    _genes = {
        "length": Gene(mutation_std=2.0, min_val=20.0, max_val=120.0),
        "width": Gene(mutation_std=1.0, min_val=5.0, max_val=30.0),
    }


class EvolvableFingerConfig(FingerConfig, EvolvableConfig):
    segments: Tuple[EvolvableSegmentConfig, ...]

    _genes = {
        "segments": GeneList(
            structure=GeneList.Structure.CHAIN,
            min_len=1,
            max_len=10,
            add_prob=0.1,
            remove_prob=0.1,
        )
    }


class EvolvableRobotConfig(RobotConfig, EvolvableConfig):
    base: EvolvableBaseConfig
    fingers: Tuple[EvolvableFingerConfig, ...]

    _genes = {
        "fingers": GeneList(
            structure=GeneList.Structure.PARALLEL,
            min_len=1,
            max_len=5,
            add_prob=0.1,
            remove_prob=0.05,
        )
    }
