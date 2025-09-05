import math
from typing import Tuple, Union

from evodex.evolution.types import EvolvableConfig, Gene, GeneList

from .config import (
    BoxConfig,
    CapsuleConfig,
    CylinderConfig,
    SphereConfig,
    FingerAttachmentConfig,
    FingerConfig,
    RobotConfig,
    LinkConfig,
    BaseConfig,
)


class EvolvableBoxConfig(BoxConfig, EvolvableConfig):
    _genes = {
        "width": Gene(mutation_std=0.02, min_val=0.01, max_val=0.5),
        "length": Gene(mutation_std=0.02, min_val=0.01, max_val=0.5),
        "depth": Gene(mutation_std=0.01, min_val=0.01, max_val=0.1),
    }


class EvolvableCapsuleConfig(CapsuleConfig, EvolvableConfig):
    _genes = {
        "radius": Gene(mutation_std=0.01, min_val=0.005, max_val=0.5),
        "length": Gene(mutation_std=0.02, min_val=0.01, max_val=0.2),
    }


class EvolvableCylinderConfig(CylinderConfig, EvolvableConfig):
    _genes = {
        "radius": Gene(mutation_std=0.01, min_val=0.005, max_val=0.5),
        "length": Gene(mutation_std=0.02, min_val=0.01, max_val=0.2),
    }


class EvolvableSphereConfig(SphereConfig, EvolvableConfig):
    _genes = {
        "radius": Gene(mutation_std=0.02, min_val=0.005, max_val=0.1),
    }


AllEvolvableGeometryConfigs = Union[
    EvolvableBoxConfig,
    EvolvableSphereConfig,
    EvolvableCylinderConfig,
    EvolvableCapsuleConfig,
]


class EvolvableLinkConfig(LinkConfig, EvolvableConfig):
    geometry: AllEvolvableGeometryConfigs


class EvolvableBaseConfig(EvolvableLinkConfig, BaseConfig):
    pass


class EvolvableFingerAttachmentConfig(FingerAttachmentConfig, EvolvableConfig):
    _genes = {
        "angle": Gene(mutation_std=0.1, min_val=-math.pi, max_val=math.pi),
        "radius": Gene(mutation_std=0.02, min_val=0.02, max_val=0.2),
        "yaw_offset": Gene(mutation_std=0.1, min_val=-math.pi, max_val=math.pi),
    }


class EvolvableFingerConfig(FingerConfig, EvolvableConfig):
    attachment: EvolvableFingerAttachmentConfig
    segments: Tuple[EvolvableLinkConfig, ...]

    _genes = {
        "segments": GeneList(
            structure=GeneList.Structure.CHAIN,
            min_len=1,
            max_len=8,
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
            max_len=8,
            add_prob=0.1,
            remove_prob=0.05,
        ),
    }
