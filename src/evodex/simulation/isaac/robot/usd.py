import yaml
import os

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def convert_urdf_to_usd(urdf_path: str, output_path: str) -> UrdfConverter:
    """
    Converts a URDF file to USD format using Isaac Sim's command-line tool.

    Args:
        urdf_path (str): The path to the input URDF file.
        usd_path (str): The path where the output USD file will be saved.
    """
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)

    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    output_dir = os.path.dirname(output_path)
    output_file_name = os.path.basename(output_path)

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name=output_file_name,
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=50.0,
                damping=1.0,
            ),
            target_type="position",
            drive_type="acceleration",
        ),
    )

    urdf_converter = UrdfConverter(urdf_converter_cfg)
    return urdf_converter
