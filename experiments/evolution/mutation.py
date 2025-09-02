import os
import argparse
import numpy as np

from evodex.evolution.core import mutate
from evodex.evolution.robot import EvolvableRobotConfig
from experiments.utils import load_config, save_config


def generate_mutations(
    config_path: str, output_dir: str, num_mutations: int, seed: int
):
    """
    Loads a base robot configuration, applies mutations a specified number of times,
    and saves each new configuration to a file in the output directory.
    """
    np.random.seed(seed)

    print(f"Loading base robot configuration from: {config_path}")
    base_config_dict = load_config(config_path)
    base_robot_config = EvolvableRobotConfig(**base_config_dict)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {num_mutations} mutated configurations to: {output_dir}")

    for i in range(num_mutations):
        # Always mutate from a clean copy of the original config
        mutated_config = mutate(base_robot_config)

        # Use zfill for zero-padded filenames (e.g., mutant_001.yaml) for better sorting
        output_filename = f"mutant_{str(i).zfill(3)}.yaml"
        output_path = os.path.join(output_dir, output_filename)

        save_config(mutated_config.model_dump(), output_path)
        print(f"  - Saved {output_path}")

    print("\n✅ Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/robot/base_robot.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mutation/",
        help="Path to the output config file",
    )
    parser.add_argument(
        "--num-mutations",
        "-n",
        default=1,
        type=int,
        help="Number of mutations to apply",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    config_path = args.config
    output_dir = args.output_dir
    n = args.num_mutations
    seed = args.seed

    generate_mutations(config_path, output_dir, n, seed)
