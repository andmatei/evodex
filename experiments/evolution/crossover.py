import os
import argparse
import numpy as np

# Assuming your core and utils are in the correct paths
from evodex.evolution.core import crossover
from evodex.evolution.robot import EvolvableRobotConfig
from experiments.utils import load_config, save_config


def generate_crossover(
    config_path_a: str, config_path_b: str, output_dir: str, n: int, seed: int
):
    """
    Loads two parent robot configurations, applies crossover a specified number of times,
    and saves each new child configuration to a file in the output directory.
    """
    np.random.seed(seed)

    # 1. Load the two parent configurations
    print(f"Loading Parent A from: {config_path_a}")
    parent_a_dict = load_config(config_path_a)
    parent_a = EvolvableRobotConfig(**parent_a_dict)

    print(f"Loading Parent B from: {config_path_b}")
    parent_b_dict = load_config(config_path_b)
    parent_b = EvolvableRobotConfig(**parent_b_dict)

    # 2. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {n} children and saving to: {output_dir}")

    # 3. Loop to generate and save children
    for i in range(n):
        # Apply crossover between the two parents
        child = crossover(parent_a, parent_b)

        # Save the child config to a new file
        output_filename = f"child_{str(i).zfill(3)}.yaml"
        output_path = os.path.join(output_dir, output_filename)
        save_config(child.model_dump(), output_path)
        print(f"  - Saved {output_path}")

    print("\n✅ Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate robot configurations using crossover."
    )
    parser.add_argument(
        "--config-a",
        type=str,
        required=True,
        help="Path to the configuration file for Parent A.",
    )
    parser.add_argument(
        "--config-b",
        type=str,
        required=True,
        help="Path to the configuration file for Parent B.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/crossover/",
        help="Directory to save the output configuration files.",
    )
    parser.add_argument(
        "--num-crossover",
        "-n",
        default=10,
        type=int,
        help="Number of crossover operations to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    generate_crossover(
        config_path_a=args.config_a,
        config_path_b=args.config_b,
        output_dir=args.output_dir,
        n=args.num_crossover,
        seed=args.seed,
    )
