import os
from pathlib import Path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Finds the root directory of the project by looking for a marker file.

    Args:
        marker (str): The filename to look for that indicates the project root.

    Returns:
        Path: The path to the project root directory, or None if not found.
    """
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"Project root with marker '{marker}' not found above {__file__}."
    )


def get_project_root(marker: str = "pyproject.toml") -> Path:
    """Returns the project root directory, caching the result for future calls.

    Args:
        marker (str): The filename to look for that indicates the project root.

    Returns:
        Path: The path to the project root directory.
    """
    env_root = os.environ.get("EVODEX_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return find_project_root(marker=marker)


PROJECT_ROOT = get_project_root()
GENERATED_DIR = os.environ.get("GENERATED_DIR", PROJECT_ROOT / "generated")
LOGS_DIR = os.environ.get("LOGS_DIR", PROJECT_ROOT / "logs")
CONFIGS_DIR = os.environ.get("CONFIGS_DIR", PROJECT_ROOT / "configs")
