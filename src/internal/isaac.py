import os
import sys
import subprocess
from pathlib import Path


def load_env(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    load_env(env_file)

    isaaclab_home = os.environ.get("ISAACLAB_HOME")
    if not isaaclab_home:
        sys.exit("❌ ISAACLAB_HOME not set in .env")

    if os.name == "nt":
        isaac_cmd = str(Path(isaaclab_home) / "isaaclab.bat")
    else:
        isaac_cmd = str(Path(isaaclab_home) / "isaaclab.sh")

    cmd = [isaac_cmd] + sys.argv[1:]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
