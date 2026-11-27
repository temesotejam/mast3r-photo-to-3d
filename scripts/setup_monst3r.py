"""Prepare a CPU installation of MonST3R's dynamic-scene pairwise pointmap model."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "monst3r"
REVISION = "574cc77ad278bad582f470e5382624e01f8769a7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-only", action="store_true", help="skip package installation")
    args = parser.parse_args()
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/Junyi42/monst3r",
                        str(SOURCE)], check=True)
        subprocess.run(["git", "checkout", "--detach", REVISION], cwd=SOURCE, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected {SOURCE} at {REVISION}, found {revision}.")
    subprocess.run(["git", "submodule", "update", "--init", "croco"], cwd=SOURCE, check=True)
    if args.checkout_only:
        return
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run([*pip, "--index-url", "https://download.pytorch.org/whl/cpu",
                    "torch==2.5.1", "torchvision==0.20.1"], check=True)
    subprocess.run([*pip, "-r", str(ROOT / "scripts/requirements-inference.txt"), "evo==1.31.1"], check=True)


if __name__ == "__main__":
    main()
