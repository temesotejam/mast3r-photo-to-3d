"""Prepare the official MASt3R source tree and its CPU inference environment."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "mast3r"
REVISION = "f5209afc300cec36239a7ac992263f36847bbba0"


def checkout_source():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", "https://github.com/naver/mast3r.git", str(SOURCE)],
            check=True,
        )
        subprocess.run(["git", "checkout", "--detach", REVISION], cwd=SOURCE, check=True)
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip()
        if revision != REVISION:
            raise SystemExit(
                f"{SOURCE} is at {revision}. Check out {REVISION} in this directory "
                "or move the existing checkout before running setup again."
            )
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=SOURCE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-only", action="store_true", help="skip package installation")
    args = parser.parse_args()
    checkout_source()
    if args.checkout_only:
        return
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run(
        [*pip, "--index-url", "https://download.pytorch.org/whl/cpu",
         "torch==2.5.1", "torchvision==0.20.1"],
        check=True,
    )
    subprocess.run([*pip, "-r", str(ROOT / "scripts" / "requirements-inference.txt")], check=True)


if __name__ == "__main__":
    main()
