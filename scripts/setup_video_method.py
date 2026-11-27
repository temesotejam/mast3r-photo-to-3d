"""Install the CPU runtime for a selected video method, without downloading unrelated models."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_methods import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    args = parser.parse_args()
    scripts = {"mast3r": "setup_cpu.py", "monst3r": "setup_monst3r.py"}
    if args.method in scripts:
        subprocess.run([sys.executable, str(ROOT / "scripts" / scripts[args.method])], check=True)
        return
    pip = [sys.executable, "-m", "pip", "install"]
    packages = ["numpy==1.26.4", "opencv-python-headless==4.10.0.84", "scipy==1.13.1", "Pillow==11.3.0"]
    if args.method == "depth-anything-v2":
        subprocess.run([*pip, "--index-url", "https://download.pytorch.org/whl/cpu",
                        "torch==2.5.1", "torchvision==0.20.1"], check=True)
        packages += ["transformers==4.48.3", "huggingface-hub==0.28.1", "safetensors==0.5.3"]
    subprocess.run([*pip, *packages], check=True)


if __name__ == "__main__":
    main()
