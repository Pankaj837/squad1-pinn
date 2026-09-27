"""Run the device benchmark suite (G1-G8) and write JSON.  Usage:  python scripts/run_gpu_suite.py [--smoke] [--out results.json]"""

import sys

from squad1.cli import main

if __name__ == "__main__":
    sys.exit(main(["gpu-suite", *sys.argv[1:]]))
