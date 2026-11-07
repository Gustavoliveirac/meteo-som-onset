"""
run_supplementary.py - material outside the main text: SOM grids from 10x10 to
100x100, the K-means reference and their comparison with the 20x20 map.

    nohup python run_supplementary.py > /dev/null 2>&1 &

Needs the arrays of som/prepare.py and the 20x20 seed 42 map. Each step skips
what is already done. Output of every step in logs/run_supplementary.log.
"""
import subprocess
import sys
from datetime import datetime

import config

STEPS = [
    ["supplementary/kmeans.py"],
    ["som/train.py", "--grids", "10", "30", "40", "50", "60", "70", "80", "90", "100"],
    ["supplementary/compare.py"],
]


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(config.LOG_DIR / "run_supplementary.log", "a")
    for step in STEPS:
        line = f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] python {' '.join(step)}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        r = subprocess.run([sys.executable, str(config.ROOT / step[0])] + step[1:],
                           stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
        if r.returncode:
            print(f"failed, see {config.LOG_DIR / 'run_supplementary.log'}", flush=True)
            sys.exit(r.returncode)
    log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] done\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
