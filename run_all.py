"""
run_all.py - runs the whole sequence, from the BDMEP files to the figures.

    python run_all.py
    python run_all.py --from analysis/coverage.py

Each step skips what is already done (parquets, trained maps). The output of
every step goes to logs/run_all.log.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config

STEPS = [
    ["data/build_features.py"],
    ["som/prepare.py"],
    ["som/train.py"],
    ["som/train.py", "--seeds", "43", "44", "45", "46", "47"],
    ["analysis/coverage.py"],
    ["analysis/unit_stats.py"],
    ["analysis/text_numbers.py"],
    ["analysis/regional.py"],
    ["analysis/states.py"],
    ["analysis/refined_groups.py"],
    ["analysis/groups.py"],
    ["analysis/seed_groups.py"],
    ["analysis/group_profiles.py"],
    ["analysis/group_conditions.py"],
    ["figures/make_figures.py"],
]


def main():
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(config.LOG_DIR / "run_all.log", "a")
    active = start is None
    for step in STEPS:
        active = active or step[0] == start
        if not active:
            continue
        line = f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] python {' '.join(step)}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        r = subprocess.run([sys.executable, str(config.ROOT / step[0])] + step[1:],
                           stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
        if r.returncode:
            print(f"failed, see {config.LOG_DIR / 'run_all.log'}", flush=True)
            sys.exit(r.returncode)
    log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] done\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
