"""Run every experiment in dependency order and write each part's outputs/ folder.

python run_all.py              # everything, about 8 minutes
python run_all.py e4 e5        # only the named steps, still in dependency order

E1 writes the shared DAGs, E2 and E3 read them, E3 writes the fitted CPTs, and the
VE check, E4 and E5 read those CPTs. Running a later step alone reuses the committed
upstream outputs.
"""
import argparse
import importlib
from time import perf_counter

from inference import check_setup, run_e4
from parameter_learning import run_part_c
from prediction import run_e5
from structure import run_e1

run_e2 = importlib.import_module("complete-case_MLE_and_EM.run_e2")

STEPS = {
    "e1": ("E1 structure learning", run_e1.run),
    "e2": ("E2 missing data", run_e2.run),
    "e3": ("CPT fitting and E3 latent variable", run_part_c.run),
    "check": ("VE check against pgmpy", check_setup.main),
    "e4": ("E4 exact vs approximate inference", run_e4.run),
    "e5": ("E5 price-tier prediction", run_e5.run),
}


def main(selected):
    for key in selected:
        title, step = STEPS[key]
        print(f"\n=== {title}", flush=True)
        start = perf_counter()
        step()
        print(f"=== {title} done in {perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", metavar="step", help=f"any of {', '.join(STEPS)} (default: all)")
    args = parser.parse_args()
    unknown = sorted(set(args.steps) - set(STEPS))
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}")
    main([key for key in STEPS if key in args.steps or not args.steps])
