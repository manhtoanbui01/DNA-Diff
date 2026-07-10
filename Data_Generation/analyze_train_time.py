"""
Estimate wall-clock TRAINING TIME per stage from TensorBoard event files.

Training was done with many restarts, so we:
  - gather every (wall_time, step) scalar point across all event files in a stage dir,
  - sort by wall_time,
  - ACTIVE time  = sum of consecutive dt where dt < --gap (bridges within-run logging
                   intervals, drops the idle gaps between separate runs),
  - SPAN time    = last_wall - first_wall (calendar span, includes idle),
  - report max step reached + throughput.

Usage:
    python analyze_train_time.py <stage_dir> [<stage_dir> ...] [--gap 600]
"""
import argparse
import glob
import os
from datetime import timedelta

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def hms(seconds):
    return str(timedelta(seconds=int(seconds)))


def collect_points(stage_dir):
    files = glob.glob(os.path.join(stage_dir, "**", "*tfevents*"), recursive=True)
    pts = []  # (wall_time, step)
    for f in files:
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                pts.append((e.wall_time, e.step))
    # dedup + sort by wall time
    pts = sorted(set(pts), key=lambda x: x[0])
    return pts


def analyze(stage_dir, gap):
    pts = collect_points(stage_dir)
    if len(pts) < 2:
        print(f"  {os.path.basename(stage_dir):55s} : no scalar timing data")
        return 0.0
    wall = [p[0] for p in pts]
    steps = [p[1] for p in pts]
    active = 0.0
    n_breaks = 0
    for i in range(1, len(wall)):
        dt = wall[i] - wall[i - 1]
        if dt < gap:
            active += dt
        else:
            n_breaks += 1
    span = wall[-1] - wall[0]
    max_step = max(steps)
    thr = max_step / active if active > 0 else 0
    print(f"  {os.path.basename(stage_dir):55s}")
    print(f"      max step reached : {max_step:,}")
    print(f"      ACTIVE compute   : {hms(active)}  ({active/3600:.2f} h)")
    print(f"      calendar span    : {hms(span)}  ({span/86400:.2f} days, {n_breaks} restart gaps)")
    print(f"      throughput       : {thr:.2f} steps/s  ({1/thr:.2f} s/step)" if thr else "")
    return active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--gap", type=float, default=600,
                    help="seconds; gaps larger than this are treated as idle (between-run) and excluded")
    args = ap.parse_args()
    total = 0.0
    print("=" * 70)
    for d in args.dirs:
        a = analyze(d, args.gap)
        total += a
        print("-" * 70)
    print(f"  SUM of ACTIVE compute over listed stages : {hms(total)}  ({total/3600:.2f} h)")
    print("=" * 70)


if __name__ == "__main__":
    main()
