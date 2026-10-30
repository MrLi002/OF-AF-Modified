#!/usr/bin/env python3
"""Check assumptions behind frame-interval GOPRO/BS-ERGB event loading.

Usage:
    python check_events.py /path/to/scene
    python check_events.py /path/to/bsergb/scene --bsergb --skip 3

Checks cannot prove camera-frame synchronization without frame timestamps.
"""
import argparse
from pathlib import Path
import numpy as np
from util.alignment import natural_sort


def inspect_scene(scene, bsergb=False, skip=31, bins=128):
    scene = Path(scene)
    images = list((scene / "images").glob("*.png"))
    events = list((scene / "events").glob("*.npz"))
    print(f"Scene: {scene}\nImages: {len(images)} | event intervals: {len(events)}")
    bad = []
    if len(images) < skip + 2:
        bad.append(f"not enough images for skip={skip}")
    if len(events) != len(images) - 1:
        bad.append(f"expected images - 1 events files; got {len(events)} vs {len(images)} images")
    lexical = sorted(p.name for p in events)
    numeric = natural_sort([p.name for p in events])
    if lexical != numeric:
        bad.append("event names are not zero-padded: lexicographic order differs from numeric order")
    events = [scene / "events" / name for name in numeric]
    time_key = "timestamp" if bsergb else "t"
    pol_key = "polarity" if bsergb else "p"
    bounds = []
    sampling = events[:3] + events[max(3, len(events) // 2):max(3, len(events) // 2) + 2] + events[-3:]
    for f in dict.fromkeys(sampling):
        with np.load(f, allow_pickle=False) as arr:
            if not all(k in arr for k in (time_key, pol_key)):
                bad.append(f"{f.name}: missing expected keys {time_key}/{pol_key}")
                continue
            t = np.asarray(arr[time_key]).reshape(-1)
            p = np.asarray(arr[pol_key]).reshape(-1)
            if t.size == 0:
                print(f"[warn] {f.name}: empty event interval (can be valid for static scenes)")
                continue
            unique_pol = np.unique(p)
            if not np.all(np.isin(unique_pol, (-1, 0, 1))):
                bad.append(f"{f.name}: unsupported polarity values {unique_pol[:10]}")
            elif 0 in unique_pol:
                print(f"[info] {f.name}: 0/1 polarity; loader converts to -1/+1")
            if len(np.unique(t)) < max(2, bins // (skip + 1)):
                bad.append(f"{f.name}: unusually few distinct timestamps; verify 8x V2E generation")
            if np.any(np.diff(t) < 0):
                bad.append(f"{f.name}: timestamps not sorted")
            print(f"{f.name}: time=[{float(t.min()):.6g},{float(t.max()):.6g}] "
                  f"unique_t={np.unique(t).size}, polarity={unique_pol.tolist()}")
    # Inspect every file's first/last timestamp, without concatenating huge arrays.
    for f in events:
        with np.load(f, allow_pickle=False) as arr:
            if time_key not in arr:
                continue
            t = np.asarray(arr[time_key]).reshape(-1)
            bounds.append((f.name, float(t.min()), float(t.max())) if t.size else None)
    for left, right in zip(bounds, bounds[1:]):
        if left is not None and right is not None and right[1] < left[2]:
            bad.append(f"adjacent event intervals overlap: {left[0]} -> {right[0]}")
            if len(bad) > 25:
                break
    if bad:
        for item in bad:
            print("[!] " + item)
        print(f"Found {len(bad)} potential problems. No training should start until checked.")
        return 1
    print("No structural problems detected; exact event/frame alignment is still unverified.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--bsergb", action="store_true")
    parser.add_argument("--skip", type=int, default=31)
    parser.add_argument("--bins", type=int, default=128)
    args = parser.parse_args()
    raise SystemExit(inspect_scene(args.scene, args.bsergb, args.skip, args.bins))
