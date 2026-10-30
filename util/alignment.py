"""Helpers for verifying and ordering per-frame event files."""
import re


def natural_sort(names):
    """Sort 1.npz, 2.npz, 10.npz in numeric order."""
    return sorted(names, key=lambda name: [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ])
