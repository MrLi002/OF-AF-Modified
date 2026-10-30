"""Helpers for verifying and ordering per-frame event files."""
import re


def natural_sort(names):
    """Sort 1.npz, 2.npz, 10.npz in numeric order."""
    return sorted(names, key=lambda name: [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ])

import warnings


def validate_interval_file_count(image_count, event_count, scene):
    """Ensure enough adjacent-frame event intervals, ignoring *trailing* extras.

    This only checks counts. It cannot establish the first-file offset or
    prove timestamp alignment with video frames.
    """
    required = max(0, image_count - 1)
    if event_count < required:
        raise ValueError(
            f"{scene}: need at least images - 1 event files "
            f"({required} needed for {image_count} images, found {event_count})"
        )
    if event_count > required:
        warnings.warn(
            f"{scene}: ignoring {event_count - required} extra trailing event file(s); "
            "verify that event file 0 corresponds to image frame 0 -> 1",
            UserWarning,
            stacklevel=2,
        )
    return required
