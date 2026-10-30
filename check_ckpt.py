#!/usr/bin/env python3
"""Diagnose Adam momentum magnitudes in the forward-flow heads vs other layers.

These values are optimizer first moments, NOT instantaneous gradients.
The relative scale is a heuristic only, not a proof of correct learning.
"""
import argparse
import torch


def is_forward_flow_parameter(name):
    return (name.startswith(("bidir_flow0.", "bidir_flow1."))
            and (".flow." in name or ".gru." in name))


def analyze_adam_moments(named_parameters, optimizer_state):
    """Map optimizer parameter IDs to names and summarize RMS(exp_avg)."""
    names = list(named_parameters)
    groups = optimizer_state["param_groups"]
    flat_ids = [param_id for group in groups for param_id in group["params"]]
    if len(flat_ids) != len(names):
        raise ValueError(
            f"Saved optimizer has {len(flat_ids)} parameters; "
            f"model has {len(names)}. Cannot safely match optimizer states by order."
        )
    forward, other = [], []
    rows = []
    for (name, _), param_id in zip(names, flat_ids):
        state = optimizer_state["state"].get(param_id, {})
        momentum = state.get("exp_avg")
        if momentum is None:
            continue
        rms = float(momentum.detach().float().square().mean().sqrt())
        if not torch.isfinite(torch.tensor(rms)):
            continue
        rows.append((name, rms))
        if rms > 0:
            (forward if is_forward_flow_parameter(name) else other).append(rms)
    def median(values):
        return float(torch.tensor(values, dtype=torch.float64).median()) if values else None
    fw, ref = median(forward), median(other)
    ratio = fw / ref if fw is not None and ref not in (None, 0) else None
    return rows, fw, ref, ratio


def inspect(path):
    from types import SimpleNamespace
    from models.model import MyModel

    checkpoint = torch.load(path, map_location="cpu")
    print("Saved fixes:", checkpoint.get("preprocessing_flags", "(legacy checkpoint)"))
    opt = checkpoint.get("optimizer")
    if opt is None:
        print("No Adam optimizer state saved; cannot inspect optimizer moments.")
        return
    model_weights = {key.replace("module.", ""): value
                     for key, value in checkpoint["model"].items()}
    channels = model_weights["events_encoder.ct"].shape[1]
    # Parameter registration/order is independent of the temporal step count.
    model = MyModel(SimpleNamespace(voxel_bins=channels, nb_of_flow=16))
    named = list(model.named_parameters())
    rows, fw, ref, ratio = analyze_adam_moments(named, opt)
    for name, rms in rows:
        if is_forward_flow_parameter(name):
            print(f"{name:65s} Adam exp_avg RMS={rms:.3e}")
    if ratio is None:
        print("No usable forward/reference moment ratio; missing/zero Adam states.")
    else:
        print(f"Forward-flow median Adam moment: {fw:.3e}")
        print(f"Other-layer median Adam moment:  {ref:.3e}")
        print(f"Forward / other median ratio:    {ratio:.3e}")
        print("Rule of thumb only: a ratio above 1e-3 suggests the forward head")
        print("is receiving a meaningful update signal; verify on GPU with actual")
        print("gradient norms and flow/PSNR diagnostics before drawing conclusions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    inspect(args.checkpoint)
