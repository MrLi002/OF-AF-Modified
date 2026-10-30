#!/usr/bin/env python3
"""Inspect relative Adam moment scales of OF-AF forward-flow parameters."""
import argparse
import torch
from models.model import MyModel


def inspect(path):
    checkpoint = torch.load(path, map_location="cpu")
    print("Saved fixes:", checkpoint.get("preprocessing_flags", "(legacy checkpoint)"))
    opt = checkpoint.get("optimizer")
    if opt is None:
        print("No Adam optimizer state saved; weight-only checkpoints cannot show gradient history.")
        return
    from types import SimpleNamespace
    model_weights = checkpoint["model"]
    channels = model_weights["events_encoder.ct"].shape[1]
    flow_steps = 16
    model = MyModel(SimpleNamespace(voxel_bins=channels, nb_of_flow=flow_steps))
    named = list(model.named_parameters())
    param_ids = opt["param_groups"][0]["params"]
    for idx, (name, param) in enumerate(named):
        if ".flow." not in name and ".gru." not in name:
            continue
        state = opt["state"].get(param_ids[idx], {})
        if "exp_avg" not in state:
            continue
        print(f"{name:65s} Adam exp_avg RMS={state['exp_avg'].float().square().mean().sqrt():.3e}")
    print("These are Adam running moments, not raw per-step gradients.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    inspect(args.checkpoint)
