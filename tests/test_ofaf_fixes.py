import numpy as np
import torch

from models.BiFEB import Query
from models.blocks import BackwardWarp
from util.event import load_events, EventSequence
from util.alignment import natural_sort, validate_interval_file_count
from models.blocks import splat_valid_mask
from check_ckpt import analyze_adam_moments
import pytest
from util.voxelization import to_voxel_grid


def test_natural_sort():
    assert natural_sort(["10.npz", "2.npz", "1.npz"]) == ["1.npz", "2.npz", "10.npz"]


def test_query_legacy_and_corrected():
    a = torch.tensor([[[[4.0]]]])
    b = torch.tensor([[[[12.0]]]])
    alpha = torch.tensor([[[[0.25]]]])
    legacy_flow, _ = Query(32, fix_query=False)(None, a, b, a, b, alpha)
    fixed_flow, _ = Query(32, fix_query=True)(None, a, b, a, b, alpha)
    assert torch.allclose(legacy_flow, torch.tensor([[[[10.0]]]]))
    assert torch.allclose(fixed_flow, torch.tensor([[[[6.0]]]]))


def test_zero_flow_warp():
    torch.manual_seed(1)
    img = torch.rand(1, 3, 32, 48)
    flow = torch.zeros(1, 2, 32, 48)
    fixed = BackwardWarp(fix_half_pixel=True)(img, flow)
    legacy = BackwardWarp(fix_half_pixel=False)(img, flow)
    assert torch.allclose(fixed, img, atol=2e-6)
    assert (legacy - img).abs().max() > 1e-2


def test_polarity_conversion(tmp_path):
    path = tmp_path / "events.npz"
    np.savez(path, x=np.array([0, 1]), y=np.array([0, 1]),
             t=np.array([0., 1.]), p=np.array([0, 1]))
    result = load_events(path)
    assert result[:, 3].tolist() == [-1, 1]
    np.savez(path, x=np.array([0, 1]), y=np.array([0, 1]),
             t=np.array([0., 1.]), p=np.array([-1, 1]))
    result = load_events(path)
    assert result[:, 3].tolist() == [-1, 1]


def test_interval_voxel_alignment(tmp_path):
    paths = []
    for idx in range(2):
        path = tmp_path / f"{idx:04}.npz"
        np.savez(path, x=np.array([1, 2]), y=np.array([1, 2]),
                 t=np.array([10 + idx * 4, 11 + idx * 4]),
                 p=np.array([-1, 1]))
        paths.append(str(path))
    seq = EventSequence.from_frame_interval_files(paths, 4, 4)
    assert seq.start_time() == 0
    assert seq.end_time() == 2
    assert np.all(seq._features[:2, 2] < 1)
    assert np.all(seq._features[2:, 2] >= 1)
    vox = to_voxel_grid(seq, 8)
    assert torch.isfinite(vox).all()


def test_empty_interval(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, x=np.array([]), y=np.array([]),
             t=np.array([]), p=np.array([]))
    assert EventSequence.from_frame_interval_files([str(path)], 4, 4) is None


def test_crop_time_normalization_uses_full_file_timestamps(tmp_path):
    # The crop contains only events at 40% and 60% of the full interval.
    # An incorrect crop-first implementation would map them to 0% and 100%.
    path = tmp_path / "frame_interval.npz"
    np.savez(path,
             x=np.array([8, 1, 1, 8]),
             y=np.array([8, 1, 1, 8]),
             t=np.array([0.0, 40.0, 60.0, 100.0]),
             p=np.array([1, -1, 1, 1]))
    seq = EventSequence.from_frame_interval_files([str(path)], 4, 4,
                                                   size=[0, 0, 4, 4])
    assert seq is not None
    assert np.allclose(seq._features[:, 2], [0.4, 0.6])
    assert np.allclose(seq._features[:, :2], [[1, 1], [1, 1]])


def test_extra_trailing_events_allowed_and_missing_rejected():
    assert validate_interval_file_count(4, 3, "scene") == 3
    with pytest.warns(UserWarning, match="extra trailing"):
        assert validate_interval_file_count(4, 4, "scene") == 3
    with pytest.raises(ValueError, match="need at least images - 1"):
        validate_interval_file_count(4, 2, "scene")


def test_splat_valid_mask_has_safe_threshold_and_no_grad():
    occupancy = torch.tensor([0.0, 1e-4, 0.49, 0.5, 1.0],
                             requires_grad=True)
    mask = splat_valid_mask(occupancy)
    assert mask.tolist() == [0, 0, 0, 1, 1]
    assert not mask.requires_grad


def test_check_ckpt_moment_ratio():
    named = [
        ("bidir_flow0.gru.convz1.weight", torch.nn.Parameter(torch.zeros(1))),
        ("bidir_flow1.flow.0.weight", torch.nn.Parameter(torch.zeros(1))),
        ("events_encoder.ct", torch.nn.Parameter(torch.zeros(1))),
        ("biw.conv_out.weight", torch.nn.Parameter(torch.zeros(1))),
    ]
    optimizer = {
        "param_groups": [{"params": [0, 1, 2, 3]}],
        "state": {
            0: {"exp_avg": torch.tensor([0.01])},
            1: {"exp_avg": torch.tensor([0.01])},
            2: {"exp_avg": torch.tensor([1.0])},
            3: {"exp_avg": torch.tensor([1.0])},
        }
    }
    rows, forward_median, reference_median, ratio = analyze_adam_moments(named, optimizer)
    assert len(rows) == 4
    assert abs(forward_median - 0.01) < 1e-6
    assert reference_median == 1.0
    assert abs(ratio - 0.01) < 1e-6
