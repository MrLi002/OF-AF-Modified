import numpy as np
import torch

from models.BiFEB import Query
from models.blocks import BackwardWarp
from util.event import load_events, EventSequence
from util.alignment import natural_sort
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
