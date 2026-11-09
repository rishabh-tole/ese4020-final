import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from slotcourse.baselines.hierarchical import hierarchical_frames
from slotcourse.baselines.kmeans import kmeans_frames, temporal_kmeans
from slotcourse.data.prepare import downsample_seg, downsample_video
from slotcourse.features import PCAFeatures, labels_to_pixels, patch_features, patchify


def test_patchify_numpy_and_torch_agree_and_invert() -> None:
    frames = np.random.default_rng(0).integers(0, 255, (2, 64, 64, 3), dtype=np.uint8)
    p_np = patchify(frames)
    p_t = patchify(torch.from_numpy(frames)).numpy()
    assert p_np.shape == (2, 1024, 12)
    np.testing.assert_array_equal(p_np, p_t)
    # patch 33 = row 1, col 1 -> pixels [2:4, 2:4]
    np.testing.assert_array_equal(p_np[0, 33].reshape(2, 2, 3), frames[0, 2:4, 2:4])


def test_labels_to_pixels_roundtrip() -> None:
    labels = np.arange(1024).reshape(1, 1024)
    px = labels_to_pixels(labels)
    assert px.shape == (1, 64, 64)
    assert px[0, 2, 3] == 33 and px[0, 63, 63] == 1023


def test_patch_features_include_positions() -> None:
    f = patch_features(np.zeros((64, 64, 3), dtype=np.uint8))
    assert f.shape == (1024, 14)
    assert f[0, -2:].tolist() == [0.0, 0.0] and f[-1, -2:].tolist() == [1.0, 1.0]


def test_pca_whitens_and_orders_variance() -> None:
    feats = np.random.default_rng(0).normal(size=(5000, 14)).astype(np.float32)
    pca = PCAFeatures.fit(feats, 5)
    z = pca.transform(feats)
    assert z.shape == (5000, 5)
    assert np.all(np.diff(z.var(0)) <= 1e-6)
    with pytest.raises(ValueError):
        PCAFeatures.fit(feats, 20)


def test_downsampling_shapes_and_values() -> None:
    vid = np.full((24, 128, 128, 3), 200, dtype=np.uint8)
    seg = np.zeros((24, 128, 128, 1), dtype=np.uint8)
    seg[:, :64] = 3
    assert downsample_video(vid).shape == (24, 64, 64, 3) and downsample_video(vid).max() == 200
    s = downsample_seg(seg)
    assert s.shape == (24, 64, 64) and s[0, 0, 0] == 3 and s[0, 63, 0] == 0


def _blobs(b: int, seed: int = 0) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, (b, 300))
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    x = centers[labels] + rng.normal(0, 0.5, (b, 300, 2)).astype(np.float32)
    return torch.from_numpy(x), labels


def test_kmeans_recovers_separated_blobs() -> None:
    x, labels = _blobs(8)
    pred = kmeans_frames(x, 3).numpy()
    for i in range(8):
        assert adjusted_rand_score(labels[i], pred[i]) > 0.99


def test_temporal_kmeans_keeps_ids_across_frames() -> None:
    x, labels = _blobs(1)
    video = x.unsqueeze(0).repeat(1, 5, 1, 1) + torch.linspace(0, 1, 5)[None, :, None, None]
    pred = temporal_kmeans(video, 3).numpy()[0]
    assert all((pred[t] == pred[0]).all() for t in range(5))


def test_hierarchical_recovers_blobs() -> None:
    x, labels = _blobs(2)
    pred = hierarchical_frames(x.numpy(), 3, n_jobs=1)
    assert adjusted_rand_score(labels[0], pred[0]) > 0.99
