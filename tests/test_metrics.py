import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from slotcourse.bootstrap import bootstrap_ci, paired_diff_ci
from slotcourse.metrics import ari, fg_ari, score_video


def _random_video(rng: np.random.Generator, t: int = 6, n_obj: int = 4) -> np.ndarray:
    gt = np.zeros((t, 16, 16), dtype=np.uint8)
    for k in range(1, n_obj + 1):
        r, c = rng.integers(0, 12, 2)
        for i in range(t):
            gt[i, r : r + 4, (c + i) % 12 : (c + i) % 12 + 4] = k
    return gt


@pytest.mark.parametrize("seed", range(5))
def test_ari_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 5, 300), rng.integers(0, 7, 300)
    assert ari(a, b) == pytest.approx(adjusted_rand_score(a, b))


def test_ari_degenerate_cases_match_sklearn() -> None:
    one = np.zeros(10, dtype=int)
    split = np.arange(10) % 3
    assert ari(one, one) == pytest.approx(adjusted_rand_score(one, one))
    assert ari(one, split) == pytest.approx(adjusted_rand_score(one, split))


def test_perfect_masks_score_100_even_when_permuted() -> None:
    gt = _random_video(np.random.default_rng(0))
    perm = np.array([7, 3, 9, 1, 5, 2, 8, 0, 4, 6])
    pred = perm[gt]  # same partition, different ids
    scores = score_video(gt, pred)
    for name, value in scores.items():
        assert value * 100 == pytest.approx(100.0), name


def test_identity_switch_hurts_video_metrics_only() -> None:
    gt = _random_video(np.random.default_rng(1), t=6, n_obj=2)
    pred = gt.copy().astype(np.int64)
    swap = pred[3:].copy()
    pred[3:][swap == 1], pred[3:][swap == 2] = 2, 1  # swap ids halfway through
    s = score_video(gt, pred)
    assert s["fg_ari"] == pytest.approx(1.0)
    assert s["mbo_f"] == pytest.approx(1.0)
    assert s["mbo_v"] < 0.9
    assert s["fg_ari_v"] < 0.9


def test_background_is_ignored_by_fg_ari() -> None:
    gt = np.array([0, 0, 0, 1, 1, 2, 2])
    pred = np.array([5, 6, 7, 1, 1, 2, 2])
    assert fg_ari(gt, pred) == pytest.approx(1.0)


def test_empty_foreground_is_nan() -> None:
    gt = np.zeros((2, 4, 4), dtype=np.uint8)
    s = score_video(gt, gt)
    assert np.isnan(s["fg_ari"]) and np.isnan(s["mbo_v"])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        score_video(np.zeros((2, 4, 4)), np.zeros((2, 4, 5)))


def test_bootstrap_interval_contains_mean_and_paired_diff() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 250)
    b = a + 0.05 + rng.normal(0, 0.01, 250)
    ci = bootstrap_ci(a)
    assert ci.lo < ci.mean < ci.hi
    diff = paired_diff_ci(a, b)
    assert diff.excludes_zero() and diff.mean == pytest.approx(0.05, abs=0.01)
    assert not paired_diff_ci(a, a + rng.normal(0, 0.01, 250)).excludes_zero()
