import pytest
import torch

from slotcourse.models.video import (
    ModelConfig,
    SlotVideoModel,
    random_frame_mask,
    slot_similarity_penalty,
)


@pytest.mark.parametrize("variant", ["L1", "L2", "L3", "L4"])
def test_forward_shapes_and_backward(variant: str) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig.for_variant(variant, num_slots=4, hidden=16, slot_dim=16, decoder_hidden=16, gru_hidden=8)
    model = SlotVideoModel(cfg)
    video = torch.rand(2, 3, 64, 64, 3)
    out = model(video)
    assert out.recon.shape == (2, 3, 1024, 12)
    assert out.masks.shape == (2, 3, 1024, 4)
    torch.testing.assert_close(out.masks.sum(-1), torch.ones(2, 3, 1024))
    loss, logs = model.loss(video)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert ("slot_sim" in logs) == (variant == "L4")
    assert model.segment(video).shape == (2, 3, 1024)


def test_variant_defaults_are_cumulative() -> None:
    l1, l2, l3, l4 = (ModelConfig.for_variant(v) for v in ("L1", "L2", "L3", "L4"))
    assert not l1.recurrent and l2.recurrent and not l2.temporal and l3.temporal
    assert l1.mask_ratio == l2.mask_ratio == 0 and l3.mask_ratio > 0
    assert l3.slot_penalty == 0 and l4.slot_penalty > 0
    with pytest.raises(ValueError):
        ModelConfig.for_variant("L9")


def test_frame_mask_leaves_a_visible_frame() -> None:
    m = random_frame_mask(100, 6, 0.99, torch.device("cpu"))
    assert (m.sum(1) == 5).all()
    assert not random_frame_mask(3, 6, 0.0, torch.device("cpu")).any()


def test_slot_penalty_zero_for_orthogonal_slots() -> None:
    eye = torch.eye(4).unsqueeze(0)
    assert slot_similarity_penalty(eye).item() == pytest.approx(0.0)
    same = torch.ones(1, 4, 4)
    assert slot_similarity_penalty(same).item() == pytest.approx(1.0)


def test_recurrent_model_handles_longer_clips_than_training() -> None:
    model = SlotVideoModel(ModelConfig.for_variant("L3", num_slots=3, hidden=8, slot_dim=8, decoder_hidden=8))
    assert model.segment(torch.rand(1, 24, 64, 64, 3)).shape == (1, 24, 1024)
