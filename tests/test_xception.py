import pytest
import torch
from torch import nn

from eotorch.models.xception import SeparableConv2d, XceptionS2, XceptionSDB

SMALL = dict(num_sepconv_blocks=2, num_sepconv_filters=32)


@pytest.mark.parametrize(
    "returns, n_outputs",
    [("targets", 1), ("variances_exp", 2), ("variances", 2), ("var_from_second_moments", 3)],
)
@pytest.mark.parametrize("out_channels", [1, 3])
def test_forward_shapes(returns, n_outputs, out_channels):
    """Each return mode yields the expected number of full-resolution outputs."""
    model = XceptionSDB(in_channels=12, out_channels=out_channels, returns=returns, **SMALL)
    out = model(torch.randn(2, 12, 16, 16))
    out = out if isinstance(out, tuple) else (out,)

    assert len(out) == n_outputs
    assert all(o.shape == (2, out_channels, 16, 16) for o in out)


@pytest.mark.parametrize("returns", ["variances_exp", "variances"])
def test_variances_are_non_negative(returns):
    model = XceptionSDB(in_channels=4, returns=returns, **SMALL)
    _, variances = model(torch.randn(2, 4, 8, 8))
    assert (variances >= 0).all()


def test_min_var_shifts_variances():
    model = XceptionSDB(in_channels=4, returns="variances", min_var=0.5, **SMALL)
    _, variances = model(torch.randn(2, 4, 8, 8))
    assert (variances >= 0.5).all()


def test_long_skip_changes_output():
    torch.manual_seed(0)
    plain = XceptionSDB(in_channels=4, **SMALL).eval()
    skip = XceptionSDB(in_channels=4, long_skip=True, **SMALL).eval()
    skip.load_state_dict(plain.state_dict())
    x = torch.randn(1, 4, 8, 8)
    assert not torch.allclose(plain(x), skip(x))


def test_architecture_mirrors_reference():
    """Entry block widths, bias-free separable convs and head names follow XceptionS2."""
    model = XceptionSDB(in_channels=12, **SMALL)

    assert model.entry_block.filters == [128, 256, 32]
    assert len(model.sepconv_blocks) == 2
    sepconvs = [m for m in model.modules() if isinstance(m, SeparableConv2d)]
    assert sepconvs and all(
        m.depthwise.bias is None and m.pointwise.bias is None for m in sepconvs
    )
    for head in ("predictions", "variances", "second_moments"):
        assert isinstance(getattr(model, head), nn.Conv2d)
    assert XceptionS2 is XceptionSDB


def test_used_parameters_all_receive_gradients():
    """Identity shortcuts must not leave unused projection weights (DDP-safe)."""
    model = XceptionSDB(in_channels=12, returns="variances_exp", long_skip=True, **SMALL)
    mean, var = model(torch.randn(2, 12, 8, 8))
    (mean.sum() + var.sum()).backward()

    no_grad = {n for n, p in model.named_parameters() if p.grad is None}
    # Only the head that this return mode doesn't use.
    assert no_grad == {"second_moments.weight", "second_moments.bias"}


def test_freeze_features_keeps_last_layers_trainable():
    model = XceptionSDB(in_channels=4, freeze_features=True, freeze_last_var=True, **SMALL)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"predictions.weight", "predictions.bias"}


def test_manual_init_zeroes_biases():
    model = XceptionSDB(in_channels=4, manual_init=True, **SMALL)
    assert torch.count_nonzero(model.predictions.bias) == 0


def test_loads_reference_checkpoint_layout(tmp_path):
    """Reference checkpoints carry unused shortcut weights in every sepconv block; they must load."""
    source = XceptionSDB(in_channels=4, **SMALL)
    state = source.state_dict()
    for i in range(SMALL["num_sepconv_blocks"]):
        n = SMALL["num_sepconv_filters"]
        prefix = f"sepconv_blocks.{i}."
        state[prefix + "conv_shortcut.weight"] = torch.randn(n, n, 1, 1)
        state[prefix + "conv_shortcut.bias"] = torch.randn(n)
        for k, v in nn.BatchNorm2d(n).state_dict().items():
            state[prefix + "bn_shortcut." + k] = v
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": state}, ckpt)

    loaded = XceptionSDB(in_channels=4, model_weights_path=ckpt, **SMALL)
    x = torch.randn(1, 4, 8, 8)
    assert torch.allclose(source.eval()(x), loaded.eval()(x))


@pytest.mark.parametrize(
    "kwargs", [{"returns": "unknown"}, {"var_activation": "softmax"}]
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(ValueError):
        XceptionSDB(in_channels=4, **SMALL, **kwargs)
