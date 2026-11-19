import pytest
import torch
from torch import nn

from eotorch.data import RegressionTask
from eotorch.data.tasks import create_model
from eotorch.models.xception import SeparableConv2d, XceptionSDB

SMALL = dict(num_sepconv_blocks=2, num_sepconv_filters=32)


@pytest.mark.parametrize(
    "returns, n_outputs",
    [("targets", 1), ("variances_exp", 2), ("variances", 2), ("var_from_second_moments", 3)],
)
@pytest.mark.parametrize("num_classes", [1, 3])
def test_forward_shapes(returns, n_outputs, num_classes):
    """Each return mode yields the expected number of full-resolution outputs."""
    model = XceptionSDB(in_channels=12, num_classes=num_classes, returns=returns, **SMALL)
    out = model(torch.randn(2, 12, 16, 16))
    out = out if isinstance(out, tuple) else (out,)

    assert len(out) == n_outputs
    assert all(o.shape == (2, num_classes, 16, 16) for o in out)


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


def _with_aux(x: torch.Tensor, values: list[float]) -> torch.Tensor:
    """Append one constant plane per auxiliary value to a batch of images."""
    planes = torch.tensor(values, dtype=x.dtype)[None, :, None, None]
    return torch.cat([x, planes.expand(x.shape[0], -1, *x.shape[2:])], dim=1)


@pytest.mark.parametrize("positions", [("entry",), ("head",), ("entry", "head")])
def test_aux_output_shapes(positions):
    model = XceptionSDB(
        in_channels=6, aux_channels=2, aux_film_positions=positions, returns="variances_exp", **SMALL
    )
    mean, var = model(_with_aux(torch.randn(2, 4, 8, 8), [0.3, -1.0]))

    assert model.entry_block.in_channels == 4
    assert mean.shape == var.shape == (2, 1, 8, 8)


def test_aux_conditioning_is_identity_at_init():
    """Zero-initialised FiLM layers leave the features unchanged, whatever the aux values."""
    torch.manual_seed(0)
    plain = XceptionSDB(in_channels=4, **SMALL).eval()
    with_aux = XceptionSDB(in_channels=6, aux_channels=2, aux_filters=8, **SMALL).eval()
    with_aux.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 4, 8, 8)

    torch.testing.assert_close(with_aux(_with_aux(x, [5.0, -3.0])), plain(x))


def test_aux_values_change_the_output_after_training_step():
    model = XceptionSDB(in_channels=5, aux_channels=1, aux_filters=8, **SMALL)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 4, 8, 8)
    loss = model(_with_aux(x, [1.0])).pow(2).mean()
    loss.backward()
    optimizer.step()

    model.eval()
    assert not torch.allclose(model(_with_aux(x, [1.0])), model(_with_aux(x, [-1.0])))


def test_patch_aux_values_use_median_and_mean_fallback():
    aux = torch.full((2, 2, 8, 8), 3.0)
    aux[0, 0, :2, :2] = 0.0  # corner padded by a rotation
    aux[1, 1] = torch.nan  # missing value

    values = XceptionSDB.patch_aux_values(aux)

    torch.testing.assert_close(values, torch.tensor([[3.0, 3.0], [3.0, 0.0]]))


def test_missing_aux_equals_training_mean():
    model = XceptionSDB(in_channels=5, aux_channels=1, aux_filters=8, **SMALL).eval()
    for layer in model.aux_conditioning.scales.values():
        nn.init.normal_(layer.weight)
    x = torch.randn(1, 4, 8, 8)

    torch.testing.assert_close(model(_with_aux(x, [float("nan")])), model(_with_aux(x, [0.0])))


def test_aux_module_receives_gradients_and_stays_trainable_when_frozen():
    model = XceptionSDB(in_channels=5, aux_channels=1, aux_filters=8, **SMALL)
    model(_with_aux(torch.randn(2, 4, 8, 8), [0.5])).sum().backward()
    assert all(p.grad is not None for p in model.aux_conditioning.parameters())

    frozen = XceptionSDB(in_channels=5, aux_channels=1, freeze_features=True, **SMALL)
    assert all(p.requires_grad for p in frozen.aux_conditioning.parameters())
    assert not any(p.requires_grad for p in frozen.entry_block.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aux_channels": 4},
        {"aux_channels": -1},
        {"aux_channels": 1, "aux_film_positions": ()},
        {"aux_channels": 1, "aux_film_positions": ("middle",)},
    ],
)
def test_invalid_aux_options_raise(kwargs):
    with pytest.raises(ValueError):
        XceptionSDB(in_channels=4, **SMALL, **kwargs)


def test_regression_task_trains_xception_with_aux():
    task = RegressionTask(
        in_channels=5,
        model="xception",
        loss="gaussian_nll",
        model_kwargs={**SMALL, "aux_channels": 2, "aux_filters": 8},
    )
    x = _with_aux(torch.randn(2, 3, 16, 16), [0.2, -0.4])

    loss = task.training_step({"image": x, "mask": torch.randn(2, 1, 16, 16)}, 0)

    assert torch.isfinite(loss)
    loss.backward()


def test_create_model_builds_xception():
    """The registry name works with create_model, which passes num_classes."""
    model = create_model(
        "xception", in_channels=4, out_channels=2, num_filters=8, model_kwargs=SMALL
    )
    assert isinstance(model, XceptionSDB)
    assert model(torch.randn(2, 4, 8, 8)).shape == (2, 2, 8, 8)


def test_regression_task_trains_xception():
    task = RegressionTask(in_channels=3, model="xception", model_kwargs=SMALL)
    x = torch.randn(2, 3, 16, 16)
    y = torch.randn(2, 1, 16, 16)

    loss = task.training_step({"image": x, "mask": y}, 0)

    assert torch.isfinite(loss)
    loss.backward()
