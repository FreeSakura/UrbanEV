import pytest
torch=pytest.importorskip("torch")
from urbanev_forecast.models import build_model


@pytest.mark.parametrize("name",["seasonal_linear","seasonal_mlp","innovation_attention","level_attention","timexer"])
def test_multivariate_forecast_and_finite_gradients(name):
    torch.set_num_threads(2)
    model=build_model(name,168,12,4)
    x=torch.rand(2,168,4)
    y=model(x)
    assert y.shape==(2,12,4) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_residual_returns_correct_daily_anchor():
    model=build_model("seasonal_linear",48,3,1)
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.zero_()
    x=torch.arange(48.).reshape(1,48,1)
    assert torch.equal(model(x),torch.tensor([24.,25.,26.]).reshape(1,3,1))


def test_single_channel_attention_has_no_nan():
    model=build_model("innovation_attention",48,3,1)
    assert torch.isfinite(model(torch.rand(2,48,1))).all()
