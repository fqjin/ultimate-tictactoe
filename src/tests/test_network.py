import pytest
import torch
from network import UTTTNet


def test_UTTTNet_shape():
    x = torch.randn((1, 8, 9, 9))
    m = UTTTNet()
    with torch.no_grad():
        p, v = m(x)

    assert p.shape == (1, 9, 9)
    assert p.exp().sum().item() == pytest.approx(1.0, abs=1e-6)
    assert v.shape == (1, 1)
    assert -1.0 <= v <= 1.0


def test_UTTTNet_changes():
    x = torch.randn((1, 8, 9, 9))
    y_p = torch.randn((1, 9, 9))
    y_v = torch.randn((1, 1))
    m = UTTTNet()

    params = [p for p in m.named_parameters() if p[1].requires_grad]
    old_params = [(name, p.clone()) for (name, p) in params]
    m.train()
    optimizer = torch.optim.SGD(m.parameters(), lr=1.0)
    optimizer.zero_grad()
    p, v = m(x)
    loss = torch.nn.MSELoss()(v, y_v) - torch.mean(y_p * p)
    loss.backward()
    optimizer.step()

    for (name, p0), (_, p1) in zip(old_params, params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            raise AssertionError(name + ' did not change')
