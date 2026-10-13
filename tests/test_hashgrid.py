import torch

from saga.model.embed import HashGrid


def test_dense_level_reproduces_linear_field():
    g = HashGrid(n_levels=1, table_size=10 ** 6, feat_dim=1, res_xyz=(4, 4), res_t=(3, 3))
    assert g.dense == [True] and g.size == [5 ** 3 * 4]
    ids = torch.arange(g.size[0])
    c = torch.stack([ids % 5, ids // 5 % 5, ids // 25 % 5, ids // 125], 1).float()
    coef = torch.tensor([0.3, -1.2, 0.7, 2.0])
    with torch.no_grad():
        g.table[:, 0] = (c / torch.tensor([4.0, 4, 4, 3]) * coef).sum(1)
    x = torch.rand(1000, 4)
    x[:10] = 1.0  # upper boundary maps into the last cell
    assert torch.allclose(g(x)[:, 0], (x * coef).sum(1), atol=1e-5)


def test_hashed_levels_get_gradient():
    torch.manual_seed(0)
    g = HashGrid(n_levels=4, table_size=2 ** 10, feat_dim=2, res_xyz=(8, 64), res_t=(2, 16))
    assert not all(g.dense) and g.table.shape[0] == sum(g.size)
    out = g(torch.rand(500, 4))
    assert out.shape == (500, 8)
    out.sum().backward()
    for l in range(4):
        assert g.table.grad[g.offsets[l]:g.offsets[l + 1]].abs().sum() > 0
