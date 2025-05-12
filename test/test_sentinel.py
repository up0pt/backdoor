import os
import pytest
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from sentinel import (
    cosine_similarity,
    compute_bootstrap_loss,
    map_loss_distance,
    normalize_model,
    sentinel_aggregation
)

# テスト用の簡易モデル
class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


def test_cosine_similarity_identical():
    params_m = {'w': torch.tensor([1., 0.])}
    params_p = {'w': torch.tensor([1., 0.])}
    assert cosine_similarity(params_p, params_m) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    params_m = {'w': torch.tensor([1., 0.])}
    params_p = {'w': torch.tensor([0., 1.])}
    assert cosine_similarity(params_p, params_m) == pytest.approx(0.0)


def test_compute_bootstrap_loss_zero():
    # IdentityModel に対し、x=y のデータセットなら損失は 0
    model = IdentityModel()
    x = torch.tensor([[1.], [2.]], dtype=torch.float32)
    y = x.clone()
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    loss_fn = nn.MSELoss()
    loss = compute_bootstrap_loss(model, loader, loss_fn)
    assert loss == pytest.approx(0.0)


def test_map_loss_distance_equal_histories():
    history_i = [1.0, 1.0, 1.0]
    history_j = [1.0, 1.0, 1.0]
    # 平均差は 0 -> w = exp(0) = 1
    w = map_loss_distance(history_i, history_j, tau_L=0.5)
    assert w == pytest.approx(1.0)


def test_map_loss_distance_below_threshold():
    history_i = [0.1]
    history_j = [1.0]
    # 大きな差分で小さな w -> 閾値 tau_L = 0.9 を超えない場合 0
    w = map_loss_distance(history_i, history_j, tau_L=0.9)
    assert w == 0.0


def test_normalize_model_scaling():
    params_m = {'a': torch.tensor([5., 0.])}
    params_p = {'a': torch.tensor([3., 0.])}
    normalized = normalize_model(params_p, params_m)
    # norm_p/norm_m = 3/5 = 0.6 -> 3 * 0.6 = 1.8
    assert torch.allclose(normalized['a'], torch.tensor([1.8, 0.]))


def test_sentinel_aggregation_identity():
    # local と neighbor が同じパラメータの場合、同じモデルが返る
    local_model = IdentityModel()
    # state_dict をそのまま利用
    state = local_model.state_dict()
    neighbor_params = {0: state.copy(), 1: state.copy()}
    # 初期履歴: 各クライアントにひとつの履歴要素を用意
    loss_history = {0: [0.0], 1: [0.0]}
    # ブートストラップ用データ: x=y
    x = torch.tensor([[1.], [2.]], dtype=torch.float32)
    y = x.clone()
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    loss_fn = nn.MSELoss()

    aggregated = sentinel_aggregation(
        neighbor_params,
        local_model,
        loss_history,
        loader,
        loss_fn,
        tau_S=0.0,
        tau_L=0.0
    )
    # aggregated は state と同一の辞書
    for k, v in state.items():
        assert torch.allclose(aggregated[k], v)
