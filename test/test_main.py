import torch
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import clip_weights


def assert_tensor_dict_allclose(d1, d2, tol=1e-6):
    """
    辞書の中の全テンソルがほぼ等しいことを検証
    """
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        assert torch.allclose(d1[k], d2[k], atol=tol), f"Key {k} differs: {d1[k]} vs {d2[k]}"


def test_no_clipping_when_const_large():
    # norm_all = sqrt(3^2+4^2) = 5
    weights = {'w1': torch.tensor([3.0, 4.0])}
    const = 100.0  # 大きな定数 → factor = max(1, 5/100)=1.0 → クリップなし
    clipped = clip_weights(weights, const)
    assert_tensor_dict_allclose(clipped, weights)


def test_scaling_when_norm_exceeds_const():
    # norm_all = sqrt(6^2+8^2) = 10
    weights = {'w2': torch.tensor([6.0, 8.0])}
    const = 2.0
    # factor = max(1, 10/2) = 5.0
    expected = {'w2': torch.tensor([6.0/5.0, 8.0/5.0])}
    clipped = clip_weights(weights, const)
    assert_tensor_dict_allclose(clipped, expected)


def test_zero_weights_stay_zero():
    weights = {
        'a': torch.zeros(3),
        'b': torch.zeros(2, 2)
    }
    const = 0.1
    # norm_all = 0 → factor = max(1, 0/0.1)=1 → zero のまま
    clipped = clip_weights(weights, const)
    assert_tensor_dict_allclose(clipped, weights)


def test_multiple_keys_behavior():
    w1 = torch.tensor([1.0, 2.0, 2.0])  # norm = 3
    w2 = torch.tensor([-3.0, 4.0])      # norm = 5
    weights = {'w1': w1, 'w2': w2}
    norm_all = np.linalg.norm([3.0, 5.0])  # ≈5.8309518948
    const = 4.0
    factor = max(1.0, norm_all/const)
    expected = { 'w1': w1 / factor, 'w2': w2 / factor }
    clipped = clip_weights(weights, const)
    assert_tensor_dict_allclose(clipped, expected)