import torch
import numpy as np
import pytest
import sys
import os

from model import CNN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@pytest.fixture
def get_initial_models():
    num_clients = 5
    return [CNN() for _ in range(num_clients)]

def test_initial_weights_are_equal(get_initial_models):
    models = get_initial_models
    ref_state_dict = models[0].state_dict()

    for i, model in enumerate(models[1:], start=1):
        for key in ref_state_dict:
            assert torch.equal(ref_state_dict[key], model.state_dict()[key]), \
                f"Model at index {i} differs in parameter '{key}'"

