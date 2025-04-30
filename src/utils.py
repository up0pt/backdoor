import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import random

def load_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train, test


def partition_dataset(dataset, num_clients):
    # TODO: partitionではなく、固定数（例えば6000枚）にする
    size = 6000
    subsets = []
    for i in range(num_clients):
        #TODO: indices = list(range(i * size, (i + 1) * size))になおす
        indices = list(range(0, size))
        subsets.append(torch.utils.data.Subset(dataset, indices))
    return subsets

def assign_random_data_to_clients(dataset, num_clients, samples_per_client=6000, sample_num_dict: dict[str, int] = {}):
    indices = list(range(len(dataset)))
    subsets = []

    for client_id in range(num_clients):
        if str(client_id) in sample_num_dict:
            selected_indices = random.sample(indices, sample_num_dict[str(client_id)])
        else:
            selected_indices = random.sample(indices, samples_per_client)
        subsets.append(torch.utils.data.Subset(dataset, selected_indices))

    print(f"subset is equal? {subsets_equal(subsets[0], subsets[1])}")

    return subsets


def create_backdoor_testloader(test_dataset, target_class, batch_size=64, device=None):
    poisoned, targets = [], []
    for data, _ in test_dataset:
        img = data.clone()
        img[:, -2:, -2:] = 1.0  # inject 2x2 white patch
        poisoned.append(img)
        targets.append(target_class)
    tensor_x = torch.stack(poisoned)
    tensor_y = torch.tensor(targets)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if device:
        for batch in loader:
            yield (batch[0].to(device), batch[1].to(device))
    else:
        yield from loader

def corrcoef_numpy(x, y):
    """
    NumPy の corrcoef で相関行列をつくり、
    [0,1] 成分を返す。
    """
    # pandasのDataFrameを作成
    df = pd.DataFrame({'list1': x, 'list2': y})

    # NaNを含む行を削除
    df = df.dropna()

    # 相関係数を計算
    correlation = np.corrcoef(df['list1'], df['list2'])[0, 1]


def subsets_equal(sub1, sub2, atol=1e-6):
    # １）長さが同じか
    if len(sub1) != len(sub2):
        return False

    # ２）全サンプルを順番に比較
    for i in range(len(sub1)):
        x1, y1 = sub1[i]
        x2, y2 = sub2[i]
        # テンソル部分は allclose で比較（浮動小数点誤差を許容）
        if not torch.allclose(x1, x2, atol=atol):
            return False
        # ラベル部分は厳密一致
        if y1 != y2:
            return False

    return True
