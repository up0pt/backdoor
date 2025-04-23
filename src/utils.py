import torch
from torchvision import datasets, transforms

def load_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train, test


def partition_dataset(dataset, num_clients):
    size = len(dataset) // num_clients
    subsets = []
    for i in range(num_clients):
        indices = list(range(i * size, (i + 1) * size))
        subsets.append(torch.utils.data.Subset(dataset, indices))
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