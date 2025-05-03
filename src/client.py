import torch
import torch.nn as nn
from model import CNN

class Client:
    def __init__(self, cid, dataset, neighbors, device='cpu', malicious=False, pdr=0):
        self.id = cid
        self.device = torch.device(device)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        self.model = CNN().to(self.device)
        self.neighbors = neighbors           # list of neighbor client indices
        self.malicious = malicious           # whether this client injects backdoor
        self.pdr = pdr                       # Poisoned Data Ratio
        self.grad = None

    def train(self, epochs=1, mal_epochs = 5):
        old_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        if self.malicious:
            epochs = mal_epochs
        for _ in range(epochs):
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                if self.malicious and self.pdr > 0:
                    data, target = self.inject_backdoor(data, target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        new_params = self.model.state_dict()
        self.grad = {k: (new_params[k].cpu().detach().clone() - old_params[k].cpu().detach().clone()) for k in old_params}

    def inject_backdoor(self, data, target):
        batch_size = data.size(0)
        mask = torch.rand(batch_size, device=self.device) < self.pdr
        if mask.any():
            data[mask, :, -2:, -2:] = 1.0
            target[mask] = 7  # forced target class
        return data, target

    def get_weights(self):
        return {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
    
    def get_grad(self):
        return self.grad
    
    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)