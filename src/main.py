import argparse
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from utils import load_dataset, partition_dataset, create_backdoor_testloader
from client import Client


def parse_args():
    parser = argparse.ArgumentParser(description='DFL Backdoor Attack Simulation')
    parser.add_argument('--clients', type=int, default=5,
                        help='number of total clients')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--num_attackers', type=int,
                     help='number of malicious clients')
    grp.add_argument('--attacker_ratio', type=float, default=0.2,
                     help='fraction of malicious clients (0-1)')
    parser.add_argument('--attack_selection', type=str, default='random',
                        choices=['random','pagerank'],
                        help='selection strategy for attackers')
    parser.add_argument('--rounds', type=int, default=10,
                        help='number of communication rounds')
    parser.add_argument('--pdr', type=float, default=0.3,
                        help='Poisoned Data Ratio for malicious clients')
    parser.add_argument('--boost', type=float, default=1.0,
                        help='weight boosting factor for malicious clients')
    parser.add_argument('--clip_global', type=float, default=None,
                        help='clipping constant for received models')
    parser.add_argument('--clip_local', type=float, default=None,
                        help='clipping constant for own model')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='compute device')
    parser.add_argument('--topology', type=str, default='ring',
                        choices=['ring','full','random','watts','barabasi'],
                        help='client topology type')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--k', type=int, default=2,
                        help='neighbor count for ring/Watts-Strogatz')
    parser.add_argument('--m', type=int, default=2,
                        help='edge count for Barabási–Albert')
    return parser.parse_args()


def build_topology(args):
    n, seed = args.clients, args.seed
    np.random.seed(seed)
    if args.topology == 'full':
        return nx.complete_graph(n)
    if args.topology == 'ring':
        return nx.cycle_graph(n)
    if args.topology == 'random':
        return nx.erdos_renyi_graph(n, p=args.k/n, seed=seed)
    if args.topology == 'watts':
        return nx.watts_strogatz_graph(n, k=args.k, p=0.1, seed=seed)
    if args.topology == 'barabasi':
        return nx.barabasi_albert_graph(n, args.m, seed=seed)
    raise ValueError('Unknown topology')


def select_attackers(G, args):
    n = args.clients
    if args.num_attackers is not None:
        k = min(args.num_attackers, n)
    else:
        k = max(1, int(args.attacker_ratio * n))
    if args.attack_selection == 'random':
        rng = np.random.RandomState(args.seed)
        return list(rng.choice(n, size=k, replace=False))
    pr = nx.pagerank(G)
    return [node for node, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]]


def average_weights(w1, w2):
    return {k: (w1[k] + w2[k]) / 2.0 for k in w1}


def clip_weights(weights, const):
    clipped = {}
    for k, v in weights.items():
        norm = torch.norm(v)
        factor = max(1.0, (norm / const).item())
        clipped[k] = v / factor
    return clipped


def evaluate_backdoor_success(clients, backdoor_loader):
    rates = []
    for c in clients:
        c.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in backdoor_loader:
                data, target = data.to(c.device), target.to(c.device)
                pred = c.model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        rates.append(correct / total)
    return sum(rates) / len(rates)


def simulate(args):
    train_set, test_set = load_dataset()
    subsets = partition_dataset(train_set, args.clients)
    backdoor_loader = list(create_backdoor_testloader(test_set, target_class=7, batch_size=64, device=args.device))

    G = build_topology(args)
    attackers = set(select_attackers(G, args))
    print(f"Topology: {args.topology}, edges: {G.number_of_edges()}")
    print(f"Attackers (malicious clients): {sorted(attackers)}")

    clients = []
    for i in range(args.clients):
        is_mal = (i in attackers)
        neighbors = list(G.neighbors(i))
        clients.append(Client(i, subsets[i], neighbors, device=args.device, malicious=is_mal, pdr=args.pdr if is_mal else 0.0))

    success_rates = []
    for r in range(args.rounds):
        print(f"--- Round {r+1} ---")
        for c in clients:
            c.train(epochs=1)
        new_weights = []
        for c in clients:
            own = c.get_weights()
            if args.clip_local:
                own = clip_weights(own, args.clip_local)
            if c.malicious:
                own = {k: v * args.boost for k, v in own.items()}
            if c.neighbors:
                for nid in c.neighbors:
                    w_n = clients[nid].get_weights()
                    if args.clip_global:
                        w_n = clip_weights(w_n, args.clip_global)
                    own = average_weights(own, w_n)
            new_weights.append(own)
        for c, w in zip(clients, new_weights):
            c.set_weights(w)
        rate = evaluate_backdoor_success(clients, backdoor_loader)
        success_rates.append(rate)
        print(f"Backdoor success rate: {rate:.2f}")

    plt.figure()
    plt.plot(range(1, args.rounds+1), success_rates)
    plt.xticks(range(1, args.rounds+1))
    plt.xlabel('Round')
    plt.ylabel('Backdoor Success Rate')
    plt.title('Backdoor Success vs. Rounds')
    plt.grid(True)
    plt.show()
    plt.savefig("plot.png")

if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {args.device}, PDR: {args.pdr}, Boost: {args.boost}, ClipGlobal: {args.clip_global}, ClipLocal: {args.clip_local}")
    simulate(args)