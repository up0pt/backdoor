import argparse
import os
import csv
import json
from datetime import datetime
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
    parser.add_argument('--output_csv', type=str, default='results.csv',
                        help='CSV file name to store metrics')
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
    k = args.num_attackers if args.num_attackers is not None else max(1, int(args.attacker_ratio * n))
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


def evaluate_clean_accuracy(clients, clean_loader):
    accs = []
    for c in clients:
        c.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in clean_loader:
                data, target = data.to(c.device), target.to(c.device)
                pred = c.model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        accs.append(correct / total)
    return sum(accs) / len(accs)


def simulate(args):
    # setup experiment directory
    base_dir = 'results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    # save config
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # load data
    train_set, test_set = load_dataset()
    subsets = partition_dataset(train_set, args.clients)
    backdoor_loader = list(create_backdoor_testloader(test_set, target_class=7, batch_size=64, device=args.device))
    clean_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # build topology and select attackers
    G = build_topology(args)
    attackers = set(select_attackers(G, args))
    print(f"Topology: {args.topology}, edges: {G.number_of_edges()}")
    print(f"Attackers (malicious clients): {sorted(attackers)}")

    # initialize clients
    clients = []
    for i in range(args.clients):
        is_mal = (i in attackers)
        neighbors = list(G.neighbors(i))
        clients.append(Client(i, subsets[i], neighbors, device=args.device, malicious=is_mal, pdr=args.pdr if is_mal else 0.0))

    # metrics storage
    success_rates = []
    clean_accuracies = []

    # run rounds
    for r in range(1, args.rounds+1):
        print(f"--- Round {r} ---")
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

        # evaluate
        bs = evaluate_backdoor_success(clients, backdoor_loader)
        cs = evaluate_clean_accuracy(clients, clean_loader)
        success_rates.append(bs)
        clean_accuracies.append(cs)
        print(f"Backdoor success rate: {bs:.2f}, Clean test accuracy: {cs:.2f}")

    # save CSV
    csv_path = os.path.join(run_dir, args.output_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'backdoor_success', 'test_accuracy'])
        for i, (bs, cs) in enumerate(zip(success_rates, clean_accuracies), start=1):
            writer.writerow([i, bs, cs])
    print(f"Results CSV saved to {csv_path}")

    # plot and save figure
    plt.figure()
    plt.plot(range(1, args.rounds+1), success_rates, label='Backdoor Success')
    plt.plot(range(1, args.rounds+1), clean_accuracies, label='Clean Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Rate')
    plt.title('Backdoor Success and Test Accuracy by Round')
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(run_dir, 'metrics.png')
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {args.device}, PDR: {args.pdr}, Boost: {args.boost}, ClipGlobal: {args.clip_global}, ClipLocal: {args.clip_local}, CSV: {args.output_csv}")
    simulate(args)