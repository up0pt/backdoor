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
from utils import load_dataset, partition_dataset
from client import Client


def parse_args():
    parser = argparse.ArgumentParser(description='DFL Backdoor Attack Simulation')
    parser.add_argument('--clients', type=int, default=5)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--num_attackers', type=int)
    grp.add_argument('--attacker_ratio', type=float, default=0.2)
    parser.add_argument('--attack_selection', type=str, default='random', choices=['random','pagerank'])
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--pdr', type=float, default=0.3)
    parser.add_argument('--boost', type=float, default=1.0)
    parser.add_argument('--clip_global', type=float, default=None)
    parser.add_argument('--clip_local', type=float, default=None)
    parser.add_argument('--output_csv', type=str, default='results.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topology', type=str, default='ring', choices=['ring','full','random','watts','barabasi'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    return parser.parse_args()


def build_topology(args):
    n, seed = args.clients, args.seed
    np.random.seed(seed)
    if args.topology == 'full': return nx.complete_graph(n)
    if args.topology == 'ring': return nx.cycle_graph(n)
    if args.topology == 'random': return nx.erdos_renyi_graph(n, p=args.k/n, seed=seed)
    if args.topology == 'watts': return nx.watts_strogatz_graph(n, k=args.k, p=0.1, seed=seed)
    if args.topology == 'barabasi': return nx.barabasi_albert_graph(n, args.m, seed=seed)
    raise ValueError('Unknown topology')


def select_attackers(G, args):
    n = args.clients
    k = args.num_attackers if args.num_attackers is not None else max(1, int(args.attacker_ratio * n))
    if args.attack_selection == 'random':
        rng = np.random.RandomState(args.seed)
        return list(rng.choice(n, size=k, replace=False))
    pr = nx.pagerank(G)
    return [node for node,_ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]]


def average_weights(w1, w2):
    return {k:(w1[k]+w2[k])/2.0 for k in w1}


def clip_weights(weights, const):
    clipped = {}
    for k,v in weights.items():
        norm = torch.norm(v)
        factor = max(1.0, (norm/const).item())
        clipped[k] = v/factor
    return clipped


def evaluate_attack_success(clients, test_dataset, target_class, batch_size=64):
    # false-positive rate: non-target samples misclassified as target
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    rates = []
    for c in clients:
        c.model.eval()
        fp = total = 0
        device = c.device
        with torch.no_grad():
            for data, label in loader:
                mask = (label != target_class)
                if not mask.any(): continue
                x = data[mask].to(device)
                # inject patch
                x[:, :, -2:, -2:] = 1.0
                preds = c.model(x).argmax(dim=1)
                fp += (preds == target_class).sum().item()
                total += x.size(0)
        rates.append(fp/total if total>0 else 0)
    return sum(rates)/len(rates)


def evaluate_clean_accuracy(clients, clean_loader):
    accs=[]
    for c in clients:
        c.model.eval()
        correct=total=0
        device=c.device
        with torch.no_grad():
            for data,label in clean_loader:
                d=data.to(device); l=label.to(device)
                pred=c.model(d).argmax(dim=1)
                correct+=(pred==l).sum().item(); total+=l.size(0)
        accs.append(correct/total)
    return sum(accs)/len(accs)


def simulate(args):
    # setup
    base='results'
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir=os.path.join(base,ts)
    os.makedirs(run_dir,exist_ok=True)
    with open(os.path.join(run_dir,'args.json'),'w') as f: 
        json.dump(vars(args),f,indent=2)
    logf=open(os.path.join(run_dir,'log.txt'),'w')
    def log(m): 
        print(m)
        logf.write(m+'\n')
        logf.flush()
    log(f"Device:{args.device},PDR:{args.pdr},Boost:{args.boost},ClipG:{args.clip_global},ClipL:{args.clip_local}")

    train,test=load_dataset()
    subsets=partition_dataset(train,args.clients)
    clean_loader=DataLoader(test,batch_size=64,shuffle=False)

    G=build_topology(args)
    attackers=set(select_attackers(G,args))
    log(f"Topology:{args.topology},edges:{G.number_of_edges()}")
    log(f"Attackers:{sorted(attackers)}")

    clients=[]
    for i in range(args.clients):
        is_m=i in attackers
        nb=list(G.neighbors(i))
        clients.append(Client(i,subsets[i],nb,device=args.device,malicious=is_m,pdr=args.pdr if is_m else 0))

    brs=[]; accs=[]
    for r in range(1,args.rounds+1):
        log(f"--- Round {r} ---")
        for c in clients: c.train(1)
        new_w=[]
        for c in clients:
            w=c.get_weights()
            if args.clip_local: 
                w=clip_weights(w,args.clip_local)
            if c.malicious: 
                w={k:v*args.boost for k,v in w.items()}
            for nid in c.neighbors:
                wn=clients[nid].get_weights()
                if args.clip_global: 
                    wn=clip_weights(wn,args.clip_global)
                w=average_weights(w,wn)
            new_w.append(w)
        for c,w in zip(clients,new_w): 
            c.set_weights(w)

        bs=evaluate_attack_success(clients,test,7)
        cs=evaluate_clean_accuracy(clients,clean_loader)
        brs.append(bs)
        accs.append(cs)
        log(f"Attack FP rate: {bs:.2f}, Clean acc: {cs:.2f}")

    # save CSV
    csvp=os.path.join(run_dir,args.output_csv)
    with open(csvp,'w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['round','attack_fp','test_acc'])
        for i,(bs,cs) in enumerate(zip(brs,accs),1): 
            wr.writerow([i,bs,cs])
    log(f"CSV saved to {csvp}")

    # plot
    plt.figure()
    plt.plot(range(1,args.rounds+1),brs,label='Attack FP')
    plt.plot(range(1,args.rounds+1),accs,label='Clean Acc')
    plt.xlabel('Round')
    plt.ylabel('Rate')
    plt.title('Attack FP & Test Acc')
    plt.legend();plt.grid(True)
    plt.xticks(rounds)
    figp=os.path.join(run_dir,'metrics.png')
    plt.savefig(figp)
    log(f"Figure saved to {figp}")
    plt.show()
    logf.close()

if __name__=='__main__': args=parse_args(); simulate(args)