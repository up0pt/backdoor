import argparse
import ast
import os
import csv
import json
from datetime import datetime
import math
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import load_dataset, assign_random_data_to_clients, corrcoef_numpy
from client import Client
from fix_seed import fix_seeds
from plot_utils import plot_pagerank_vs_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='DFL Backdoor Attack Simulation')
    parser.add_argument('--com_grad', type=bool, default=True)
    parser.add_argument('--clients', type=int, default=5)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--num_attackers', type=int, default=0)
    grp.add_argument('--attacker_ratio', type=float, default=0)
    parser.add_argument('--attack_selection', type=str, default='random', choices=['random','pagerank'])
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--pdr', type=float, default=0)
    parser.add_argument('--boost', type=float, default=1.0)
    parser.add_argument('--clip_global', type=float, default=None)
    parser.add_argument('--clip_local', type=float, default=None)
    parser.add_argument('--output_csv', type=str, default='results.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topology', type=str, default='ring', choices=['ring','full','random','watts','barabasi'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--graph_path', type=str, default='graph.png')
    parser.add_argument('--client_extra_data', type=json.loads, default='{}')
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

def save_topology(topo, run_dir, rel_path):

    # -----------------------------
    # 1) PageRank を計算
    # -----------------------------
    pr = nx.pagerank(topo, alpha=0.85)

    # -----------------------------
    # 2) レイアウト＆ラベル位置調整
    # -----------------------------
    pos = nx.spring_layout(topo)

    # ノードラベルを y 方向に少し上げるオフセット
    label_offset = 0.03
    label_pos = {
        n: (x, y + label_offset)
        for n, (x, y) in pos.items()
    }

    # ラベル文字列：ID と PageRank を改行で結合
    labels = {
        n: f"{n}\n{pr[n]:.2f}"
        for n in topo.nodes()
    }

    # -----------------------------
    # 3) 描画
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # ノード・エッジ
    nx.draw_networkx_nodes(topo, pos=pos, ax=ax, node_size=100)
    nx.draw_networkx_edges(topo, pos=pos, ax=ax, alpha=0.5)

    # ラベル（少し上にオフセット）
    nx.draw_networkx_labels(
        topo,
        pos=label_pos,
        labels=labels,
        font_size=10,
        verticalalignment='bottom',  # ラベル文字をノード上部に寄せる
    )

    ax.set_axis_off()

    # -----------------------------
    # 4) ファイルに保存
    # -----------------------------
    path = os.path.join(run_dir, rel_path)
    fig.savefig(path, dpi=300, bbox_inches="tight")



def select_attackers(G, args):
    n = args.clients
    k = args.num_attackers if args.num_attackers is not None else max(1, int(args.attacker_ratio * n))
    if args.attack_selection == 'random':
        rng = np.random.RandomState(args.seed)
        return list(rng.choice(n, size=k, replace=False))
    pr = nx.pagerank(G)
    return [node for node,_ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]]


def average_weights(w1, wlist):
    return {k:(w1[k]+sum([w[k] for w in wlist]))/(len(wlist) + 1) for k in w1}

def add_weights(w1, wdict: dict):
    return {k:(v+wdict[k]) for k, v in w1.items()}

def clip_weights(weights, const):
    clipped = {}
    norms = {k: v.norm(p=2).item() for k, v in weights.items()}
    norm_all = np.linalg.norm(list(norms.values()), ord = 2)
    factor = max(1.0, norm_all/const)
    for k,v in weights.items():
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
                # TODO: inject を別の関数に切り分ける
                x[:, :, -2:, -2:] = 1.0
                preds = c.model(x).argmax(dim=1)
                fp += (preds == target_class).sum().item()
                total += x.size(0)
        rates.append(fp/total if total>0 else 0)
    return sum(rates)/len(rates)


def evaluate_clean_accuracy(clients, clean_loader):# 
    accs=[]
    for c in clients:
        c.model.eval()
        correct=0
        total=0
        device=c.device
        if c.malicious:
            accs.append(float('nan'))
            continue
        with torch.no_grad():
            for data,label in clean_loader:
                d=data.to(device)
                l=label.to(device)
                pred=c.model(d).argmax(dim=1)
                correct+=(pred==l).sum().item()
                total+=l.size(0)
        accs.append(correct/total)
    without_none = [a for a in accs if not math.isnan(a)]
    return sum(without_none)/len(without_none), accs

def simulate(args):
    # setup
    fix_seeds(args.seed)
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
    subsets=assign_random_data_to_clients(train,args.clients, sample_num_dict=args.client_extra_data)
    clean_loader=DataLoader(test,batch_size=64,shuffle=False)

    G=build_topology(args)
    save_topology(G, run_dir, args.graph_path)
    attackers=set(select_attackers(G,args))
    log(f"Topology:{args.topology},edges:{G.number_of_edges()}")
    log(f"Attackers:{sorted(attackers)}")

    clients=[]
    for i in range(args.clients):
        is_m=i in attackers
        nb=list(G.neighbors(i))
        clients.append(Client(i,subsets[i],nb,device=args.device,malicious=is_m,pdr=args.pdr if is_m else 0))

    brs=[]; accs=[]
    acc_clients = []
    coef = []

    client_pageranks = {i: pr_score for i, pr_score in enumerate(nx.pagerank(G).values())}
    pagerank_list = [client_pageranks[i] for i in range(len(clients))]
    for r in range(1,args.rounds+1):
        log(f"--- Round {r} ---")
        for c in clients: 
            c.train(epochs = 1)
        new_w=[]
        if args.com_grad:
            # w はgradient
            for c in clients:
                w=c.get_grad()
                if args.clip_local: 
                    w=clip_weights(w,args.clip_local)
                if c.malicious: 
                    # TODO: 攻撃者の重みの更新はこれであっているのか？
                    w={k:v*args.boost for k,v in w.items()}
                wns=[clients[nid].get_grad() for nid in c.neighbors]
                # TODO: 下のclipを追加する
                # if args.clip_global: 
                #     wn=clip_weights(wn,args.clip_global)
                grad_w=average_weights(w,wns)
                weight = add_weights(c.get_weights(), grad_w)
                new_w.append(weight)
        else:
            for c in clients:
                w=c.get_weights()
                if args.clip_local: 
                    w=clip_weights(w,args.clip_local)
                if c.malicious: 
                    # TODO: 攻撃者の重みの更新はこれであっているのか？
                    w={k:v*args.boost for k,v in w.items()}
                wns=[clients[nid].get_weights() for nid in c.neighbors]
                # TODO: 下のclipを追加する
                # if args.clip_global: 
                #     wn=clip_weights(wn,args.clip_global)
                w=average_weights(w,wns) 
                new_w.append(w)
        # 同期して集約しているので、set_weightsを最後に行う。
        for c,w in zip(clients,new_w): 
            c.set_weights(w)

        bs=evaluate_attack_success(clients,test,7)
        cs, acc_list=evaluate_clean_accuracy(clients,clean_loader)
        brs.append(bs)
        accs.append(cs)
        client_accuracies = {i: acc for i, acc in enumerate(acc_list)}
        print(client_accuracies)
        accuracy_list = [client_accuracies[i] for i in range(len(clients))]
        coef.append(corrcoef_numpy(pagerank_list, accuracy_list))
        acc_clients.append(accuracy_list)
        plot_pagerank_vs_accuracy(pagerank_list, accuracy_list, os.path.join(run_dir,f'pagerank_vs_accuracy_{r}.png'))
        log(f"Attack FP rate: {bs:.2f}, Clean acc: {cs:.2f}")

        # Cosine similarity graph
        flat_weights=[]
        for c in clients:
            ws=c.get_weights()
            flat=torch.cat([v.view(-1) for v in ws.values()])
            flat_weights.append(flat)
        edge_labels={}
        for u,v in G.edges():
            cos=F.cosine_similarity(flat_weights[u].to(args.device), flat_weights[v].to(args.device), dim=0).item()
            edge_labels[(u,v)]=f"{cos:.2f}"
        plt.figure()
        pos=nx.spring_layout(G,seed=args.seed)
        nx.draw(G,pos,with_labels=True,node_color='lightblue')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        plt.title(f"Cosine Similarity Round {r}")
        sim_path=os.path.join(run_dir, f"sim_round_{r}.png")
        plt.savefig(sim_path)
        log(f"Similarity graph saved to {sim_path}")
        plt.close()


    # save CSV
    csvp=os.path.join(run_dir,args.output_csv)
    with open(csvp,'w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(pagerank_list)
        for i, acces in enumerate(acc_clients, 1): 
            wr.writerow(acces)
    log(f"CSV saved to {csvp}")

    # save CSV
    csvp=os.path.join(run_dir,'pagerank_accuracy.csv')
    with open(csvp,'w',newline='') as f:
        wr=csv.writer(f)
        for i,(bs,cs) in enumerate(zip(brs,accs),1): 
            wr.writerow([i,bs,cs])
    log(f"CSV saved to {csvp}")

    # plot
    plt.figure()
    plt.plot(range(1,args.rounds+1),coef)
    plt.xlabel('Round')
    plt.ylabel('coef')
    plt.title('coef between acc and pagerank')
    plt.grid(True)
    plt.xticks(range(1,args.rounds+1))
    plt.ylim(-1.09, 1.09)
    coefp=os.path.join(run_dir,'coef.png')
    plt.savefig(coefp)
    log(f"Figure saved to {coefp}")
    plt.show()

    # plot
    plt.figure()
    plt.plot(range(1,args.rounds+1),brs,label='Attack FP')
    plt.plot(range(1,args.rounds+1),accs,label='Clean Acc')
    plt.xlabel('Round')
    plt.ylabel('Rate')
    plt.title('Attack FP & Test Acc')
    plt.legend();plt.grid(True)
    plt.xticks(range(1,args.rounds+1))
    plt.ylim(0, 1.09)
    figp=os.path.join(run_dir,'metrics.png')
    plt.savefig(figp)
    log(f"Figure saved to {figp}")
    logf.close()
    plt.show()

if __name__=='__main__': # 
    args=parse_args()
    simulate(args)