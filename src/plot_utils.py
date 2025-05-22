import matplotlib.pyplot as plt
import re

def plot_pagerank_vs_accuracy(pageranks, accuracies, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pageranks, accuracies, s=100, alpha=0.7)   
    for xi, yi, label in zip(pageranks, accuracies, range(len(pageranks))):
        ax.annotate(
            label,                # 表示する文字列
            (xi, yi),             # アノテーションを付ける座標
            textcoords="offset points",  # 座標系：点からのオフセット距離
            xytext=(5, 5),        # オフセット量 (x方向, y方向)
            ha='left',            # horizontal alignment
            va='bottom',          # vertical alignment
            fontsize=18
        )
    plt.xlabel("PageRank Score")
    plt.ylabel("Test Accuracy")
    plt.title("Client PageRank vs. Model Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_acc():

    # ここに全文のログを貼り付け
    log = """
    --- Round 1 ---
    Attack FP rate: 0.34, Clean acc: 0.24
    Similarity graph saved to results/20250513_003503/sim_round_1.png
    --- Round 2 ---
    Attack FP rate: 0.03, Clean acc: 0.49
    Similarity graph saved to results/20250513_003503/sim_round_2.png
    --- Round 3 ---
    Attack FP rate: 0.17, Clean acc: 0.73
    Similarity graph saved to results/20250513_003503/sim_round_3.png
    --- Round 4 ---
    Attack FP rate: 0.14, Clean acc: 0.83
    Similarity graph saved to results/20250513_003503/sim_round_4.png
    --- Round 5 ---
    Attack FP rate: 0.14, Clean acc: 0.86
    Similarity graph saved to results/20250513_003503/sim_round_5.png
    --- Round 6 ---
    Attack FP rate: 0.14, Clean acc: 0.89
    Similarity graph saved to results/20250513_003503/sim_round_6.png
    --- Round 7 ---
    Attack FP rate: 0.14, Clean acc: 0.90
    Similarity graph saved to results/20250513_003503/sim_round_7.png
    --- Round 8 ---
    Attack FP rate: 0.14, Clean acc: 0.91
    Similarity graph saved to results/20250513_003503/sim_round_8.png
    --- Round 9 ---
    Attack FP rate: 0.14, Clean acc: 0.92
    Similarity graph saved to results/20250513_003503/sim_round_9.png
    --- Round 10 ---
    Attack FP rate: 0.14, Clean acc: 0.93
    Similarity graph saved to results/20250513_003503/sim_round_10.png
    --- Round 11 ---
    Attack FP rate: 0.14, Clean acc: 0.94
    Similarity graph saved to results/20250513_003503/sim_round_11.png
    --- Round 12 ---
    Attack FP rate: 0.14, Clean acc: 0.94
    Similarity graph saved to results/20250513_003503/sim_round_12.png
    --- Round 13 ---
    Attack FP rate: 0.14, Clean acc: 0.94
    Similarity graph saved to results/20250513_003503/sim_round_13.png
    --- Round 14 ---
    Attack FP rate: 0.14, Clean acc: 0.95
    Similarity graph saved to results/20250513_003503/sim_round_14.png
    --- Round 15 ---
    Attack FP rate: 0.14, Clean acc: 0.95

    --- Round 1 ---
    Attack FP rate: 0.28, Clean acc: 0.25
    Similarity graph saved to results/20250513_001847/sim_round_1.png
    --- Round 2 ---
    Attack FP rate: 0.03, Clean acc: 0.47
    Similarity graph saved to results/20250513_001847/sim_round_2.png
    --- Round 3 ---
    Attack FP rate: 0.18, Clean acc: 0.69
    Similarity graph saved to results/20250513_001847/sim_round_3.png
    --- Round 4 ---
    Attack FP rate: 0.08, Clean acc: 0.81
    Similarity graph saved to results/20250513_001847/sim_round_4.png
    --- Round 5 ---
    Attack FP rate: 0.16, Clean acc: 0.86
    Similarity graph saved to results/20250513_001847/sim_round_5.png
    --- Round 6 ---
    Attack FP rate: 0.09, Clean acc: 0.88
    Similarity graph saved to results/20250513_001847/sim_round_6.png
    --- Round 7 ---
    Attack FP rate: 0.14, Clean acc: 0.90
    Similarity graph saved to results/20250513_001847/sim_round_7.png
    --- Round 8 ---
    Attack FP rate: 0.11, Clean acc: 0.91
    Similarity graph saved to results/20250513_001847/sim_round_8.png
    --- Round 9 ---
    Attack FP rate: 0.14, Clean acc: 0.92
    Similarity graph saved to results/20250513_001847/sim_round_9.png
    --- Round 10 ---
    Attack FP rate: 0.11, Clean acc: 0.94
    Similarity graph saved to results/20250513_001847/sim_round_10.png
    --- Round 11 ---
    Attack FP rate: 0.14, Clean acc: 0.93
    Similarity graph saved to results/20250513_001847/sim_round_11.png
    --- Round 12 ---
    Attack FP rate: 0.12, Clean acc: 0.93
    Similarity graph saved to results/20250513_001847/sim_round_12.png
    --- Round 13 ---
    Attack FP rate: 0.14, Clean acc: 0.94
    Similarity graph saved to results/20250513_001847/sim_round_13.png
    --- Round 14 ---
    Attack FP rate: 0.12, Clean acc: 0.95
    Similarity graph saved to results/20250513_001847/sim_round_14.png
    --- Round 15 ---
    Attack FP rate: 0.14, Clean acc: 0.95

    --- Round 1 ---
    Attack FP rate: 0.18, Clean acc: 0.26
    Similarity graph saved to results/20250513_062605/sim_round_1.png
    --- Round 2 ---
    Attack FP rate: 0.04, Clean acc: 0.48
    Similarity graph saved to results/20250513_062605/sim_round_2.png
    --- Round 3 ---
    Attack FP rate: 0.17, Clean acc: 0.69
    Similarity graph saved to results/20250513_062605/sim_round_3.png
    --- Round 4 ---
    Attack FP rate: 0.07, Clean acc: 0.83
    Similarity graph saved to results/20250513_062605/sim_round_4.png
    --- Round 5 ---
    Attack FP rate: 0.15, Clean acc: 0.86
    Similarity graph saved to results/20250513_062605/sim_round_5.png
    --- Round 6 ---
    Attack FP rate: 0.08, Clean acc: 0.89
    Similarity graph saved to results/20250513_062605/sim_round_6.png
    --- Round 7 ---
    Attack FP rate: 0.14, Clean acc: 0.90
    Similarity graph saved to results/20250513_062605/sim_round_7.png
    --- Round 8 ---
    Attack FP rate: 0.14, Clean acc: 0.91
    Similarity graph saved to results/20250513_062605/sim_round_8.png
    --- Round 9 ---
    Attack FP rate: 0.14, Clean acc: 0.92
    Similarity graph saved to results/20250513_062605/sim_round_9.png
    --- Round 10 ---
    Attack FP rate: 0.13, Clean acc: 0.93
    Similarity graph saved to results/20250513_062605/sim_round_10.png
    --- Round 11 ---
    Attack FP rate: 0.14, Clean acc: 0.93
    Similarity graph saved to results/20250513_062605/sim_round_11.png
    --- Round 12 ---
    Attack FP rate: 0.13, Clean acc: 0.94
    Similarity graph saved to results/20250513_062605/sim_round_12.png
    --- Round 13 ---
    Attack FP rate: 0.14, Clean acc: 0.94
    Similarity graph saved to results/20250513_062605/sim_round_13.png
    --- Round 14 ---
    Attack FP rate: 0.11, Clean acc: 0.94
    Similarity graph saved to results/20250513_062605/sim_round_14.png
    --- Round 15 ---
    Attack FP rate: 0.14, Clean acc: 0.95
    """

    # 空行で区切って実験ブロックごとに分割
    blocks = [b for b in log.strip().split("\n\n") if b]

    # 抽出用の正規表現パターン
    pattern = re.compile(r"--- Round (\d+) ---\s*Attack FP rate: ([0-9.]+), Clean acc: ([0-9.]+)")

    label = ["Page Rank", "Random", "Rev Page Rank"]
    plt.figure(figsize=(8, 5))
    for i, block in enumerate(blocks):
        rounds = []
        accs = []
        BA = []
        for m in pattern.finditer(block):
            rounds.append(int(m.group(1)))
            BA.append(float(m.group(2)))
            accs.append(float(m.group(3)))
        plt.plot(list(range(len(accs))), accs, marker='o', label=f"clean accuracy {label[i]}")
        plt.plot(list(range(len(BA))), BA, marker='o', label=f"Backdoored rate {label[i]}")

    plt.xlabel("Round")
    plt.ylabel("accuracy")
    plt.title("Clean Accuracy & Backdoor rate per Round")
    plt.ylim(0, 1)           # 0〜1の範囲に固定
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("clean_accuracy_per_round.png")


if __name__ == "__main__":
    plot_acc()