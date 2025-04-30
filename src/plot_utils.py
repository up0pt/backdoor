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
    Attack FP rate: 0.01, Clean acc: 0.11
    --- Round 2 ---
    Attack FP rate: 0.00, Clean acc: 0.13
    --- Round 3 ---
    Attack FP rate: 0.00, Clean acc: 0.13
    --- Round 4 ---
    Attack FP rate: 0.00, Clean acc: 0.17
    --- Round 5 ---
    Attack FP rate: 0.01, Clean acc: 0.36
    --- Round 6 ---
    Attack FP rate: 0.02, Clean acc: 0.63
    --- Round 7 ---
    Attack FP rate: 0.03, Clean acc: 0.78
    --- Round 8 ---
    Attack FP rate: 0.01, Clean acc: 0.83
    --- Round 9 ---
    Attack FP rate: 0.01, Clean acc: 0.86
    --- Round 10 ---
    Attack FP rate: 0.01, Clean acc: 0.88
    --- Round 11 ---
    Attack FP rate: 0.01, Clean acc: 0.90
    --- Round 12 ---
    Attack FP rate: 0.01, Clean acc: 0.90
    --- Round 13 ---
    Attack FP rate: 0.01, Clean acc: 0.92
    --- Round 14 ---
    Attack FP rate: 0.01, Clean acc: 0.92
    --- Round 15 ---
    Attack FP rate: 0.01, Clean acc: 0.93
    --- Round 16 ---
    Attack FP rate: 0.01, Clean acc: 0.94
    --- Round 17 ---
    Attack FP rate: 0.00, Clean acc: 0.94
    --- Round 18 ---
    Attack FP rate: 0.00, Clean acc: 0.95
    --- Round 19 ---
    Attack FP rate: 0.00, Clean acc: 0.95
    --- Round 20 ---
    Attack FP rate: 0.00, Clean acc: 0.95

    --- Round 1 ---
    Attack FP rate: 0.00, Clean acc: 0.11
    --- Round 2 ---
    Attack FP rate: 0.00, Clean acc: 0.13
    --- Round 3 ---
    Attack FP rate: 0.00, Clean acc: 0.17
    --- Round 4 ---
    Attack FP rate: 0.00, Clean acc: 0.18
    --- Round 5 ---
    Attack FP rate: 0.00, Clean acc: 0.20
    --- Round 6 ---
    Attack FP rate: 0.00, Clean acc: 0.31
    --- Round 7 ---
    Attack FP rate: 0.01, Clean acc: 0.52
    --- Round 8 ---
    Attack FP rate: 0.02, Clean acc: 0.72
    --- Round 9 ---
    Attack FP rate: 0.02, Clean acc: 0.82
    --- Round 10 ---
    Attack FP rate: 0.01, Clean acc: 0.84
    --- Round 11 ---
    Attack FP rate: 0.01, Clean acc: 0.87
    --- Round 12 ---
    Attack FP rate: 0.01, Clean acc: 0.89
    --- Round 13 ---
    Attack FP rate: 0.01, Clean acc: 0.90
    --- Round 14 ---
    Attack FP rate: 0.01, Clean acc: 0.91
    --- Round 15 ---
    Attack FP rate: 0.01, Clean acc: 0.92
    --- Round 16 ---
    Attack FP rate: 0.00, Clean acc: 0.93
    --- Round 17 ---
    Attack FP rate: 0.00, Clean acc: 0.93
    --- Round 18 ---
    Attack FP rate: 0.01, Clean acc: 0.94
    --- Round 19 ---
    Attack FP rate: 0.00, Clean acc: 0.94
    --- Round 20 ---
    Attack FP rate: 0.01, Clean acc: 0.95

    --- Round 1 ---
    Attack FP rate: 0.00, Clean acc: 0.11
    --- Round 2 ---
    Attack FP rate: 0.00, Clean acc: 0.11
    --- Round 3 ---
    Attack FP rate: 0.00, Clean acc: 0.11
    --- Round 4 ---
    Attack FP rate: 0.00, Clean acc: 0.12
    --- Round 5 ---
    Attack FP rate: 0.00, Clean acc: 0.13
    --- Round 6 ---
    Attack FP rate: 0.00, Clean acc: 0.18
    --- Round 7 ---
    Attack FP rate: 0.00, Clean acc: 0.30
    --- Round 8 ---
    Attack FP rate: 0.01, Clean acc: 0.47
    --- Round 9 ---
    Attack FP rate: 0.03, Clean acc: 0.73
    --- Round 10 ---
    Attack FP rate: 0.02, Clean acc: 0.82
    --- Round 11 ---
    Attack FP rate: 0.01, Clean acc: 0.85
    --- Round 12 ---
    Attack FP rate: 0.01, Clean acc: 0.86
    --- Round 13 ---
    Attack FP rate: 0.01, Clean acc: 0.88
    --- Round 14 ---
    Attack FP rate: 0.01, Clean acc: 0.90
    --- Round 15 ---
    Attack FP rate: 0.01, Clean acc: 0.91
    --- Round 16 ---
    Attack FP rate: 0.01, Clean acc: 0.91
    --- Round 17 ---
    Attack FP rate: 0.01, Clean acc: 0.92
    --- Round 18 ---
    Attack FP rate: 0.00, Clean acc: 0.93
    --- Round 19 ---
    Attack FP rate: 0.01, Clean acc: 0.93
    --- Round 20 ---
    Attack FP rate: 0.01, Clean acc: 0.94
    """

    # 空行で区切って実験ブロックごとに分割
    blocks = [b for b in log.strip().split("\n\n") if b]

    # 抽出用の正規表現パターン
    pattern = re.compile(r"--- Round (\d+) ---\s*Attack FP rate: [0-9.]+, Clean acc: ([0-9.]+)")

    label = ["0","2", "8"]
    plt.figure(figsize=(8, 5))
    for i, block in enumerate(blocks):
        rounds = []
        accs = []
        for m in pattern.finditer(block):
            rounds.append(int(m.group(1)))
            accs.append(float(m.group(2)))
        plt.plot(list(range(20)), accs, marker='o', label=f"extra data at Client {label[i]}")

    plt.xlabel("Round")
    plt.ylabel("Clean Accuracy")
    plt.title("Clean Accuracy per Round")
    plt.xticks(range(1, 21))  # ラウンド1〜8をx軸に
    plt.ylim(0, 1)           # 0〜1の範囲に固定
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("clean_accuracy_per_round.png")


if __name__ == "__main__":
    plot_acc()