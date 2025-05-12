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
    Attack FP rate: 0.60, Clean acc: 0.22
    Similarity graph saved to results/20250512_120246/sim_round_1.png
    --- Round 2 ---
    Attack FP rate: 0.14, Clean acc: 0.42
    Similarity graph saved to results/20250512_120246/sim_round_2.png
    --- Round 3 ---
    Attack FP rate: 0.81, Clean acc: 0.37
    Similarity graph saved to results/20250512_120246/sim_round_3.png
    --- Round 4 ---
    Attack FP rate: 0.62, Clean acc: 0.22
    Similarity graph saved to results/20250512_120246/sim_round_4.png
    --- Round 5 ---
    Attack FP rate: 0.87, Clean acc: 0.37
    Similarity graph saved to results/20250512_120246/sim_round_5.png
    --- Round 6 ---
    Attack FP rate: 0.70, Clean acc: 0.43
    Similarity graph saved to results/20250512_120246/sim_round_6.png
    --- Round 7 ---
    Attack FP rate: 0.82, Clean acc: 0.56
    Similarity graph saved to results/20250512_120246/sim_round_7.png
    --- Round 8 ---
    Attack FP rate: 0.77, Clean acc: 0.64
    Similarity graph saved to results/20250512_120246/sim_round_8.png
    --- Round 9 ---
    Attack FP rate: 0.75, Clean acc: 0.74
    Similarity graph saved to results/20250512_120246/sim_round_9.png
    --- Round 10 ---
    Attack FP rate: 0.70, Clean acc: 0.80
    Similarity graph saved to results/20250512_120246/sim_round_10.png
    --- Round 11 ---
    Attack FP rate: 0.70, Clean acc: 0.83
    Similarity graph saved to results/20250512_120246/sim_round_11.png
    --- Round 12 ---
    Attack FP rate: 0.67, Clean acc: 0.86
    Similarity graph saved to results/20250512_120246/sim_round_12.png
    --- Round 13 ---
    Attack FP rate: 0.67, Clean acc: 0.87
    Similarity graph saved to results/20250512_120246/sim_round_13.png
    --- Round 14 ---
    Attack FP rate: 0.64, Clean acc: 0.89
    Similarity graph saved to results/20250512_120246/sim_round_14.png
    --- Round 15 ---
    Attack FP rate: 0.66, Clean acc: 0.90
    Similarity graph saved to results/20250512_120246/sim_round_15.png
    --- Round 16 ---
    Attack FP rate: 0.64, Clean acc: 0.90
    Similarity graph saved to results/20250512_120246/sim_round_16.png
    --- Round 17 ---
    Attack FP rate: 0.65, Clean acc: 0.91
    Similarity graph saved to results/20250512_120246/sim_round_17.png
    --- Round 18 ---
    Attack FP rate: 0.64, Clean acc: 0.92
    Similarity graph saved to results/20250512_120246/sim_round_18.png
    --- Round 19 ---
    Attack FP rate: 0.64, Clean acc: 0.93
    Similarity graph saved to results/20250512_120246/sim_round_19.png
    --- Round 20 ---
    Attack FP rate: 0.64, Clean acc: 0.93
    Similarity graph saved to results/20250512_120246/sim_round_20.png
    --- Round 21 ---
    Attack FP rate: 0.63, Clean acc: 0.94
    Similarity graph saved to results/20250512_120246/sim_round_21.png
    --- Round 22 ---
    Attack FP rate: 0.63, Clean acc: 0.94
    Similarity graph saved to results/20250512_120246/sim_round_22.png
    --- Round 23 ---
    Attack FP rate: 0.64, Clean acc: 0.94
    Similarity graph saved to results/20250512_120246/sim_round_23.png
    --- Round 24 ---
    Attack FP rate: 0.62, Clean acc: 0.94
    Similarity graph saved to results/20250512_120246/sim_round_24.png
    --- Round 25 ---
    Attack FP rate: 0.63, Clean acc: 0.94
    Similarity graph saved to results/20250512_120246/sim_round_25.png
    --- Round 26 ---
    Attack FP rate: 0.62, Clean acc: 0.95
    Similarity graph saved to results/20250512_120246/sim_round_26.png
    --- Round 27 ---
    Attack FP rate: 0.62, Clean acc: 0.95
    Similarity graph saved to results/20250512_120246/sim_round_27.png
    --- Round 28 ---
    Attack FP rate: 0.62, Clean acc: 0.95
    Similarity graph saved to results/20250512_120246/sim_round_28.png
    --- Round 29 ---
    Attack FP rate: 0.62, Clean acc: 0.95
    Similarity graph saved to results/20250512_120246/sim_round_29.png
    --- Round 30 ---
    Attack FP rate: 0.61, Clean acc: 0.95
    Similarity graph saved to results/20250512_120246/sim_round_30.png
    --- Round 31 ---
    Attack FP rate: 0.62, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_31.png
    --- Round 32 ---
    Attack FP rate: 0.62, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_32.png
    --- Round 33 ---
    Attack FP rate: 0.62, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_33.png
    --- Round 34 ---
    Attack FP rate: 0.61, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_34.png
    --- Round 35 ---
    Attack FP rate: 0.62, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_35.png
    --- Round 36 ---
    Attack FP rate: 0.62, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_36.png
    --- Round 37 ---
    Attack FP rate: 0.61, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_37.png
    --- Round 38 ---
    Attack FP rate: 0.61, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_38.png
    --- Round 39 ---
    Attack FP rate: 0.61, Clean acc: 0.96
    Similarity graph saved to results/20250512_120246/sim_round_39.png
    --- Round 40 ---
    Attack FP rate: 0.61, Clean acc: 0.96
    """

    # 空行で区切って実験ブロックごとに分割
    blocks = [b for b in log.strip().split("\n\n") if b]

    # 抽出用の正規表現パターン
    pattern = re.compile(r"--- Round (\d+) ---\s*Attack FP rate: ([0-9.]+), Clean acc: ([0-9.]+)")

    label = ["0","2", "8"]
    plt.figure(figsize=(8, 5))
    for i, block in enumerate(blocks):
        rounds = []
        accs = []
        BA = []
        for m in pattern.finditer(block):
            rounds.append(int(m.group(1)))
            BA.append(float(m.group(2)))
            accs.append(float(m.group(3)))
        plt.plot(list(range(len(accs))), accs, marker='o', label=f"clean accuracy")
        plt.plot(list(range(len(BA))), BA, marker='o', label=f"Backdoored rate")

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