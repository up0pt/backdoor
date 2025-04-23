import matplotlib.pyplot as plt

def plot_pagerank_vs_accuracy(pageranks, accuracies, output_path):
    plt.figure(figsize=(6, 4))
    plt.scatter(pageranks, accuracies, alpha=0.7)
    plt.xlabel("PageRank Score")
    plt.ylabel("Test Accuracy")
    plt.title("Client PageRank vs. Model Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()