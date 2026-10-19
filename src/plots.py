"""Matplotlib figures saved to artifacts/plots/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def plot_residuals(actual, predicted, path: Path, code: str) -> None:
    resid = np.asarray(actual) - np.asarray(predicted)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(predicted, resid, alpha=0.7)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set(xlabel="Predicted yield (tons)", ylabel="Residual (actual - predicted)",
              title="Residuals vs predictions")
    ax[1].hist(resid, bins=15, edgecolor="black")
    ax[1].set(xlabel="Residual (tons)", ylabel="Farms", title="Residual distribution")
    fig.suptitle(f"Regression residuals - {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path: Path, code: str, model_name: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not high risk", "High risk"],
           yticklabels=["Not high risk", "High risk"], xlabel="Predicted", ylabel="Actual",
           title=f"{model_name}\nConfusion matrix - {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clusters(Ztr, train_labels, Zte, test_labels, path: Path, code: str) -> None:
    pca = PCA(n_components=2, random_state=0).fit(Ztr)
    P, T = pca.transform(Ztr), pca.transform(Zte)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(P[:, 0], P[:, 1], c=train_labels, cmap="tab10", alpha=0.6, label="train")
    ax.scatter(T[:, 0], T[:, 1], c=test_labels, cmap="tab10", marker="x", s=60, label="test")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Farm clusters (PCA view) - {code}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
