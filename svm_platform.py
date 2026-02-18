import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


SEED = 0
N = 40
SPREAD = 0.5
C = 1e6
SAVE_PATH = "svm_demo.png"
SHOW = False


def make_linearly_separable():
    rng = np.random.default_rng(SEED)
    k = N // 2
    m0 = np.array([-2.0, -1.0])
    m1 = np.array([2.0, 1.0])
    x0 = rng.normal(loc=m0, scale=SPREAD, size=(k, 2))
    x1 = rng.normal(loc=m1, scale=SPREAD, size=(N - k, 2))
    x = np.vstack([x0, x1])
    y = np.hstack([-np.ones(k, dtype=int), np.ones(N - k, dtype=int)])
    p = rng.permutation(N)
    return x[p], y[p]


def fit_hard_margin_like(x: np.ndarray, y: np.ndarray):
    clf = SVC(kernel="linear", C=C)
    clf.fit(x, y)
    w = clf.coef_.reshape(-1)
    b = float(clf.intercept_.reshape(-1)[0])
    return clf, w, b


def plot_hyperplane_and_margins(ax, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, support_vectors: np.ndarray):
    ax.scatter(x[y == -1, 0], x[y == -1, 1], s=50, marker="o", label="y=-1")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], s=50, marker="^", label="y=+1")
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=140, facecolors="none", edgecolors="k", linewidths=1.5, label="support vectors")

    xmin, xmax = ax.get_xlim()
    xs = np.linspace(xmin, xmax, 400)

    if abs(w[1]) < 1e-12:
        x0 = -b / w[0]
        ax.axvline(x0, color="k", linewidth=2)
    else:
        ys0 = -(w[0] * xs + b) / w[1]
        ys1 = -(w[0] * xs + b - 1.0) / w[1]
        ys2 = -(w[0] * xs + b + 1.0) / w[1]
        ax.plot(xs, ys0, color="k", linewidth=2, label="w·x + b = 0")
        ax.plot(xs, ys1, color="k", linestyle="--", linewidth=1.5, label="w·x + b = ±1")
        ax.plot(xs, ys2, color="k", linestyle="--", linewidth=1.5)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")


def main():
    x, y = make_linearly_separable()
    clf, w, b = fit_hard_margin_like(x, y)

    norm_w = float(np.linalg.norm(w))
    margin_half_width = 1.0 / norm_w

    f = x @ w + b
    signed_dist = f / norm_w

    print("w=", w)
    print("b=", b)
    print("||w||=", norm_w)
    print("margin half-width (distance from boundary to margin line)=", margin_half_width)
    print("support vector count=", int(clf.support_vectors_.shape[0]))
    print("min signed distance=", float(np.min(signed_dist)))
    print("max signed distance=", float(np.max(signed_dist)))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    plot_hyperplane_and_margins(ax, x, y, w, b, clf.support_vectors_)
    ax.set_title("Linear SVM (C large) with max-margin geometry")
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(SAVE_PATH, dpi=160, bbox_inches="tight")
    print("saved:", SAVE_PATH)

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
