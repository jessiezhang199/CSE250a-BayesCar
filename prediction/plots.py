"""Metric comparison and reliability figures for E5."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
LABELS = {
    "logistic": "Logistic regression",
    "hill_climb_bic": "Learned DAG",
    "expert": "Expert DAG",
    "naive_bayes": "Naive Bayes",
    "chow_liu": "Chow-Liu tree",
    "empty": "Prior only",
}
COLORS = {"expert": "#2a78d6", "hill_climb_bic": "#eb6834", "chow_liu": "#1baf7a",
          "naive_bayes": "#eda100", "logistic": "#e87ba4"}
MARKERS = {"expert": "o", "hill_climb_bic": "s", "chow_liu": "^", "naive_bayes": "D", "logistic": "v"}
EVIDENCE_LABELS = {"all_features": "All 11 features",
                   "no_condition_history": "Without condition and history"}
MIN_BIN_COUNT = 30


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)


def metric_figure(summary, path):
    """Dumbbell per model with filled points for all features and hollow points for reduced evidence. The prior is left off the axis."""
    order = [m for m in LABELS if m != "empty"]
    data = summary.set_index(["model", "evidence"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE, sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "log_loss"),
                                 ("Accuracy (higher is better)", "Log-loss, nats (lower is better)")):
        _style(ax)
        for y, model in enumerate(order):
            full = data.loc[(model, "all_features"), metric]
            reduced = data.loc[(model, "no_condition_history"), metric]
            ax.plot([full, reduced], [y, y], color=INK_2, linewidth=2, zorder=1)
            ax.scatter(reduced, y, s=64, facecolor=SURFACE, edgecolor=INK, linewidth=2, zorder=2,
                       label=EVIDENCE_LABELS["no_condition_history"] if y == 0 else None)
            ax.scatter(full, y, s=64, color=INK, edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=EVIDENCE_LABELS["all_features"] if y == 0 else None)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [LABELS[m] for m in order])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               labelcolor=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def calibration_figure(calibration, path):
    """Top-label reliability per evidence set, hiding bins with fewer than MIN_BIN_COUNT rows."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE, sharey=True)
    for ax, (evidence, label) in zip(axes, EVIDENCE_LABELS.items()):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1, label="Perfect calibration")
        for model, color in COLORS.items():
            rows = calibration[(calibration.evidence == evidence) & (calibration.model == model)
                               & (calibration["count"] >= MIN_BIN_COUNT)]
            ax.plot(rows.mean_confidence, rows.accuracy, color=color, linewidth=2,
                    marker=MARKERS[model], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=LABELS[model])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Mean top-tier probability", color=INK)
    axes[0].set_ylabel("Observed accuracy", color=INK)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
