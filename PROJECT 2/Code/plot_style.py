from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# GLOBAL STYLE
# -----------------------------
mpl.rcParams.update({
    # LaTeX-like look without requiring TeX installed
    "text.usetex": False,
    "mathtext.fontset": "cm",

    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],   # always available

    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "axes.linewidth": 0.9,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,

    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# LaTeX column width sizing
PT_PER_INCH = 72.27
COLUMN_PT = 246.0
GOLDEN = 1.618


def make_fig(width_pt=COLUMN_PT, golden=GOLDEN):
    """Create a latex-column sized figure (width/golden ratio)."""
    w_in = width_pt / PT_PER_INCH
    h_in = w_in / golden
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    return fig, ax


def save_pdf(name, folder="Plots"):
    """Save current figure to folder/name.pdf (folder created if needed)."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{folder}/{name}.pdf")


# Color palette for professional plots
palette = {
    "blue":      "#1f77b4",
    "orange":    "#ff7f0e",
    "green":     "#2ca02c",
    "red":       "#d62728",
    "purple":    "#9467bd",
    "brown":     "#8c564b",
    "pink":      "#e377c2",
    "gray":      "#7f7f7f",
    "olive":     "#bcbd22",
    "cyan":      "#17becf",
    "black":     "#000000",
}


# -----------------------------
# HIGH-LEVEL REUSABLE HELPERS
# -----------------------------
def quick_plot(x, y, label=None, color="blue", name=None, show=True):
    """One-line plot + optional PDF save."""
    fig, ax = make_fig()
    ax.plot(x, y, color=palette[color], label=label)
    if label:
        ax.legend()
    ax.grid(True)
    if name:
        save_pdf(name)
    if show:
        plt.show()
    return fig, ax


def scatter_plot(x, y, label=None, color="black", s=18, name=None, show=True):
    """Quick scatter plot + optional save."""
    fig, ax = make_fig()
    ax.scatter(x, y, s=s, color=palette.get(color, color), label=label)
    if label:
        ax.legend()
    ax.grid(True)
    if name:
        save_pdf(name)
    if show:
        plt.show()
    return fig, ax


def compare_plot(x, y1, y2, labels=("True", "Pred"), colors=("blue", "orange"),
                 x_train=None, y_train=None, scatter_color="black",
                 title=None, name=None, show=True):
    """
    Common comparison plot:
    true curve + predicted curve + (optional) training points.
    """
    fig, ax = make_fig()

    ax.plot(x, y1, color=palette[colors[0]], linewidth=2, label=labels[0])
    ax.plot(x, y2, color=palette[colors[1]], linewidth=2, linestyle="--", label=labels[1])

    if x_train is not None and y_train is not None:
        ax.scatter(x_train, y_train, s=18, color=scatter_color, alpha=0.7, label="Training data")

    if title:
        ax.set_title(title)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()
    ax.grid(True)

    if name:
        save_pdf(name)
    if show:
        plt.show()

    return fig, ax
