import matplotlib.pyplot as plt
from pathlib import Path  # <-- needed

# LaTeX / REVTeX style
plt.rcParams.update({
    "text.usetex": True,        # requires TeX installed
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

PT_PER_INCH = 72.27
COLUMNWIDTH_PT = 246.0   # LaTeX \columnwidth
GOLDEN = 1.618

def new_fig(width_pt=COLUMNWIDTH_PT, golden=GOLDEN):
    """Create a figure sized to match LaTeX columnwidth."""
    w_in = width_pt / PT_PER_INCH
    h_in = w_in / golden
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    return fig, ax

palette = {  # (optional) fixed spelling
    "red":        "#E3120B",
    "deep_blue":  "#005566",
    "bright_blue":"#008DC3",
    "dark_gray":  "#4A4A4A",
    "light_gray": "#F5F6F5",
    "yellow":     "#FFC107",
    "green":      "#007A3D",
    "dark_red":   "#A10000",
    "sky_blue":   "#40C4FF",
    "charcoal":   "#2E2E2E",
    "orange":     "#FF6F00",
    "lime_green": "#4CAF50",
    "pink":       "#D81B60",
    "cerulean":   "#0288D1",
    "purple":     "#6A1B9A",
}

def save_pdf(basename, folder="Plots"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{folder}/{basename}.pdf", bbox_inches="tight")
