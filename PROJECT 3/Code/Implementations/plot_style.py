from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_latex_style(usetex: bool = True, base_fontsize: int = 9) -> None:
    """
    Apply a consistent LaTeX-like style for all matplotlib figures.
    Set usetex=False if your environment lacks a LaTeX installation.
    """
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        # If usetex is off, this still helps emulate Computer Modern:
        "mathtext.fontset": "cm",

        "axes.labelsize": base_fontsize,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,

        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

def fig_ax(width: str = "onecol", height_scale: float = 1.0):
    """
    width: 'onecol' (~3.35in) or 'twocol' (~6.9in)
    """
    if width == "onecol":
        w = 3.35
    elif width == "twocol":
        w = 6.9
    else:
        raise ValueError("width must be 'onecol' or 'twocol'")

    h = 2.2 * height_scale if width == "onecol" else 3.2 * height_scale
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax

def save_pdf(fig, filename: str, folder: str = "../Plots") -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    return path
