def set_mpl_latex_style(
    mode: str = "usetex",        # "usetex", "pgf", or "mathtext"
    fontsize: int = 11,
    figsize=(6, 4),
    linewidth: float = 1.4,
    tex_preamble: str = r"\usepackage{amsmath,amssymb}\usepackage{siunitx}\sisetup{detect-all}"
):
    """
    Configure Matplotlib for LaTeX-style figures.

    Parameters
    ----------
    mode : {"usetex", "pgf", "mathtext"}
        - "usetex": render all text with LaTeX (requires LaTeX + dvipng/ghostscript).
        - "pgf": use the PGF backend for LaTeX documents (great for Overleaf).
        - "mathtext": LaTeX-like look using Matplotlib's built-in mathtext (no LaTeX install).
    fontsize : int
        Base font size (pt).
    figsize : tuple
        Default figure size in inches (width, height).
    linewidth : float
        Default line width.
    tex_preamble : str
        Extra LaTeX preamble (ignored in "mathtext" mode).
    """
    import matplotlib as mpl

    # Common aesthetics
    params = {
        # Fonts
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],  # aligns with LaTeX default
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "legend.fontsize": max(8, fontsize - 1),
        "xtick.labelsize": max(8, fontsize - 1),
        "ytick.labelsize": max(8, fontsize - 1),

        # Lines and axes
        "lines.linewidth": linewidth,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,

        # Figure
        "figure.figsize": figsize,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

    mode = mode.lower()
    if mode == "usetex":
        params.update({
            "text.usetex": True,
            "text.latex.preamble": tex_preamble,
        })
    elif mode == "pgf":
        # Use PGF for perfect LaTeX integration (pdf/tex sync)
        # Note: setting backend after importing pyplot can cause warnings—set early in your program.
        mpl.rcParams["backend"] = "pgf"
        params.update({
            "text.usetex": True,               # routed through pgf
            "pgf.texsystem": "pdflatex",       # or "lualatex"/"xelatex"
            "pgf.rcfonts": False,
            "pgf.preamble": tex_preamble,
        })
    elif mode == "mathtext":
        params.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",          # Computer Modern
            "mathtext.rm": "serif",
        })
    else:
        raise ValueError("mode must be one of {'usetex','pgf','mathtext'}")

    mpl.rcParams.update(params)