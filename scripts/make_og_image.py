"""Generate the social-share preview image (Open Graph / Twitter Card).

Renders a 1200x630 dark cover with the project title, subtitle and a
magnifying-glass motif matching ``frontend/public/favicon.svg``. Text is kept
in English so it renders with matplotlib's bundled font on any machine.
Saves ``frontend/public/og-cover.png``.

Usage:
    uv run python scripts/make_og_image.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

_OUT = Path("frontend/public/og-cover.png")

_BG = "#0f172a"        # slate-900
_TEAL = "#2dd4bf"      # accent
_AMBER = "#f59e0b"     # lesion dot
_WHITE = "#f8fafc"
_SLATE = "#94a3b8"
_MUTED = "#64748b"


def main() -> None:
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.set_facecolor(_BG)
    ax.axis("off")

    # Left accent bar.
    ax.add_patch(Rectangle((90, 250), 11, 195, facecolor=_TEAL, edgecolor="none"))

    # Magnifying-glass motif in the top-right corner (mirrors the favicon).
    cx, cy = 1085, 495
    ax.add_patch(Circle((cx, cy), 58, facecolor="none", edgecolor=_TEAL, linewidth=10))
    ax.add_patch(Circle((cx, cy), 21, facecolor=_AMBER, edgecolor="none"))
    ax.plot([cx + 41, cx + 86], [cy - 41, cy - 86],
            color=_TEAL, linewidth=15, solid_capstyle="round")

    # Text block.
    ax.text(126, 415, "Skin Lesion Classifier", color=_WHITE,
            fontsize=48, fontweight="bold", va="center")
    ax.text(130, 352, "CNN  vs.  Vision Transformer  ·  HAM10000",
            color=_TEAL, fontsize=26, fontweight="bold", va="center")
    ax.text(130, 308, "Multi-class dermoscopy classification under class imbalance",
            color=_SLATE, fontsize=20, va="center")
    ax.text(130, 70, "Research & educational demo — not a certified medical device",
            color=_MUTED, fontsize=19, va="center")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, facecolor=_BG)
    plt.close(fig)
    print(f"wrote {_OUT} ({_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
