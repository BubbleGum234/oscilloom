"""
Generate Oscilloom logo concepts as PNG files.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "branding")
os.makedirs(OUT, exist_ok=True)

# Brand colors
NAVY = "#0f172a"
DARK = "#1e293b"
TEAL = "#06b6d4"
TEAL_LIGHT = "#67e8f9"
SLATE = "#94a3b8"
WHITE = "#f1f5f9"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"


# =====================================================================
# LOGO 1: The Signal Thread (messy → nodes → clean)
# =====================================================================
def logo_signal_thread():
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, 2)
    ax.axis("off")

    # Messy signal (left side)
    t_messy = np.linspace(0, 3, 500)
    np.random.seed(42)
    messy = (0.5 * np.sin(2 * np.pi * 2 * t_messy)
             + 0.3 * np.sin(2 * np.pi * 7 * t_messy)
             + 0.2 * np.sin(2 * np.pi * 13 * t_messy)
             + 0.15 * np.random.randn(500))
    ax.plot(t_messy, messy, color=SLATE, linewidth=1.5, alpha=0.7)

    # Three nodes
    node_x = [3.5, 5.5, 7.5]
    node_colors = [TEAL, BLUE, PURPLE]
    for x, c in zip(node_x, node_colors):
        circle = plt.Circle((x, 0), 0.35, facecolor=c, edgecolor=WHITE,
                           linewidth=2, zorder=10, alpha=0.9)
        ax.add_patch(circle)
        # Glow effect
        glow = plt.Circle((x, 0), 0.5, facecolor=c, edgecolor="none",
                          alpha=0.15, zorder=5)
        ax.add_patch(glow)

    # Connection lines between nodes
    for i in range(len(node_x) - 1):
        t_conn = np.linspace(node_x[i] + 0.35, node_x[i + 1] - 0.35, 100)
        wave = 0.15 * np.sin(2 * np.pi * 3 * (t_conn - node_x[i]))
        ax.plot(t_conn, wave, color=TEAL_LIGHT, linewidth=2, alpha=0.6)

    # Arrow from messy to first node
    t_arr = np.linspace(3.0, 3.15, 50)
    ax.annotate("", xy=(3.15, 0), xytext=(3.0, 0),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))

    # Clean signal (right side)
    t_clean = np.linspace(8, 11, 500)
    clean = 0.8 * np.sin(2 * np.pi * 1.5 * (t_clean - 8))
    ax.plot(t_clean, clean, color=TEAL, linewidth=2.5, alpha=0.9)

    # Arrow from last node to clean
    ax.annotate("", xy=(8.0, 0), xytext=(7.85, 0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))

    # Text
    ax.text(5.5, -1.6, "O S C I L L O O M", fontsize=24, fontweight="bold",
            color=WHITE, ha="center", va="center",
            fontfamily="monospace")

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "logo_signal_thread.png"),
                dpi=300, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print("  [1] Signal Thread saved")


# =====================================================================
# LOGO 2: The Waveform Weave (two interlocking waves)
# =====================================================================
def logo_waveform_weave():
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # Two interlocking sine waves in a circular arrangement
    theta = np.linspace(0, 2 * np.pi, 1000)
    r_base = 2.0

    # Wave 1: oscillates outward/inward
    r1 = r_base + 0.35 * np.sin(8 * theta)
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    # Wave 2: phase-shifted
    r2 = r_base + 0.35 * np.sin(8 * theta + np.pi)
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)

    ax.plot(x1, y1, color=TEAL, linewidth=3, alpha=0.9)
    ax.plot(x2, y2, color=PURPLE, linewidth=3, alpha=0.7)

    # Glow ring
    glow_circle = plt.Circle((0, 0), r_base, facecolor="none",
                             edgecolor=TEAL, linewidth=0.5, alpha=0.15)
    ax.add_patch(glow_circle)

    # Center dot
    center = plt.Circle((0, 0), 0.25, facecolor=WHITE, edgecolor="none", alpha=0.9)
    ax.add_patch(center)

    # Node dots at intersections (every 45 degrees)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        nx = r_base * np.cos(angle)
        ny = r_base * np.sin(angle)
        dot = plt.Circle((nx, ny), 0.12, facecolor=TEAL_LIGHT,
                         edgecolor="none", alpha=0.7)
        ax.add_patch(dot)

    # Text below
    ax.text(0, -3.2, "OSCILLOOM", fontsize=22, fontweight="bold",
            color=WHITE, ha="center", va="center", fontfamily="monospace")

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "logo_waveform_weave.png"),
                dpi=300, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print("  [2] Waveform Weave saved")


# =====================================================================
# LOGO 3: Wordmark with O's as nodes
# =====================================================================
def logo_wordmark():
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    # Letter positions and characters
    text = "OSCILLOOM"
    x_positions = np.linspace(1.5, 10.5, len(text))

    for i, (ch, x) in enumerate(zip(text, x_positions)):
        if ch == "O":
            # Draw O as a node circle
            circle = plt.Circle((x, 0), 0.42, facecolor="none",
                               edgecolor=TEAL, linewidth=3, zorder=10)
            ax.add_patch(circle)
            # Inner dot
            inner = plt.Circle((x, 0), 0.1, facecolor=TEAL,
                              edgecolor="none", zorder=11)
            ax.add_patch(inner)
            # Glow
            glow = plt.Circle((x, 0), 0.55, facecolor=TEAL,
                             edgecolor="none", alpha=0.08, zorder=5)
            ax.add_patch(glow)
        else:
            ax.text(x, 0, ch, fontsize=36, fontweight="bold",
                    color=WHITE, ha="center", va="center",
                    fontfamily="monospace")

    # Sine wave connecting the two O's
    o1_x = x_positions[text.index("O")]
    o2_x = x_positions[len(text) - 1 - text[::-1].index("O")]

    t_wave = np.linspace(o1_x + 0.45, o2_x - 0.45, 300)
    wave = 0.12 * np.sin(2 * np.pi * 2.5 * (t_wave - o1_x) / (o2_x - o1_x))
    ax.plot(t_wave, wave - 0.7, color=TEAL, linewidth=1.5, alpha=0.4)

    # Tagline
    ax.text(6, -1.2, "Visual EEG Pipeline Builder", fontsize=12,
            color=SLATE, ha="center", va="center", fontfamily="monospace",
            style="italic")

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "logo_wordmark.png"),
                dpi=300, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print("  [3] Wordmark saved")


# =====================================================================
# LOGO 4: The Shuttle Icon (compact mark for favicon/app icon)
# =====================================================================
def logo_shuttle():
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")
    ax.set_aspect("equal")

    # Shuttle / eye shape using bezier curves
    # Top curve
    verts_top = [(-2, 0), (-1, 1.5), (1, 1.5), (2, 0)]
    codes_top = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    # Bottom curve
    verts_bot = [(-2, 0), (-1, -1.5), (1, -1.5), (2, 0)]
    codes_bot = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    path_top = MplPath(verts_top, codes_top)
    path_bot = MplPath(verts_bot, codes_bot)

    patch_top = patches.PathPatch(path_top, facecolor="none",
                                  edgecolor=TEAL, linewidth=3)
    patch_bot = patches.PathPatch(path_bot, facecolor="none",
                                  edgecolor=TEAL, linewidth=3)
    ax.add_patch(patch_top)
    ax.add_patch(patch_bot)

    # Sine wave through the center
    t = np.linspace(-1.8, 1.8, 300)
    wave = 0.5 * np.sin(2 * np.pi * 1.5 * t / 3.6)
    ax.plot(t, wave, color=TEAL_LIGHT, linewidth=2.5, alpha=0.9)

    # Center node
    center = plt.Circle((0, 0), 0.2, facecolor=WHITE, edgecolor="none")
    ax.add_patch(center)

    # Tip dots
    for x in [-2, 2]:
        dot = plt.Circle((x, 0), 0.12, facecolor=TEAL, edgecolor="none")
        ax.add_patch(dot)

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "logo_shuttle_icon.png"),
                dpi=300, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print("  [4] Shuttle Icon saved")


# =====================================================================
# LOGO 5: Combined — Icon + Wordmark (horizontal lockup)
# =====================================================================
def logo_combined():
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(-1, 14)
    ax.set_ylim(-2, 2)
    ax.axis("off")

    # --- Icon (left side): simplified signal-through-nodes ---
    # Three nodes in a slight arc
    icon_nodes = [(1.0, 0.3), (2.2, -0.1), (3.4, 0.2)]
    node_colors_icon = [TEAL, BLUE, PURPLE]

    for (nx, ny), c in zip(icon_nodes, node_colors_icon):
        circle = plt.Circle((nx, ny), 0.28, facecolor=c, edgecolor=WHITE,
                           linewidth=1.5, zorder=10, alpha=0.9)
        ax.add_patch(circle)
        glow = plt.Circle((nx, ny), 0.4, facecolor=c, edgecolor="none",
                          alpha=0.1, zorder=5)
        ax.add_patch(glow)

    # Connecting wave through nodes
    all_x = [n[0] for n in icon_nodes]
    t_full = np.linspace(all_x[0] - 0.8, all_x[-1] + 0.8, 300)
    wave_full = 0.2 * np.sin(2 * np.pi * 2 * (t_full - all_x[0]))

    # Messy part before first node
    t_pre = t_full[t_full < all_x[0] - 0.28]
    np.random.seed(7)
    messy_pre = wave_full[:len(t_pre)] + 0.15 * np.random.randn(len(t_pre))
    ax.plot(t_pre, messy_pre, color=SLATE, linewidth=1.2, alpha=0.5)

    # Clean part after last node
    t_post = t_full[t_full > all_x[-1] + 0.28]
    clean_post = 0.3 * np.sin(2 * np.pi * 2 * (t_post - all_x[-1]))
    ax.plot(t_post, clean_post + 0.2, color=TEAL_LIGHT, linewidth=2, alpha=0.8)

    # Lines between nodes
    for i in range(len(icon_nodes) - 1):
        x1, y1 = icon_nodes[i]
        x2, y2 = icon_nodes[i + 1]
        t_seg = np.linspace(x1 + 0.3, x2 - 0.3, 80)
        y_seg = np.linspace(y1, y2, 80) + 0.08 * np.sin(2 * np.pi * 4 * (t_seg - x1))
        ax.plot(t_seg, y_seg, color=TEAL_LIGHT, linewidth=1.5, alpha=0.5)

    # --- Wordmark (right side) ---
    ax.text(5.2, 0.15, "OSCILLOOM", fontsize=38, fontweight="bold",
            color=WHITE, ha="left", va="center", fontfamily="monospace")

    ax.text(5.2, -0.85, "Visual EEG Pipeline Builder", fontsize=13,
            color=SLATE, ha="left", va="center", fontfamily="monospace")

    # Subtle underline accent
    ax.plot([5.2, 11.8], [-0.45, -0.45], color=TEAL, linewidth=1.5, alpha=0.3)

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "logo_combined.png"),
                dpi=300, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print("  [5] Combined Lockup saved")


# =====================================================================
# LOGO 6: Light version (white background for papers/docs)
# =====================================================================
def logo_light():
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-1, 14)
    ax.set_ylim(-2, 2)
    ax.axis("off")

    DARK_TEXT = "#1e293b"
    TEAL_DARK = "#0891b2"
    BLUE_DARK = "#2563eb"
    PURPLE_DARK = "#7c3aed"

    # Icon nodes
    icon_nodes = [(1.0, 0.3), (2.2, -0.1), (3.4, 0.2)]
    node_colors_light = [TEAL_DARK, BLUE_DARK, PURPLE_DARK]

    for (nx, ny), c in zip(icon_nodes, node_colors_light):
        circle = plt.Circle((nx, ny), 0.28, facecolor=c, edgecolor="white",
                           linewidth=1.5, zorder=10, alpha=0.9)
        ax.add_patch(circle)

    # Connecting waves
    for i in range(len(icon_nodes) - 1):
        x1, y1 = icon_nodes[i]
        x2, y2 = icon_nodes[i + 1]
        t_seg = np.linspace(x1 + 0.3, x2 - 0.3, 80)
        y_seg = np.linspace(y1, y2, 80) + 0.08 * np.sin(2 * np.pi * 4 * (t_seg - x1))
        ax.plot(t_seg, y_seg, color=TEAL_DARK, linewidth=1.5, alpha=0.4)

    # Messy input
    t_pre = np.linspace(-0.5, 0.72, 100)
    np.random.seed(7)
    messy = 0.2 * np.sin(2 * np.pi * 3 * t_pre) + 0.12 * np.random.randn(100)
    ax.plot(t_pre, messy + 0.3, color="#94a3b8", linewidth=1, alpha=0.5)

    # Clean output
    t_post = np.linspace(3.68, 4.5, 100)
    clean = 0.3 * np.sin(2 * np.pi * 2 * (t_post - 3.68))
    ax.plot(t_post, clean + 0.2, color=TEAL_DARK, linewidth=2, alpha=0.7)

    # Wordmark
    ax.text(5.2, 0.15, "OSCILLOOM", fontsize=38, fontweight="bold",
            color=DARK_TEXT, ha="left", va="center", fontfamily="monospace")
    ax.text(5.2, -0.85, "Visual EEG Pipeline Builder", fontsize=13,
            color="#64748b", ha="left", va="center", fontfamily="monospace")
    ax.plot([5.2, 11.8], [-0.45, -0.45], color=TEAL_DARK, linewidth=1.5, alpha=0.2)

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, "logo_light.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  [6] Light Version saved")


# =====================================================================
# LOGO 7: Favicon / App Icon (square, minimal)
# =====================================================================
def logo_favicon():
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # Three nodes in triangle
    nodes = [(0, 1.2), (-1.1, -0.7), (1.1, -0.7)]
    colors = [TEAL, BLUE, PURPLE]

    for (nx, ny), c in zip(nodes, colors):
        circle = plt.Circle((nx, ny), 0.35, facecolor=c, edgecolor=WHITE,
                           linewidth=2, zorder=10)
        ax.add_patch(circle)
        glow = plt.Circle((nx, ny), 0.5, facecolor=c, edgecolor="none",
                          alpha=0.12, zorder=5)
        ax.add_patch(glow)

    # Connecting sine wave lines between nodes (triangle path)
    for i in range(3):
        x1, y1 = nodes[i]
        x2, y2 = nodes[(i + 1) % 3]
        t = np.linspace(0, 1, 100)
        # Straight line with sine perturbation perpendicular
        mx = x1 + t * (x2 - x1)
        my = y1 + t * (y2 - y1)
        dx = -(y2 - y1)
        dy = (x2 - x1)
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / length, dy / length
        perturb = 0.12 * np.sin(2 * np.pi * 3 * t)
        # Trim to avoid overlapping with node circles
        mask = (t > 0.2) & (t < 0.8)
        ax.plot(mx[mask] + perturb[mask] * dx,
                my[mask] + perturb[mask] * dy,
                color=TEAL_LIGHT, linewidth=1.5, alpha=0.5)

    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(OUT, "logo_favicon.png"),
                dpi=300, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print("  [7] Favicon saved")


# =====================================================================
# Run all
# =====================================================================
print("Generating Oscilloom logos...\n")
logo_signal_thread()
logo_waveform_weave()
logo_wordmark()
logo_shuttle()
logo_combined()
logo_light()
logo_favicon()
print(f"\nAll logos saved to: {OUT}/")
