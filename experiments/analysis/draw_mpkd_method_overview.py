from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image, ImageEnhance, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

NAVY = "#253449"
INK = "#263445"
MUTED = "#64748B"
TEACHER = "#DCEED7"
STUDENT = "#DCEAF7"
WEAK = "#E8F2FA"
STRONG = "#FBE8DD"
PURPLE = "#7764B8"
PURPLE_LIGHT = "#F0ECFA"
ORANGE = "#C56C3D"
GREEN = "#4C8C68"
PAPER = "#FFFFFF"


def example_views():
    image_path = ROOT / "data" / "imagenet" / "train" / "n02058221" / "n02058221_10006.JPEG"
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        original = image.resize((192, 192), Image.Resampling.LANCZOS)
        weak = image.crop((side // 16, side // 16, side - side // 16, side - side // 16))
        weak = weak.resize((192, 192), Image.Resampling.LANCZOS)
        strong = image.crop((side // 12, 0, side, side - side // 12))
        strong = ImageOps.mirror(strong)
        strong = ImageEnhance.Color(strong).enhance(1.8)
        strong = ImageEnhance.Contrast(strong).enhance(1.35)
        strong = strong.resize((192, 192), Image.Resampling.LANCZOS)
        ImageDraw.Draw(strong).rectangle((118, 75, 169, 126), fill=(190, 190, 190))
        return original, weak, strong

    data_dir = ROOT / "data" / "cifar-100-python"
    meta = pickle.load(open(data_dir / "meta", "rb"), encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"fine_label_names"]]
    class_id = class_names.index("aquarium_fish")
    image_array = None
    for batch_name in ("train", "test"):
        batch = pickle.load(open(data_dir / batch_name, "rb"), encoding="bytes")
        labels = batch.get(b"fine_labels", batch.get(b"labels"))
        for idx, label in enumerate(labels):
            if label == class_id:
                image_array = batch[b"data"][idx].reshape(3, 32, 32).transpose(1, 2, 0)
                break
        if image_array is not None:
            break
    if image_array is None:
        raise RuntimeError("Could not find a CIFAR-100 example image.")

    image = Image.fromarray(image_array)
    weak = image.crop((1, 1, 31, 31)).resize((192, 192), Image.Resampling.LANCZOS)
    strong = image.crop((2, 0, 32, 30)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    strong = ImageEnhance.Color(strong).enhance(1.75)
    strong = ImageEnhance.Contrast(strong).enhance(1.35)
    strong = strong.resize((192, 192), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(strong)
    draw.rectangle((116, 72, 170, 126), fill=(190, 190, 190))
    original = image.resize((192, 192), Image.Resampling.LANCZOS)
    return original, weak, strong


def rounded_box(ax, x, y, w, h, *, face, edge=NAVY, lw=1.6, radius=0.12,
                linestyle="solid", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=NAVY, lw=1.7, rad=0.0, ls="solid",
          mutation=14, zorder=1):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(ax, points, color=PURPLE, lw=1.45, ls="dashed", mutation=13):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=mutation,
                            linewidth=lw, color=color, linestyle=ls, zorder=1)
    ax.add_patch(patch)
    return patch


def add_view(ax, image, x, y, title, face):
    w, h = 1.35, 1.10
    rounded_box(ax, x, y, w, h, face=face, lw=1.2, radius=0.07)
    ax.imshow(image, extent=(x + 0.05, x + w - 0.05, y + 0.12, y + h - 0.10),
              interpolation="lanczos", zorder=3, aspect="auto")
    ax.text(x + w / 2, y + h + 0.15, title, ha="center", va="bottom",
            fontsize=14, color=INK, weight="semibold")
    return w, h


def add_model(ax, x, y, w, h, title, status, face, status_color):
    rounded_box(ax, x, y, w, h, face=face, lw=2.0, radius=0.16, zorder=3)
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center",
            fontsize=20, color=INK, weight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.23, status, ha="center", va="center",
            fontsize=11.5, color=status_color, weight="bold", zorder=4)


def add_distribution(ax, x, y, label, values, face, bar_color):
    w, h = 1.70, 0.88
    rounded_box(ax, x, y, w, h, face=face, lw=1.15, radius=0.08, zorder=3)
    ax.text(x + 0.12, y + h - 0.12, label, ha="left", va="top",
            fontsize=13, color=INK, zorder=4)
    left, right = x + 0.12, x + w - 0.12
    base, chart_h = y + 0.12, h - 0.43
    gap = 0.035
    bar_w = (right - left - gap * (len(values) - 1)) / len(values)
    for idx, value in enumerate(values):
        bx = left + idx * (bar_w + gap)
        bh = chart_h * value / max(values)
        ax.add_patch(Rectangle((bx, base), bar_w, bh, facecolor=bar_color,
                               edgecolor="none", zorder=4))
    ax.plot([left, right], [base, base], color="#91A0B1", lw=0.55, zorder=4)
    return w, h


def add_proto_bank(ax):
    x, y, w, h = 10.0, 6.72, 8.35, 0.90
    rounded_box(ax, x, y, w, h, face="#F7F5FC", edge=PURPLE, lw=1.4, radius=0.1)
    ax.text(x + 0.20, y + h - 0.18, "OFFLINE  |  Fixed prototype bank",
            fontsize=14, weight="bold", color=PURPLE, va="top")
    ax.text(x + 0.20, y + 0.20,
            "Reliable weak-view features  →  class-wise K-Means (K = 3)",
            fontsize=10.5, color=INK, va="bottom")
    # Small centers visualize multiple local modes without implying learned parameters.
    groups = [16.55, 17.35, 18.10]
    colors = ["#6D5AAE", "#548B70", "#C67B4B", "#4D85A8", "#B05E78"]
    for group_idx, (gx, color) in enumerate(zip(groups, colors)):
        for k, offset in enumerate((-0.16, 0.0, 0.16)):
            ax.scatter(gx + offset, y + 0.48 + (0.045 if k == 1 else 0),
                       s=27, color=color, edgecolors="white", linewidths=0.5,
                       zorder=4)
        if group_idx < len(groups) - 1:
            ax.text(gx + 0.40, y + 0.48, "···", fontsize=13,
                    color=MUTED, ha="center", va="center")


def create_figure():
    original, weak, strong = example_views()
    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(16.5, 7.14))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 8.3)
    ax.axis("off")

    # Main framework: input and paired perturbation views.
    ax.text(0.98, 4.98, "Input", ha="center", va="bottom", fontsize=14,
            color=INK, weight="semibold")
    rounded_box(ax, 0.34, 3.64, 1.28, 1.12, face="white", edge="#9AA8B8",
                lw=1.2, radius=0.08)
    ax.imshow(original, extent=(0.39, 1.57, 3.69, 4.71), interpolation="lanczos",
              zorder=3, aspect="auto")
    ax.text(0.98, 3.47, r"$x_i$", ha="center", va="top", fontsize=13, color=INK)

    weak_xy = (2.05, 5.66)
    strong_xy = (2.05, 1.47)
    add_view(ax, weak, *weak_xy, "Weak perturbation", WEAK)
    add_view(ax, strong, *strong_xy, "", STRONG)
    ax.text(2.725, 2.90, "Strong\nperturbation", ha="center", va="center",
            fontsize=14, color=INK, weight="semibold", linespacing=1.0)
    arrow(ax, (1.62, 4.23), (2.02, 6.10), color=NAVY, lw=1.55, rad=-0.08)
    arrow(ax, (1.62, 4.14), (2.02, 2.03), color=NAVY, lw=1.55, rad=0.08)
    ax.text(1.83, 6.38, r"$x_{w,i}$", fontsize=12, color=INK, ha="right")
    ax.text(1.83, 1.28, r"$x_{s,i}$", fontsize=12, color=INK, ha="right")

    # Frozen teacher and trainable student both process the two views.
    teacher_box = (4.00, 4.92, 1.72, 0.94)
    student_box = (4.00, 2.02, 1.72, 0.94)
    arrow(ax, (3.40, 6.20), (3.98, 5.51), color=NAVY, lw=1.45, rad=0.02)
    arrow(ax, (3.40, 2.01), (3.98, 5.20), color=NAVY, lw=1.45, rad=-0.10)
    arrow(ax, (3.40, 6.06), (3.98, 2.56), color=NAVY, lw=1.45, rad=0.10)
    arrow(ax, (3.40, 1.98), (3.98, 2.33), color=NAVY, lw=1.45, rad=-0.02)
    add_model(ax, *teacher_box, "Teacher", "FROZEN", TEACHER, GREEN)
    add_model(ax, *student_box, "Student", "TRAINABLE", STUDENT, "#3D7397")

    # Teacher and student probability outputs for each view.
    dist_x = 6.50
    dist_nodes = {
        "tw": (dist_x, 6.30, r"$\widetilde{p}_{w,i}^{T}$", [0.07, 0.10, 0.13, 0.56, 0.08, 0.04, 0.02], "#EDF5E9", "#466A47"),
        "ts": (dist_x, 4.83, r"$\widetilde{p}_{s,i}^{T}$", [0.12, 0.17, 0.20, 0.18, 0.12, 0.11, 0.10], "#FFF0E8", "#A35D39"),
        "sw": (dist_x, 2.69, r"$\widetilde{p}_{w,i}^{S}$", [0.07, 0.10, 0.16, 0.44, 0.10, 0.08, 0.05], "#EEF5FB", "#476C8A"),
        "ss": (dist_x, 1.22, r"$\widetilde{p}_{s,i}^{S}$", [0.15, 0.16, 0.18, 0.20, 0.12, 0.11, 0.08], "#FCEEF0", "#9B5361"),
    }
    arrow(ax, (5.72, 5.62), (6.47, 6.74), color=NAVY, lw=1.3, rad=-0.10)
    arrow(ax, (5.72, 5.32), (6.47, 5.27), color=NAVY, lw=1.3)
    arrow(ax, (5.72, 2.61), (6.47, 3.13), color=NAVY, lw=1.3, rad=0.10)
    arrow(ax, (5.72, 2.35), (6.47, 1.66), color=NAVY, lw=1.3, rad=-0.10)
    for key, (x, y, label, vals, face, bar) in dist_nodes.items():
        add_distribution(ax, x, y, label, vals, face, bar)

    # Unmodified CRLD cross-view paths and weak-view same-view path.
    # Strong-view same-view KD uses the reconstructed target from the MPKD panel.
    arrow(ax, (8.23, 6.67), (8.23, 3.13), color=GREEN, lw=1.65, rad=0.19)
    arrow(ax, (8.23, 6.65), (8.23, 1.66), color=ORANGE, lw=1.45, rad=-0.11)
    arrow(ax, (8.23, 5.24), (8.23, 3.13), color=ORANGE, lw=1.45, rad=0.09)

    # MPKD module, visually separated from the unchanged two-view backbone.
    outer = FancyBboxPatch(
        (9.48, 1.10), 9.42, 7.02,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#FCFBFE", edgecolor=PURPLE, linewidth=2.0,
        linestyle=(0, (5, 4)), zorder=0,
    )
    ax.add_patch(outer)
    ax.text(9.82, 7.84, "MPKD  |  Prototype-guided target reconstruction",
            fontsize=17, weight="bold", color=PURPLE, va="center")
    add_proto_bank(ax)

    # Offline prototype mining uses weak-view teacher features; online matching
    # uses the teacher feature from the strong view.
    routed_arrow(ax, [(5.72, 5.73), (6.05, 7.55), (9.70, 7.55), (10.03, 7.14)])
    ax.text(7.15, 7.63, r"$f_{w,i}^{T}$", fontsize=12, color=PURPLE,
            ha="center", va="bottom",
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.1})

    # Strong-view teacher feature and prototype matching.
    rounded_box(ax, 9.98, 4.85, 1.35, 0.94, face="#EDF3F8", edge="#56728A", lw=1.25)
    ax.text(10.655, 5.38, "Feature", ha="center", va="center",
            fontsize=13, weight="semibold", color=INK)
    ax.text(10.655, 5.08, r"$f_{s,i}^{T}$", ha="center", va="center",
            fontsize=13, color=INK)
    routed_arrow(ax, [(5.72, 4.96), (6.05, 4.43), (9.62, 4.43), (9.98, 5.03)])

    rounded_box(ax, 11.57, 4.77, 1.78, 1.10, face=PURPLE_LIGHT, edge=PURPLE, lw=1.25)
    ax.text(12.46, 5.56, "Prototype\nmatch", ha="center", va="center",
            fontsize=13, weight="semibold", color=INK, linespacing=1.0)
    ax.text(12.46, 5.00, "Cosine sim.\nLog-Mean-Exp", ha="center", va="center",
            fontsize=10.0, color=MUTED, linespacing=1.05)
    arrow(ax, (11.34, 5.31), (11.55, 5.31), color=PURPLE, lw=1.55)
    arrow(ax, (12.65, 6.70), (12.52, 5.89), color=PURPLE, lw=1.35,
          rad=0.08, ls="dashed", mutation=12)

    rounded_box(ax, 13.58, 4.87, 1.43, 0.92, face="#F7F5FC", edge=PURPLE, lw=1.25)
    ax.text(14.295, 5.53, "Prototype", ha="center", va="center",
            fontsize=11.8, color=INK, weight="semibold")
    ax.text(14.295, 5.17, r"$p_i^{\mathrm{proto}}$", ha="center", va="center",
            fontsize=14, color=PURPLE)
    arrow(ax, (13.37, 5.31), (13.56, 5.31), color=PURPLE, lw=1.55)

    # Reliability factors are grouped as one gate, matching the paper's concise overview.
    rounded_box(ax, 15.25, 4.70, 3.22, 1.22, face="#F2EFFA", edge=PURPLE, lw=1.5)
    ax.text(16.86, 5.60, "Reliability gate", ha="center", va="center",
            fontsize=14, color=INK, weight="bold")
    ax.text(16.86, 5.16, "weak anchor  ·  uncertainty\ngain  ·  proto confidence",
            ha="center", va="center", fontsize=10.8, color=MUTED, linespacing=1.25)
    arrow(ax, (15.03, 5.31), (15.23, 5.31), color=PURPLE, lw=1.55)

    rounded_box(ax, 15.25, 2.52, 3.22, 1.38, face="#FFF5E8", edge=ORANGE, lw=1.45)
    ax.text(16.86, 3.60, "Strong-view target", ha="center", va="center",
            fontsize=13.0, color=INK, weight="semibold")
    ax.text(16.86, 3.12,
            r"$\widehat{p}_{s,i}^{T}=(1-\lambda_i)\widetilde{p}_{s,i}^{T}$"
            "\n" r"$+\;\lambda_i p_i^{\mathrm{proto}}$",
            ha="center", va="center", fontsize=13.5, color=INK, linespacing=1.18)
    arrow(ax, (16.86, 4.68), (16.86, 3.93), color=PURPLE, lw=1.65)
    ax.text(16.99, 4.27, r"$\lambda_i$", fontsize=11, color=PURPLE, va="center")

    # The corrected target returns only to strong-view same-view distillation.
    arrow(ax, (15.23, 2.75), (8.23, 1.40), color=PURPLE, lw=1.85,
          rad=-0.13, ls="dashed", mutation=15, zorder=2)
    ax.text(11.55, 1.57, r"$\widehat{p}_{s,i}^{T}$  →  $S_s$ (KD-WV)",
            fontsize=11.3, color=PURPLE, ha="center", va="bottom", weight="semibold",
            bbox={"facecolor": "#FCFBFE", "edgecolor": "none", "pad": 1.8})

    # Compact objective legend makes the exact unchanged cross-view targets explicit.
    legend_y, legend_h = 0.08, 0.76
    rounded_box(ax, 0.22, legend_y, 18.68, legend_h,
                face="#F8FAFC", edge="#CAD3DE", lw=0.9, radius=0.08, zorder=0)
    ax.text(0.52, 0.61, "TRAINING OBJECTIVES", fontsize=10.5,
            color=MUTED, weight="bold", va="center")
    ax.text(3.10, 0.59, r"CE:  $y_i \rightarrow S_w,\;S_s$",
            fontsize=12.2, color=INK, va="center")
    ax.plot([6.05, 6.05], [0.22, 0.69], color="#D2DAE3", lw=0.9)
    ax.text(6.30, 0.59,
            r"KD-WV:  $T_w\to S_w$;  $\widehat{T}_s\to S_s$",
            fontsize=12.2, color=GREEN, va="center", weight="semibold")
    ax.plot([11.45, 11.45], [0.22, 0.69], color="#D2DAE3", lw=0.9)
    ax.text(11.70, 0.59,
            r"KD-CV:  $T_w\to S_s$;  $T_s\to S_w$  (unchanged)",
            fontsize=12.0, color=ORANGE, va="center", weight="semibold")

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99)
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() * 1.25)
    png = OUT / "mpkd_method_overview.png"
    svg = OUT / "mpkd_method_overview.svg"
    fig.savefig(png, dpi=320, facecolor=PAPER, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg, facecolor=PAPER, bbox_inches="tight", pad_inches=0.05,
                metadata={"Title": "MPKD Method Overview", "Description": "Multi-center prototype knowledge distillation framework"})
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    create_figure()
