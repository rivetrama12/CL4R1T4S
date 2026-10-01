#!/usr/bin/env python3
"""Nominal section of 4.8 Lock Bolt Flat Head, measured from the STEP solid.

The STEP file stores one axisymmetric solid. No PMI, tolerances, material, or
finish. Model axis is Y, millimetres. Datum A is the head top face (Y = 0).
Positive Y runs toward the tail, left to right here.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent
STEP_PATH = ROOT / "4.8_Lock_Bolt_Flat_Head.step"
OUT_PATH = ROOT / "4.8_Lock_Bolt_Flat_Head_DIMENSIONS.pdf"
PREVIEW = ROOT / "preview" / "01-lock-bolt.png"

PAGE_W, PAGE_H = 420.0, 297.0
TOTAL = "1"

DIM = "#0e3a5d"
EXT = "#8aa0b4"
CENTER = "#a32020"
RULE = "#1c1c1c"
INK = "#1a1a1a"
MUTED = "#4a5160"
PANEL = "#f4f7fb"
LINE = "#c5d2e0"
BOLT_FC = "#d5e0ea"

FP = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Regular.ttf")
FP_MED = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Medium.ttf")
FP_BOLD = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf")
FP_MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/JetBrainsMono-Regular.ttf")
FONTS = {"regular": FP, "medium": FP_MED, "bold": FP_BOLD, "mono": FP_MONO}

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["hatch.linewidth"] = 0.3
plt.rcParams["hatch.color"] = "#7f93a8"

# Radii and stations measured from the analytic edges. The taper axial length
# is (2.45 - 2.125) * sqrt(3), because the cone is 30° from the axis.
R_FLAT = 2.5
R_FILLET = 1.5
R_HEAD = 4.0
R_SHANK = 2.45
R_TAIL = 2.125
Y_LAND = 1.5
Y_SHOULDER = 2.0
Y_TAIL = 14.7
Y_END = 47.0
TAPER_DEG = 30.0
Y_TAPER = Y_TAIL - (R_SHANK - R_TAIL) * math.sqrt(3)


def T(ax, x, y, text, size=7.0, weight="regular", color=INK, ha="left", va="center", z=5, **kw):
    ax.text(
        x, y, text,
        fontsize=size, fontproperties=FONTS[weight], color=color,
        ha=ha, va=va, zorder=z, **kw,
    )


def fmm(v: float) -> str:
    if abs(v) < 5e-7:
        v = 0.0
    if abs(v - round(v, 3)) <= 5e-7:
        return f"{v:.3f}"
    return f"{v:.5f}"


class View:
    def __init__(self, x_at_y0, y0, scale, r0=0.0):
        self.x0, self.y0, self.s, self.r0 = x_at_y0, y0, scale, r0

    def pt(self, y, r=0.0):
        return (self.x0 + y * self.s, self.y0 + (r - self.r0) * self.s)

    def label(self) -> str:
        if abs(self.s - round(self.s)) < 1e-9:
            return f"SCALE {int(round(self.s))} : 1"
        return f"SCALE {self.s:.1f} : 1"


def assert_model(text: str) -> None:
    """The drawing numbers have to be the surfaces stored in the STEP file."""
    if "SI_UNIT(.MILLI.,.METRE.)" not in text and "SI_UNIT ( .MILLI. , .METRE. )" not in text:
        raise SystemExit("STEP length unit is not millimetres")
    cyl = {float(v) for v in re.findall(r"CYLINDRICAL_SURFACE\('',#\d+,([0-9.]+)\)", text)}
    if cyl != {R_HEAD, R_SHANK, R_TAIL}:
        raise SystemExit(f"cylinder radii {cyl}")
    tori = re.findall(r"TOROIDAL_SURFACE\('',#\d+,([0-9.]+),([0-9.]+)\)", text)
    if tori != [(f"{R_FLAT:g}", f"{R_FILLET:g}")]:
        raise SystemExit(f"torus {tori}")
    cones = re.findall(r"CONICAL_SURFACE\('',#\d+,([0-9.]+),([0-9.eE+-]+)\)", text)
    if len(cones) != 1:
        raise SystemExit(f"cone count {len(cones)}")
    radius, angle = float(cones[0][0]), float(cones[0][1])
    if abs(radius - R_SHANK) > 1e-9 or abs(angle - math.radians(TAPER_DEG)) > 1e-9:
        raise SystemExit(f"cone {radius} {angle}")
    if abs((R_SHANK - R_TAIL) * math.sqrt(3) - (Y_TAIL - Y_TAPER)) > 1e-9:
        raise SystemExit("taper is not 30 degrees from the axis")


def profile():
    """Upper meridional outline, head to tail, including the closing axis."""
    pts = [(0.0, 0.0), (R_FLAT, 0.0)]
    # Quarter circle, centre (R_FLAT, Y_LAND), from the top face to the land.
    cx, cy, rad = R_FLAT, Y_LAND, R_FILLET
    for i in range(1, 25):
        ang = -math.pi / 2 + (math.pi / 2) * i / 24
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    pts.extend([
        (R_HEAD, Y_SHOULDER),
        (R_SHANK, Y_SHOULDER),
        (R_SHANK, Y_TAPER),
        (R_TAIL, Y_TAIL),
        (R_TAIL, Y_END),
        (0.0, Y_END),
    ])
    return pts


def new_page():
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def frame(ax):
    ax.add_patch(Rectangle((6, 6), 408, 285, fill=False, lw=1.1, ec=RULE, zorder=6))
    ax.add_patch(Rectangle((8, 8), 404, 281, fill=False, lw=0.35, ec=RULE, zorder=6))
    T(ax, 12, 286.4, "4.8 LOCK BOLT", size=10.2, weight="bold", va="top")
    T(ax, 58, 286.4, "FLAT HEAD   ·   REV A   ·   NOMINAL", size=7.6, weight="medium", color=DIM, va="top")
    T(ax, 406, 286.4, "SHEET 1   SECTION THROUGH AXIS", size=7.0, weight="medium", ha="right", va="top", color="#333")
    x, y, w, h = 248, 8, 164, 40
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=RULE, lw=0.7, zorder=6))
    ax.plot([x, x + w], [y + 28, y + 28], color=RULE, lw=0.35, zorder=6)
    ax.plot([x, x + w], [y + 16, y + 16], color=RULE, lw=0.35, zorder=6)
    ax.plot([x + 82, x + 82], [y, y + h], color=RULE, lw=0.35, zorder=6)
    T(ax, x + 3, y + 34, "DRAWING", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 24, y + 34, "4.8-LB-FH-A", size=7.2, weight="bold", va="center", z=7)
    T(ax, x + 85, y + 34, "SHEET", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 108, y + 34, f"1  /  {TOTAL}", size=7.4, weight="bold", va="center", z=7)
    T(ax, x + 3, y + 22, "PART", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 24, y + 22, "LOCK BOLT FLAT HEAD", size=6.2, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 22, "UNITS", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 108, y + 22, "MILLIMETRES", size=6.6, weight="medium", va="center", z=7)
    T(ax, x + 3, y + 8, "DATUM A   HEAD FACE Y = 0", size=5.2, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 8, "DATUM B   AXIS", size=5.2, weight="medium", va="center", z=7)
    cx, cy = 214, 22
    ax.add_patch(Circle((cx, cy + 8), 3.1, fill=False, lw=0.55, ec=RULE, zorder=6))
    ax.plot([cx + 5.0, cx + 12.2], [cy + 5.4, cy + 6.6], color=RULE, lw=0.5, zorder=6)
    ax.plot([cx + 5.0, cx + 12.2], [cy + 10.6, cy + 9.4], color=RULE, lw=0.5, zorder=6)
    ax.plot([cx + 12.2, cx + 12.2], [cy + 6.6, cy + 9.4], color=RULE, lw=0.5, zorder=6)
    T(ax, cx + 6.2, cy + 1.0, "3RD ANGLE", size=4.3, ha="center", color="#444", z=7)


def fill_profile(ax, view, pts):
    upper = [view.pt(y, r) for y, r in pts]
    lower = [view.pt(y, -r) for y, r in reversed(pts)]
    ax.add_patch(Polygon(upper + lower, closed=True, fc=BOLT_FC, ec=INK, lw=0.8, hatch="////", joinstyle="round", zorder=2))


def centerline(ax, view, y0, y1, pad=3.0):
    x0, y = view.pt(y0, 0)
    x1, _ = view.pt(y1, 0)
    ax.plot([x0 - pad, x1 + pad], [y, y], color=CENTER, lw=0.45, linestyle=(0, (7, 1.5, 0.9, 1.5)), zorder=3)


def hdim(ax, x1, x2, y, text, ey1=None, ey2=None, text_side="up"):
    if x1 > x2:
        x1, x2 = x2, x1
        ey1, ey2 = ey2, ey1
    over = 1.2
    for x, ey in ((x1, ey1), (x2, ey2)):
        if ey is None:
            continue
        y_end = y + over if y >= ey else y - over
        ax.plot([x, x], [ey, y_end], color=EXT, lw=0.28, zorder=3)
    gap = x2 - x1
    bbox = dict(fc="white", ec="none", pad=0.15, alpha=0.94)
    if gap >= 16:
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.2, shrinkA=0, shrinkB=0),
            zorder=4,
        )
        dy = 0.9 if text_side == "up" else -0.8
        va = "bottom" if text_side == "up" else "top"
        T(ax, (x1 + x2) / 2, y + dy, text, size=6.1, weight="medium", color=DIM, ha="center", va=va, z=5, bbox=bbox)
    else:
        wing = 6.0
        ax.plot([x1 - wing, x2 + wing], [y, y], color=DIM, lw=0.45, zorder=4)
        ax.annotate("", xy=(x1, y), xytext=(x1 - wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=4)
        ax.annotate("", xy=(x2, y), xytext=(x2 + wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=4)
        T(ax, (x1 + x2) / 2, y + (1.7 if text_side == "up" else -1.7), text, size=5.7, weight="medium",
          color=DIM, ha="center", va="bottom" if text_side == "up" else "top", z=5, bbox=bbox)


def leader(ax, x, y, tx, ty, text, ha="left", size=6.2):
    ax.annotate(
        "", xy=(x, y), xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.4, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.plot([x], [y], marker="o", ms=1.6, color=DIM, zorder=5)
    dx = 1.1 if ha == "left" else (-1.1 if ha == "right" else 0.0)
    T(ax, tx + dx, ty, text, size=size, weight="medium", color=DIM, ha=ha, va="center", z=5,
      bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.94))


def draw_table(ax, x, y_top, widths, headers, rows, row_h=5.0, size=5.4, title=None):
    if title:
        T(ax, x, y_top + 1.4, title, size=6.5, weight="bold", color=DIM, va="bottom")
    total_w = sum(widths)
    head_h = row_h + 0.6
    y = y_top - head_h
    ax.add_patch(Rectangle((x, y), total_w, head_h, fc="#e7eef6", ec=LINE, lw=0.3, zorder=2))
    cx = x
    for head, w in zip(headers, widths):
        T(ax, cx + 1.0, y + head_h / 2, head, size=size, weight="bold", color=DIM, va="center", z=3)
        cx += w
    for i, row in enumerate(rows):
        y -= row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((x, y), total_w, row_h, fc="#fbfcfe", ec="none", zorder=1))
        cx = x
        for cell, w in zip(row, widths):
            T(ax, cx + 1.0, y + row_h / 2, str(cell), size=size, weight="mono", va="center", color="#1c1c1c", z=3)
            cx += w
    ax.add_patch(Rectangle((x, y), total_w, y_top - y, fill=False, ec=LINE, lw=0.45, zorder=3))
    cx = x
    for w in widths[:-1]:
        cx += w
        ax.plot([cx, cx], [y, y_top], color=LINE, lw=0.25, zorder=3)
    return y


def head_profile(pts):
    """Head, shoulder, and a short shank stub. The tail stays on the main view."""
    stub = Y_SHOULDER + 0.55
    kept = [(y, r) for y, r in pts if y <= Y_SHOULDER + 1e-9]
    kept.append((stub, R_SHANK))
    kept.append((stub, 0.0))
    return kept


def draw_main(ax, view, pts):
    fill_profile(ax, view, pts)
    centerline(ax, view, -1.2, Y_END + 1.5, pad=4)
    top = view.y0 + R_HEAD * view.s
    bot = view.y0 - R_HEAD * view.s
    x0 = view.pt(0, 0)[0]
    x_sh = view.pt(Y_SHOULDER, 0)[0]
    x_tp = view.pt(Y_TAPER, 0)[0]
    x_tl = view.pt(Y_TAIL, 0)[0]
    x1 = view.pt(Y_END, 0)[0]
    hdim(ax, x0, x1, top + 16, fmm(Y_END), ey1=top, ey2=view.y0 + R_TAIL * view.s)
    T(ax, view.pt(28, 0)[0], top + 7, view.label() + "    ·    SECTION THROUGH AXIS",
      size=5.6, weight="medium", color=MUTED, va="center")
    hdim(ax, x_sh, x_tp, bot - 10, fmm(Y_TAPER - Y_SHOULDER), ey1=bot, ey2=view.y0 - R_SHANK * view.s, text_side="down")
    hdim(ax, x_tl, x1, bot - 10, fmm(Y_END - Y_TAIL), ey1=view.y0 - R_TAIL * view.s, ey2=view.y0 - R_TAIL * view.s, text_side="down")
    leader(ax, *view.pt(0.2, R_FLAT), x0 + 2, view.y0 + 18, f"Ø{fmm(2 * R_FLAT)}", ha="left")
    leader(ax, *view.pt(0.75, R_FLAT + 0.85), x0 + 14, top + 6, "R1.500", ha="left")
    leader(ax, *view.pt(Y_LAND + 0.2, R_HEAD), x_sh + 16, top + 5, f"Ø{fmm(2 * R_HEAD)}", ha="left")
    T(ax, *view.pt((Y_SHOULDER + Y_TAPER) / 2, 0), f"Ø{fmm(2 * R_SHANK)}", size=6.4, weight="medium",
      color=DIM, ha="center", va="center", z=5, bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.92))
    T(ax, *view.pt((Y_TAIL + Y_END) / 2, 0), f"Ø{fmm(2 * R_TAIL)}", size=6.4, weight="medium",
      color=DIM, ha="center", va="center", z=5, bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.92))
    T(ax, x0 - 1.5, view.y0 - 6, "A", size=6.5, weight="bold", color=CENTER, ha="right", va="center")
    ax.plot([x0, x0], [bot - 1, top + 1], color=CENTER, lw=0.35, zorder=3)
    T(ax, (x0 + x1) / 2, bot - 22, "HEAD AT DATUM A, LEFT.   TAIL TO THE RIGHT.", size=5.4, color=MUTED, ha="center")


def draw_head(ax, view, pts):
    fill_profile(ax, view, head_profile(pts))
    centerline(ax, view, -0.15, Y_SHOULDER + 0.7, pad=2)
    top = view.y0 + R_HEAD * view.s
    T(ax, view.pt(0.05, 0)[0], top + 16, "DETAIL A   " + view.label(),
      size=5.6, weight="medium", color=MUTED, va="bottom")
    x0 = view.pt(0, 0)[0]
    x_land = view.pt(Y_LAND, 0)[0]
    x_sh = view.pt(Y_SHOULDER, 0)[0]
    top = view.y0 + R_HEAD * view.s
    hdim(ax, x0, x_sh, top + 8, fmm(Y_SHOULDER), ey1=view.y0 + R_FLAT * view.s, ey2=top)
    hdim(ax, x_land, x_sh, view.y0 - R_HEAD * view.s - 8, fmm(Y_SHOULDER - Y_LAND),
         ey1=view.y0 - R_HEAD * view.s, ey2=view.y0 - R_HEAD * view.s, text_side="down")
    leader(ax, *view.pt(0.08, R_FLAT), x0 + 6, view.y0 + R_FLAT * view.s + 6, f"Ø{fmm(2 * R_FLAT)}", ha="left")
    leader(ax, *view.pt(Y_LAND + 0.2, R_HEAD), view.pt(Y_LAND, R_HEAD)[0] + 8, top + 3, f"Ø{fmm(2 * R_HEAD)}", ha="left")
    # Radius to the arc, away from the metal.
    mid_ang = -math.pi / 4
    mr = R_FLAT + R_FILLET * math.cos(mid_ang)
    my = Y_LAND + R_FILLET * math.sin(mid_ang)
    leader(ax, *view.pt(my, mr), *view.pt(0.55, R_HEAD + 0.55), f"R{fmm(R_FILLET)}", ha="left")
    leader(ax, *view.pt(Y_SHOULDER, (R_HEAD + R_SHANK) / 2 + 0.15), *view.pt(Y_SHOULDER + 0.35, R_SHANK - 0.8),
           f"Ø{fmm(2 * R_SHANK)}", ha="left", size=5.8)


def draw_taper(ax, view):
    # Upper half only. r0 keeps the axis off this window.
    seg = [
        (Y_TAPER - 0.55, R_SHANK),
        (Y_TAPER, R_SHANK),
        (Y_TAIL, R_TAIL),
        (Y_TAIL + 0.7, R_TAIL),
    ]
    xy = [view.pt(y, r) for y, r in seg]
    base = view.y0 - 8
    poly = xy + [(xy[-1][0], base), (xy[0][0], base)]
    ax.add_patch(Polygon(poly, closed=True, fc=BOLT_FC, ec=INK, lw=0.8, hatch="////", zorder=2))
    T(ax, xy[0][0], view.y0 + (R_SHANK - view.r0) * view.s + 10, "DETAIL B   " + view.label(),
      size=5.6, weight="medium", color=MUTED, va="bottom")
    y_dim = view.y0 - 4
    hdim(ax, view.pt(Y_TAPER, 0)[0], view.pt(Y_TAIL, 0)[0], y_dim, fmm(Y_TAIL - Y_TAPER),
         ey1=view.pt(Y_TAPER, R_SHANK)[1], ey2=view.pt(Y_TAIL, R_TAIL)[1], text_side="down")
    leader(ax, *view.pt(Y_TAPER - 0.25, R_SHANK), *view.pt(Y_TAPER - 0.45, R_SHANK + 0.22), f"Ø{fmm(2 * R_SHANK)}", ha="right")
    leader(ax, *view.pt(Y_TAIL + 0.35, R_TAIL), *view.pt(Y_TAIL + 0.55, R_TAIL + 0.28), f"Ø{fmm(2 * R_TAIL)}", ha="left")
    # 30° from the axis. The axis is below the window; the callout states it.
    leader(ax, *view.pt((Y_TAPER + Y_TAIL) / 2, (R_SHANK + R_TAIL) / 2),
           *view.pt(Y_TAPER + 0.15, R_SHANK + 0.28), "30° FROM AXIS", ha="left", size=5.8)


def segment_rows():
    return [
        ["1", "Top face", fmm(0), fmm(0), "0", "0", fmm(2 * R_FLAT), "—"],
        ["2", "Head radius", fmm(0), fmm(Y_LAND), fmm(Y_LAND), fmm(2 * R_FLAT), fmm(2 * R_HEAD), f"R{fmm(R_FILLET)}"],
        ["3", "Head land", fmm(Y_LAND), fmm(Y_SHOULDER), fmm(Y_SHOULDER - Y_LAND), fmm(2 * R_HEAD), fmm(2 * R_HEAD), "—"],
        ["4", "Shoulder", fmm(Y_SHOULDER), fmm(Y_SHOULDER), "0", fmm(2 * R_HEAD), fmm(2 * R_SHANK), "—"],
        ["5", "Shank", fmm(Y_SHOULDER), fmm(Y_TAPER), fmm(Y_TAPER - Y_SHOULDER), fmm(2 * R_SHANK), fmm(2 * R_SHANK), "—"],
        ["6", "Taper", fmm(Y_TAPER), fmm(Y_TAIL), fmm(Y_TAIL - Y_TAPER), fmm(2 * R_SHANK), fmm(2 * R_TAIL), "30°"],
        ["7", "Tail", fmm(Y_TAIL), fmm(Y_END), fmm(Y_END - Y_TAIL), fmm(2 * R_TAIL), fmm(2 * R_TAIL), "—"],
        ["8", "End face", fmm(Y_END), fmm(Y_END), "0", fmm(2 * R_TAIL), "0", "—"],
    ]


def main():
    assert_model(STEP_PATH.read_text())
    pts = profile()
    fig, ax = new_page()
    frame(ax)
    main_view = View(24, 214, 3.2)
    draw_main(ax, main_view, pts)
    head_view = View(22, 108, 10)
    draw_head(ax, head_view, pts)
    # Place the taper so Y_TAPER lands near x = 168.
    taper_scale = 28
    taper_view = View(168 - Y_TAPER * taper_scale, 92, taper_scale, r0=2.02)
    draw_taper(ax, taper_view)
    draw_table(
        ax, 228, 262,
        [8, 28, 22, 22, 22, 18, 18, 16],
        ["#", "FEATURE", "Y FROM", "Y TO", "LENGTH", "Ø FROM", "Ø TO", "ANGLE"],
        segment_rows(),
        row_h=4.8,
        title="PROFILE, HEAD TO TAIL",
    )
    T(ax, 230, 214, "Angle is from datum B.  30° from the axis is 60° included.", size=5.3, color=MUTED, va="top")
    T(ax, 230, 208, "The part is named 4.8. The solid measures Ø4.900 on the shank.", size=5.3, color=MUTED, va="top")
    T(ax, 12, 14, "NOMINAL GEOMETRY FROM THE STEP FILE.   NO PMI, TOLERANCES, OR MATERIAL ARE STORED.", size=5.2, color=MUTED, va="center", z=7)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW)
    with PdfPages(OUT_PATH) as pdf:
        pdf.infodict()["Title"] = "4.8 Lock Bolt Flat Head — nominal dimensions"
        pdf.savefig(fig)
    plt.close(fig)
    print(OUT_PATH, OUT_PATH.stat().st_size)


if __name__ == "__main__":
    main()
