#!/usr/bin/env python3
"""Nominal dimensioned drawing of part 4.8 LB, taken from the STEP solid.

The source model is a turned solid of revolution (SolidWorks name "1 SMALL NUT").
The STEP file stores geometry only: no PMI, tolerances, material, or finish.
Graphic callouts are rounded to 0.001 mm. The segment table keeps 0.00001 mm,
which matches the model's stated distance accuracy.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Circle
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

ROOT = Path(__file__).resolve().parent
STEP_PATH = ROOT / "4.8_LB.STEP"
OUT_PATH = ROOT / "4.8_LB_all_dimensions.pdf"

PAGE_W, PAGE_H = 420.0, 297.0

GEO = "#161616"
DIM = "#0e3a5d"
EXT = "#8aa0b4"
CENTER = "#a32020"
HAIR = "#9aa8b5"
RULE = "#1c1c1c"

FP = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Regular.ttf")
FP_MED = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Medium.ttf")
FP_BOLD = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf")
FP_MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/JetBrainsMono-Regular.ttf")
FONTS = {"regular": FP, "medium": FP_MED, "bold": FP_BOLD, "mono": FP_MONO}


def T(ax, x, y, text, size=7.0, weight="regular", color=RULE, ha="left", va="center", z=5, **kw):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontproperties=FONTS[weight],
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        **kw,
    )


def f3(v: float) -> str:
    return f"{v:.3f}"


def f5(v: float) -> str:
    return f"{v:.5f}"


def diam(r: float) -> float:
    return 2.0 * r


class View:
    """Maps model (z, r) onto the sheet, in millimetres."""

    def __init__(self, zx, y0, scale, z0=0.0, r0=0.0):
        self.zx = zx
        self.y0 = y0
        self.s = scale
        self.z0 = z0
        self.r0 = r0

    def pt(self, z, r):
        return (self.zx + (z - self.z0) * self.s, self.y0 + (r - self.r0) * self.s)

    def zx_of(self, z):
        return self.zx + (z - self.z0) * self.s

    def ry_of(self, r):
        return self.y0 + (r - self.r0) * self.s


def _in_sweep(a0, sweep, angle):
    rel = (angle - a0) % (2.0 * math.pi)
    if sweep < 0.0:
        rel -= 2.0 * math.pi
        return sweep - 1e-8 <= rel <= 1e-8
    return -1e-8 <= rel <= sweep + 1e-8


def arc_points(cx, cz, radius, p0, p1, n=72):
    a0 = math.atan2(p0[1] - cz, p0[0] - cx)
    a1 = math.atan2(p1[1] - cz, p1[0] - cx)
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(abs(sweep) - math.pi) < 1e-8:
        sweep = math.pi
    samples = []
    for i in range(n + 1):
        a = a0 + sweep * i / n
        samples.append((a, (cx + radius * math.cos(a), cz + radius * math.sin(a))))
    if _in_sweep(a0, sweep, 0.0):
        samples.append((0.0 if sweep >= 0 else 0.0, (cx + radius, cz)))
        # Keep the extreme in parameter order along the sweep.
        def key(item):
            rel = (item[0] - a0) % (2.0 * math.pi)
            if sweep < 0.0:
                rel -= 2.0 * math.pi
            return rel

        samples.sort(key=key)
    return [p for _, p in samples]


def _orient(edge, p_from, p_to):
    dr = p_to[0] - p_from[0]
    dz = p_to[1] - p_from[1]
    length = math.hypot(dr, dz)
    if length < 1e-9:
        ang = None
    elif abs(dz) < 1e-8:
        ang = 90.0
    else:
        ang = math.degrees(math.atan2(abs(dr), abs(dz)))
    seg = {
        "type": edge["type"],
        "p0": p_from,
        "p1": p_to,
        "length": length,
        "ang": ang,
        "radius": None,
        "center": None,
        "pts": [p_from, p_to],
    }
    if edge["type"] == "A":
        seg["radius"] = edge["r"]
        seg["center"] = (edge["cx"], edge["cz"])
        seg["pts"] = arc_points(edge["cx"], edge["cz"], edge["r"], p_from, p_to)
        seg["length"] = abs(
            ((math.atan2(p_to[1] - edge["cz"], p_to[0] - edge["cx"])
              - math.atan2(p_from[1] - edge["cz"], p_from[0] - edge["cx"]) + math.pi)
             % (2 * math.pi) - math.pi)
            * edge["r"]
        )
    return seg


def extract_profile(shape):
    section = BRepAlgoAPI_Section(shape, gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), False)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("Section of the solid failed")
    raw = []
    exp = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        curve = BRepAdaptor_Curve(edge)
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        p0, p1 = curve.Value(u0), curve.Value(u1)
        rec = {"x0": p0.X(), "z0": p0.Z(), "x1": p1.X(), "z1": p1.Z(), "type": "L"}
        if curve.GetType() == GeomAbs_Circle:
            circ = curve.Circle()
            cen = circ.Location()
            rec.update(type="A", r=circ.Radius(), cx=cen.X(), cz=cen.Z())
        raw.append(rec)
        exp.Next()

    positive = []
    for rec in raw:
        x0, z0, x1, z1 = rec["x0"], rec["z0"], rec["x1"], rec["z1"]
        if x0 >= -1e-7 and x1 >= -1e-7:
            kept = dict(rec)
            kept["x0"] = max(0.0, x0)
            kept["x1"] = max(0.0, x1)
            positive.append(kept)
            continue
        if x0 <= 1e-7 and x1 <= 1e-7:
            continue
        if abs(x1 - x0) < 1e-12:
            continue
        t = (0.0 - x0) / (x1 - x0)
        zc = z0 + t * (z1 - z0)
        kept = dict(rec)
        if x1 >= x0:
            kept.update(x0=0.0, z0=zc, x1=x1, z1=z1)
        else:
            kept.update(x0=x0, z0=z0, x1=0.0, z1=zc)
        positive.append(kept)

    positive = [e for e in positive if math.hypot(e["x1"] - e["x0"], e["z1"] - e["z0"]) > 1e-6]
    used = [False] * len(positive)

    def ends(i):
        e = positive[i]
        return (e["x0"], e["z0"]), (e["x1"], e["z1"])

    start_i, start_at_first = 0, True
    best = (1e9, 1e9)
    for i in range(len(positive)):
        for at_first, p in ((True, ends(i)[0]), (False, ends(i)[1])):
            key = (p[1], p[0])
            if key < best:
                best = key
                start_i, start_at_first = i, at_first

    ordered = []
    used[start_i] = True
    a, b = ends(start_i)
    p_cur = b if start_at_first else a
    ordered.append(_orient(positive[start_i], a if start_at_first else b, p_cur))
    tol = 2e-4
    for _ in range(len(positive) + 2):
        nxt = None
        for i, e in enumerate(positive):
            if used[i]:
                continue
            a, b = ends(i)
            if math.hypot(a[0] - p_cur[0], a[1] - p_cur[1]) < tol:
                nxt = (i, a, b)
                break
            if math.hypot(b[0] - p_cur[0], b[1] - p_cur[1]) < tol:
                nxt = (i, b, a)
                break
        if nxt is None:
            break
        i, a, b = nxt
        used[i] = True
        ordered.append(_orient(positive[i], a, b))
        p_cur = b
    if not all(used):
        missed = sum(1 for u in used if not u)
        raise RuntimeError(f"Profile chain left {missed} edges unused")
    return ordered


def props_of(shape):
    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props)
    com = volume_props.CentreOfMass()
    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area_props)
    return {
        "volume": volume_props.Mass(),
        "area": area_props.Mass(),
        "com": (com.X(), com.Y(), com.Z()),
    }


def _near_point(points, r, z, tr=0.03, tz=0.03):
    best = min(points, key=lambda p: (p[0] - r) ** 2 + (p[1] - z) ** 2)
    if abs(best[0] - r) > tr or abs(best[1] - z) > tz:
        raise RuntimeError(f"Expected a vertex near r={r}, z={z}; closest was {best}")
    return best


def _cyls(segs, radius, length=None, r_tol=0.004, l_tol=0.02):
    found = []
    for seg in segs:
        if seg["type"] != "L" or seg["ang"] is None or seg["ang"] > 0.05:
            continue
        if abs(seg["p0"][0] - radius) > r_tol:
            continue
        if length is not None and abs(seg["length"] - length) > l_tol:
            continue
        z0, z1 = sorted((seg["p0"][1], seg["p1"][1]))
        found.append({"r": seg["p0"][0], "z0": z0, "z1": z1, "L": z1 - z0, "seg": seg})
    found.sort(key=lambda c: c["z0"])
    return found


def analyze(segs, mass):
    pts = [segs[0]["p0"]]
    for seg in segs:
        pts.append(seg["p1"])
    tip = pts[0]
    end = pts[-1]
    if tip[0] > 1e-4:
        raise RuntimeError(f"Profile does not start on the axis: {tip}")
    if abs(end[0]) > 1e-3:
        raise RuntimeError(f"Profile does not finish on the axis: {end}")

    arcs = [s for s in segs if s["type"] == "A"]
    if len(arcs) != 3:
        raise RuntimeError(f"Expected 3 axial arcs, found {len(arcs)}")
    arc_nose, arc_head, arc_fillet = arcs

    max_r = max(p[0] for s in segs for p in s["pts"])
    if abs(max_r - 4.8) > 1e-3:
        raise RuntimeError(f"Maximum radius is {max_r}, expected 4.8")

    form_a = _cyls(segs, 2.35, 0.25)
    form_b = _cyls(segs, 2.25, 0.15)
    if len(form_a) != 4:
        raise RuntimeError(f"Expected 4 form-A crests, found {len(form_a)}")
    if len(form_b) != 23:
        raise RuntimeError(f"Expected 23 form-B crests, found {len(form_b)}")

    def station(r, z):
        p = _near_point(pts, r, z)
        return {"r": p[0], "z": p[1], "d": 2 * p[0]}

    # Lead-in and runout around form A, taken from the ordered profile.
    neck = _cyls(segs, 2.5, 3.3)[0]
    groove_root = _cyls(segs, 1.6)[0]
    barrel = [c for c in _cyls(segs, 2.265437, r_tol=0.01) if c["L"] > 1.0][0]
    barrel_b = [c for c in _cyls(segs, 2.265437, r_tol=0.01) if c["L"] < 1.0][0]
    land_43 = _cyls(segs, 2.15, 1.12, r_tol=0.01, l_tol=0.05)[0]
    nose_land = [c for c in _cyls(segs, 2.15, r_tol=0.01) if c["z0"] > 30][0]

    model = {
        "tip_z": tip[1],
        "end_z": end[1],
        "overall": end[1] - tip[1],
        "max_r": max_r,
        "max_d": 2 * max_r,
        "arcs": {"nose": arc_nose, "head": arc_head, "fillet": arc_fillet},
        "form_a": form_a,
        "form_b": form_b,
        "neck": neck,
        "groove_root": groove_root,
        "barrel": barrel,
        "barrel_b": barrel_b,
        "land_43": land_43,
        "nose_land": nose_land,
        "mass": mass,
        "stations": {
            "blend": station(4.667188, -1.517087),
            "face_out": station(4.5, 0.0),
            "face_in": station(3.3, 0.0),
            "fillet_end": station(2.5, 0.8),
            "neck_end": station(2.5, 4.1),
            "taper_end": station(2.327586, 6.111127),
            "chamfer_end": station(1.9, 6.323382),
            "forma_start": station(1.9, 6.553272),
            "runout_root": station(1.9, 10.916589),
            "runout_crest": station(2.35, 11.4),
            "barrel_start": station(2.265437, 11.561069),
            "groove_start": station(1.6, 13.512216),
            "groove_end": station(1.6, 13.85),
            "barrel_b_start": station(2.265437, 14.55),
            "barrel_b_end": station(2.265437, 15.25),
            "to_43": station(2.15, 15.777232),
            "land_43_end": station(2.15, 16.9),
            "pre_thread": station(2.1019, 17.246638),
            "thread_start": station(2.25, 17.4),
            "last_crest_end": station(2.25, 35.15),
            "last_root_end": station(2.0, 35.541117),
            "runout_b": station(2.15, 35.696447),
            "nose_land_end": station(2.15, 36.9),
            "end_rim": station(1.924627, 37.75),
        },
    }
    # Pitch checks against the extracted crests.
    a_pitch = form_a[1]["z0"] - form_a[0]["z0"]
    b_pitch = form_b[1]["z0"] - form_b[0]["z0"]
    if abs(a_pitch - 1.1) > 1e-6 or abs(b_pitch - 0.8) > 1e-6:
        raise RuntimeError(f"Unexpected pitches A={a_pitch} B={b_pitch}")
    model["pitch_a"] = a_pitch
    model["pitch_b"] = b_pitch
    return model


def render_iso(solid):
    verts, tris = solid.tessellate(0.15)
    xyz = np.array([(v.x, v.y, v.z) for v in verts], dtype=float)
    faces = xyz[np.asarray(tris)]
    order = faces[:, :, [2, 1, 0]]
    normals = np.cross(order[:, 1] - order[:, 0], order[:, 2] - order[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.array([0.58, 0.66, 0.74])
    rgb = np.clip(0.18 + 0.82 * shade[:, None] * base, 0, 1)
    rgba = np.concatenate([rgb, np.ones((len(order), 1))], axis=1)

    fig = plt.figure(figsize=(6.4, 3.6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(order, linewidths=0.0, antialiased=False)
    coll.set_facecolor(rgba)
    ax.add_collection3d(coll)
    zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()
    span = max(xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max() - xyz[:, 1].min())
    ax.set_xlim(zmin - 1, zmax + 1)
    ax.set_ylim(-span / 2 - 1, span / 2 + 1)
    ax.set_zlim(-span / 2 - 1, span / 2 + 1)
    ax.set_box_aspect((zmax - zmin + 2, span + 2, span + 2))
    ax.view_init(elev=18, azim=-62)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    image = plt.imread(buf)
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3]
    else:
        rgb = image[:, :, :3]
    mask = np.any(rgb < 0.97, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 8
    r0, r1 = max(0, rows[0] - pad), min(rgb.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(rgb.shape[1], cols[-1] + pad)
    return rgb[r0:r1, c0:c1]


def new_page():
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def frame(ax, page, total, view_title):
    ax.add_patch(Rectangle((6, 6), 408, 285, fill=False, lw=1.15, ec=RULE, zorder=6))
    ax.add_patch(Rectangle((8, 8), 404, 281, fill=False, lw=0.35, ec=RULE, zorder=6))
    T(ax, 12, 286.2, "4.8 LB", size=10.5, weight="bold", va="top")
    T(ax, 38, 286.2, "SMALL NUT  ·  NOMINAL DIMENSION DRAWING", size=8.0, weight="medium", color=DIM, va="top")
    T(ax, 406, 286.2, view_title, size=7.2, weight="medium", ha="right", va="top", color="#333")
    # Title block
    x, y, w, h = 248, 8, 164, 40
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=RULE, lw=0.7, zorder=4))
    ax.plot([x, x + w], [y + 28, y + 28], color=RULE, lw=0.4, zorder=4)
    ax.plot([x, x + w], [y + 16, y + 16], color=RULE, lw=0.4, zorder=4)
    ax.plot([x + 82, x + 82], [y, y + h], color=RULE, lw=0.4, zorder=4)
    T(ax, x + 3, y + 34, "PART", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 34, "4.8 LB", size=8, weight="bold", va="center", z=7)
    T(ax, x + 85, y + 34, "SHEET", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 34, f"{page}  /  {total}", size=8, weight="bold", va="center", z=7)
    T(ax, x + 3, y + 22, "NAME", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 22, "SMALL NUT", size=7.2, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 22, "UNITS", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 22, "MILLIMETRES", size=7.2, weight="medium", va="center", z=7)
    T(ax, x + 3, y + 8, "DATUM A  SHOULDER Z = 0", size=5.6, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 8, "DATUM B  AXIS", size=5.6, weight="medium", va="center", z=7)
    # Third-angle mark, kept clear of the title block.
    cx, cy = 214, 24
    ax.add_patch(Circle((cx, cy + 8), 3.1, fill=False, lw=0.55, ec=RULE, zorder=6))
    ax.plot([cx + 5.0, cx + 12.4], [cy + 5.4, cy + 6.5], color=RULE, lw=0.55, zorder=6)
    ax.plot([cx + 5.0, cx + 12.4], [cy + 10.6, cy + 9.5], color=RULE, lw=0.55, zorder=6)
    ax.plot([cx + 12.4, cx + 12.4], [cy + 6.5, cy + 9.5], color=RULE, lw=0.55, zorder=6)
    T(ax, cx + 6, cy + 1.2, "3RD ANGLE", size=4.4, ha="center", color="#444", z=7)


def panel(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fc="#f6f8fb", ec="#d5dee8", lw=0.6, zorder=1))
    T(ax, x + 2.4, y + h - 3.4, title, size=6.6, weight="bold", color=DIM, va="top")


def hdim(ax, x1, x2, y, text, ey1=None, ey2=None, text_side="up"):
    if x1 > x2:
        x1, x2 = x2, x1
        ey1, ey2 = ey2, ey1
    over = 1.3
    for x, ey in ((x1, ey1), (x2, ey2)):
        if ey is None:
            continue
        y_end = y + over if y >= ey else y - over
        ax.plot([x, x], [ey, y_end], color=EXT, lw=0.28, zorder=2)
    gap = x2 - x1
    if gap >= 11:
        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0),
            zorder=3,
        )
        va = "bottom" if text_side == "up" else "top"
        dy = 0.7 if text_side == "up" else -0.7
        T(ax, (x1 + x2) / 2, y + dy, text, size=6.2, weight="medium", color=DIM, ha="center", va=va,
          bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))
    else:
        wing = 6.5
        ax.plot([x1 - wing, x2 + wing], [y, y], color=DIM, lw=0.5, zorder=3)
        ax.annotate("", xy=(x1, y), xytext=(x1 - wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", xy=(x2, y), xytext=(x2 + wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        T(ax, x2 + wing + 1.1, y, text, size=6.0, weight="medium", color=DIM, ha="left", va="center",
          bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.92))


def vdim(ax, x, y1, y2, text, ex1=None, ex2=None, text_side="right"):
    if y1 > y2:
        y1, y2 = y2, y1
        ex1, ex2 = ex2, ex1
    over = 1.3
    for y, ex in ((y1, ex1), (y2, ex2)):
        if ex is None:
            continue
        x_end = x + over if x >= ex else x - over
        ax.plot([ex, x_end], [y, y], color=EXT, lw=0.28, zorder=2)
    gap = y2 - y1
    if gap >= 12:
        ax.annotate(
            "",
            xy=(x, y2),
            xytext=(x, y1),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0),
            zorder=3,
        )
        ha = "left" if text_side == "right" else "right"
        dx = 1.1 if text_side == "right" else -1.1
        T(ax, x + dx, (y1 + y2) / 2, text, size=6.2, weight="medium", color=DIM, ha=ha, va="center",
          bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))
    else:
        wing = 6.5
        ax.plot([x, x], [y1 - wing, y2 + wing], color=DIM, lw=0.5, zorder=3)
        ax.annotate("", xy=(x, y1), xytext=(x, y1 - wing),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", xy=(x, y2), xytext=(x, y2 + wing),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        T(ax, x + 1.2, y2 + wing + 1.5, text, size=6.0, weight="medium", color=DIM, ha="left", va="bottom",
          bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.92))


def leader(ax, x, y, tx, ty, text, ha="left"):
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.45, shrinkA=0, shrinkB=0),
        zorder=3,
    )
    ax.plot([x], [y], marker="o", ms=1.8, color=DIM, zorder=4)
    dx = 1.3 if ha == "left" else -1.3
    T(ax, tx + dx, ty, text, size=6.3, weight="medium", color=DIM, ha=ha, va="center",
      bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))


def centerline(ax, view, z0, z1):
    x0, y = view.pt(z0, 0)
    x1, _ = view.pt(z1, 0)
    ax.plot([x0, x1], [y, y], color=CENTER, lw=0.4, linestyle=(0, (7, 1.6, 0.9, 1.6)), zorder=2)


def draw_profile(ax, view, segs, both=True, lw=0.95, z0=None, z1=None, clip=False):
    for seg in segs:
        zs = (seg["p0"][1], seg["p1"][1])
        if clip:
            if z0 is not None and min(zs) < z0 - 1e-4:
                continue
            if z1 is not None and max(zs) > z1 + 1e-4:
                continue
        else:
            if z0 is not None and max(zs) < z0:
                continue
            if z1 is not None and min(zs) > z1:
                continue
        radii = (1, -1) if both else (1,)
        for sign in radii:
            xs, ys = [], []
            for r, z in seg["pts"]:
                x, y = view.pt(z, sign * r)
                xs.append(x)
                ys.append(y)
            ax.plot(xs, ys, color=GEO, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3)


def balloon(ax, x, y, label):
    ax.add_patch(Circle((x, y), 3.6, fc="white", ec=DIM, lw=0.7, zorder=5))
    T(ax, x, y, label, size=7.0, weight="bold", color=DIM, ha="center", va="center", z=6)


def draw_table(ax, x, y_top, widths, headers, rows, row_h=3.7, size=5.7):
    """Draw a table downward from y_top. Returns the y of the bottom edge."""
    total_w = sum(widths)
    head_h = row_h + 0.4
    y = y_top - head_h
    ax.add_patch(Rectangle((x, y), total_w, head_h, fc="#e7eef6", ec="#c5d2e0", lw=0.3, zorder=2))
    cx = x
    for head, w in zip(headers, widths):
        T(ax, cx + 1.1, y + head_h / 2, head, size=size, weight="bold", color=DIM, va="center")
        cx += w
    for i, row in enumerate(rows):
        y -= row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((x, y), total_w, row_h, fc="#fbfcfe", ec="none", zorder=1))
        cx = x
        for cell, w in zip(row, widths):
            T(ax, cx + 1.1, y + row_h / 2, str(cell), size=size, weight="mono", va="center", color="#1d1d1d")
            cx += w
    ax.add_patch(Rectangle((x, y), total_w, y_top - y, fill=False, ec="#c5d2e0", lw=0.45, zorder=3))
    return y


def seg_label(seg):
    if seg["type"] == "A":
        c = seg["center"]
        return f"ARC R{f5(seg['radius'])}  C({f5(c[0])}, {f5(c[1])})"
    if seg["ang"] is not None and seg["ang"] < 0.02:
        return "CYLINDER"
    if seg["ang"] is not None and abs(seg["ang"] - 90) < 0.02:
        return "RADIAL FACE"
    return "CONE / TAPER"


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def sheet_general(pdf, page, total, segs, model, iso):
    fig, ax = new_page()
    frame(ax, page, total, "SHEET 1   GENERAL ARRANGEMENT")
    scale = 4.35
    view = View(zx=52, y0=178, scale=scale)
    draw_profile(ax, view, segs, both=True, lw=0.7)
    centerline(ax, view, model["tip_z"] - 1.4, model["end_z"] + 1.6)
    T(ax, view.zx_of(model["tip_z"] - 1.3), view.y0 + 2.2, "B", size=6.5, weight="bold", color=CENTER, ha="right")

    # Datum A on the shoulder face.
    face_x = view.zx_of(0)
    ax.plot([face_x, face_x], [view.ry_of(-4.5), view.ry_of(-5.6)], color=DIM, lw=0.4, zorder=3)
    ax.add_patch(Polygon(
        [(face_x, view.ry_of(-5.6)), (face_x - 1.7, view.ry_of(-5.6) - 3.0), (face_x + 1.7, view.ry_of(-5.6) - 3.0)],
        closed=True, fc=DIM, ec=DIM, lw=0.2, zorder=4,
    ))
    ax.add_patch(Rectangle((face_x - 3.3, view.ry_of(-5.6) - 8.2), 6.6, 5.0, fc="white", ec=DIM, lw=0.55, zorder=4))
    T(ax, face_x, view.ry_of(-5.6) - 5.7, "A", size=6.2, weight="bold", color=DIM, ha="center", va="center", z=5)

    tip_z, end_z = model["tip_z"], model["end_z"]
    hdim(ax, view.zx_of(tip_z), view.zx_of(end_z), 128, f"{f3(model['overall'])} OVERALL",
         view.ry_of(-model["max_r"]), view.ry_of(0))
    hdim(ax, view.zx_of(0), view.zx_of(end_z), 118, f"{f3(end_z)}  DATUM A → END",
         view.ry_of(-4.5), view.ry_of(0))
    hdim(ax, view.zx_of(tip_z), view.zx_of(0), 108, f"{f3(-tip_z)} HEAD",
         view.ry_of(0), view.ry_of(-4.5), text_side="down")

    # Principal diameters, offset so the arrows sit just outside the metal.
    vdim(ax, view.zx_of(-0.9) - 7, view.ry_of(-4.8), view.ry_of(4.8), "Ø 9.600",
         view.zx_of(-0.9), view.zx_of(-0.9), text_side="left")
    vdim(ax, view.zx_of(2.4) + 6, view.ry_of(-2.5), view.ry_of(2.5), "Ø 5.000",
         view.zx_of(2.4), view.zx_of(2.4))
    vdim(ax, view.zx_of(26) + 7, view.ry_of(-2.25), view.ry_of(2.25), "Ø 4.500",
         view.zx_of(26), view.zx_of(26))
    vdim(ax, view.zx_of(end_z) + 6, view.ry_of(-model["stations"]["end_rim"]["r"]),
         view.ry_of(model["stations"]["end_rim"]["r"]), "Ø 3.849",
         view.zx_of(end_z), view.zx_of(end_z))

    # Zone brackets
    zones = [
        (model["stations"]["forma_start"]["z"], model["form_a"][-1]["z1"], "FORM A"),
        (model["form_b"][0]["z0"], model["form_b"][-1]["z1"], "FORM B"),
    ]
    for z0, z1, name in zones:
        y = view.ry_of(model["max_r"]) + 8
        ax.annotate("", xy=(view.zx_of(z1), y), xytext=(view.zx_of(z0), y),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0))
        T(ax, (view.zx_of(z0) + view.zx_of(z1)) / 2, y + 2.4, name, size=5.6, weight="bold",
          color=DIM, ha="center", va="bottom")

    balloon(ax, view.zx_of(-1.5), view.ry_of(4.8) + 14, "2")
    ax.plot([view.zx_of(-1.2), view.zx_of(-1.2)], [view.ry_of(4.6), view.ry_of(4.8) + 10.4], color=DIM, lw=0.35)
    balloon(ax, view.zx_of(8.6), view.ry_of(2.6) + 16, "3")
    ax.plot([view.zx_of(8.6), view.zx_of(8.6)], [view.ry_of(2.35), view.ry_of(2.6) + 12.4], color=DIM, lw=0.35)
    balloon(ax, view.zx_of(22), view.ry_of(2.5) + 16, "4")
    ax.plot([view.zx_of(22), view.zx_of(22)], [view.ry_of(2.25), view.ry_of(2.5) + 12.4], color=DIM, lw=0.35)

    T(ax, view.zx_of(tip_z), 96, "SCALE 4.35 : 1     AXIS HORIZONTAL     HEAD AT LEFT", size=6.2, color="#444")

    # Isometric
    ax.imshow(iso, extent=(248, 404, 168, 276), aspect="auto", zorder=1, interpolation="bilinear")
    T(ax, 250, 164, "PICTORIAL — NOT TO SCALE     HEAD AT LEFT", size=5.6, color="#444")

    # End views
    def end_view(cx, cy, radius_mm, label, scale_e=3.2):
        ax.add_patch(Circle((cx, cy), radius_mm * scale_e / 2, fill=False, ec=GEO, lw=0.8, zorder=3))
        ax.plot([cx - radius_mm * scale_e / 2 - 2.5, cx + radius_mm * scale_e / 2 + 2.5], [cy, cy],
                color=CENTER, lw=0.35, linestyle=(0, (5, 1.4, 0.8, 1.4)), zorder=2)
        ax.plot([cx, cx], [cy - radius_mm * scale_e / 2 - 2.5, cy + radius_mm * scale_e / 2 + 2.5],
                color=CENTER, lw=0.35, linestyle=(0, (5, 1.4, 0.8, 1.4)), zorder=2)
        T(ax, cx, cy - radius_mm * scale_e / 2 - 5.5, label, size=5.5, ha="center", color="#333")

    end_view(292, 118, 9.6, "HEAD END   ENVELOPE Ø 9.600")
    end_view(368, 118, 3.849, "SHANK END FACE Ø 3.849", scale_e=4.0)

    notes = [
        "Solid of revolution about datum B. Revolve the axial profile.",
        "Sizes are CAD nominals from 4.8_LB.STEP.",
        "No tolerance, material, or finish is stored in the model.",
        "Graphic sizes are rounded to 0.001 mm.",
        "Sheet 6 lists every profile segment to 0.00001 mm.",
        "Positive Z runs from datum A toward the shank end.",
        "Balloons 2, 3 and 4 mark the detail sheets.",
        "Form A pitch 1.100.   Form B pitch 0.800, 23 crests.",
    ]
    yy = 78
    for line in notes:
        T(ax, 14, yy, line, size=5.6, color="#333", va="center")
        yy -= 6.5
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_head(pdf, page, total, segs, model):
    fig, ax = new_page()
    frame(ax, page, total, "SHEET 2   HEAD, SHOULDER AND NECK")
    scale = 13.0
    view = View(zx=118, y0=168, scale=scale)
    draw_profile(ax, view, segs, both=True, lw=0.95, z0=-3.05, z1=6.554, clip=True)
    centerline(ax, view, -3.15, 6.9)
    # Open end where form A continues.
    for sign in (1, -1):
        bx, by = view.pt(model["stations"]["forma_start"]["z"], sign * 1.9)
        ax.plot([bx, bx + 1.2, bx - 0.4, bx + 1.2], [by, by + 1.1 * sign, by + 2.2 * sign, by + 3.3 * sign],
                color=GEO, lw=0.6, zorder=3)
    T(ax, view.zx_of(6.55) + 3, view.ry_of(2.3), "SHEET 3", size=5.4, color=DIM)
    T(ax, view.zx_of(-3.1), view.y0 + 2.3, "B", size=6.4, weight="bold", color=CENTER, ha="right")

    st = model["stations"]
    # Datum triangle on the +r shoulder, which is a vertical line at z=0.
    fx = view.zx_of(0)
    ax.plot([fx, fx + 8], [view.ry_of(3.9), view.ry_of(3.9)], color=DIM, lw=0.35, zorder=3)
    T(ax, fx + 9, view.ry_of(3.9), "DATUM A", size=6.0, weight="bold", color=DIM, va="center")

    vdim(ax, view.zx_of(-0.9) - 8, view.ry_of(-4.8), view.ry_of(4.8), "Ø 9.600",
         view.zx_of(-0.9), view.zx_of(-0.9), text_side="left")
    vdim(ax, view.zx_of(2.5) + 9, view.ry_of(-2.5), view.ry_of(2.5), "Ø 5.000",
         view.zx_of(2.5), view.zx_of(2.5))
    hdim(ax, view.zx_of(model["tip_z"]), view.zx_of(0), 96, f"{f3(-model['tip_z'])} HEAD",
         view.ry_of(0), view.ry_of(-4.5))
    hdim(ax, view.zx_of(st["fillet_end"]["z"]), view.zx_of(st["neck_end"]["z"]), 86,
         f"{f3(model['neck']['L'])}  Ø5 NECK",
         view.ry_of(-2.5), view.ry_of(-2.5))

    nose, head, fillet = model["arcs"]["nose"], model["arcs"]["head"], model["arcs"]["fillet"]
    p_nose = nose["pts"][len(nose["pts"]) // 2]
    p_head = max(head["pts"], key=lambda p: p[0])
    p_fil = fillet["pts"][len(fillet["pts"]) // 2]
    leader(ax, *view.pt(p_nose[1], p_nose[0]), view.zx_of(-2.2), view.ry_of(4.8) + 18,
           f"R {f3(nose['radius'])}", ha="left")
    leader(ax, *view.pt(p_head[1], p_head[0]), view.zx_of(-0.2), view.ry_of(4.8) + 22,
           "R 1.500   Ø 9.600 AT Z −0.900", ha="left")
    leader(ax, *view.pt(p_fil[1], -p_fil[0]), view.zx_of(1.6), 78,
           "R 0.800 QUARTER CIRCLE", ha="left")
    leader(ax, *view.pt(5.05, 2.42), view.zx_of(3.6), view.ry_of(4.55),
           "4.90° FROM AXIS", ha="left")
    leader(ax, *view.pt(6.22, 2.08), view.zx_of(6.7), view.ry_of(0.85),
           "63.60° FROM AXIS", ha="left")

    rows = [
        ["1", f"{f5(model['tip_z'])}", "0", "0", "TIP ON AXIS"],
        ["2", f"{f5(st['blend']['z'])}", f5(st["blend"]["r"]), f3(st["blend"]["d"]), "NOSE / HEAD ARC JOIN"],
        ["3", "−0.90000", "4.80000", "9.600", "LARGEST DIAMETER"],
        ["4", "0.00000", "4.50000", "9.000", "DATUM A, OUTER"],
        ["5", "0.00000", "3.30000", "6.600", "DATUM A, INNER"],
        ["6", "0.80000", "2.50000", "5.000", "NECK START, R0.8 END"],
        ["7", "4.10000", "2.50000", "5.000", "NECK END"],
        ["8", f5(st["taper_end"]["z"]), f5(st["taper_end"]["r"]), f3(st["taper_end"]["d"]), "4.90° TAPER END"],
        ["9", f5(st["chamfer_end"]["z"]), f5(st["chamfer_end"]["r"]), "3.800", "63.60° END, Ø3.800"],
        ["10", f5(st["forma_start"]["z"]), "1.90000", "3.800", "FORM A STARTS"],
    ]
    panel(ax, 228, 168, 176, 112, "HEAD STATIONS FROM DATUM A")
    draw_table(ax, 231, 272, [10, 28, 26, 22, 78],
               ["PT", "Z", "RADIUS", "Ø", "FEATURE"], rows, row_h=3.55, size=5.3)

    arc_rows = [
        ["NOSE", "10.40000", "−0.15000", "7.70000", "TORUS, CENTER PAST THE AXIS"],
        ["HEAD", "1.50000", "3.30000", "−0.90000", "THROUGH Ø9.600"],
        ["FILLET", "0.80000", "3.30000", "0.80000", "QUARTER CIRCLE, TANGENT"],
    ]
    T(ax, 231, 158, "AXIAL-SECTION ARCS", size=6.4, weight="bold", color=DIM)
    draw_table(ax, 231, 155, [22, 28, 30, 30, 62],
               ["ARC", "R", "CENTER R", "CENTER Z", "NOTE"], arc_rows, row_h=4.2, size=5.3)

    notes = [
        "Nose arc center is 0.150 mm past datum B, at Z = 7.700.",
        "R1.500 meets the nose at a corner. R0.800 is tangent",
        "to datum A and to the Ø5.000 neck.",
        "Datum A is the annular face from Ø6.600 to Ø9.000.",
    ]
    yy = 70
    for line in notes:
        T(ax, 14, yy, line, size=5.6, color="#333")
        yy -= 7.0
    T(ax, 14, 274, "DETAIL 2     SCALE 13 : 1     FULL PROFILE", size=7, weight="bold", color=DIM)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_a(pdf, page, total, segs, model):
    fig, ax = new_page()
    frame(ax, page, total, "SHEET 3   FORM A AND GROOVE")
    T(ax, 14, 274, "DETAIL 3     FORM A SCALE 24 : 1     GROOVE SCALE 18 : 1     HALF PROFILE ABOVE THE AXIS", size=6.6, weight="bold", color=DIM)

    view = View(zx=28, y0=186, scale=24, z0=6.30, r0=0.0)
    draw_profile(ax, view, segs, both=False, lw=1.0, z0=6.25, z1=11.70)
    centerline(ax, view, 6.25, 11.75)
    crests = model["form_a"]
    # One-pitch callouts on the first tooth.
    c0 = crests[0]
    hdim(ax, view.zx_of(c0["z0"]), view.zx_of(c0["z0"] + model["pitch_a"]), view.ry_of(2.7),
         f"PITCH {f3(model['pitch_a'])}    4 CRESTS", view.ry_of(2.35), view.ry_of(2.35))
    hdim(ax, view.zx_of(c0["z0"]), view.zx_of(c0["z1"]), view.ry_of(2.55),
         f"CREST {f3(c0['L'])}", view.ry_of(2.35), view.ry_of(2.35))
    leader(ax, *view.pt(6.75, 2.12), view.zx_of(6.55), view.ry_of(1.35), "48.60°", ha="left")
    leader(ax, *view.pt(7.30, 2.12), view.zx_of(7.55), view.ry_of(1.35), "63.60°", ha="right")
    leader(ax, *view.pt(c0["z0"] + 0.1, 2.35), view.zx_of(c0["z0"]), view.ry_of(3.05), "Ø 4.700 CREST", ha="left")
    leader(ax, *view.pt(7.55, 1.90), view.zx_of(7.7), view.ry_of(1.15), "Ø 3.800 ROOT", ha="left")

    # Groove
    g = View(zx=28, y0=78, scale=18, z0=11.40, r0=0.0)
    draw_profile(ax, g, segs, both=False, lw=1.0, z0=11.35, z1=15.45)
    centerline(ax, g, 11.35, 15.5)
    st = model["stations"]
    leader(ax, *g.pt(12.4, 2.265), g.zx_of(12.2), g.ry_of(3.15), f"Ø {f3(diam(model['barrel']['r']))}", ha="left")
    leader(ax, *g.pt(13.7, 1.60), g.zx_of(13.7), g.ry_of(0.7), "Ø 3.200", ha="left")
    leader(ax, *g.pt(13.4, 1.95), g.zx_of(12.6), g.ry_of(1.05), "76.30°", ha="right")
    leader(ax, *g.pt(14.2, 1.95), g.zx_of(14.7), g.ry_of(1.05), "43.55°", ha="left")
    hdim(ax, g.zx_of(st["groove_start"]["z"]), g.zx_of(st["groove_end"]["z"]), 62,
         f"ROOT {f3(model['groove_root']['L'])}", g.ry_of(1.6), g.ry_of(1.6))

    a_rows = []
    for i, c in enumerate(crests, start=1):
        a_rows.append([str(i), f5(c["z0"]), f5(c["z1"]), f5(c["z0"] - model["pitch_a"] + 0.396728 if i else 0)])
    # The root flat after each of the first 3 crests is 0.230; the 4th runout is shorter.
    # Replace the last column with the crest length, which is the controlled width.
    a_rows = [[str(i), f5(c["z0"]), f5(c["z1"]), f3(c["L"]), f5(c["z0"] + model["pitch_a"]) if i < 4 else "RUNOUT"]
              for i, c in enumerate(crests, start=1)]

    panel(ax, 214, 150, 190, 124, "FORM A — 4 CRESTS, PITCH 1.100")
    T(ax, 218, 262, "Crest Ø 4.700, flat 0.250.   Root Ø 3.800, flat 0.230.", size=5.5, color="#333")
    T(ax, 218, 256, "Rising flank 48.60° from the axis.   Falling flank 63.60°.", size=5.5, color="#333")
    T(ax, 218, 250, "First rising flank starts at Z of station 10 on sheet 2.", size=5.5, color="#333")
    draw_table(ax, 218, 246, [12, 32, 32, 22, 32],
               ["#", "CREST Z0", "CREST Z1", "FLAT", "NEXT"], a_rows, row_h=4.3, size=5.4)

    run_rows = [
        ["SHORT ROOT", f5(st["runout_root"]["z"]), "1.90000", "3.800", "AFTER 4th FALLING FLANK"],
        ["42.95° TO", f5(st["runout_crest"]["z"]), "2.35000", "4.700", "NO CREST FLAT"],
        ["27.70° TO", f5(st["barrel_start"]["z"]), f5(st["barrel_start"]["r"]), f3(st["barrel_start"]["d"]), "BARREL START"],
    ]
    T(ax, 218, 214, "FORM A RUNOUT", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 218, 211, [28, 32, 28, 20, 62],
               ["STEP", "Z", "RADIUS", "Ø", "NOTE"], run_rows, row_h=4.2, size=5.3)

    g_rows = [
        ["BARREL", f5(model["barrel"]["z0"]), f5(model["barrel"]["z1"]), f3(model["barrel"]["L"]), f3(diam(model["barrel"]["r"]))],
        ["76.30° FLANK", f5(model["barrel"]["z1"]), f5(st["groove_start"]["z"]), f3(st["groove_start"]["z"] - model["barrel"]["z1"]), "→ Ø3.200"],
        ["GROOVE ROOT", f5(st["groove_start"]["z"]), f5(st["groove_end"]["z"]), f3(model["groove_root"]["L"]), "3.200"],
        ["43.55° FLANK", f5(st["groove_end"]["z"]), f5(st["barrel_b_start"]["z"]), f3(st["barrel_b_start"]["z"] - st["groove_end"]["z"]), f"→ Ø{f3(st['barrel_b_start']['d'])}"],
        ["SHORT BARREL", f5(model["barrel_b"]["z0"]), f5(model["barrel_b"]["z1"]), f3(model["barrel_b"]["L"]), f3(diam(model["barrel_b"]["r"]))],
    ]
    T(ax, 218, 148, "GROOVE", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 218, 145, [32, 32, 32, 22, 28],
               ["FEATURE", "Z0", "Z1", "LENGTH", "Ø"], g_rows, row_h=4.4, size=5.3)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_b(pdf, page, total, segs, model):
    fig, ax = new_page()
    frame(ax, page, total, "SHEET 4   FORM B AND TIP")
    T(ax, 14, 274, "DETAIL 4     THREAD ENTRY AND TIP ARE TRUE RADIAL SCALE     TOOTH DETAIL IS ENLARGED", size=6.4, weight="bold", color=DIM)

    # Tooth window, two pitches, radial origin at the root so the tooth fills the band.
    tooth = View(zx=36, y0=206, scale=64, z0=17.40, r0=1.90)
    draw_profile(ax, tooth, segs, both=False, lw=1.15, z0=17.39, z1=19.05, clip=True)
    c0 = model["form_b"][0]
    c1 = model["form_b"][1]
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c1["z0"]), tooth.ry_of(2.72),
         f"PITCH {f3(model['pitch_b'])}", tooth.ry_of(2.25), tooth.ry_of(2.25))
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c0["z1"]), tooth.ry_of(2.46),
         f3(c0["L"]), tooth.ry_of(2.25), tooth.ry_of(2.25))
    # Root flat sits just after the first falling flank.
    root_z0 = c0["z1"] + (0.25 / math.tan(math.radians(43.85)))
    # Use the actual next segment geometry via stations of crest 0 end.
    # Falling flank ends at the root cylinder. Find it from the crest end + angle.
    leader(ax, *tooth.pt(17.68, 2.12), tooth.zx_of(17.52), tooth.ry_of(1.78), "43.85°", ha="right")
    leader(ax, *tooth.pt(18.07, 2.12), tooth.zx_of(18.22), tooth.ry_of(1.78), "44.00°", ha="left")
    leader(ax, *tooth.pt(c1["z0"] + 0.07, 2.25), tooth.zx_of(19.20), tooth.ry_of(2.48), "Ø 4.500", ha="left")
    T(ax, 16, 190, "TOOTH  SCALE 64 : 1     ROOT Ø 4.000     MATERIAL IS BELOW THE LINE     23 CRESTS", size=5.5, color="#333")

    # Entry from the Ø4.300 land through the first crest.
    entry = View(zx=20, y0=128, scale=22, z0=15.20, r0=0.0)
    draw_profile(ax, entry, segs, both=False, lw=0.95, z0=15.15, z1=17.70)
    centerline(ax, entry, 15.15, 17.75)
    st = model["stations"]
    leader(ax, *entry.pt(16.35, 2.15), entry.zx_of(15.85), entry.ry_of(2.42), "Ø 4.300", ha="left")
    leader(ax, *entry.pt(17.05, 2.12), entry.zx_of(16.45), entry.ry_of(1.45), "7.90°", ha="right")
    leader(ax, *entry.pt(17.32, 2.18), entry.zx_of(17.15), entry.ry_of(2.48), "44.00°", ha="left")
    T(ax, 20, 116, "THREAD ENTRY   SCALE 22 : 1", size=5.8, color="#333")

    tip = View(zx=118, y0=64, scale=14, z0=34.90, r0=0.0)
    draw_profile(ax, tip, segs, both=False, lw=0.95, z0=34.85, z1=37.80, clip=True)
    centerline(ax, tip, 34.85, 38.05)
    leader(ax, *tip.pt(36.3, 2.15), tip.zx_of(35.5), tip.ry_of(2.85), "Ø 4.300", ha="left")
    leader(ax, *tip.pt(37.32, 2.04), tip.zx_of(36.7), tip.ry_of(2.55), "14.85°", ha="right")
    leader(ax, *tip.pt(37.75, 0.85), tip.zx_of(37.55), tip.ry_of(1.55), "END Ø 3.849", ha="right")
    T(ax, 118, 54, "TIP   SCALE 14 : 1", size=5.8, color="#333")

    b_rows = [[str(i), f5(c["z0"]), f5(c["z1"])] for i, c in enumerate(model["form_b"], start=1)]
    panel(ax, 214, 168, 190, 104, "FORM B CREST STATIONS — ALL 23")
    # two columns of the 23 crests
    left = b_rows[:12]
    right = b_rows[12:]
    while len(right) < len(left):
        right.append(["", "", ""])
    paired = [a + b for a, b in zip(left, right)]
    draw_table(ax, 218, 264, [10, 30, 30, 10, 30, 30],
               ["#", "Z0", "Z1", "#", "Z0", "Z1"], paired, row_h=3.45, size=5.2)

    spec = [
        ["PITCH", "0.800", "CREST START TO CREST START"],
        ["CREST Ø", "4.500", "AXIAL FLAT 0.150"],
        ["ROOT Ø", "4.000", "AXIAL FLAT 0.131"],
        ["FALLING FLANK", "43.85°", "FROM THE AXIS, CREST TO ROOT"],
        ["RISING FLANK", "44.00°", "FROM THE AXIS, ROOT TO CREST"],
        ["COUNT", "23", "FIRST Z0 17.40000  LAST Z1 35.15000"],
        ["AFTER LAST ROOT", f5(st["runout_b"]["z"]), "44.00° STOPS AT Ø 4.300"],
    ]
    T(ax, 218, 160, "FORM B DEFINITION", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 218, 157, [36, 32, 92],
               ["ITEM", "VALUE", "NOTE"], spec, row_h=3.7, size=5.2)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_schedule(pdf, page, total, model):
    fig, ax = new_page()
    frame(ax, page, total, "SHEET 5   DIMENSION SCHEDULE")
    T(ax, 14, 274, "ALL CONTROLLING SIZES.  Z IS FROM DATUM A, POSITIVE TOWARD THE SHANK END.  ANGLES ARE FROM THE AXIS.", size=6.3, weight="medium", color=DIM)
    st = model["stations"]
    m = model["mass"]

    overall = [
        ["OVERALL LENGTH", f3(model["overall"]), f5(model["overall"]), "TIP TO END FACE"],
        ["TIP Z", f3(model["tip_z"]), f5(model["tip_z"]), "ON THE AXIS"],
        ["END FACE Z", f3(model["end_z"]), f5(model["end_z"]), "FULL FACE TO THE AXIS"],
        ["LARGEST Ø", "9.600", "9.60000", "AT Z = −0.900"],
        ["END FACE Ø", f3(st["end_rim"]["d"]), f5(st["end_rim"]["d"]), "SHANK END"],
        ["VOLUME", f"{m['volume']:.3f} mm³", f"{m['volume']:.5f}", "SOLID, NO BORE"],
        ["SURFACE AREA", f"{m['area']:.3f} mm²", f"{m['area']:.3f}", ""],
        ["CENTRE OF MASS Z", f3(m["com"][2]), f5(m["com"][2]), "ON THE AXIS"],
    ]
    T(ax, 12, 266, "1   OVERALL", size=6.6, weight="bold", color=DIM)
    y = draw_table(ax, 12, 263, [42, 32, 32, 48],
                   ["ITEM", "0.001 mm", "MODEL", "NOTE"], overall, row_h=3.6, size=5.2)

    head = [
        ["TIP", f5(model["tip_z"]), "0.00000", "0.000", "R10.400 START"],
        ["ARC JOIN", f5(st["blend"]["z"]), f5(st["blend"]["r"]), f3(st["blend"]["d"]), "NOSE TO R1.500"],
        ["MAX Ø", "−0.90000", "4.80000", "9.600", "ON THE R1.500 ARC"],
        ["DATUM A OUT", "0.00000", "4.50000", "9.000", "SHOULDER OUTER"],
        ["DATUM A IN", "0.00000", "3.30000", "6.600", "SHOULDER INNER"],
        ["NECK START", "0.80000", "2.50000", "5.000", "R0.800 END"],
        ["NECK END", "4.10000", "2.50000", "5.000", "LENGTH 3.300"],
        ["TAPER END", f5(st["taper_end"]["z"]), f5(st["taper_end"]["r"]), f3(st["taper_end"]["d"]), "4.90°"],
        ["CHAMFER END", f5(st["chamfer_end"]["z"]), f5(st["chamfer_end"]["r"]), "3.800", "63.60°"],
        ["FORM A START", f5(st["forma_start"]["z"]), "1.90000", "3.800", "LEAD-IN LAND END"],
    ]
    T(ax, 175, 266, "2   HEAD AND NECK", size=6.6, weight="bold", color=DIM)
    draw_table(ax, 175, 263, [36, 32, 30, 22, 48],
               ["POINT", "Z", "RADIUS", "Ø", "NOTE"], head, row_h=3.6, size=5.2)

    arcs = [
        ["NOSE", "10.40000", "−0.15000", "7.70000", f5(model["arcs"]["nose"]["length"])],
        ["HEAD", "1.50000", "3.30000", "−0.90000", f5(model["arcs"]["head"]["length"])],
        ["FILLET", "0.80000", "3.30000", "0.80000", f5(model["arcs"]["fillet"]["length"])],
    ]
    T(ax, 12, y - 6, "3   ARCS IN THE AXIAL SECTION", size=6.6, weight="bold", color=DIM)
    y = draw_table(ax, 12, y - 9, [24, 32, 32, 32, 32],
                   ["ARC", "R", "CENTER R", "CENTER Z", "ARC LENGTH"], arcs, row_h=3.8, size=5.3)

    mid = [
        ["A CREST Ø / FLAT", "4.700 / 0.250", "4 CRESTS"],
        ["A ROOT Ø / FLAT", "3.800 / 0.230", "PITCH 1.100"],
        ["A FLANKS", "48.60° / 63.60°", "RISE / FALL, FROM AXIS"],
        ["A FIRST CREST", f5(model["form_a"][0]["z0"]), "Z0"],
        ["A LAST CREST END", f5(model["form_a"][-1]["z1"]), "Z1"],
        ["BARREL Ø", f3(diam(model["barrel"]["r"])), f"Z {f5(model['barrel']['z0'])} → {f5(model['barrel']['z1'])}"],
        ["GROOVE Ø", "3.200", f"FLAT {f3(model['groove_root']['L'])} AT Z {f5(model['groove_root']['z0'])}"],
        ["GROOVE FLANKS", "76.30° / 43.55°", "INTO / OUT OF Ø3.200"],
        ["SHORT BARREL", f3(diam(model["barrel_b"]["r"])), f"LENGTH {f3(model['barrel_b']['L'])}"],
        ["Ø4.300 LAND", "4.300", f"Z {f5(model['land_43']['z0'])} → {f5(model['land_43']['z1'])}"],
        ["ENTRY TAPERS", "12.35° THEN 7.90°", "ONTO THE PRE-THREAD Ø"],
        ["PRE-THREAD Ø", f3(st["pre_thread"]["d"]), f"Z {f5(st['pre_thread']['z'])}"],
    ]
    T(ax, 175, 168, "4   FORM A, GROOVE, ENTRY", size=6.6, weight="bold", color=DIM)
    draw_table(ax, 175, 165, [42, 48, 78],
               ["ITEM", "VALUE", "WHERE"], mid, row_h=3.55, size=5.15)

    tip_rows = [
        ["B PITCH", "0.80000", "23 CRESTS"],
        ["B CREST", "Ø 4.500   FLAT 0.150", f"Z {f5(model['form_b'][0]['z0'])} FIRST"],
        ["B ROOT", "Ø 4.000   FLAT 0.131", "BETWEEN FLANKS"],
        ["B FLANKS", "43.85° AND 44.00°", "FROM THE AXIS"],
        ["B LAST CREST END", f5(model["form_b"][-1]["z1"]), "Z"],
        ["B RUNOUT", f"Ø 4.300 AT Z {f5(st['runout_b']['z'])}", "44.00° PARTIAL FLANK"],
        ["TIP LAND", f"Ø 4.300  L {f3(model['nose_land']['L'])}", f"TO Z {f5(model['nose_land']['z1'])}"],
        ["TIP TAPER", "14.85°", f"TO Ø {f3(st['end_rim']['d'])}"],
        ["END FACE", f"Z {f5(model['end_z'])}", f"Ø {f3(st['end_rim']['d'])} TO THE AXIS"],
    ]
    T(ax, 12, y - 6, "5   FORM B AND TIP", size=6.6, weight="bold", color=DIM)
    draw_table(ax, 12, y - 9, [40, 62, 52],
               ["ITEM", "VALUE", "NOTE"], tip_rows, row_h=3.7, size=5.2)

    T(ax, 12, 20, "ANGLES ARE THE MODEL VALUES.  A 0.001 mm CALLOUT IS THE MODEL SIZE ROUNDED TO THE NEAREST MICROMETRE.", size=5.4, color="#444")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_segments(pdf, page, total, segs):
    """One or more sheets listing every axial-profile segment in order from the tip."""
    rows = []
    for i, seg in enumerate(segs, start=1):
        kind = "ARC" if seg["type"] == "A" else ("FACE" if seg["ang"] and abs(seg["ang"] - 90) < 0.02 else ("CYL" if seg["ang"] is not None and seg["ang"] < 0.02 else "TAPER"))
        ang = "" if seg["type"] == "A" else f"{seg['ang']:.2f}°"
        extra = f"R{f5(seg['radius'])}" if seg["type"] == "A" else ""
        rows.append([
            f"{i:03d}",
            kind,
            f5(seg["p0"][1]),
            f5(seg["p0"][0]),
            f5(seg["p1"][1]),
            f5(seg["p1"][0]),
            f5(seg["length"]),
            ang,
            extra,
        ])
    per_col = 48
    per_page = per_col * 2
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)] or [[]]
    headers = ["#", "KIND", "Z1", "R1", "Z2", "R2", "LENGTH", "ANGLE", "ARC"]
    widths = [11, 16, 26, 24, 26, 24, 24, 18, 23]
    for offset, chunk in enumerate(pages):
        fig, ax = new_page()
        frame(ax, page + offset, total, "SHEET 6   PROFILE SEGMENT TABLE" if offset == 0 else "PROFILE SEGMENT TABLE, CONTINUED")
        T(ax, 12, 274, "EVERY SEGMENT OF THE +Z AXIAL PROFILE, TIP TO END FACE. REVOLVE ABOUT DATUM B. VALUES IN MILLIMETRES TO 0.00001.", size=6.0, weight="medium", color=DIM)
        left = chunk[:per_col]
        right = chunk[per_col:]
        draw_table(ax, 12, 266, widths, headers, left, row_h=3.55, size=4.6)
        if right:
            draw_table(ax, 212, 266, widths, headers, right, row_h=3.55, size=4.6)
        T(ax, 12, 16, f"SEGMENTS {chunk[0][0]}–{chunk[-1][0]} OF {rows[-1][0]}    ANGLE IS FROM THE AXIS    ARC CENTERS ARE ON SHEET 5", size=5.3, color="#444")
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    return len(pages)


def build():
    print("Loading", STEP_PATH)
    wp = cq.importers.importStep(str(STEP_PATH))
    solid = wp.val()
    shape = solid.wrapped
    print("Sectioning profile")
    segs = extract_profile(shape)
    mass = props_of(shape)
    model = analyze(segs, mass)
    print(f"segments {len(segs)}  overall {model['overall']:.5f}  volume {mass['volume']:.3f}")
    print("Rendering pictorial")
    iso = render_iso(solid)
    # Sheet count: 5 graphic/schedule sheets + segment pages
    per_page = 48 * 2
    seg_pages = max(1, math.ceil(len(segs) / per_page))
    total = 5 + seg_pages
    print(f"Writing {total} sheets")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "4.8 LB small nut — all nominal dimensions"
        info["Subject"] = "Dimensioned drawing extracted from 4.8_LB.STEP"
        sheet_general(pdf, 1, total, segs, model, iso)
        sheet_head(pdf, 2, total, segs, model)
        sheet_form_a(pdf, 3, total, segs, model)
        sheet_form_b(pdf, 4, total, segs, model)
        sheet_schedule(pdf, 5, total, model)
        sheet_segments(pdf, 6, total, segs)
    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    build()
