#!/usr/bin/env python3
"""Nominal dimension drawing of the turned part "2 MEDIUM NUT".

The STEP file stores geometry only: no PMI, tolerances, material, or finish.
The solid is a surface of revolution. Sizes are measured in the part frame:
Z along the axis, origin on the head shoulder (datum A), positive Z toward
the small end. R is the perpendicular distance from the axis.

Graphic callouts are rounded to 0.001 mm. Tables keep 0.00001 mm, which
matches the model's stated distance accuracy.
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
from matplotlib.patches import Circle, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder, GeomAbs_Line
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

ROOT = Path(__file__).resolve().parent
STEP_PATH = ROOT / "2_MEDIUM_NUT.STEP"
OUT_PATH = ROOT / "2_MEDIUM_NUT_all_dimensions.pdf"

PAGE_W, PAGE_H = 420.0, 297.0

GEO = "#161616"
DIM = "#0e3a5d"
EXT = "#8aa0b4"
CENTER = "#a32020"
RULE = "#1c1c1c"

FP = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Regular.ttf")
FP_MED = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Medium.ttf")
FP_BOLD = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf")
FP_MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/JetBrainsMono-Regular.ttf")
FONTS = {"regular": FP, "medium": FP_MED, "bold": FP_BOLD, "mono": FP_MONO}

TOTAL = 7


def T(ax, x, y, text, size=7.0, weight="regular", color=RULE, ha="left", va="center", z=5, **kw):
    ax.text(
        x, y, text, fontsize=size, fontproperties=FONTS[weight], color=color,
        ha=ha, va=va, zorder=z, **kw,
    )


def f3(v: float) -> str:
    return f"{v:.3f}"


def f5(v: float) -> str:
    if abs(v) < 5e-6:
        v = 0.0
    return f"{v:.5f}"


def diam(r: float) -> float:
    return 2.0 * r


def ang_txt(seg):
    return f"{seg['ang']:.2f}°"


def f6(v: float) -> str:
    if abs(v) < 5e-7:
        v = 0.0
    return f"{v:.6f}"


class View:
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


def _circle_from_3(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return None
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    return ux, uy, math.hypot(x1 - ux, y1 - uy)


def _arc_points(cx, cz, radius, p0, p1, n=64):
    a0 = math.atan2(p0[1] - cz, p0[0] - cx)
    a1 = math.atan2(p1[1] - cz, p1[0] - cx)
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(abs(sweep) - math.pi) < 1e-9:
        sweep = math.pi if a1 >= a0 else -math.pi
    pts = []
    for i in range(n + 1):
        a = a0 + sweep * i / n
        pts.append((cx + radius * math.cos(a), cz + radius * math.sin(a)))
    return pts, sweep


def _kind_and_angle(p0, p1, is_arc):
    if is_arc:
        return "ARC", None
    dr = p1[0] - p0[0]
    dz = p1[1] - p0[1]
    length = math.hypot(dr, dz)
    if length < 1e-9:
        return "CYL", 0.0
    ang = math.degrees(math.atan2(abs(dr), abs(dz)))
    if ang < 0.02:
        return "CYL", ang
    if abs(ang - 90.0) < 0.02:
        return "FACE", ang
    return "TAPER", ang


def _axis_of(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() == GeomAbs_Cylinder:
            axis = surf.Cylinder().Axis()
            loc, direc = axis.Location(), axis.Direction()
            origin = gp_Pnt(loc.X(), loc.Y(), loc.Z())
            u = gp_Vec(direc.X(), direc.Y(), direc.Z())
            u.Normalize()
            ref = gp_Vec(1.0, 0.0, 0.0) if abs(u.X()) < 0.9 else gp_Vec(0.0, 1.0, 0.0)
            n = u.Crossed(ref)
            n.Normalize()
            w = n.Crossed(u)
            w.Normalize()
            return origin, u, n, w
        exp.Next()
    raise RuntimeError("No cylindrical face found to define the axis")


def _to_rz(origin, u, w, point):
    vec = gp_Vec(origin, point)
    return (vec.Dot(w), vec.Dot(u))


def _seg_from_ends(p0, p1, arc):
    is_arc = arc is not None
    kind, ang = _kind_and_angle(p0, p1, is_arc)
    seg = {
        "kind": kind,
        "p0": p0,
        "p1": p1,
        "r0": p0[0],
        "z0": p0[1],
        "r1": p1[0],
        "z1": p1[1],
        "length": math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
        "ang": ang,
        "radius": None,
        "center": None,
        "pts": [p0, p1],
    }
    if is_arc:
        cx, cz, radius = arc
        pts, sweep = _arc_points(cx, cz, radius, p0, p1)
        seg["radius"] = radius
        seg["center"] = (cx, cz)
        seg["pts"] = pts
        seg["length"] = abs(sweep) * radius
    return seg


def extract_profile(shape):
    origin, u, n, w = _axis_of(shape)
    section = BRepAlgoAPI_Section(shape, gp_Pln(origin, gp_Dir(n.X(), n.Y(), n.Z())), False)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("Section of the solid failed")

    raw = []
    exp = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        curve = BRepAdaptor_Curve(edge)
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        p0 = _to_rz(origin, u, w, curve.Value(u0))
        p1 = _to_rz(origin, u, w, curve.Value(u1))
        rec = {"p0": p0, "p1": p1, "arc": None}
        if curve.GetType() == GeomAbs_Circle:
            cr, cz = _to_rz(origin, u, w, curve.Circle().Location())
            rec["arc"] = (cr, cz, curve.Circle().Radius())
        elif curve.GetType() != GeomAbs_Line:
            mid = _to_rz(origin, u, w, curve.Value(0.5 * (u0 + u1)))
            fit = _circle_from_3(p0, mid, p1)
            samples = [
                _to_rz(origin, u, w, curve.Value(u0 + (u1 - u0) * i / 16.0))
                for i in range(17)
            ]
            if fit is None:
                raise RuntimeError("Section curve is neither a line nor a circular arc")
            cx, cz, radius = fit
            err = max(abs(math.hypot(r - cx, z - cz) - radius) for r, z in samples)
            if err > 1e-4:
                raise RuntimeError(f"Non-circular section curve, fit error {err}")
            rec["arc"] = (cx, cz, radius)
        raw.append(rec)
        exp.Next()

    positive = []
    for rec in raw:
        r0, z0 = rec["p0"]
        r1, z1 = rec["p1"]
        if r0 >= -1e-7 and r1 >= -1e-7:
            kept = dict(rec)
            kept["p0"] = (max(0.0, r0), z0)
            kept["p1"] = (max(0.0, r1), z1)
            positive.append(kept)
            continue
        if r0 <= 1e-7 and r1 <= 1e-7:
            continue
        if abs(r1 - r0) < 1e-12:
            continue
        t = (0.0 - r0) / (r1 - r0)
        zc = z0 + t * (z1 - z0)
        kept = dict(rec)
        if r1 >= r0:
            kept["p0"] = (0.0, zc)
            kept["p1"] = (r1, z1)
        else:
            kept["p0"] = (r0, z0)
            kept["p1"] = (0.0, zc)
        positive.append(kept)

    def span(rec):
        return math.hypot(rec["p1"][0] - rec["p0"][0], rec["p1"][1] - rec["p0"][1])

    positive = [e for e in positive if span(e) > 1e-6]
    used = [False] * len(positive)

    def ends(i):
        e = positive[i]
        return e["p0"], e["p1"]

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
    if not start_at_first:
        a, b = b, a
    p_cur = b
    ordered.append((a, b, positive[start_i]))
    for _ in range(len(positive) + 2):
        nxt = None
        for i, e in enumerate(positive):
            if used[i]:
                continue
            if math.hypot(e["p0"][0] - p_cur[0], e["p0"][1] - p_cur[1]) < 2e-4:
                nxt = (i, e["p0"], e["p1"], e)
                break
            if math.hypot(e["p1"][0] - p_cur[0], e["p1"][1] - p_cur[1]) < 2e-4:
                nxt = (i, e["p1"], e["p0"], e)
                break
        if nxt is None:
            break
        i, a, b, e = nxt
        used[i] = True
        ordered.append((a, b, e))
        p_cur = b
    if not all(used):
        raise RuntimeError(f"Profile chain left {sum(1 for flag in used if not flag)} edges unused")

    segs = [_seg_from_ends(p0, p1, rec["arc"]) for p0, p1, rec in ordered]

    # Datum A is the head shoulder: the radial face of greatest radius.
    faces = [s for s in segs if s["kind"] == "FACE" and min(s["r0"], s["r1"]) > 0.2]
    if not faces:
        raise RuntimeError("No shoulder face found")
    shoulder = max(faces, key=lambda s: max(s["r0"], s["r1"]))
    z_shoulder = 0.5 * (shoulder["z0"] + shoulder["z1"])

    # Reverse so Z increases from the head end toward the small end, with the shoulder at 0.
    flipped = []
    for seg in reversed(segs):
        def fz(z, z0=z_shoulder):
            return z0 - z

        p0 = (seg["r1"], fz(seg["z1"]))
        p1 = (seg["r0"], fz(seg["z0"]))
        arc = None
        if seg["center"] is not None:
            arc = (seg["center"][0], fz(seg["center"][1]), seg["radius"])
        flipped.append(_seg_from_ends(p0, p1, arc))
    for i, seg in enumerate(flipped, start=1):
        seg["n"] = i

    axis = {
        "origin": (origin.X(), origin.Y(), origin.Z()),
        "direction": (u.X(), u.Y(), u.Z()),
        "z_shoulder_profile": z_shoulder,
    }
    return flipped, axis


def _revolution_volume(segs):
    acc = 0.0
    for seg in segs:
        pts = seg["pts"]
        for (r0, z0), (r1, z1) in zip(pts, pts[1:]):
            dz = z1 - z0
            dr = r1 - r0
            acc += dz * (r0 * r0 + r0 * dr + dr * dr / 3.0)
    return math.pi * acc


def _cyls(segs, radius=None, length=None, r_tol=2e-4, l_tol=2e-4):
    found = []
    for seg in segs:
        if seg["kind"] != "CYL":
            continue
        if radius is not None and abs(seg["r0"] - radius) > r_tol:
            continue
        if length is not None and abs(seg["length"] - length) > l_tol:
            continue
        found.append(seg)
    return found


def _on_arc(arc, r, z):
    a0 = math.atan2(arc["p0"][1] - arc["center"][1], arc["p0"][0] - arc["center"][0])
    a1 = math.atan2(arc["p1"][1] - arc["center"][1], arc["p1"][0] - arc["center"][0])
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    aa = math.atan2(z - arc["center"][1], r - arc["center"][0])
    rel = (aa - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if sweep < 0:
        return sweep - 1e-6 <= rel <= 1e-6
    return -1e-6 <= rel <= sweep + 1e-6


def _extreme_on_arc(arc):
    """Greatest radius that lies on the arc, not on the rest of the parent circle."""
    r = arc["center"][0] + arc["radius"]
    z = arc["center"][1]
    if _on_arc(arc, r, z):
        return (r, z)
    return max((arc["p0"], arc["p1"]), key=lambda p: p[0])


def analyze(segs, shape, axis):
    if segs[0]["r0"] > 1e-4 or segs[-1]["r1"] > 1e-4:
        raise RuntimeError("Profile does not run from axis to axis")
    if any(seg["z1"] + 1e-6 < seg["z0"] for seg in segs):
        raise RuntimeError("Profile Z is not monotonic")
    arcs = [s for s in segs if s["kind"] == "ARC"]
    if len(arcs) != 3:
        raise RuntimeError(f"Expected 3 axial arcs, found {len(arcs)}")

    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props)
    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area_props)
    com = volume_props.CentreOfMass()
    origin = gp_Pnt(*axis["origin"])
    u = gp_Vec(*axis["direction"])
    profile_z = gp_Vec(origin, gp_Pnt(com.X(), com.Y(), com.Z())).Dot(u)
    com_z = axis["z_shoulder_profile"] - profile_z
    n = u.Crossed(gp_Vec(1.0, 0.0, 0.0))
    if n.Magnitude() < 1e-9:
        n = u.Crossed(gp_Vec(0.0, 1.0, 0.0))
    n.Normalize()
    w = n.Crossed(u)
    w.Normalize()
    vec = gp_Vec(origin, gp_Pnt(com.X(), com.Y(), com.Z()))
    com_r = math.hypot(vec.Dot(w), vec.Dot(n))
    rev = _revolution_volume(segs)
    if abs(rev - volume_props.Mass()) / volume_props.Mass() > 0.002:
        raise RuntimeError(f"Profile volume {rev:.3f} disagrees with the solid {volume_props.Mass():.3f}")

    form_a = _cyls(segs, 3.8, 0.15, r_tol=1e-4, l_tol=1e-3)
    form_b = _cyls(segs, 3.9, 0.20, r_tol=1e-4, l_tol=1e-3)
    if len(form_a) != 20 or len(form_b) != 6:
        raise RuntimeError(f"Unexpected crests A={len(form_a)} B={len(form_b)}")
    pitch_a = [form_a[i + 1]["z0"] - form_a[i]["z0"] for i in range(len(form_a) - 1)]
    pitch_b = [form_b[i + 1]["z0"] - form_b[i]["z0"] for i in range(len(form_b) - 1)]
    if any(abs(p - 1.2) > 1e-4 for p in pitch_a):
        raise RuntimeError(f"Form A pitch is not 1.200: {pitch_a[:3]}")
    if any(abs(p - 1.75) > 1e-4 for p in pitch_b):
        raise RuntimeError(f"Form B pitch is not 1.750: {pitch_b}")

    necks = _cyls(segs, 4.15, r_tol=1e-3)
    if len(necks) != 1:
        raise RuntimeError(f"Expected one Ø8.300 cylinder, found {len(necks)}")

    roots = _cyls(segs, 3.35, r_tol=1e-3)
    root_groups = []
    for seg in roots:
        for group in root_groups:
            if abs(group[0]["length"] - seg["length"]) < 5e-4:
                group.append(seg)
                break
        else:
            root_groups.append([seg])

    faces = [s for s in segs if s["kind"] == "FACE"]
    shoulder = min(faces, key=lambda s: abs(s["z0"]))
    head_face = min(faces, key=lambda s: s["z0"])
    tip_face = max(faces, key=lambda s: s["z0"])

    crown_arc = max(arcs, key=lambda s: _extreme_on_arc(s)[0])
    crown = _extreme_on_arc(crown_arc)
    if abs(crown[0] - 8.0) > 1e-4 or abs(crown[1] + 1.6) > 1e-4:
        raise RuntimeError(f"Crown is not Ø16.000 at Z −1.600: {crown}")

    model = {
        "segs": segs,
        "axis": axis,
        "head_z": head_face["z0"],
        "tip_z": tip_face["z0"],
        "overall": tip_face["z0"] - head_face["z0"],
        "head_len": -head_face["z0"],
        "arcs": arcs,
        "crown_arc": crown_arc,
        "crown": crown,
        "shoulder": shoulder,
        "head_face": head_face,
        "tip_face": tip_face,
        "neck": necks[0],
        "form_a": form_a,
        "form_b": form_b,
        "pitch_a": pitch_a[0],
        "pitch_b": pitch_b[0],
        "root_groups": root_groups,
        "mass": {
            "volume": volume_props.Mass(),
            "area": area_props.Mass(),
            "com_z": com_z,
            "com_r": com_r,
        },
    }
    if abs(model["overall"] - 61.5) > 1e-3 or abs(model["head_z"] + 5.0) > 1e-3:
        raise RuntimeError(f"Unexpected envelope head {model['head_z']} overall {model['overall']}")
    if abs(model["tip_z"] - 56.5) > 1e-3:
        raise RuntimeError(f"Tip station {model['tip_z']}")
    if model["mass"]["com_r"] > 1e-3:
        raise RuntimeError("Centre of mass is off the axis")
    return model


def render_iso(solid):
    """World Z is already the drawing Z. The axis sits at X = 0.1."""
    verts, tris = solid.tessellate(0.2)
    xyz = np.array([(v.z, v.x - 0.1, v.y) for v in verts], dtype=float)
    faces = xyz[np.asarray(tris)]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.array([0.58, 0.66, 0.74])
    rgb = np.clip(0.18 + 0.82 * shade[:, None] * base, 0, 1)
    rgba = np.concatenate([rgb, np.ones((len(faces), 1))], axis=1)

    fig = plt.figure(figsize=(7.6, 2.8), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(faces, linewidths=0.0, antialiased=False)
    coll.set_facecolor(rgba)
    ax.add_collection3d(coll)
    xmin, xmax = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    span = max(float(xyz[:, 1].max() - xyz[:, 1].min()), float(xyz[:, 2].max() - xyz[:, 2].min()))
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(-span / 2 - 1, span / 2 + 1)
    ax.set_zlim(-span / 2 - 1, span / 2 + 1)
    ax.set_box_aspect((xmax - xmin + 2, span + 2, span + 2))
    ax.view_init(elev=16, azim=-90)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=130, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    image = plt.imread(buf)[:, :, :3]
    mask = np.any(image < 0.97, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 6
    return image[max(0, rows[0] - pad):rows[-1] + pad, max(0, cols[0] - pad):cols[-1] + pad]


def new_page():
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def frame(ax, page, view_title):
    ax.add_patch(Rectangle((6, 6), 408, 285, fill=False, lw=1.15, ec=RULE, zorder=6))
    ax.add_patch(Rectangle((8, 8), 404, 281, fill=False, lw=0.35, ec=RULE, zorder=6))
    T(ax, 12, 286.2, "2 MEDIUM NUT", size=10.5, weight="bold", va="top")
    T(ax, 62, 286.2, "TURNED SOLID  ·  NOMINAL DIMENSION DRAWING", size=8.0, weight="medium", color=DIM, va="top")
    T(ax, 406, 286.2, view_title, size=7.2, weight="medium", ha="right", va="top", color="#333")
    x, y, w, h = 248, 8, 164, 40
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=RULE, lw=0.7, zorder=4))
    ax.plot([x, x + w], [y + 28, y + 28], color=RULE, lw=0.4, zorder=4)
    ax.plot([x, x + w], [y + 16, y + 16], color=RULE, lw=0.4, zorder=4)
    ax.plot([x + 82, x + 82], [y, y + h], color=RULE, lw=0.4, zorder=4)
    T(ax, x + 3, y + 34, "PART", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 34, "2 MEDIUM NUT", size=7.2, weight="bold", va="center", z=7)
    T(ax, x + 85, y + 34, "SHEET", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 34, f"{page}  /  {TOTAL}", size=8, weight="bold", va="center", z=7)
    T(ax, x + 3, y + 22, "FILE", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 22, "2 MEDIUM NUT.STEP", size=6.4, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 22, "UNITS", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 22, "MILLIMETRES", size=7.2, weight="medium", va="center", z=7)
    T(ax, x + 3, y + 8, "DATUM A  SHOULDER Z = 0", size=5.2, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 8, "DATUM B  AXIS", size=5.2, weight="medium", va="center", z=7)
    cx, cy = 214, 24
    ax.add_patch(Circle((cx, cy + 8), 3.1, fill=False, lw=0.55, ec=RULE, zorder=6))
    ax.plot([cx + 5.0, cx + 12.4], [cy + 5.4, cy + 6.5], color=RULE, lw=0.55, zorder=6)
    ax.plot([cx + 5.0, cx + 12.4], [cy + 10.6, cy + 9.5], color=RULE, lw=0.55, zorder=6)
    ax.plot([cx + 12.4, cx + 12.4], [cy + 6.5, cy + 9.5], color=RULE, lw=0.55, zorder=6)
    T(ax, cx + 6, cy + 1.2, "3RD ANGLE", size=4.4, ha="center", color="#444", z=7)


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
    if gap >= 14:
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0),
            zorder=3,
        )
        va = "bottom" if text_side == "up" else "top"
        dy = 0.7 if text_side == "up" else -0.7
        T(ax, (x1 + x2) / 2, y + dy, text, size=6.0, weight="medium", color=DIM, ha="center", va=va,
          bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))
    else:
        wing = 5.5
        ax.plot([x1 - wing, x2 + wing], [y, y], color=DIM, lw=0.5, zorder=3)
        ax.annotate("", xy=(x1, y), xytext=(x1 - wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", xy=(x2, y), xytext=(x2 + wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        va = "bottom" if text_side == "up" else "top"
        dy = 1.3 if text_side == "up" else -1.3
        T(ax, (x1 + x2) / 2, y + dy, text, size=5.6, weight="medium", color=DIM, ha="center", va=va,
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
            "", xy=(x, y2), xytext=(x, y1),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0),
            zorder=3,
        )
        ha = "left" if text_side == "right" else "right"
        dx = 1.1 if text_side == "right" else -1.1
        T(ax, x + dx, (y1 + y2) / 2, text, size=6.0, weight="medium", color=DIM, ha=ha, va="center",
          bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))
    else:
        wing = 6.0
        ax.plot([x, x], [y1 - wing, y2 + wing], color=DIM, lw=0.5, zorder=3)
        ax.annotate("", xy=(x, y1), xytext=(x, y1 - wing),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", xy=(x, y2), xytext=(x, y2 + wing),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        T(ax, x + 1.2, y2 + wing + 1.3, text, size=5.8, weight="medium", color=DIM, ha="left", va="bottom",
          bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.92))


def leader(ax, x, y, tx, ty, text, ha="left"):
    ax.annotate(
        "", xy=(x, y), xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.45, shrinkA=0, shrinkB=0), zorder=3,
    )
    ax.plot([x], [y], marker="o", ms=1.7, color=DIM, zorder=4)
    dx = 1.2 if ha == "left" else -1.2
    T(ax, tx + dx, ty, text, size=6.0, weight="medium", color=DIM, ha=ha, va="center",
      bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))


def centerline(ax, view, z0, z1):
    x0, y = view.pt(z0, 0)
    x1, _ = view.pt(z1, 0)
    ax.plot([x0, x1], [y, y], color=CENTER, lw=0.4, linestyle=(0, (7, 1.6, 0.9, 1.6)), zorder=2)


def _clip_pts(pts, z0, z1):
    kept = []
    for i, (r, z) in enumerate(pts):
        if z0 is not None and z < z0 - 1e-9:
            nxt = pts[i + 1] if i + 1 < len(pts) else None
            if nxt is not None and nxt[1] >= z0 - 1e-9 and abs(nxt[1] - z) > 1e-12:
                t = (z0 - z) / (nxt[1] - z)
                kept.append((r + t * (nxt[0] - r), z0))
            continue
        if z1 is not None and z > z1 + 1e-9:
            if kept and abs(z - kept[-1][1]) > 1e-12:
                r0, z_prev = kept[-1]
                t = (z1 - z_prev) / (z - z_prev)
                kept.append((r0 + t * (r - r0), z1))
            break
        kept.append((r, z))
    return kept


def draw_profile(ax, view, segs, both=True, lw=0.9, z0=None, z1=None):
    for seg in segs:
        pts = _clip_pts(seg["pts"], z0, z1)
        if len(pts) < 2:
            continue
        for sign in ((1, -1) if both else (1,)):
            xs, ys = [], []
            for r, z in pts:
                x, y = view.pt(z, sign * r)
                xs.append(x)
                ys.append(y)
            ax.plot(xs, ys, color=GEO, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3)


def balloon(ax, x, y, label):
    ax.add_patch(Circle((x, y), 3.4, fc="white", ec=DIM, lw=0.7, zorder=5))
    T(ax, x, y, label, size=6.6, weight="bold", color=DIM, ha="center", va="center", z=6)


def draw_table(ax, x, y_top, widths, headers, rows, row_h=3.5, size=5.1):
    total_w = sum(widths)
    head_h = row_h + 0.45
    y = y_top - head_h
    ax.add_patch(Rectangle((x, y), total_w, head_h, fc="#e7eef6", ec="#c5d2e0", lw=0.3, zorder=2))
    cx = x
    for head, w in zip(headers, widths):
        T(ax, cx + 1.0, y + head_h / 2, head, size=size, weight="bold", color=DIM, va="center")
        cx += w
    for i, row in enumerate(rows):
        y -= row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((x, y), total_w, row_h, fc="#fbfcfe", ec="none", zorder=1))
        cx = x
        for cell, w in zip(row, widths):
            T(ax, cx + 1.0, y + row_h / 2, str(cell), size=size, weight="mono", va="center", color="#1d1d1d")
            cx += w
    ax.add_patch(Rectangle((x, y), total_w, y_top - y, fill=False, ec="#c5d2e0", lw=0.45, zorder=3))
    return y


def midpt(seg):
    if seg["kind"] == "ARC":
        return seg["pts"][len(seg["pts"]) // 2]
    return ((seg["r0"] + seg["r1"]) / 2.0, (seg["z0"] + seg["z1"]) / 2.0)


def find_ang(segs, angle, tol=0.03):
    return next(s for s in segs if s["ang"] is not None and abs(s["ang"] - angle) < tol)


def break_mark(ax, view, z, r):
    bx, by = view.pt(z, r)
    for sign in (1, -1):
        yy = view.ry_of(sign * r)
        ax.plot([bx, bx + 1.3, bx - 0.45, bx + 1.3], [yy, yy + 1.1 * sign, yy + 2.2 * sign, yy + 3.3 * sign],
                color=GEO, lw=0.5, zorder=3)


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def sheet_general(pdf, model, iso):
    fig, ax = new_page()
    frame(ax, 1, "SHEET 1   GENERAL ARRANGEMENT")
    segs = model["segs"]
    scale = 4.4
    view = View(zx=52, y0=164, scale=scale)
    draw_profile(ax, view, segs, both=True, lw=0.5)
    centerline(ax, view, model["head_z"] - 1.5, model["tip_z"] + 1.6)
    T(ax, view.zx_of(model["head_z"] - 1.4), view.y0 + 2.0, "B", size=6.2, weight="bold", color=CENTER, ha="right")

    face_x = view.zx_of(0)
    ax.plot([face_x, face_x], [view.ry_of(-7.2), view.ry_of(-8.3)], color=DIM, lw=0.4, zorder=3)
    ax.add_patch(Polygon(
        [(face_x, view.ry_of(-8.3)), (face_x - 1.5, view.ry_of(-8.3) - 2.6), (face_x + 1.5, view.ry_of(-8.3) - 2.6)],
        closed=True, fc=DIM, ec=DIM, lw=0.2, zorder=4,
    ))
    ax.add_patch(Rectangle((face_x - 3.1, view.ry_of(-8.3) - 7.2), 6.2, 4.4, fc="white", ec=DIM, lw=0.55, zorder=4))
    T(ax, face_x, view.ry_of(-8.3) - 5.0, "A", size=5.8, weight="bold", color=DIM, ha="center", va="center", z=5)

    hdim(ax, view.zx_of(model["head_z"]), view.zx_of(model["tip_z"]), 108,
         f"{f3(model['overall'])} OVERALL", view.ry_of(-model["crown"][0]), view.ry_of(0))
    hdim(ax, view.zx_of(0), view.zx_of(model["tip_z"]), 96,
         f"{f3(model['tip_z'])}   DATUM A TO TIP", view.ry_of(-4.2), view.ry_of(0))
    hdim(ax, view.zx_of(model["head_z"]), view.zx_of(0), 84,
         f"{f3(model['head_len'])} HEAD", view.ry_of(0), view.ry_of(-6))

    crown_r, crown_z = model["crown"]
    leader(ax, *view.pt(crown_z, crown_r), view.zx_of(crown_z) - 6, view.ry_of(crown_r) + 12,
           f"Ø {f3(diam(crown_r))}", ha="right")
    neck = model["neck"]
    leader(ax, *view.pt(neck["z0"] + 2.0, neck["r0"]), view.zx_of(4.5), view.ry_of(neck["r0"]) + 14,
           f"Ø {f3(diam(neck['r0']))}", ha="left")
    crest = model["form_a"][10]
    leader(ax, *view.pt(0.5 * (crest["z0"] + crest["z1"]), crest["r0"]),
           view.zx_of(crest["z0"]), view.ry_of(5.2), "Ø 7.600 CREST", ha="left")
    tip_r = max(model["tip_face"]["r0"], model["tip_face"]["r1"])
    leader(ax, *view.pt(model["tip_z"], tip_r), view.zx_of(model["tip_z"]) + 1.5, view.ry_of(tip_r) + 10,
           f"Ø {f3(diam(tip_r))} TIP", ha="left")

    zones = [
        (model["form_b"][0]["z0"], model["form_b"][-1]["z1"], "FORM B"),
        (model["form_a"][0]["z0"], model["form_a"][-1]["z1"], "FORM A"),
    ]
    for z0, z1, name in zones:
        y = view.ry_of(crown_r) + 5.5
        ax.annotate("", xy=(view.zx_of(z1), y), xytext=(view.zx_of(z0), y),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0))
        T(ax, (view.zx_of(z0) + view.zx_of(z1)) / 2, y + 2.0, name, size=5.3, weight="bold",
          color=DIM, ha="center", va="bottom")

    balloon(ax, view.zx_of(-3.4), view.ry_of(crown_r) + 24, "2")
    balloon(ax, view.zx_of(17.6), view.ry_of(crown_r) + 24, "3")
    balloon(ax, view.zx_of(41.5), view.ry_of(crown_r) + 24, "4")
    T(ax, 14, 74, "SCALE 4.4 : 1     AXIS HORIZONTAL     HEAD AT LEFT     FULL PROFILE", size=5.8, color="#333")

    ih, iw = iso.shape[0], iso.shape[1]
    box_w, box_h = 96.0, 62.0
    aspect = iw / ih
    if box_w / box_h > aspect:
        dh, dw = box_h, box_h * aspect
    else:
        dw, dh = box_w, box_w / aspect
    x0, y0 = 404 - dw, 268 - dh
    ax.imshow(iso, extent=(x0, x0 + dw, y0, y0 + dh), aspect="equal", zorder=1, interpolation="bilinear")
    T(ax, x0, y0 - 3.0, "PICTORIAL — NOT TO SCALE     HEAD AT LEFT", size=5.0, color="#444")

    m = model["mass"]
    notes = [
        "Solid of revolution about datum B. Revolve the +R profile.",
        "Sizes are CAD nominals from 2 MEDIUM NUT.STEP (SolidWorks 2023, 2024-10-24).",
        "No tolerance, material, or finish is stored in the model.",
        "Graphic sizes are rounded to 0.001 mm. Sheets 6 and 7 list every segment to 0.00001 mm.",
        "Positive Z runs from datum A toward the small end.",
        f"Volume {m['volume']:.3f} mm³     area {m['area']:.3f} mm²     centre of mass Z {f3(m['com_z'])} on the axis.",
        "Balloons 2, 3 and 4 mark the detail sheets.",
        "Form B: 6 crests, pitch 1.750, crest Ø 7.800.   Form A: 20 crests, pitch 1.200, crest Ø 7.600.",
    ]
    yy = 66
    for line in notes:
        T(ax, 14, yy, line, size=5.3, color="#333", va="center")
        yy -= 5.3
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_head(pdf, model):
    fig, ax = new_page()
    frame(ax, 2, "SHEET 2   HEAD, SHOULDER AND NECK")
    segs = model["segs"]
    scale = 10.0
    view = View(zx=96, y0=152, scale=scale)
    neck = model["neck"]
    draw_profile(ax, view, segs, both=True, lw=0.9, z0=model["head_z"] - 0.05, z1=neck["z1"])
    centerline(ax, view, model["head_z"] - 0.5, neck["z1"] + 0.4)
    T(ax, view.zx_of(model["head_z"] - 0.45), view.y0 + 2.2, "B", size=6.2, weight="bold", color=CENTER, ha="right")
    break_mark(ax, view, neck["z1"], neck["r0"])
    T(ax, view.zx_of(neck["z1"]) + 3.5, view.ry_of(neck["r0"]) + 6, "SHEET 3", size=5.2, color=DIM)

    fx = view.zx_of(0)
    ax.plot([fx, fx + 8], [view.ry_of(7.0), view.ry_of(7.0)], color=DIM, lw=0.35, zorder=3)
    T(ax, fx + 9, view.ry_of(7.0), "DATUM A", size=5.8, weight="bold", color=DIM, va="center")

    crown_r, crown_z = model["crown"]
    head_r = max(model["head_face"]["r0"], model["head_face"]["r1"])
    vdim(ax, view.zx_of(model["head_z"]) - 10, view.ry_of(-crown_r), view.ry_of(crown_r),
         f"Ø {f3(diam(crown_r))}", view.zx_of(crown_z), view.zx_of(crown_z), text_side="left")
    vdim(ax, view.zx_of(4.5), view.ry_of(-neck["r0"]), view.ry_of(neck["r0"]),
         f"Ø {f3(diam(neck['r0']))}", view.zx_of(4.5), view.zx_of(4.5))
    hdim(ax, view.zx_of(model["head_z"]), view.zx_of(0), 58,
         f"{f3(model['head_len'])} HEAD", view.ry_of(-1.5), view.ry_of(-min(model["shoulder"]["r0"], model["shoulder"]["r1"])))
    hdim(ax, view.zx_of(neck["z0"]), view.zx_of(neck["z1"]), 48,
         f"{f3(neck['length'])} NECK", view.ry_of(-neck["r0"]), view.ry_of(-neck["r0"]))

    for arc in model["arcs"]:
        pr, pz = midpt(arc)
        label = f"R {f3(arc['radius'])}"
        if arc is model["crown_arc"]:
            label = f"R {f3(arc['radius'])}   Ø {f3(diam(crown_r))} AT Z {f3(crown_z)}"
            leader(ax, *view.pt(crown_z, crown_r), view.zx_of(crown_z) + 4, view.ry_of(crown_r) + 10, label, ha="left")
        elif arc["radius"] > 8:
            leader(ax, *view.pt(pz, pr), view.zx_of(pz) - 2, view.ry_of(pr) + 8, label, ha="right")
        else:
            leader(ax, *view.pt(pz, pr), view.zx_of(pz) - 8, view.ry_of(pr) + 8, label, ha="right")
    leader(ax, *view.pt(model["head_z"], head_r * 0.55), view.zx_of(model["head_z"]) - 12, view.ry_of(head_r * 0.55),
           f"END Ø {f3(diam(head_r))}", ha="right")

    sh = model["shoulder"]
    rows = [
        ["HEAD FACE", f5(model["head_z"]), "0.00000", "0.000", "ON THE AXIS"],
        ["HEAD RIM", f5(model["head_z"]), f5(head_r), f5(diam(head_r)), "R6 MEETS THE FACE"],
        ["CROWN", f5(crown_z), f5(crown_r), f5(diam(crown_r)), "LARGEST DIAMETER"],
        ["DATUM A OUT", "0.00000", f5(max(sh["r0"], sh["r1"])), f5(diam(max(sh["r0"], sh["r1"]))), "SHOULDER OUTER"],
        ["DATUM A IN", "0.00000", f5(min(sh["r0"], sh["r1"])), f5(diam(min(sh["r0"], sh["r1"]))), "SHOULDER INNER"],
        ["NECK END", f5(neck["z1"]), f5(neck["r1"]), f5(diam(neck["r1"])), "13.60° TAPER STARTS"],
    ]
    draw_table(ax, 214, 266, [32, 28, 28, 28, 64],
               ["POINT", "Z", "RADIUS", "Ø", "NOTE"], rows, row_h=4.3, size=5.1)
    arc_rows = []
    for arc in model["arcs"]:
        c = arc["center"]
        arc_rows.append([f5(arc["radius"]), f5(c[0]), f5(c[1]), f5(arc["z0"]), f5(arc["z1"]), f5(arc["length"])])
    T(ax, 214, 228, "AXIAL-SECTION ARCS", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 214, 225, [24, 28, 28, 28, 28, 30],
               ["R", "CENTER R", "CENTER Z", "Z0", "Z1", "ARC LEN"], arc_rows, row_h=4.6, size=5.1)
    notes = [
        "The R3.100 arc carries Ø 16.000.",
        "R6.000 is tangent to the head end face.",
        "Datum A is the annular face from Ø 8.300 to the head arc.",
        "The neck is a true cylinder. The 13.60° taper is on sheet 3.",
    ]
    yy = 188
    for line in notes:
        T(ax, 214, yy, line, size=5.3, color="#333")
        yy -= 6.0
    T(ax, 14, 274, "DETAIL 2     SCALE 10 : 1     FULL PROFILE", size=6.6, weight="bold", color=DIM)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_b(pdf, model):
    fig, ax = new_page()
    frame(ax, 3, "SHEET 3   FORM B AND GROOVE")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 3     ONE PITCH ENLARGED     GROOVE AT TRUE RADIAL SCALE     ANGLES FROM THE AXIS", size=6.2, weight="bold", color=DIM)

    b0, b1 = model["form_b"][0], model["form_b"][1]
    tooth = View(zx=22, y0=198, scale=34, z0=b0["z0"] - 0.05, r0=3.15)
    draw_profile(ax, tooth, segs, both=False, lw=1.05, z0=b0["z0"], z1=b1["z0"])
    hdim(ax, tooth.zx_of(b0["z0"]), tooth.zx_of(b1["z0"]), tooth.ry_of(4.15),
         f"PITCH {f3(model['pitch_b'])}", tooth.ry_of(3.9), tooth.ry_of(3.9))
    hdim(ax, tooth.zx_of(b0["z0"]), tooth.zx_of(b0["z1"]), tooth.ry_of(4.02),
         f"FLAT {f3(b0['length'])}", tooth.ry_of(3.9), tooth.ry_of(3.9))
    fall = next(s for s in segs if abs(s["z0"] - b0["z1"]) < 1e-6)
    rise = next(s for s in segs if abs(s["z1"] - b1["z0"]) < 1e-6)
    fx, fy = tooth.pt(*midpt(fall)[::-1])
    rx, ry = tooth.pt(*midpt(rise)[::-1])
    leader(ax, fx, fy, fx - 8, fy - 12, ang_txt(fall), ha="right")
    leader(ax, rx, ry, rx + 8, ry - 12, ang_txt(rise), ha="left")
    root = next(s for s in segs if s["kind"] == "CYL" and abs(s["r0"] - 3.35) < 1e-3 and b0["z1"] - 1e-6 < s["z0"] < b1["z0"])
    hdim(ax, tooth.zx_of(root["z0"]), tooth.zx_of(root["z1"]), tooth.ry_of(3.05),
         f"ROOT {f5(root['length'])}", tooth.ry_of(3.35), tooth.ry_of(3.35))
    leader(ax, *tooth.pt(b0["z0"] + 0.05, 3.9), tooth.zx_of(b0["z0"]) - 0.15, tooth.ry_of(4.28), "Ø 7.800", ha="left")
    T(ax, 16, 178, "FORM B TOOTH   SCALE 34 : 1   MATERIAL BELOW THE LINE   Ø 6.700 ROOT   6 CRESTS", size=5.2, color="#333")

    g0, g1 = 22.2, 31.2
    groove = View(zx=18, y0=92, scale=11, z0=g0, r0=0.0)
    draw_profile(ax, groove, segs, both=False, lw=0.9, z0=g0, z1=g1)
    centerline(ax, groove, g0 - 0.1, g1 + 0.15)
    g_fall = find_ang(segs, 69.65)
    g_rise = find_ang(segs, 57.4)
    fx, fy = groove.pt(*midpt(g_fall)[::-1])
    rx, ry = groove.pt(*midpt(g_rise)[::-1])
    leader(ax, fx, fy, fx - 18, fy + 14, ang_txt(g_fall), ha="right")
    leader(ax, rx, ry, rx - 2, ry + 18, ang_txt(g_rise), ha="left")
    vx, vy = groove.pt(g_fall["z1"], g_fall["r1"])
    leader(ax, vx, vy, vx - 16, vy - 9, f"Ø {f3(diam(g_fall['r1']))}", ha="right")
    lx, ly = groove.pt(g_rise["z1"] + 0.55, g_rise["r1"])
    leader(ax, lx, ly, lx + 14, ly + 8, f"Ø {f3(diam(g_rise['r1']))}", ha="left")
    T(ax, 16, 80, "GROOVE BETWEEN THE FORMS   SCALE 11 : 1   HALF PROFILE   REMAINING TRANSITIONS ON SHEET 5",
      size=5.2, color="#333")

    rows = []
    prev = None
    for i, c in enumerate(model["form_b"], start=1):
        pitch = "—" if prev is None else f5(c["z0"] - prev)
        rows.append([str(i), f5(c["z0"]), f5(c["z1"]), pitch])
        prev = c["z0"]
    draw_table(ax, 214, 266, [12, 36, 36, 36],
               ["#", "Z0", "Z1", "PITCH"], rows, row_h=4.2, size=5.2)
    T(ax, 214, 268, "", size=1)  # keep the table under the frame
    spec = [
        ["CREST Ø", "7.800", "FLAT 0.20000"],
        ["ROOT Ø", "6.700", f"FLAT {f5(root['length'])}  ×5"],
        ["FLANKS", f"{fall['ang']:.2f}° / {rise['ang']:.2f}°", "LEAVING / APPROACHING, +Z"],
        ["PITCH", "1.75000", "CREST START TO CREST START"],
        ["COUNT", "6", f"Z {f5(model['form_b'][0]['z0'])} TO {f5(model['form_b'][-1]['z1'])}"],
    ]
    # The root after the last crest is longer than the five repeating roots.
    odd = [
        g for g in model["root_groups"]
        if abs(g[0]["r0"] - 3.35) < 1e-3 and len(g) == 1
        and model["form_b"][-1]["z1"] < g[0]["z0"] < model["form_a"][0]["z0"]
    ]
    if odd:
        spec.append(["RUNOUT ROOT", f5(odd[0][0]["length"]), f"Z {f5(odd[0][0]['z0'])}  Ø 6.700"])
    T(ax, 214, 228, "FORM B DEFINITION", size=6.2, weight="bold", color=DIM)
    y = draw_table(ax, 214, 225, [36, 40, 78], ["ITEM", "VALUE", "NOTE"], spec, row_h=4.0, size=5.1)

    trans = [
        s for s in segs
        if s["kind"] in ("TAPER", "CYL") and model["neck"]["z1"] - 1e-6 <= s["z0"] < model["form_b"][0]["z0"]
    ]
    t_rows = []
    for s in trans:
        label = ang_txt(s) if s["kind"] == "TAPER" else f"CYL L {f5(s['length'])}"
        t_rows.append([label, f5(s["z0"]), f5(s["z1"]), f5(diam(s["r0"])), f5(diam(s["r1"]))])
    T(ax, 214, y - 5, "NECK TO FIRST FORM B CREST", size=6.0, weight="bold", color=DIM)
    draw_table(ax, 214, y - 8, [32, 30, 30, 28, 28],
               ["FEATURE", "Z0", "Z1", "Ø START", "Ø END"], t_rows, row_h=4.0, size=4.8)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_a(pdf, model):
    fig, ax = new_page()
    frame(ax, 4, "SHEET 4   FORM A AND TIP")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 4     REGULAR CREST ENLARGED     TIP AT TRUE RADIAL SCALE", size=6.2, weight="bold", color=DIM)

    # Second crest is a fully repeated tooth; the first crest has a shortened head-side flank.
    a0, a1 = model["form_a"][1], model["form_a"][2]
    tooth = View(zx=20, y0=200, scale=46, z0=a0["z0"] - 0.02, r0=3.15)
    draw_profile(ax, tooth, segs, both=False, lw=1.05, z0=a0["z0"], z1=a1["z0"])
    hdim(ax, tooth.zx_of(a0["z0"]), tooth.zx_of(a1["z0"]), tooth.ry_of(4.05),
         f"PITCH {f3(model['pitch_a'])}", tooth.ry_of(3.8), tooth.ry_of(3.8))
    hdim(ax, tooth.zx_of(a0["z0"]), tooth.zx_of(a0["z1"]), tooth.ry_of(3.92),
         f"FLAT {f3(a0['length'])}", tooth.ry_of(3.8), tooth.ry_of(3.8))
    fall = next(s for s in segs if abs(s["z0"] - a0["z1"]) < 1e-6)
    rise = next(s for s in segs if abs(s["z1"] - a1["z0"]) < 1e-6)
    fx, fy = tooth.pt(*midpt(fall)[::-1])
    rx, ry = tooth.pt(*midpt(rise)[::-1])
    leader(ax, fx, fy, fx - 6, fy - 10, ang_txt(fall), ha="right")
    leader(ax, rx, ry, rx + 6, ry - 10, ang_txt(rise), ha="left")
    root = next(s for s in segs if s["kind"] == "CYL" and abs(s["r0"] - 3.35) < 1e-3 and a0["z1"] < s["z0"] < a1["z0"])
    hdim(ax, tooth.zx_of(root["z0"]), tooth.zx_of(root["z1"]), tooth.ry_of(2.95),
         f"ROOT {f5(root['length'])}", tooth.ry_of(3.35), tooth.ry_of(3.35))
    T(ax, 16, 182, "CRESTS 2–20   SCALE 46 : 1   Ø 7.600 CREST   Ø 6.700 ROOT   20 CRESTS AT PITCH 1.200", size=5.1, color="#333")

    tip = View(zx=18, y0=78, scale=12, z0=51.5, r0=0.0)
    draw_profile(ax, tip, segs, both=False, lw=0.9, z0=51.4, z1=model["tip_z"] + 0.02)
    centerline(ax, tip, 51.3, model["tip_z"] + 0.3)
    chamfer = find_ang(segs, 31.2)
    leader(ax, *tip.pt(*midpt(chamfer)[::-1]), tip.zx_of(54.6), tip.ry_of(1.2), ang_txt(chamfer), ha="right")
    tip_r = max(model["tip_face"]["r0"], model["tip_face"]["r1"])
    leader(ax, *tip.pt(model["tip_z"], tip_r * 0.65), tip.zx_of(model["tip_z"]) + 0.3, tip.ry_of(2.4),
           f"TIP Ø {f3(diam(tip_r))}", ha="left")
    leader(ax, *tip.pt(model["form_a"][-1]["z0"] + 0.05, 3.8), tip.zx_of(52.2), tip.ry_of(4.5), "Ø 7.600", ha="left")
    T(ax, 16, 66, "TIP   SCALE 12 : 1   HALF PROFILE", size=5.2, color="#333")

    rows = []
    prev = None
    for i, c in enumerate(model["form_a"], start=1):
        pitch = "—" if prev is None else f5(c["z0"] - prev)
        rows.append([str(i), f5(c["z0"]), f5(c["z1"]), pitch])
        prev = c["z0"]
    left, right = rows[:10], rows[10:]
    paired = [a + b for a, b in zip(left, right)]
    draw_table(ax, 168, 266, [10, 30, 30, 24, 10, 30, 30, 24],
               ["#", "Z0", "Z1", "PITCH", "#", "Z0", "Z1", "PITCH"], paired, row_h=3.7, size=4.8)

    chamfer = find_ang(segs, 31.2)
    tip_root = next(s for s in segs if s["kind"] == "CYL" and abs(s["z1"] - chamfer["z0"]) < 1e-6)
    first = model["form_a"][0]
    prev = segs[first["n"] - 2]
    notes = [
        ["CREST / ROOT", "Ø 7.600 / Ø 6.700", f"FLAT {f5(a0['length'])} / {f5(root['length'])}"],
        ["FLANKS", f"{fall['ang']:.2f}° / {rise['ang']:.2f}°", "LEAVING / APPROACHING A CREST, +Z"],
        ["CREST 1 HEAD SIDE", ang_txt(prev), f"STOPS AT Ø {f3(diam(min(prev['r0'], prev['r1'])))}"],
        ["TIP ROOT", f5(tip_root["length"]), f"Z {f5(tip_root['z0'])} BEFORE THE 31.20°"],
        ["TIP FACE", f5(model["tip_z"]), f"Ø {f5(diam(tip_r))} TO THE AXIS"],
    ]
    T(ax, 168, 214, "FORM A NOTES", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 168, 211, [42, 48, 92], ["ITEM", "VALUE", "NOTE"], notes, row_h=4.2, size=5.0)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_schedule(pdf, model):
    fig, ax = new_page()
    frame(ax, 5, "SHEET 5   DIMENSION SCHEDULE")
    T(ax, 14, 274, "CONTROLLING SIZES.  Z FROM DATUM A, POSITIVE TOWARD THE SMALL END.  ANGLES FROM THE AXIS.  MODEL VALUES TO 0.00001 mm.",
      size=5.8, weight="medium", color=DIM)
    m = model["mass"]
    sh = model["shoulder"]
    head_r = max(model["head_face"]["r0"], model["head_face"]["r1"])
    tip_r = max(model["tip_face"]["r0"], model["tip_face"]["r1"])
    crown_r, crown_z = model["crown"]
    direc = model["axis"]["direction"]
    origin = model["axis"]["origin"]

    overall = [
        ["OVERALL", f5(model["overall"]), "HEAD FACE TO TIP FACE"],
        ["HEAD FACE Z", f5(model["head_z"]), f"Ø {f5(diam(head_r))}"],
        ["DATUM A", "0.00000", f"Ø {f5(diam(min(sh['r0'], sh['r1'])))} TO {f5(diam(max(sh['r0'], sh['r1'])))}"],
        ["LARGEST Ø", f5(diam(crown_r)), f"AT Z {f5(crown_z)}"],
        ["TIP FACE Z", f5(model["tip_z"]), f"Ø {f5(diam(tip_r))}"],
        ["VOLUME", f"{m['volume']:.5f}", "mm³   SOLID, NO BORE"],
        ["SURFACE", f"{m['area']:.5f}", "mm²"],
        ["COM Z", f5(m["com_z"]), "ON THE AXIS"],
    ]
    T(ax, 12, 266, "1   OVERALL", size=6.3, weight="bold", color=DIM)
    y = draw_table(ax, 12, 263, [32, 36, 78], ["ITEM", "MODEL", "NOTE"], overall, row_h=3.45, size=4.9)

    arc_rows = []
    for arc in model["arcs"]:
        c = arc["center"]
        arc_rows.append([f5(arc["radius"]), f5(c[0]), f5(c[1]), f5(arc["z0"]), f5(arc["z1"])])
    T(ax, 12, y - 5, "2   ARCS", size=6.3, weight="bold", color=DIM)
    y = draw_table(ax, 12, y - 8, [24, 28, 28, 28, 28],
                   ["R", "CTR R", "CTR Z", "Z0", "Z1"], arc_rows, row_h=3.6, size=4.9)

    neck = model["neck"]
    T(ax, 12, y - 5, "3   NECK", size=6.3, weight="bold", color=DIM)
    y = draw_table(ax, 12, y - 8, [28, 32, 32, 54], ["ITEM", "Z0", "Z1", "SIZE"],
                   [["CYLINDER", f5(neck["z0"]), f5(neck["z1"]), f"Ø {f5(diam(neck['r0']))}  L {f5(neck['length'])}"]],
                   row_h=3.8, size=4.9)

    T(ax, 12, y - 5, "4   AXIS IN THE STEP FILE", size=6.3, weight="bold", color=DIM)
    y = draw_table(ax, 12, y - 8, [36, 36, 36, 38], ["ITEM", "X", "Y", "Z"], [
        ["DIRECTION", f6(direc[0]), f6(direc[1]), f6(direc[2])],
        ["POINT ON AXIS", f5(origin[0]), f5(origin[1]), f5(origin[2])],
    ], row_h=3.7, size=4.9)
    T(ax, 12, y - 4, "The point is the head-end centre. World +Z runs from the head toward the tip.", size=4.7, color="#444")

    right = []
    for arc in model["arcs"]:
        right.append([f"ARC R{f3(arc['radius'])}", f5(arc["z0"]), f5(arc["z1"]),
                      f"CTR ({f5(arc['center'][0])}, {f5(arc['center'][1])})"])
    right.append(["NECK", f5(neck["z0"]), f5(neck["z1"]), f"Ø {f5(diam(neck['r0']))}"])
    for s in segs_between(model["segs"], neck["z1"], model["form_b"][0]["z0"]):
        if s["kind"] == "TAPER":
            right.append([f"TAP {s['ang']:.2f}°", f5(s["z0"]), f5(s["z1"]), f"Ø {f5(diam(s['r0']))} → {f5(diam(s['r1']))}"])
        elif s["kind"] == "CYL":
            right.append(["CYL", f5(s["z0"]), f5(s["z1"]), f"Ø {f5(diam(s['r0']))}  L {f5(s['length'])}"])
    right.append(["B CREST 1", f5(model["form_b"][0]["z0"]), f5(model["form_b"][0]["z1"]), "Ø 7.80000  FLAT 0.20000"])
    right.append(["B CREST 6", f5(model["form_b"][-1]["z0"]), f5(model["form_b"][-1]["z1"]), "PITCH 1.75000"])
    for s in segs_between(model["segs"], model["form_b"][-1]["z1"], model["form_a"][0]["z0"]):
        if s["kind"] == "TAPER":
            right.append([f"TAP {s['ang']:.2f}°", f5(s["z0"]), f5(s["z1"]), f"Ø {f5(diam(s['r0']))} → {f5(diam(s['r1']))}"])
        elif s["kind"] == "CYL":
            right.append(["CYL", f5(s["z0"]), f5(s["z1"]), f"Ø {f5(diam(s['r0']))}  L {f5(s['length'])}"])
    right.append(["A CREST 1", f5(model["form_a"][0]["z0"]), f5(model["form_a"][0]["z1"]), "Ø 7.60000  FLAT 0.15000"])
    right.append(["A CREST 20", f5(model["form_a"][-1]["z0"]), f5(model["form_a"][-1]["z1"]), "PITCH 1.20000"])
    chamfer = find_ang(model["segs"], 31.2)
    right.append([f"TIP {chamfer['ang']:.2f}°", f5(chamfer["z0"]), f5(chamfer["z1"]),
                  f"Ø {f5(diam(chamfer['r0']))} → {f5(diam(chamfer['r1']))}"])
    right.append(["TIP FACE", f5(model["tip_z"]), f5(model["tip_z"]), f"Ø {f5(diam(tip_r))} TO THE AXIS"])
    for g in sorted(model["root_groups"], key=lambda g: g[0]["z0"]):
        right.append([f"ROOT ×{len(g)}", f5(g[0]["z0"]), f5(g[-1]["z1"]),
                      f"FLAT {f5(g[0]['length'])}  Ø {f5(diam(g[0]['r0']))}"])

    T(ax, 168, 266, "5   EVERY CONTROLLING FEATURE", size=6.3, weight="bold", color=DIM)
    y_right = draw_table(ax, 168, 263, [32, 32, 32, 84],
                         ["ITEM", "Z0", "Z1", "SIZE"], right, row_h=3.15, size=4.35)
    T(ax, 12, 18, "SHEETS 6 AND 7 LIST ALL 126 PROFILE SEGMENTS, INCLUDING EVERY FLANK.", size=5.1, color="#444")
    if y_right < 50:
        raise RuntimeError(f"Schedule table collides with the title block at y={y_right:.1f}")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def segs_between(segs, z_lo, z_hi):
    return [s for s in segs if s["z1"] > z_lo + 1e-6 and s["z0"] < z_hi - 1e-6]


def sheet_segments(pdf, model, page, rows, first, last, continued):
    fig, ax = new_page()
    title = "SHEET 6   EVERY PROFILE SEGMENT" if not continued else "SHEET 7   PROFILE SEGMENTS, CONTINUED"
    frame(ax, page, title)
    T(ax, 12, 274,
      "ORDERED +R PROFILE, HEAD FACE TO TIP. REVOLVE ABOUT DATUM B. MILLIMETRES TO 0.00001. ANGLE IS FROM THE AXIS.",
      size=5.7, weight="medium", color=DIM)
    headers = ["#", "KIND", "Z1", "R1", "Z2", "R2", "LENGTH", "ANGLE", "ARC"]
    widths = [12, 16, 26, 24, 26, 24, 24, 16, 24]
    per_col = 48
    left, right = rows[:per_col], rows[per_col:]
    draw_table(ax, 12, 266, widths, headers, left, row_h=3.35, size=4.35)
    if right:
        draw_table(ax, 214, 266, widths, headers, right, row_h=3.35, size=4.35)
    T(ax, 12, 72, f"SEGMENTS {first}–{last} OF 126", size=5.2, color="#333")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def build():
    global TOTAL
    print("Loading", STEP_PATH)
    solid = cq.importers.importStep(str(STEP_PATH)).val()
    shape = solid.wrapped
    print("Sectioning")
    segs, axis = extract_profile(shape)
    model = analyze(segs, shape, axis)
    print(
        f"segments {len(segs)}  overall {model['overall']:.5f}  "
        f"volume {model['mass']['volume']:.3f}  crests A {len(model['form_a'])} B {len(model['form_b'])}"
    )
    print("Pictorial")
    iso = render_iso(solid)
    rows = []
    for seg in segs:
        if seg["kind"] == "ARC":
            note = f"R{f5(seg['radius'])}"
            ang = ""
        else:
            note = ""
            ang = f"{seg['ang']:.2f}"
        rows.append([
            f"{seg['n']:03d}", seg["kind"], f5(seg["z0"]), f5(seg["r0"]),
            f5(seg["z1"]), f5(seg["r1"]), f5(seg["length"]), ang, note,
        ])
    per_page = 96
    chunks = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    TOTAL = 5 + len(chunks)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "2 MEDIUM NUT — all nominal dimensions"
        info["Subject"] = "Dimensioned drawing extracted from 2 MEDIUM NUT.STEP"
        sheet_general(pdf, model, iso)
        sheet_head(pdf, model)
        sheet_form_b(pdf, model)
        sheet_form_a(pdf, model)
        sheet_schedule(pdf, model)
        for i, chunk in enumerate(chunks):
            sheet_segments(pdf, model, 6 + i, chunk, chunk[0][0], chunk[-1][0], continued=i > 0)
    print("Wrote", OUT_PATH, "sheets", TOTAL)


if __name__ == "__main__":
    build()
