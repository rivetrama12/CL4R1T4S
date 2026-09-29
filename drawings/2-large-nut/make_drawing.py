#!/usr/bin/env python3
"""Nominal dimension drawing of the turned part "2 large nut".

The STEP file stores geometry only: no PMI, tolerances, material, or finish.
The solid is a surface of revolution. Sizes are measured in the part frame:
Z along the axis, origin on the head end face (datum A), positive Z toward
the small end. R is the perpendicular distance from the axis.

There is no annular shoulder. Datum A is the head end face, the radial face
of greatest radius.

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
STEP_PATH = ROOT / "2_large_nut.STEP"
OUT_PATH = ROOT / "2_large_nut_all_dimensions.pdf"

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

TOTAL = 8


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


def f6(v: float) -> str:
    if abs(v) < 5e-7:
        v = 0.0
    return f"{v:.6f}"


def diam(r: float) -> float:
    return 2.0 * r


def ang_txt(seg):
    angle = seg["ang"]
    hundredths = round(angle, 2)
    if abs(angle - hundredths) < 0.001:
        return f"{hundredths:.2f}°"
    return f"{angle:.3f}°"


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

    # Datum A is the radial face of greatest radius. On this part that face
    # is the head end, and it meets the axis.
    faces = [s for s in segs if s["kind"] == "FACE"]
    if not faces:
        raise RuntimeError("No radial face found")
    datum = max(faces, key=lambda s: max(s["r0"], s["r1"]))
    z_datum = 0.5 * (datum["z0"] + datum["z1"])

    flipped = []
    for seg in reversed(segs):
        def fz(z, z0=z_datum):
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
        "z_shoulder_profile": z_datum,
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


def _same_pitch(crests, pitch, tol=2e-4):
    gaps = [crests[i + 1]["z0"] - crests[i]["z0"] for i in range(len(crests) - 1)]
    if any(abs(gap - pitch) > tol for gap in gaps):
        raise RuntimeError(f"Pitch {pitch} does not hold: {gaps[:3]}")
    return gaps[0] if gaps else pitch


def analyze(segs, shape, axis):
    if segs[0]["r0"] > 1e-4 or segs[-1]["r1"] > 1e-4:
        raise RuntimeError("Profile does not run from axis to axis")
    if any(seg["z1"] + 1e-6 < seg["z0"] for seg in segs):
        raise RuntimeError("Profile Z is not monotonic")
    if len(segs) != 168:
        raise RuntimeError(f"Expected 168 segments, found {len(segs)}")

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

    arcs = [s for s in segs if s["kind"] == "ARC"]
    if len(arcs) != 4:
        raise RuntimeError(f"Expected 4 axial arcs, found {len(arcs)}")

    faces = [s for s in segs if s["kind"] == "FACE"]
    head_face = min(faces, key=lambda s: s["z0"])
    tip_face = max(faces, key=lambda s: s["z0"])
    head_cyl = _cyls(segs, 14.3, r_tol=1e-4)
    neck = _cyls(segs, 7.65, r_tol=1e-3)
    if len(head_cyl) != 1 or len(neck) != 1:
        raise RuntimeError("Head land or neck cylinder was not found")
    head_taper = next(s for s in segs if s["kind"] == "TAPER" and abs(s["ang"] - 42.7) < 0.02)

    # Form B is the head-side 53° thread. Two crest flats, pointed root.
    b_short = _cyls(segs, 7.45, 0.31673542, r_tol=1e-3, l_tol=2e-4)
    b_long = _cyls(segs, 7.45, 0.32925678, r_tol=1e-3, l_tol=2e-4)
    if len(b_short) != 9 or len(b_long) != 8:
        raise RuntimeError(f"Form B crests {len(b_short)} / {len(b_long)}")
    pitch_b1 = _same_pitch(b_short, b_short[1]["z0"] - b_short[0]["z0"])
    pitch_b2 = _same_pitch(b_long, b_long[1]["z0"] - b_long[0]["z0"])
    junction_b = b_long[0]["z0"] - b_short[-1]["z0"]
    if abs(junction_b - pitch_b1) > 2e-4:
        raise RuntimeError(f"Form B group junction pitch {junction_b}")

    # Form A is the tip-side thread. Two crest flats, root flat Ø13.
    a_long = _cyls(segs, 7.2, 0.50458910, r_tol=1e-3, l_tol=2e-4)
    a_short = _cyls(segs, 7.2, 0.48749385, r_tol=1e-3, l_tol=2e-4)
    if len(a_long) != 15 or len(a_short) != 8:
        raise RuntimeError(f"Form A crests {len(a_long)} / {len(a_short)}")
    pitch_a1 = _same_pitch(a_long, a_long[1]["z0"] - a_long[0]["z0"])
    pitch_a2 = _same_pitch(a_short, a_short[1]["z0"] - a_short[0]["z0"])
    junction_a = a_short[0]["z0"] - a_long[-1]["z0"]
    if abs(junction_a - pitch_a1) > 2e-4:
        raise RuntimeError(f"Form A group junction pitch {junction_a}")
    a_roots = _cyls(segs, 6.5, r_tol=1e-3)
    if len(a_roots) != 24 or any(abs(s["length"] - a_roots[0]["length"]) > 5e-4 for s in a_roots):
        raise RuntimeError("Form A root flats are not one repeated length")

    run_in = next(s for s in segs if s["kind"] == "CYL" and abs(s["r0"] - 7.45) < 1e-3 and s["z0"] < b_short[0]["z0"])
    exit_land = next(s for s in segs if s["kind"] == "CYL" and abs(s["r0"] - 7.45) < 1e-3 and s["z0"] > b_long[-1]["z0"])
    b_root_r = min(s["r0"] for s in segs if s["kind"] == "TAPER" and abs(s["ang"] - 53.0) < 0.02 and s["z0"] > run_in["z1"])
    groove_r, groove_z = min((p for arc in arcs for p in arc["pts"]), key=lambda p: p[0])

    overall = tip_face["z0"] - head_face["z0"]
    if abs(overall - 100.62043915) > 1e-4 or abs(head_face["z0"]) > 1e-4:
        raise RuntimeError(f"Unexpected envelope head {head_face['z0']} overall {overall}")
    if com_r > 1e-3:
        raise RuntimeError("Centre of mass is off the axis")

    return {
        "segs": segs,
        "axis": axis,
        "head_z": head_face["z0"],
        "tip_z": tip_face["z0"],
        "overall": overall,
        "head_face": head_face,
        "tip_face": tip_face,
        "head_cyl": head_cyl[0],
        "head_taper": head_taper,
        "neck": neck[0],
        "arcs": arcs,
        "form_b_short": b_short,
        "form_b_long": b_long,
        "form_a_long": a_long,
        "form_a_short": a_short,
        "form_a_root": a_roots[0],
        "form_a_root_n": len(a_roots),
        "form_b_root_r": b_root_r,
        "pitch_b1": pitch_b1,
        "pitch_b2": pitch_b2,
        "pitch_a1": pitch_a1,
        "pitch_a2": pitch_a2,
        "run_in": run_in,
        "exit_land": exit_land,
        "groove": (groove_r, groove_z),
        "mass": {
            "volume": volume_props.Mass(),
            "area": area_props.Mass(),
            "com_z": com_z,
            "com_r": com_r,
        },
    }


def render_iso(solid, axis):
    """Plot so drawing +Z runs to the right and the head end is at the left."""
    ox, oy, oz = axis["origin"]
    ux, uy, uz = axis["direction"]
    verts, tris = solid.tessellate(0.35)
    xyz = []
    for v in verts:
        dx, dy, dz = v.x - ox, v.y - oy, v.z - oz
        along = dx * ux + dy * uy + dz * uz
        drawing_z = axis["z_shoulder_profile"] - along
        rx = dx - along * ux
        ry = dy - along * uy
        rz = dz - along * uz
        xyz.append((drawing_z, ry, rz))
    xyz = np.array(xyz, dtype=float)
    faces = xyz[np.asarray(tris)]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.array([0.58, 0.66, 0.74])
    rgb = np.clip(0.18 + 0.82 * shade[:, None] * base, 0, 1)
    rgba = np.concatenate([rgb, np.ones((len(faces), 1))], axis=1)

    fig = plt.figure(figsize=(7.2, 2.6), dpi=130)
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
    T(ax, 12, 286.2, "2 LARGE NUT", size=10.5, weight="bold", va="top")
    T(ax, 58, 286.2, "TURNED SOLID  ·  NOMINAL DIMENSION DRAWING", size=8.0, weight="medium", color=DIM, va="top")
    T(ax, 406, 286.2, view_title, size=7.2, weight="medium", ha="right", va="top", color="#333")
    x, y, w, h = 248, 8, 164, 40
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=RULE, lw=0.7, zorder=4))
    ax.plot([x, x + w], [y + 28, y + 28], color=RULE, lw=0.4, zorder=4)
    ax.plot([x, x + w], [y + 16, y + 16], color=RULE, lw=0.4, zorder=4)
    ax.plot([x + 82, x + 82], [y, y + h], color=RULE, lw=0.4, zorder=4)
    T(ax, x + 3, y + 34, "PART", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 34, "2 LARGE NUT", size=7.2, weight="bold", va="center", z=7)
    T(ax, x + 85, y + 34, "SHEET", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 34, f"{page}  /  {TOTAL}", size=8, weight="bold", va="center", z=7)
    T(ax, x + 3, y + 22, "FILE", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 22, "2 large nut.STEP", size=6.4, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 22, "UNITS", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 22, "MILLIMETRES", size=7.2, weight="medium", va="center", z=7)
    T(ax, x + 3, y + 8, "DATUM A  HEAD FACE Z = 0", size=5.0, weight="medium", va="center", z=7)
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


def find_ang(segs, angle, tol=0.05):
    return next(s for s in segs if s["ang"] is not None and abs(s["ang"] - angle) < tol)


def break_mark(ax, view, z, r):
    bx, by = view.pt(z, r)
    for sign in (1, -1):
        yy = view.ry_of(sign * r)
        ax.plot([bx, bx + 1.3, bx - 0.45, bx + 1.3], [yy, yy + 1.1 * sign, yy + 2.2 * sign, yy + 3.3 * sign],
                color=GEO, lw=0.5, zorder=3)


def datum_a(ax, view, z, r_arrow):
    face_x = view.zx_of(z)
    y_tip = view.ry_of(-r_arrow)
    ax.plot([face_x, face_x], [y_tip, y_tip - 3.2], color=DIM, lw=0.4, zorder=3)
    ax.add_patch(Polygon(
        [(face_x, y_tip), (face_x - 1.6, y_tip - 2.8), (face_x + 1.6, y_tip - 2.8)],
        closed=True, fc=DIM, ec=DIM, lw=0.2, zorder=4,
    ))
    ax.add_patch(Rectangle((face_x - 3.2, y_tip - 7.6), 6.4, 4.5, fc="white", ec=DIM, lw=0.55, zorder=4))
    T(ax, face_x, y_tip - 5.35, "A", size=5.8, weight="bold", color=DIM, ha="center", va="center", z=5)


def sheet_general(pdf, model, iso):
    fig, ax = new_page()
    frame(ax, 1, "SHEET 1   GENERAL ARRANGEMENT")
    segs = model["segs"]
    scale = 2.55
    view = View(zx=36, y0=158, scale=scale)
    draw_profile(ax, view, segs, both=True, lw=0.45)
    centerline(ax, view, -2.2, model["tip_z"] + 2.2)
    T(ax, view.zx_of(-2.0), view.y0 + 2.0, "B", size=6.2, weight="bold", color=CENTER, ha="right")
    datum_a(ax, view, 0, 13.2)

    hdim(ax, view.zx_of(0), view.zx_of(model["tip_z"]), 96,
         f"{f3(model['overall'])} OVERALL", view.ry_of(-14.3), view.ry_of(0))

    head_r = 14.3
    leader(ax, *view.pt(0.6, head_r), view.zx_of(-1.2), view.ry_of(head_r) + 10, "Ø 28.600", ha="right")
    neck = model["neck"]
    leader(ax, *view.pt(0.5 * (neck["z0"] + neck["z1"]), neck["r0"]),
           view.zx_of(neck["z0"]), view.ry_of(neck["r0"]) + 12, "Ø 15.300", ha="left")
    crest_b = model["form_b_short"][4]
    leader(ax, *view.pt(0.5 * (crest_b["z0"] + crest_b["z1"]), crest_b["r0"]),
           view.zx_of(crest_b["z0"]), view.ry_of(10.2), "Ø 14.900 CREST", ha="left")
    crest_a = model["form_a_long"][7]
    leader(ax, *view.pt(0.5 * (crest_a["z0"] + crest_a["z1"]), crest_a["r0"]),
           view.zx_of(crest_a["z1"]) + 4, view.ry_of(10.4), "Ø 14.400 CREST", ha="left")
    tip_r = max(model["tip_face"]["r0"], model["tip_face"]["r1"])
    leader(ax, *view.pt(model["tip_z"], tip_r), view.zx_of(model["tip_z"]) + 1.2, view.ry_of(tip_r) + 11,
           f"Ø {f3(diam(tip_r))} TIP", ha="left")

    zones = [
        (model["form_b_short"][0]["z0"], model["form_b_long"][-1]["z1"], "FORM B"),
        (model["form_a_long"][0]["z0"], model["form_a_short"][-1]["z1"], "FORM A"),
    ]
    for z0, z1, name in zones:
        y = view.ry_of(head_r) + 4.2
        ax.annotate("", xy=(view.zx_of(z1), y), xytext=(view.zx_of(z0), y),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0))
        T(ax, (view.zx_of(z0) + view.zx_of(z1)) / 2, y + 1.7, name, size=5.2, weight="bold",
          color=DIM, ha="center", va="bottom")

    balloon(ax, view.zx_of(4), view.ry_of(head_r) + 22, "2")
    balloon(ax, view.zx_of(27), view.ry_of(head_r) + 22, "3")
    balloon(ax, view.zx_of(46), view.ry_of(head_r) + 22, "4")
    balloon(ax, view.zx_of(74), view.ry_of(head_r) + 22, "5")
    T(ax, 14, 78, "SCALE 2.55 : 1     AXIS HORIZONTAL     HEAD AT LEFT     FULL PROFILE", size=5.6, color="#333")

    ih, iw = iso.shape[0], iso.shape[1]
    box_w, box_h = 92.0, 52.0
    aspect = iw / ih
    if box_w / box_h > aspect:
        dh, dw = box_h, box_h * aspect
    else:
        dw, dh = box_w, box_w / aspect
    x0, y0 = 404 - dw, 268 - dh
    ax.imshow(iso, extent=(x0, x0 + dw, y0, y0 + dh), aspect="equal", zorder=1, interpolation="bilinear")
    T(ax, x0, y0 - 3.0, "PICTORIAL — NOT TO SCALE     HEAD AT LEFT", size=5.0, color="#444")

    mass = model["mass"]
    notes = [
        "Solid of revolution about datum B. Revolve the +R profile.",
        "Sizes are CAD nominals from 2 large nut.STEP (SolidWorks 2018, 2024-10-24).",
        "No tolerance, material, or finish is stored in the model.",
        "Graphic sizes are rounded to 0.001 mm. Sheets 7 and 8 list every segment to 0.00001 mm.",
        "Datum A is the head end face. Positive Z runs from that face toward the small end.",
        f"Volume {mass['volume']:.3f} mm³     area {mass['area']:.3f} mm²     centre of mass Z {f3(mass['com_z'])} on the axis.",
        "Balloons 2, 3, 4 and 5 mark the detail sheets.",
        "Form B: 17 crests, Ø 14.900, 53° both flanks.    Form A: 23 crests, Ø 14.400, root Ø 13.000.",
    ]
    yy = 68
    for line in notes:
        T(ax, 14, yy, line, size=5.2, color="#333", va="center")
        yy -= 5.0
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_head(pdf, model):
    fig, ax = new_page()
    frame(ax, 2, "SHEET 2   HEAD, NECK AND ENTRY")
    segs = model["segs"]
    scale = 6.0
    view = View(zx=78, y0=162, scale=scale)
    end_z = model["run_in"]["z1"]
    draw_profile(ax, view, segs, both=True, lw=0.85, z0=-0.05, z1=end_z)
    centerline(ax, view, -0.6, end_z + 0.35)
    T(ax, view.zx_of(-0.5), view.y0 + 2.2, "B", size=6.2, weight="bold", color=CENTER, ha="right")
    break_mark(ax, view, end_z, 7.45)
    T(ax, view.zx_of(end_z) + 3.2, view.ry_of(7.45) + 8, "SHEET 3", size=5.2, color=DIM)
    datum_a(ax, view, 0, 12.6)
    T(ax, view.zx_of(0) + 8, view.ry_of(13.4), "DATUM A", size=5.6, weight="bold", color=DIM, va="center")

    head = model["head_cyl"]
    neck = model["neck"]
    taper = model["head_taper"]
    vdim(ax, view.zx_of(-0.15) - 12, view.ry_of(-14.3), view.ry_of(14.3),
         "Ø 28.600", view.zx_of(0.4), view.zx_of(0.4), text_side="left")
    vdim(ax, view.zx_of(10.2), view.ry_of(-neck["r0"]), view.ry_of(neck["r0"]),
         "Ø 15.300", view.zx_of(10.2), view.zx_of(10.2))
    hdim(ax, view.zx_of(head["z0"]), view.zx_of(head["z1"]), 58,
         f"{f3(head['length'])} LAND", view.ry_of(-14.3), view.ry_of(-14.3))
    hdim(ax, view.zx_of(neck["z0"]), view.zx_of(neck["z1"]), 46,
         f"{f3(neck['length'])} NECK", view.ry_of(-neck["r0"]), view.ry_of(-neck["r0"]))
    pr, pz = midpt(taper)
    leader(ax, *view.pt(pz, pr), view.zx_of(pz) - 2, view.ry_of(pr) + 10, ang_txt(taper), ha="right")

    rows = [
        ["HEAD FACE", f5(model["head_z"]), "0.00000", "0.000", "ON THE AXIS"],
        ["HEAD RIM", f5(model["head_z"]), "14.30000", "28.60000", "LARGEST DIAMETER"],
        ["LAND END", f5(head["z1"]), "14.30000", "28.60000", "42.70° STARTS"],
        ["NECK START", f5(neck["z0"]), f5(neck["r0"]), f5(diam(neck["r0"])), "TRUE CYLINDER"],
        ["NECK END", f5(neck["z1"]), f5(neck["r1"]), f5(diam(neck["r1"])), "ENTRY TAPERS"],
        ["RUN-IN", f5(model["run_in"]["z0"]), "7.45000", "14.90000", f"L {f5(model['run_in']['length'])}"],
    ]
    draw_table(ax, 214, 266, [32, 28, 28, 28, 48],
               ["POINT", "Z", "RADIUS", "Ø", "NOTE"], rows, row_h=4.2, size=5.0)
    entry = [s for s in segs if s["z0"] >= neck["z1"] - 1e-6 and s["z1"] <= model["run_in"]["z1"] + 1e-6]
    entry_rows = []
    for seg in entry:
        if seg["kind"] == "TAPER":
            entry_rows.append([ang_txt(seg), f5(seg["z0"]), f5(seg["z1"]), f5(diam(seg["r0"])), f5(diam(seg["r1"]))])
        else:
            entry_rows.append([f"CYL L {f5(seg['length'])}", f5(seg["z0"]), f5(seg["z1"]),
                               f5(diam(seg["r0"])), f5(diam(seg["r1"]))])
    T(ax, 214, 228, "NECK END TO FORM B RUN-IN", size=6.0, weight="bold", color=DIM)
    draw_table(ax, 214, 225, [32, 28, 28, 28, 28],
               ["FEATURE", "Z0", "Z1", "Ø START", "Ø END"], entry_rows, row_h=4.0, size=4.7)
    notes = [
        "The head end face is datum A. It meets the axis.",
        "Ø 28.600 is a true cylinder, then a 42.70° cone.",
        "The neck is Ø 15.300. The entry tapers are tabulated.",
        "The 53° form continues on sheet 3.",
    ]
    yy = 168
    for line in notes:
        T(ax, 214, yy, line, size=5.2, color="#333")
        yy -= 5.6
    T(ax, 14, 274, "DETAIL 2     SCALE 6 : 1     FULL PROFILE", size=6.4, weight="bold", color=DIM)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def _crest_rows(crests):
    rows = []
    prev = None
    for i, crest in enumerate(crests, start=1):
        pitch = "—" if prev is None else f5(crest["z0"] - prev)
        rows.append([str(i), f5(crest["z0"]), f5(crest["z1"]), f5(crest["length"]), pitch])
        prev = crest["z0"]
    return rows


def sheet_form_b(pdf, model):
    fig, ax = new_page()
    frame(ax, 3, "SHEET 3   FORM B")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 3     ONE REPEATED PITCH     POINTED ROOT     ANGLES FROM THE AXIS",
      size=6.2, weight="bold", color=DIM)

    c0, c1 = model["form_b_short"][1], model["form_b_short"][2]
    tooth = View(zx=24, y0=198, scale=36, z0=c0["z0"] - 0.04, r0=6.35)
    draw_profile(ax, tooth, segs, both=False, lw=1.05, z0=c0["z0"], z1=c1["z0"])
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c1["z0"]), tooth.ry_of(8.05),
         f"PITCH {f3(model['pitch_b1'])}", tooth.ry_of(7.45), tooth.ry_of(7.45))
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c0["z1"]), tooth.ry_of(7.62),
         f"FLAT {f3(c0['length'])}", tooth.ry_of(7.45), tooth.ry_of(7.45))
    fall = next(s for s in segs if abs(s["z0"] - c0["z1"]) < 1e-6)
    rise = next(s for s in segs if abs(s["z1"] - c1["z0"]) < 1e-6)
    fx, fy = tooth.pt(*midpt(fall)[::-1])
    rx, ry = tooth.pt(*midpt(rise)[::-1])
    leader(ax, fx, fy, fx - 10, fy - 8, ang_txt(fall), ha="right")
    leader(ax, rx, ry, rx + 8, ry - 8, ang_txt(rise), ha="left")
    leader(ax, *tooth.pt(fall["z1"], fall["r1"]), tooth.zx_of(fall["z1"]) - 0.15, tooth.ry_of(6.15),
           f"Ø {f3(diam(model['form_b_root_r']))}", ha="right")
    leader(ax, *tooth.pt(c0["z0"] + 0.08, 7.45), tooth.zx_of(c0["z0"]) - 0.35, tooth.ry_of(7.45),
           "Ø 14.900", ha="right")
    T(ax, 16, 176, "FORM B TOOTH   SCALE 36 : 1   MATERIAL BELOW THE LINE   17 CRESTS   BOTH FLANKS 53.00°",
      size=5.1, color="#333")

    crests = model["form_b_short"] + model["form_b_long"]
    y = draw_table(ax, 188, 266, [12, 32, 32, 28, 32],
                   ["#", "Z0", "Z1", "FLAT", "PITCH"], _crest_rows(crests), row_h=3.45, size=4.6)
    spec = [
        ["CREST Ø", "14.90000", "TRUE CYLINDER"],
        ["ROOT", f5(diam(model["form_b_root_r"])), "POINTED, NO FLAT"],
        ["FLANKS", "53.00° / 53.00°", "LEAVING / APPROACHING, +Z"],
        ["CRESTS 1–9", f5(model["pitch_b1"]), f"FLAT {f5(model['form_b_short'][0]['length'])}  ×9"],
        ["CRESTS 10–17", f5(model["pitch_b2"]), f"FLAT {f5(model['form_b_long'][0]['length'])}  ×8"],
        ["RUN-IN", f5(model["run_in"]["length"]), f"Z {f5(model['run_in']['z0'])}  Ø 14.900"],
        ["EXIT LAND", f5(model["exit_land"]["length"]), f"Z {f5(model['exit_land']['z0'])}  Ø 14.900"],
    ]
    T(ax, 188, y - 4, "FORM B DEFINITION", size=6.0, weight="bold", color=DIM)
    y = draw_table(ax, 188, y - 7, [32, 36, 78], ["ITEM", "VALUE", "NOTE"], spec, row_h=3.7, size=4.7)
    T(ax, 188, y - 4, "Pitch is crest start to crest start. The interval from crest 9 to crest 10 is 1.40239.",
      size=4.7, color="#333")
    if y < 56:
        raise RuntimeError(f"Form B tables collide with the title block at y={y:.1f}")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_groove(pdf, model):
    fig, ax = new_page()
    frame(ax, 4, "SHEET 4   GROOVE BETWEEN THE FORMS")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 4     TRUE RADIAL SCALE     HALF PROFILE     ANGLES FROM THE AXIS",
      size=6.2, weight="bold", color=DIM)
    z0, z1 = 42.4, 51.7
    view = View(zx=28, y0=118, scale=9.2, z0=z0, r0=0.0)
    draw_profile(ax, view, segs, both=False, lw=0.95, z0=z0, z1=z1)
    centerline(ax, view, z0 - 0.15, z1 + 0.2)
    T(ax, view.zx_of(z0 - 0.12), view.y0 + 2.0, "B", size=6.0, weight="bold", color=CENTER, ha="right")

    groove_r, groove_z = model["groove"]
    by_r = {round(arc["radius"], 3): arc for arc in model["arcs"]}
    r11, r09, r05 = by_r[11.0], by_r[0.9], by_r[0.5]
    steep = find_ang(segs, 57.65)
    shallow = find_ang(segs, 3.2)
    px, py = view.pt(*midpt(r11)[::-1])
    leader(ax, px, py, px - 6, py + 16, "R 11.000", ha="right")
    gx, gy = view.pt(groove_z, groove_r)
    leader(ax, gx, gy, gx - 8, gy - 12, f"R 0.900    Ø {f3(diam(groove_r))}", ha="right")
    sx, sy = view.pt(*midpt(steep)[::-1])
    leader(ax, sx, sy, sx + 8, sy - 10, ang_txt(steep), ha="left")
    fx, fy = view.pt(*midpt(r05)[::-1])
    leader(ax, fx, fy, fx + 4, fy + 14, "R 0.500", ha="left")
    qx, qy = view.pt(*midpt(shallow)[::-1])
    leader(ax, qx, qy, qx + 2, qy + 12, ang_txt(shallow), ha="left")
    T(ax, 16, 78, "GROOVE   SCALE 9.2 : 1   HALF PROFILE   EVERY STATION IS IN THE TABLE", size=5.2, color="#333")

    window = [s for s in segs if s["z1"] > model["exit_land"]["z0"] + 1e-6 and s["z0"] < model["form_a_long"][0]["z0"] - 1e-9]
    rows = []
    for seg in window:
        if seg["kind"] == "ARC":
            kind = f"R {f5(seg['radius'])}"
            size = f"CTR ({f5(seg['center'][0])}, {f5(seg['center'][1])})"
        elif seg["kind"] == "CYL":
            kind = "CYL"
            size = f"Ø {f5(diam(seg['r0']))}  L {f5(seg['length'])}"
        else:
            kind = ang_txt(seg)
            size = f"Ø {f5(diam(seg['r0']))} → {f5(diam(seg['r1']))}"
        rows.append([kind, f5(seg["z0"]), f5(seg["z1"]), f5(seg["r0"]), f5(seg["r1"]), size])
    T(ax, 168, 266, "EXIT LAND THROUGH THE FIRST FORM A ROOT", size=6.0, weight="bold", color=DIM)
    y = draw_table(ax, 148, 263, [22, 26, 26, 24, 24, 62],
                   ["KIND", "Z0", "Z1", "R0", "R1", "SIZE"], rows, row_h=4.15, size=4.5)
    if y < 52:
        raise RuntimeError(f"Groove table collides with the title block at y={y:.1f}")
    T(ax, 148, y - 6, "The two R 0.900 arcs meet at the groove bottom. R 11.000 leaves the Ø 14.900 land.",
      size=4.8, color="#333")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_a(pdf, model):
    fig, ax = new_page()
    frame(ax, 5, "SHEET 5   FORM A AND TIP")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 5     ONE REPEATED PITCH     TIP AT TRUE RADIAL SCALE",
      size=6.2, weight="bold", color=DIM)

    c0, c1 = model["form_a_long"][1], model["form_a_long"][2]
    tooth = View(zx=18, y0=206, scale=26, z0=c0["z0"] - 0.04, r0=6.15)
    draw_profile(ax, tooth, segs, both=False, lw=1.05, z0=c0["z0"], z1=c1["z0"])
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c1["z0"]), tooth.ry_of(8.15),
         f"PITCH {f3(model['pitch_a1'])}", tooth.ry_of(7.2), tooth.ry_of(7.2))
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c0["z1"]), tooth.ry_of(7.55),
         f"FLAT {f3(c0['length'])}", tooth.ry_of(7.2), tooth.ry_of(7.2))
    fall = next(s for s in segs if abs(s["z0"] - c0["z1"]) < 1e-6)
    rise = next(s for s in segs if abs(s["z1"] - c1["z0"]) < 1e-6)
    root = next(s for s in segs if s["kind"] == "CYL" and abs(s["r0"] - 6.5) < 1e-3 and c0["z1"] < s["z0"] < c1["z0"])
    fx, fy = tooth.pt(*midpt(fall)[::-1])
    rx, ry = tooth.pt(*midpt(rise)[::-1])
    leader(ax, fx, fy, fx - 4, fy - 10, ang_txt(fall), ha="right")
    leader(ax, rx, ry, rx + 4, ry - 10, ang_txt(rise), ha="left")
    hdim(ax, tooth.zx_of(root["z0"]), tooth.zx_of(root["z1"]), tooth.ry_of(5.85),
         f"ROOT {f5(root['length'])}", tooth.ry_of(6.5), tooth.ry_of(6.5))
    T(ax, 14, 188, "FORM A TOOTH   SCALE 26 : 1   Ø 14.400 CREST   Ø 13.000 ROOT   23 CRESTS",
      size=5.0, color="#333")

    tip = View(zx=16, y0=72, scale=10, z0=95.6, r0=0.0)
    draw_profile(ax, tip, segs, both=False, lw=0.9, z0=95.8, z1=model["tip_z"] + 0.02)
    centerline(ax, tip, 95.5, model["tip_z"] + 0.25)
    partial = next(s for s in segs if s["kind"] == "TAPER" and abs(s["ang"] - 59.7) < 0.05 and max(s["r0"], s["r1"]) < 7.0)
    chamfer = find_ang(segs, 29.8118)
    leader(ax, *tip.pt(*midpt(partial)[::-1]), tip.zx_of(partial["z0"]) - 0.05, tip.ry_of(4.2), ang_txt(partial), ha="right")
    leader(ax, *tip.pt(*midpt(chamfer)[::-1]), tip.zx_of(99.2), tip.ry_of(2.4), ang_txt(chamfer), ha="right")
    tip_r = max(model["tip_face"]["r0"], model["tip_face"]["r1"])
    leader(ax, *tip.pt(model["tip_z"], tip_r * 0.55), tip.zx_of(model["tip_z"]) + 0.15, tip.ry_of(4.6),
           f"TIP Ø {f3(diam(tip_r))}", ha="left")
    T(ax, 14, 58, "TIP   SCALE 10 : 1   HALF PROFILE   LAST ROOT, THEN 59.70° STOPS SHORT OF THE CREST",
      size=5.0, color="#333")

    crests = model["form_a_long"] + model["form_a_short"]
    rows = _crest_rows(crests)
    left, right = rows[:12], rows[12:]
    while len(right) < len(left):
        right.append(["", "", "", "", ""])
    paired = [a + b for a, b in zip(left, right)]
    draw_table(ax, 150, 266, [8, 24, 24, 20, 22, 8, 24, 24, 20, 22],
               ["#", "Z0", "Z1", "FLAT", "PITCH", "#", "Z0", "Z1", "FLAT", "PITCH"],
               paired, row_h=3.35, size=4.15)
    root = model["form_a_root"]
    notes = [
        ["CREST Ø", "14.40000", "TWO FLAT LENGTHS"],
        ["ROOT Ø", "13.00000", f"FLAT {f5(root['length'])}  ×{model['form_a_root_n']}"],
        ["FLANKS", f"{fall['ang']:.2f}° / {rise['ang']:.2f}°", "LEAVING / APPROACHING, +Z"],
        ["CRESTS 1–15", f5(model["pitch_a1"]), f"FLAT {f5(model['form_a_long'][0]['length'])}"],
        ["CRESTS 16–23", f5(model["pitch_a2"]), f"FLAT {f5(model['form_a_short'][0]['length'])}"],
        ["TIP FACE", f5(model["tip_z"]), f"Ø {f5(diam(tip_r))} TO THE AXIS"],
    ]
    T(ax, 150, 208, "FORM A DEFINITION", size=6.0, weight="bold", color=DIM)
    y = draw_table(ax, 150, 205, [32, 36, 88], ["ITEM", "VALUE", "NOTE"], notes, row_h=3.6, size=4.5)
    T(ax, 150, y - 4, "The interval from crest 15 to crest 16 stays on the 2.01724 pitch.",
      size=4.6, color="#333")
    if y < 56:
        raise RuntimeError(f"Form A tables collide with the title block at y={y:.1f}")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def _describe(seg):
    if seg["kind"] == "ARC":
        return [f"R {f3(seg['radius'])}", f5(seg["z0"]), f5(seg["z1"]),
                f"CTR ({f5(seg['center'][0])}, {f5(seg['center'][1])})"]
    if seg["kind"] == "CYL":
        return ["CYL", f5(seg["z0"]), f5(seg["z1"]), f"Ø {f5(diam(seg['r0']))}  L {f5(seg['length'])}"]
    if seg["kind"] == "FACE":
        return ["FACE", f5(seg["z0"]), f5(seg["z1"]), f"Ø {f5(diam(max(seg['r0'], seg['r1'])))} TO THE AXIS"]
    return [ang_txt(seg), f5(seg["z0"]), f5(seg["z1"]),
            f"Ø {f5(diam(seg['r0']))} → {f5(diam(seg['r1']))}"]


def _repeating_ids(model):
    ids = set()
    for crest in model["form_b_short"] + model["form_b_long"]:
        ids.add(crest["n"])
    root_r = model["form_b_root_r"]
    for seg in model["segs"]:
        if seg["kind"] == "TAPER" and abs(seg["ang"] - 53.0) < 0.02 and abs(min(seg["r0"], seg["r1"]) - root_r) < 1e-3:
            ids.add(seg["n"])
        if seg["kind"] == "CYL" and abs(seg["r0"] - 7.2) < 1e-3:
            ids.add(seg["n"])
        if seg["kind"] == "CYL" and abs(seg["r0"] - 6.5) < 1e-3 and abs(seg["length"] - model["form_a_root"]["length"]) < 5e-4:
            ids.add(seg["n"])
        if seg["kind"] == "TAPER" and abs(min(seg["r0"], seg["r1"]) - 6.5) < 1e-3 and abs(max(seg["r0"], seg["r1"]) - 7.2) < 1e-3:
            ids.add(seg["n"])
    return ids


def _summary_rows(model, run):
    z0 = run[0]["z0"]
    if z0 < 45:
        b0, b1 = model["form_b_short"], model["form_b_long"]
        return [
            ["B ×9", f5(b0[0]["z0"]), f5(b0[-1]["z1"]),
             f"Ø 14.900  FLAT {f5(b0[0]['length'])}  PITCH {f5(model['pitch_b1'])}"],
            ["B ×8", f5(b1[0]["z0"]), f5(b1[-1]["z1"]),
             f"FLAT {f5(b1[0]['length'])}  PITCH {f5(model['pitch_b2'])}"],
            ["B FLANK", f5(run[0]["z0"]), f5(run[-1]["z1"]),
             f"53.00° / 53.00°  ROOT Ø {f5(diam(model['form_b_root_r']))}"],
        ]
    a0, a1 = model["form_a_long"], model["form_a_short"]
    root = model["form_a_root"]
    return [
        ["A ×15", f5(a0[0]["z0"]), f5(a0[-1]["z1"]),
         f"Ø 14.400  FLAT {f5(a0[0]['length'])}  PITCH {f5(model['pitch_a1'])}"],
        ["A ×8", f5(a1[0]["z0"]), f5(a1[-1]["z1"]),
         f"FLAT {f5(a1[0]['length'])}  PITCH {f5(model['pitch_a2'])}"],
        ["A ROOT", f5(root["z0"]), f5(run[-1]["z1"]),
         f"Ø 13.000  FLAT {f5(root['length'])}  ×{model['form_a_root_n']}"],
        ["A FLANK", f5(run[0]["z0"]), f5(run[-1]["z1"]), "35.80° LEAVING / 59.70° APPROACHING"],
    ]


def sheet_schedule(pdf, model):
    fig, ax = new_page()
    frame(ax, 6, "SHEET 6   DIMENSION SCHEDULE")
    T(ax, 14, 274, "CONTROLLING SIZES.  Z FROM DATUM A, POSITIVE TOWARD THE SMALL END.  MODEL VALUES TO 0.00001 mm.",
      size=5.7, weight="medium", color=DIM)
    mass = model["mass"]
    head_r = max(model["head_face"]["r0"], model["head_face"]["r1"])
    tip_r = max(model["tip_face"]["r0"], model["tip_face"]["r1"])
    direc = model["axis"]["direction"]
    origin = model["axis"]["origin"]
    groove_r, groove_z = model["groove"]

    overall = [
        ["OVERALL", f5(model["overall"]), "HEAD FACE TO TIP FACE"],
        ["HEAD FACE Z", f5(model["head_z"]), f"Ø {f5(diam(head_r))}"],
        ["LARGEST Ø", "28.60000", f"LAND L {f5(model['head_cyl']['length'])}"],
        ["GROOVE Ø", f5(diam(groove_r)), f"AT Z {f5(groove_z)}"],
        ["TIP FACE Z", f5(model["tip_z"]), f"Ø {f5(diam(tip_r))}"],
        ["VOLUME", f"{mass['volume']:.5f}", "mm³   SOLID"],
        ["SURFACE", f"{mass['area']:.5f}", "mm²"],
        ["COM Z", f5(mass["com_z"]), "ON THE AXIS"],
    ]
    T(ax, 12, 266, "1   OVERALL", size=6.2, weight="bold", color=DIM)
    y = draw_table(ax, 12, 263, [32, 36, 72], ["ITEM", "MODEL", "NOTE"], overall, row_h=3.35, size=4.7)

    arc_rows = []
    for arc in model["arcs"]:
        c = arc["center"]
        arc_rows.append([f5(arc["radius"]), f5(c[0]), f5(c[1]), f5(arc["z0"]), f5(arc["z1"])])
    T(ax, 12, y - 4, "2   ARCS", size=6.2, weight="bold", color=DIM)
    y = draw_table(ax, 12, y - 7, [22, 26, 28, 28, 28],
                   ["R", "CTR R", "CTR Z", "Z0", "Z1"], arc_rows, row_h=3.5, size=4.6)

    T(ax, 12, y - 4, "3   AXIS IN THE STEP FILE", size=6.2, weight="bold", color=DIM)
    y = draw_table(ax, 12, y - 7, [36, 32, 32, 36], ["ITEM", "X", "Y", "Z"], [
        ["DIRECTION", f6(direc[0]), f6(direc[1]), f6(direc[2])],
        ["POINT ON AXIS", f5(origin[0]), f5(origin[1]), f5(origin[2])],
    ], row_h=3.6, size=4.6)
    T(ax, 12, y - 4, "The point is the head-end centre. Drawing +Z is opposite this direction.", size=4.6, color="#444")

    repeating = _repeating_ids(model)
    right = []
    i = 0
    segs = model["segs"]
    while i < len(segs):
        if segs[i]["n"] not in repeating:
            right.append(_describe(segs[i]))
            i += 1
            continue
        j = i
        while j < len(segs) and segs[j]["n"] in repeating:
            j += 1
        right.extend(_summary_rows(model, segs[i:j]))
        i = j

    T(ax, 168, 266, "4   EVERY CONTROLLING FEATURE", size=6.2, weight="bold", color=DIM)
    y_right = draw_table(ax, 168, 263, [24, 28, 28, 90],
                         ["ITEM", "Z0", "Z1", "SIZE"], right, row_h=3.05, size=4.15)
    T(ax, 12, 18, "SHEETS 7 AND 8 LIST ALL 168 PROFILE SEGMENTS, INCLUDING EVERY FLANK.", size=5.0, color="#444")
    if y_right < 50:
        raise RuntimeError(f"Schedule table collides with the title block at y={y_right:.1f}")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_segments(pdf, page, rows, first, last, continued):
    fig, ax = new_page()
    title = "SHEET 7   EVERY PROFILE SEGMENT" if not continued else "SHEET 8   PROFILE SEGMENTS, CONTINUED"
    frame(ax, page, title)
    T(ax, 12, 274,
      "ORDERED +R PROFILE, HEAD FACE TO TIP. REVOLVE ABOUT DATUM B. MILLIMETRES TO 0.00001. ANGLE IS FROM THE AXIS.",
      size=5.6, weight="medium", color=DIM)
    headers = ["#", "KIND", "Z1", "R1", "Z2", "R2", "LENGTH", "ANGLE", "ARC"]
    widths = [12, 16, 26, 24, 26, 24, 24, 18, 22]
    per_col = 48
    left, right = rows[:per_col], rows[per_col:]
    draw_table(ax, 12, 266, widths, headers, left, row_h=3.35, size=4.25)
    if right:
        draw_table(ax, 214, 266, widths, headers, right, row_h=3.35, size=4.25)
    T(ax, 12, 72, f"SEGMENTS {first}–{last} OF 168", size=5.2, color="#333")
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
        f"volume {model['mass']['volume']:.3f}  "
        f"B {len(model['form_b_short'])}+{len(model['form_b_long'])}  "
        f"A {len(model['form_a_long'])}+{len(model['form_a_short'])}"
    )
    print("Pictorial")
    iso = render_iso(solid, axis)
    rows = []
    for seg in segs:
        if seg["kind"] == "ARC":
            note = f"R{f5(seg['radius'])}"
            ang = ""
        else:
            note = ""
            ang = ang_txt(seg).rstrip("°")
        rows.append([
            f"{seg['n']:03d}", seg["kind"], f5(seg["z0"]), f5(seg["r0"]),
            f5(seg["z1"]), f5(seg["r1"]), f5(seg["length"]), ang, note,
        ])
    per_page = 96
    chunks = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    TOTAL = 6 + len(chunks)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "2 LARGE NUT — all nominal dimensions"
        info["Subject"] = "Dimensioned drawing extracted from 2 large nut.STEP"
        sheet_general(pdf, model, iso)
        sheet_head(pdf, model)
        sheet_form_b(pdf, model)
        sheet_groove(pdf, model)
        sheet_form_a(pdf, model)
        sheet_schedule(pdf, model)
        for i, chunk in enumerate(chunks):
            sheet_segments(pdf, 7 + i, chunk, chunk[0][0], chunk[-1][0], continued=i > 0)
    print("Wrote", OUT_PATH, "sheets", TOTAL)


if __name__ == "__main__":
    build()
