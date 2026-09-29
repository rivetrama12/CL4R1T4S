#!/usr/bin/env python3
"""Nominal dimension drawing of the turned part "1 mid L".

The STEP file stores geometry only: no PMI, tolerances, material, or finish.
The solid is a surface of revolution about an axis that is slightly tilted in
the SolidWorks coordinate system. Every size in this drawing is measured in
the part frame: Z along that axis, origin on the head shoulder (datum A),
positive Z toward the shank end. R is the perpendicular distance from the axis.

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
STEP_PATH = ROOT / "1_mid_L.STEP"
OUT_PATH = ROOT / "1_mid_L_all_dimensions.pdf"

PAGE_W, PAGE_H = 420.0, 297.0
TOTAL = 6

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


def T(ax, x, y, text, size=7.0, weight="regular", color=RULE, ha="left", va="center", z=5, **kw):
    ax.text(
        x, y, text,
        fontsize=size, fontproperties=FONTS[weight], color=color,
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


class View:
    """Maps part-frame (z, r) onto the sheet, in millimetres."""

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
            n = u.Crossed(gp_Vec(1.0, 0.0, 0.0))
            n.Normalize()
            w = n.Crossed(u)
            w.Normalize()
            return origin, u, n, w
        exp.Next()
    raise RuntimeError("No cylindrical face found to define the axis")


def _to_rz(origin, u, w, point):
    vec = gp_Vec(origin, point)
    return (vec.Dot(w), vec.Dot(u))


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
            circ = curve.Circle()
            cr, cz = _to_rz(origin, u, w, circ.Location())
            rec["arc"] = (cr, cz, circ.Radius())
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
    tol = 2e-4
    for _ in range(len(positive) + 2):
        nxt = None
        for i, e in enumerate(positive):
            if used[i]:
                continue
            if math.hypot(e["p0"][0] - p_cur[0], e["p0"][1] - p_cur[1]) < tol:
                nxt = (i, e["p0"], e["p1"], e)
                break
            if math.hypot(e["p1"][0] - p_cur[0], e["p1"][1] - p_cur[1]) < tol:
                nxt = (i, e["p1"], e["p0"], e)
                break
        if nxt is None:
            break
        i, a, b, e = nxt
        used[i] = True
        ordered.append((a, b, e))
        p_cur = b
    if not all(used):
        missed = sum(1 for flag in used if not flag)
        raise RuntimeError(f"Profile chain left {missed} edges unused")

    segs = []
    for p0, p1, rec in ordered:
        is_arc = rec["arc"] is not None
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
            cx, cz, radius = rec["arc"]
            pts, sweep = _arc_points(cx, cz, radius, p0, p1)
            seg["radius"] = radius
            seg["center"] = (cx, cz)
            seg["pts"] = pts
            seg["length"] = abs(sweep) * radius
        segs.append(seg)

    # Datum A is the head shoulder: the radial face with the greatest radius.
    faces = [s for s in segs if s["kind"] == "FACE" and min(s["r0"], s["r1"]) > 0.2]
    if not faces:
        raise RuntimeError("No shoulder face found")
    shoulder = max(faces, key=lambda s: max(s["r0"], s["r1"]))
    z_datum = 0.5 * (shoulder["z0"] + shoulder["z1"])
    for seg in segs:
        seg["z0"] -= z_datum
        seg["z1"] -= z_datum
        seg["p0"] = (seg["r0"], seg["z0"])
        seg["p1"] = (seg["r1"], seg["z1"])
        seg["pts"] = [(r, z - z_datum) for r, z in seg["pts"]]
        if seg["center"] is not None:
            seg["center"] = (seg["center"][0], seg["center"][1] - z_datum)
    for i, seg in enumerate(segs, start=1):
        seg["n"] = i

    axis = {
        "origin": (origin.X(), origin.Y(), origin.Z()),
        "direction": (u.X(), u.Y(), u.Z()),
        "z_datum_model": z_datum,
    }
    return segs, axis


def _revolution_volume(segs):
    acc = 0.0
    for seg in segs:
        pts = seg["pts"]
        for (r0, z0), (r1, z1) in zip(pts, pts[1:]):
            dz = z1 - z0
            dr = r1 - r0
            acc += dz * (r0 * r0 + r0 * dr + dr * dr / 3.0)
    return math.pi * acc


def _cyls(segs, radius=None, length=None, r_tol=2e-4, l_tol=2e-4, z_min=None, z_max=None):
    found = []
    for seg in segs:
        if seg["kind"] != "CYL":
            continue
        if radius is not None and abs(seg["r0"] - radius) > r_tol:
            continue
        if length is not None and abs(seg["length"] - length) > l_tol:
            continue
        z0, z1 = seg["z0"], seg["z1"]
        if z_min is not None and z1 < z_min:
            continue
        if z_max is not None and z0 > z_max:
            continue
        found.append(seg)
    return found


def analyze(segs, shape, axis):
    if segs[0]["r0"] > 1e-4 or segs[-1]["r1"] > 1e-4:
        raise RuntimeError("Profile does not run from axis to axis")
    if any(seg["z1"] + 1e-6 < seg["z0"] for seg in segs):
        raise RuntimeError("Profile Z is not monotonic")
    arcs = [s for s in segs if s["kind"] == "ARC"]
    if len(arcs) != 2:
        raise RuntimeError(f"Expected 2 axial arcs, found {len(arcs)}")
    nose, head = arcs
    if abs(nose["radius"] - 4.0) > 1e-4 or abs(head["radius"] - 1.95) > 1e-4:
        raise RuntimeError("Unexpected arc radii")

    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props)
    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area_props)
    com = volume_props.CentreOfMass()
    origin = gp_Pnt(*axis["origin"])
    u = gp_Vec(*axis["direction"])
    w = gp_Vec(1.0, 0.0, 0.0)
    # Radial offset of the centre of mass, using the same section plane basis.
    # Recomputed against the section plane so a bore or a lean would show up.
    com_r, com_z_model = _to_rz(origin, u, w, gp_Pnt(com.X(), com.Y(), com.Z()))
    # w above is not the section radial. Use both perpendiculars.
    n = u.Crossed(gp_Vec(1.0, 0.0, 0.0))
    n.Normalize()
    w = n.Crossed(u)
    w.Normalize()
    vec = gp_Vec(origin, gp_Pnt(com.X(), com.Y(), com.Z()))
    com_axial = vec.Dot(u) - axis["z_datum_model"]
    com_rad = math.hypot(vec.Dot(w), vec.Dot(n))
    rev = _revolution_volume(segs)
    if abs(rev - volume_props.Mass()) / volume_props.Mass() > 0.002:
        raise RuntimeError(
            f"Profile volume {rev:.3f} disagrees with the solid {volume_props.Mass():.3f}"
        )

    crests_350 = _cyls(segs, 3.05, 0.35, r_tol=1e-4, l_tol=1e-3)
    crests_250 = _cyls(segs, 3.05, 0.25, r_tol=1e-4, l_tol=1e-3)
    form_a = [c for c in crests_350 if c["z0"] < 16.0]
    entry = [c for c in crests_350 if c["z0"] > 16.0]
    if len(form_a) != 5 or len(entry) != 1 or len(crests_250) != 15:
        raise RuntimeError(
            f"Unexpected crests A={len(form_a)} entry={len(entry)} B={len(crests_250)}"
        )
    pitch_a = [form_a[i + 1]["z0"] - form_a[i]["z0"] for i in range(len(form_a) - 1)]
    pitch_b = [crests_250[i + 1]["z0"] - crests_250[i]["z0"] for i in range(len(crests_250) - 1)]

    roots = _cyls(segs, 2.6, r_tol=1e-4)
    root_groups = []
    for seg in roots:
        for group in root_groups:
            if abs(group[0]["length"] - seg["length"]) < 5e-4:
                group.append(seg)
                break
        else:
            root_groups.append([seg])

    neck = _cyls(segs, 3.3242421489, r_tol=1e-4)
    if len(neck) != 1:
        raise RuntimeError(f"Expected one neck cylinder, found {len(neck)}")
    land = _cyls(segs, 3.05, 1.5, r_tol=1e-4, l_tol=1e-3)
    barrel = _cyls(segs, 3.05, 2.9, r_tol=1e-4, l_tol=1e-3)
    tip_land = _cyls(segs, 2.85, 0.25, r_tol=1e-4, l_tol=1e-3)
    if len(land) != 1 or len(barrel) != 1 or len(tip_land) != 1:
        raise RuntimeError("Missing land, barrel, or tip cylinder")

    faces = [s for s in segs if s["kind"] == "FACE"]
    shoulder = min(faces, key=lambda s: abs(s["z0"]))
    head_face = min(faces, key=lambda s: s["z0"])
    end_face = max(faces, key=lambda s: s["z0"])
    step = [s for s in faces if s is not shoulder and s is not head_face and s is not end_face]
    if len(step) != 1:
        raise RuntimeError(f"Expected one intermediate radial step, found {len(step)}")

    # Point of greatest radius on the head arc.
    crown_r = head["center"][0] + head["radius"]
    crown_z = head["center"][1]

    model = {
        "segs": segs,
        "axis": axis,
        "tip_z": head_face["z0"],
        "end_z": end_face["z0"],
        "overall": end_face["z0"] - head_face["z0"],
        "head_len": -head_face["z0"],
        "nose": nose,
        "head": head,
        "crown": (crown_r, crown_z),
        "shoulder": shoulder,
        "head_face": head_face,
        "end_face": end_face,
        "step": step[0],
        "neck": neck[0],
        "form_a": form_a,
        "entry": entry[0],
        "form_b": crests_250,
        "pitch_a": pitch_a,
        "pitch_b": pitch_b,
        "roots": roots,
        "root_groups": root_groups,
        "land": land[0],
        "barrel": barrel[0],
        "tip_land": tip_land[0],
        "mass": {
            "volume": volume_props.Mass(),
            "area": area_props.Mass(),
            "com_z": com_axial,
            "com_r": com_rad,
        },
    }
    if abs(model["overall"] - 45.9) > 1e-4:
        raise RuntimeError(f"Unexpected overall length {model['overall']}")
    if abs(model["tip_z"] + 3.7) > 1e-4 or abs(model["end_z"] - 42.2) > 1e-4:
        raise RuntimeError("Datum shift did not land on the expected stations")
    if model["mass"]["com_r"] > 1e-3:
        raise RuntimeError(f"Centre of mass is off the axis by {model['mass']['com_r']}")
    # Silence unused local from the first radial attempt.
    del com_r, com_z_model
    return model


def render_iso(solid, axis):
    origin = gp_Pnt(*axis["origin"])
    u = gp_Vec(*axis["direction"])
    n = u.Crossed(gp_Vec(1.0, 0.0, 0.0))
    n.Normalize()
    w = n.Crossed(u)
    w.Normalize()
    z0 = axis["z_datum_model"]

    def loc(v):
        d = gp_Vec(origin, gp_Pnt(v.x, v.y, v.z))
        return (d.Dot(u) - z0, d.Dot(w), d.Dot(n))

    verts, tris = solid.tessellate(0.18)
    xyz = np.array([loc(v) for v in verts], dtype=float)
    faces = xyz[np.asarray(tris)]
    order = faces[:, :, [0, 1, 2]]
    normals = np.cross(order[:, 1] - order[:, 0], order[:, 2] - order[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.array([0.58, 0.66, 0.74])
    rgb = np.clip(0.18 + 0.82 * shade[:, None] * base, 0, 1)
    rgba = np.concatenate([rgb, np.ones((len(order), 1))], axis=1)

    fig = plt.figure(figsize=(7.2, 3.2), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(order, linewidths=0.0, antialiased=False)
    coll.set_facecolor(rgba)
    ax.add_collection3d(coll)
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    span = max(xyz[:, 1].max() - xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min())
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(-span / 2 - 1, span / 2 + 1)
    ax.set_zlim(-span / 2 - 1, span / 2 + 1)
    ax.set_box_aspect((xmax - xmin + 2, span + 2, span + 2))
    ax.view_init(elev=18, azim=-118)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=140, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    image = plt.imread(buf)
    rgb = image[:, :, :3]
    mask = np.any(rgb < 0.97, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 6
    r0, r1 = max(0, rows[0] - pad), min(rgb.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(rgb.shape[1], cols[-1] + pad)
    return rgb[r0:r1, c0:c1]


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
    T(ax, 12, 286.2, "1 MID L", size=10.5, weight="bold", va="top")
    T(ax, 42, 286.2, "TURNED SOLID  ·  NOMINAL DIMENSION DRAWING", size=8.0, weight="medium", color=DIM, va="top")
    T(ax, 406, 286.2, view_title, size=7.2, weight="medium", ha="right", va="top", color="#333")
    x, y, w, h = 248, 8, 164, 40
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=RULE, lw=0.7, zorder=4))
    ax.plot([x, x + w], [y + 28, y + 28], color=RULE, lw=0.4, zorder=4)
    ax.plot([x, x + w], [y + 16, y + 16], color=RULE, lw=0.4, zorder=4)
    ax.plot([x + 82, x + 82], [y, y + h], color=RULE, lw=0.4, zorder=4)
    T(ax, x + 3, y + 34, "PART", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 34, "1 MID L", size=8, weight="bold", va="center", z=7)
    T(ax, x + 85, y + 34, "SHEET", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 34, f"{page}  /  {TOTAL}", size=8, weight="bold", va="center", z=7)
    T(ax, x + 3, y + 22, "FILE", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 22, y + 22, "1 mid L.STEP", size=7.2, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 22, "UNITS", size=5.2, color="#666", va="center", z=7)
    T(ax, x + 108, y + 22, "MILLIMETRES", size=7.2, weight="medium", va="center", z=7)
    T(ax, x + 3, y + 8, "DATUM A  SHOULDER Z = 0", size=5.4, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 8, "DATUM B  AXIS", size=5.4, weight="medium", va="center", z=7)
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
    if gap >= 12:
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0),
            zorder=3,
        )
        va = "bottom" if text_side == "up" else "top"
        dy = 0.7 if text_side == "up" else -0.7
        T(ax, (x1 + x2) / 2, y + dy, text, size=6.2, weight="medium", color=DIM, ha="center", va=va,
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
        T(ax, (x1 + x2) / 2, y + dy, text, size=5.8, weight="medium", color=DIM, ha="center", va=va,
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
        T(ax, x + dx, (y1 + y2) / 2, text, size=6.2, weight="medium", color=DIM, ha=ha, va="center",
          bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))
    else:
        wing = 6.5
        ax.plot([x, x], [y1 - wing, y2 + wing], color=DIM, lw=0.5, zorder=3)
        ax.annotate("", xy=(x, y1), xytext=(x, y1 - wing),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", xy=(x, y2), xytext=(x, y2 + wing),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.5, mutation_scale=6.5, shrinkA=0, shrinkB=0), zorder=3)
        T(ax, x + 1.2, y2 + wing + 1.4, text, size=6.0, weight="medium", color=DIM, ha="left", va="bottom",
          bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.92))


def leader(ax, x, y, tx, ty, text, ha="left"):
    ax.annotate(
        "", xy=(x, y), xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.45, shrinkA=0, shrinkB=0),
        zorder=3,
    )
    ax.plot([x], [y], marker="o", ms=1.8, color=DIM, zorder=4)
    dx = 1.2 if ha == "left" else -1.2
    T(ax, tx + dx, ty, text, size=6.2, weight="medium", color=DIM, ha=ha, va="center",
      bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.92))


def centerline(ax, view, z0, z1):
    x0, y = view.pt(z0, 0)
    x1, _ = view.pt(z1, 0)
    ax.plot([x0, x1], [y, y], color=CENTER, lw=0.4, linestyle=(0, (7, 1.6, 0.9, 1.6)), zorder=2)


def _clip_pts(pts, z0, z1):
    """Keep the portion of a monotonic polyline inside [z0, z1]."""
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
    ax.add_patch(Circle((x, y), 3.5, fc="white", ec=DIM, lw=0.7, zorder=5))
    T(ax, x, y, label, size=6.8, weight="bold", color=DIM, ha="center", va="center", z=6)


def draw_table(ax, x, y_top, widths, headers, rows, row_h=3.55, size=5.2):
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


def ang_txt(seg):
    return f"{seg['ang']:.2f}°"


def find_ang(segs, angle, tol=0.02):
    return next(s for s in segs if s["ang"] is not None and abs(s["ang"] - angle) < tol)


def seg_between(segs, z_lo, z_hi):
    return [s for s in segs if s["z1"] > z_lo + 1e-6 and s["z0"] < z_hi - 1e-6]


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def sheet_general(pdf, model, iso):
    fig, ax = new_page()
    frame(ax, 1, "SHEET 1   GENERAL ARRANGEMENT")
    segs = model["segs"]
    scale = 6.0
    view = View(zx=46, y0=168, scale=scale)
    draw_profile(ax, view, segs, both=True, lw=0.55)
    centerline(ax, view, model["tip_z"] - 1.2, model["end_z"] + 1.4)
    T(ax, view.zx_of(model["tip_z"] - 1.15), view.y0 + 2.2, "B", size=6.4, weight="bold", color=CENTER, ha="right")

    face_x = view.zx_of(0)
    ax.plot([face_x, face_x], [view.ry_of(-5.6), view.ry_of(-6.6)], color=DIM, lw=0.4, zorder=3)
    ax.add_patch(Polygon(
        [(face_x, view.ry_of(-6.6)), (face_x - 1.6, view.ry_of(-6.6) - 2.8), (face_x + 1.6, view.ry_of(-6.6) - 2.8)],
        closed=True, fc=DIM, ec=DIM, lw=0.2, zorder=4,
    ))
    ax.add_patch(Rectangle((face_x - 3.2, view.ry_of(-6.6) - 7.6), 6.4, 4.6, fc="white", ec=DIM, lw=0.55, zorder=4))
    T(ax, face_x, view.ry_of(-6.6) - 5.3, "A", size=6.0, weight="bold", color=DIM, ha="center", va="center", z=5)

    hdim(ax, view.zx_of(model["tip_z"]), view.zx_of(model["end_z"]), 112,
         f"{f3(model['overall'])} OVERALL", view.ry_of(-model["crown"][0]), view.ry_of(0))
    hdim(ax, view.zx_of(0), view.zx_of(model["end_z"]), 100,
         f"{f3(model['end_z'])}   DATUM A TO END", view.ry_of(-model["shoulder"]["r0"]), view.ry_of(0))
    hdim(ax, view.zx_of(model["tip_z"]), view.zx_of(0), 88,
         f"{f3(model['head_len'])} HEAD", view.ry_of(0), view.ry_of(-3.5))

    crown_r, crown_z = model["crown"]
    leader(ax, *view.pt(crown_z, crown_r), view.zx_of(crown_z) - 18, view.ry_of(crown_r) + 14,
           f"Ø {f3(diam(crown_r))}", ha="left")
    neck = model["neck"]
    leader(ax, *view.pt(0.5 * (neck["z0"] + neck["z1"]), neck["r0"]),
           view.zx_of(neck["z1"]) + 6, view.ry_of(neck["r0"]) + 16,
           f"Ø {f3(diam(neck['r0']))} NECK", ha="left")
    crest = model["form_b"][4]
    leader(ax, *view.pt(0.5 * (crest["z0"] + crest["z1"]), crest["r0"]),
           view.zx_of(crest["z0"]), view.ry_of(3.6),
           "Ø 6.100 CREST", ha="left")
    end_r = max(model["end_face"]["r0"], model["end_face"]["r1"])
    leader(ax, *view.pt(model["end_z"], end_r), view.zx_of(model["end_z"]) + 2, view.ry_of(end_r) + 12,
           f"Ø {f3(diam(end_r))} END", ha="left")

    zones = [
        (model["form_a"][0]["z0"], model["form_a"][-1]["z1"], "FORM A"),
        (model["form_b"][0]["z0"], model["form_b"][-1]["z1"], "FORM B"),
    ]
    for z0, z1, name in zones:
        y = view.ry_of(crown_r) + 6
        ax.annotate("", xy=(view.zx_of(z1), y), xytext=(view.zx_of(z0), y),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0))
        T(ax, (view.zx_of(z0) + view.zx_of(z1)) / 2, y + 2.2, name, size=5.5, weight="bold",
          color=DIM, ha="center", va="bottom")

    balloon(ax, view.zx_of(-1.6), view.ry_of(crown_r) + 22, "2")
    ax.plot([view.zx_of(-1.15), view.zx_of(-1.15)], [view.ry_of(crown_r), view.ry_of(crown_r) + 18.5], color=DIM, lw=0.35)
    balloon(ax, view.zx_of(10.5), view.ry_of(crown_r) + 22, "3")
    ax.plot([view.zx_of(10.2), view.zx_of(10.2)], [view.ry_of(3.05), view.ry_of(crown_r) + 18.5], color=DIM, lw=0.35)
    balloon(ax, view.zx_of(30), view.ry_of(crown_r) + 22, "4")
    ax.plot([view.zx_of(30), view.zx_of(30)], [view.ry_of(3.05), view.ry_of(crown_r) + 18.5], color=DIM, lw=0.35)

    T(ax, 14, 78, "SCALE 6 : 1     AXIS HORIZONTAL     HEAD AT LEFT     FULL PROFILE", size=6.0, color="#333")

    ih, iw = iso.shape[0], iso.shape[1]
    box_w, box_h = 100.0, 78.0
    aspect = iw / ih
    if box_w / box_h > aspect:
        dh, dw = box_h, box_h * aspect
    else:
        dw, dh = box_w, box_w / aspect
    x0, y0 = 404 - dw, 270 - dh
    ax.imshow(iso, extent=(x0, x0 + dw, y0, y0 + dh), aspect="equal", zorder=1, interpolation="bilinear")
    T(ax, x0, y0 - 3.2, "PICTORIAL — NOT TO SCALE     HEAD AT LEFT", size=5.2, color="#444")

    m = model["mass"]
    notes = [
        "Solid of revolution about datum B. Revolve the +R profile.",
        "Sizes are CAD nominals from 1 mid L.STEP (SolidWorks 2023, 2024-10-24).",
        "No tolerance, material, or finish is stored in the model.",
        "Graphic sizes are rounded to 0.001 mm. Sheet 6 lists every segment to 0.00001 mm.",
        "Positive Z runs from datum A toward the shank end.",
        f"Volume {m['volume']:.3f} mm³     area {m['area']:.3f} mm²     centre of mass Z {f3(m['com_z'])} on the axis.",
        "Balloons 2, 3 and 4 mark the detail sheets.",
        "Form A: 5 crests, pitch 1.300 then 1.450.   Form B: 15 crests, pitch 1.26250 then 1.25000.",
    ]
    yy = 70
    for line in notes:
        T(ax, 14, yy, line, size=5.5, color="#333", va="center")
        yy -= 5.6
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_head(pdf, model):
    fig, ax = new_page()
    frame(ax, 2, "SHEET 2   HEAD, SHOULDER AND NECK")
    segs = model["segs"]
    scale = 14.0
    view = View(zx=118, y0=150, scale=scale)
    draw_profile(ax, view, segs, both=True, lw=0.95, z0=model["tip_z"] - 0.02, z1=model["neck"]["z1"])
    centerline(ax, view, model["tip_z"] - 0.35, model["neck"]["z1"] + 0.55)
    T(ax, view.zx_of(model["tip_z"] - 0.32), view.y0 + 2.4, "B", size=6.4, weight="bold", color=CENTER, ha="right")
    # Open end where the lead-in taper continues.
    bx, by = view.pt(model["neck"]["z1"], model["neck"]["r0"])
    for sign in (1, -1):
        yy = view.ry_of(sign * model["neck"]["r0"])
        ax.plot([bx, bx + 1.4, bx - 0.5, bx + 1.4], [yy, yy + 1.15 * sign, yy + 2.3 * sign, yy + 3.45 * sign],
                color=GEO, lw=0.55, zorder=3)
    T(ax, view.zx_of(model["neck"]["z1"]) + 4, view.ry_of(model["neck"]["r0"]) + 6, "SHEET 3", size=5.3, color=DIM)

    fx = view.zx_of(0)
    ax.plot([fx, fx + 10], [view.ry_of(5.2), view.ry_of(5.2)], color=DIM, lw=0.35, zorder=3)
    T(ax, fx + 11, view.ry_of(5.2), "DATUM A", size=6.0, weight="bold", color=DIM, va="center")

    crown_r, crown_z = model["crown"]
    head_r = max(model["head_face"]["r0"], model["head_face"]["r1"])
    vdim(ax, view.zx_of(1.6), view.ry_of(-model["neck"]["r0"]), view.ry_of(model["neck"]["r0"]),
         f"Ø {f3(diam(model['neck']['r0']))}", view.zx_of(1.6), view.zx_of(1.6))
    hdim(ax, view.zx_of(model["tip_z"]), view.zx_of(0), 52,
         f"{f3(model['head_len'])} HEAD", view.ry_of(-1.2), view.ry_of(-min(model["shoulder"]["r0"], model["shoulder"]["r1"])))
    hdim(ax, view.zx_of(model["neck"]["z0"]), view.zx_of(model["neck"]["z1"]), 42,
         f"{f3(model['neck']['length'])} NECK", view.ry_of(-model["neck"]["r0"]), view.ry_of(-model["neck"]["r0"]))

    nose, head = model["nose"], model["head"]
    pn, ph = midpt(nose), midpt(head)
    leader(ax, *view.pt(pn[1], pn[0]), view.zx_of(pn[1]) - 6, view.ry_of(pn[0]) + 10,
           f"R {f3(nose['radius'])}", ha="right")
    leader(ax, *view.pt(crown_z, crown_r), view.zx_of(crown_z) + 8, view.ry_of(crown_r) + 12,
           f"R {f3(head['radius'])}    Ø {f3(diam(crown_r))} AT Z {f3(crown_z)}", ha="left")
    leader(ax, *view.pt(model["tip_z"], head_r * 0.62), view.zx_of(model["tip_z"]) - 14, view.ry_of(head_r * 0.62),
           f"END Ø {f3(diam(head_r))}", ha="right")

    sh = model["shoulder"]
    rows = [
        ["HEAD FACE", f5(model["tip_z"]), "0.00000", "0.000", "ON THE AXIS"],
        ["HEAD RIM", f5(model["tip_z"]), f5(head_r), f5(diam(head_r)), "R4 MEETS THE FACE"],
        ["ARC JOIN", f5(nose["z1"]), f5(nose["r1"]), f5(diam(nose["r1"])), "NOSE TO HEAD ARC"],
        ["CROWN", f5(crown_z), f5(crown_r), f5(diam(crown_r)), "LARGEST DIAMETER"],
        ["DATUM A OUT", "0.00000", f5(max(sh["r0"], sh["r1"])), f5(diam(max(sh["r0"], sh["r1"]))), "SHOULDER OUTER"],
        ["DATUM A IN", "0.00000", f5(min(sh["r0"], sh["r1"])), f5(diam(min(sh["r0"], sh["r1"]))), "SHOULDER INNER"],
        ["NECK END", f5(model["neck"]["z1"]), f5(model["neck"]["r1"]), f5(diam(model["neck"]["r1"])), "LEAD-IN STARTS"],
    ]
    panel(ax, 214, 230, 190, 42, "HEAD STATIONS FROM DATUM A")
    draw_table(ax, 218, 264, [32, 28, 28, 28, 62],
               ["POINT", "Z", "RADIUS", "Ø", "NOTE"], rows, row_h=4.0, size=5.2)

    arc_rows = []
    for name, arc in (("NOSE", nose), ("HEAD", head)):
        c = arc["center"]
        arc_rows.append([name, f5(arc["radius"]), f5(c[0]), f5(c[1]), f5(arc["length"])])
    T(ax, 218, 158, "AXIAL-SECTION ARCS  —  BOTH ARE TORUS SECTIONS", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 218, 155, [22, 28, 32, 32, 36],
               ["ARC", "R", "CENTER R", "CENTER Z", "ARC LENGTH"], arc_rows, row_h=5.0, size=5.3)

    notes = [
        "Nose-arc center is inside the metal, 0.180 mm past datum A.",
        "The head arc carries the largest diameter.",
        "Datum A runs from the neck out to the head arc.",
        "Lead-in taper, 1.87°, continues on sheet 3.",
    ]
    yy = 128
    for line in notes:
        T(ax, 218, yy, line, size=5.4, color="#333")
        yy -= 6.4
    T(ax, 14, 274, "DETAIL 2     SCALE 14 : 1     FULL PROFILE", size=7, weight="bold", color=DIM)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_a(pdf, model):
    fig, ax = new_page()
    frame(ax, 3, "SHEET 3   LEAD-IN, FORM A AND GROOVE")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 3     TOOTH AND GROOVE ARE ENLARGED     ANGLES FROM THE AXIS", size=6.4, weight="bold", color=DIM)

    c0, c1 = model["form_a"][0], model["form_a"][1]
    tooth = View(zx=28, y0=196, scale=58, z0=c0["z0"], r0=2.20)
    draw_profile(ax, tooth, segs, both=False, lw=1.05, z0=c0["z0"], z1=c1["z0"])
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c1["z0"]), tooth.ry_of(3.28),
         f"PITCH {f3(c1['z0'] - c0['z0'])}    CRESTS 1–4", tooth.ry_of(3.05), tooth.ry_of(3.05))
    hdim(ax, tooth.zx_of(c0["z0"]), tooth.zx_of(c0["z1"]), tooth.ry_of(3.16),
         f"FLAT {f3(c0['length'])}", tooth.ry_of(3.05), tooth.ry_of(3.05))
    fall = next(s for s in segs if s["n"] == c0["n"] + 1)
    rise = next(s for s in segs if s["n"] == c0["n"] + 2)
    leader(ax, *tooth.pt(*midpt(fall)[::-1]), tooth.zx_of(7.7), tooth.ry_of(2.15), ang_txt(fall), ha="right")
    leader(ax, *tooth.pt(*midpt(rise)[::-1]), tooth.zx_of(8.7), tooth.ry_of(2.15), ang_txt(rise), ha="left")
    leader(ax, *tooth.pt(c0["z0"] + 0.1, 3.05), tooth.zx_of(7.45), tooth.ry_of(3.42), "Ø 6.100", ha="left")
    root_r = min(fall["r1"], rise["r0"])
    leader(ax, *tooth.pt(fall["z1"], root_r), tooth.zx_of(8.15), tooth.ry_of(2.05),
           f"Ø {f3(diam(root_r))} POINTED", ha="left")
    T(ax, 16, 176, "FORM A TOOTH   SCALE 58 : 1   MATERIAL BELOW THE LINE   CRESTS 1–3 SHARE THIS ROOT", size=5.3, color="#333")

    groove = View(zx=22, y0=108, scale=16, z0=14.2, r0=0.0)
    draw_profile(ax, groove, segs, both=False, lw=0.95, z0=14.2, z1=21.3)
    centerline(ax, groove, 14.15, 21.4)
    g_down = next(s for s in segs if abs(s["z0"] - model["land"]["z1"]) < 1e-6)
    g_up = next(s for s in segs if abs(s["z1"] - model["barrel"]["z0"]) < 1e-6)
    leader(ax, *groove.pt(*midpt(g_down)[::-1]), groove.zx_of(16.2), groove.ry_of(1.15), ang_txt(g_down), ha="right")
    leader(ax, *groove.pt(*midpt(g_up)[::-1]), groove.zx_of(17.6), groove.ry_of(1.15), ang_txt(g_up), ha="left")
    leader(ax, *groove.pt(model["land"]["z0"] + 0.4, model["land"]["r0"]),
           groove.zx_of(14.5), groove.ry_of(3.55), f"Ø {f3(diam(model['land']['r0']))}", ha="left")
    root_pt = (g_down["z1"], g_down["r1"])
    leader(ax, *groove.pt(root_pt[0], root_pt[1]), groove.zx_of(root_pt[0]), groove.ry_of(0.55),
           f"Ø {f3(diam(root_pt[1]))}", ha="left")
    T(ax, 16, 96, "LAND, GROOVE, BARREL, STEP   SCALE 16 : 1   HALF PROFILE", size=5.3, color="#333")

    lead = seg_between(segs, model["neck"]["z1"] - 1e-6, model["form_a"][0]["z0"] + 1e-6)
    lead = [s for s in lead if s["kind"] == "TAPER"]
    lead_rows = [[ang_txt(s), f5(s["z0"]), f5(s["z1"]), f5(diam(s["r0"])), f5(diam(s["r1"]))] for s in lead]
    panel(ax, 214, 168, 190, 104, "LEAD-IN TAPERS, NECK END TO FIRST CREST")
    draw_table(ax, 218, 264, [24, 32, 32, 32, 32],
               ["ANGLE", "Z0", "Z1", "Ø START", "Ø END"], lead_rows, row_h=4.2, size=5.3)

    a_rows = []
    for i, c in enumerate(model["form_a"], start=1):
        pitch = f5(c["z0"] - model["form_a"][i - 2]["z0"]) if i > 1 else "—"
        a_rows.append([str(i), f5(c["z0"]), f5(c["z1"]), f5(c["length"]), pitch])
    T(ax, 218, 214, "FORM A CRESTS   Ø 6.100   FLAT 0.350", size=6.2, weight="bold", color=DIM)
    T(ax, 218, 208, "Crests 1–4: fall 45.60°, pointed root Ø 5.176, rise 42.85°.", size=5.2, color="#333")
    T(ax, 218, 203, "Crest 4 to 5 drops to Ø 5.031, so that pitch opens to 1.450.", size=5.2, color="#333")
    draw_table(ax, 218, 200, [12, 36, 36, 28, 36],
               ["#", "Z0", "Z1", "FLAT", "PITCH"], a_rows, row_h=4.0, size=5.2)

    trans = [
        ["25.05° FLANK", f5(model["land"]["z0"] - (model["land"]["z0"] - 13.66108)), "SEE SEG 025", "ONTO THE LAND"],
        ["LAND", f5(model["land"]["z0"]), f5(model["land"]["z1"]), f"L {f3(model['land']['length'])}  Ø {f3(diam(model['land']['r0']))}"],
        ["GROOVE ROOT", f5(root_pt[0]), f5(root_pt[0]), f"R {f5(root_pt[1])}   Ø {f5(diam(root_pt[1]))}  POINTED"],
        ["BARREL", f5(model["barrel"]["z0"]), f5(model["barrel"]["z1"]), f"L {f3(model['barrel']['length'])}  Ø {f3(diam(model['barrel']['r0']))}"],
        ["STEP FACE", f5(model["step"]["z0"]), f5(model["step"]["z0"]),
         f"Ø {f3(diam(model['barrel']['r0']))} TO Ø {f3(diam(min(model['step']['r0'], model['step']['r1'])))}"],
    ]
    # The 25.05 flank end is the land start; show the real segment instead of the placeholder.
    flank = find_ang(segs, 25.05)
    trans[0] = ["25.05° FLANK", f5(flank["z0"]), f5(flank["z1"]), f"R {f5(flank['r0'])} TO {f5(flank['r1'])}"]
    T(ax, 218, 148, "AFTER CREST 5", size=6.2, weight="bold", color=DIM)
    draw_table(ax, 218, 145, [36, 36, 36, 70],
               ["FEATURE", "Z0", "Z1", "NOTE"], trans, row_h=4.3, size=5.15)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_form_b(pdf, model):
    fig, ax = new_page()
    frame(ax, 4, "SHEET 4   FORM B AND TIP")
    segs = model["segs"]
    T(ax, 14, 274, "DETAIL 4     REGULAR CREST ENLARGED     ENTRY AND TIP AT TRUE RADIAL SCALE", size=6.4, weight="bold", color=DIM)

    b0, b1 = model["form_b"][0], model["form_b"][1]
    tooth = View(zx=24, y0=214, scale=58, z0=b0["z0"] - 0.15, r0=2.40)
    draw_profile(ax, tooth, segs, both=False, lw=1.1, z0=b0["z0"] - 0.02, z1=b1["z1"] + 0.02)
    hdim(ax, tooth.zx_of(b0["z0"]), tooth.zx_of(b1["z0"]), tooth.ry_of(3.28),
         f"PITCH {f5(b1['z0'] - b0['z0'])}", tooth.ry_of(3.05), tooth.ry_of(3.05))
    hdim(ax, tooth.zx_of(b0["z0"]), tooth.zx_of(b0["z1"]), tooth.ry_of(3.16),
         f"FLAT {f3(b0['length'])}", tooth.ry_of(3.05), tooth.ry_of(3.05))
    fall = next(s for s in segs if abs(s["z0"] - b0["z1"]) < 1e-6)
    rise = next(s for s in segs if abs(s["z1"] - b1["z0"]) < 1e-6)
    fx, fy = tooth.pt(*midpt(fall)[::-1])
    rx, ry = tooth.pt(*midpt(rise)[::-1])
    leader(ax, fx, fy, fx - 14, fy - 14, ang_txt(fall), ha="right")
    leader(ax, rx, ry, rx + 12, ry - 14, ang_txt(rise), ha="left")
    root = next(s for s in segs if s["kind"] == "CYL" and abs(s["r0"] - 2.6) < 1e-4 and s["z0"] > b0["z1"] and s["z1"] < b1["z0"])
    hdim(ax, tooth.zx_of(root["z0"]), tooth.zx_of(root["z1"]), tooth.ry_of(2.22),
         f"ROOT {f5(root['length'])}", tooth.ry_of(2.6), tooth.ry_of(2.6))
    T(ax, 16, 196, "FIRST 0.250 CREST   SCALE 58 : 1   Ø 6.100 CREST   Ø 5.200 ROOT   8 CRESTS AT THIS PITCH", size=5.2, color="#333")

    entry = View(zx=18, y0=108, scale=18, z0=20.85, r0=0.0)
    draw_profile(ax, entry, segs, both=False, lw=0.9, z0=20.8, z1=23.3)
    centerline(ax, entry, 20.8, 23.35)
    leader(ax, *entry.pt(model["barrel"]["z1"] - 0.3, model["barrel"]["r0"]),
           entry.zx_of(20.9), entry.ry_of(3.45), "Ø 6.100", ha="left")
    step_r = min(model["step"]["r0"], model["step"]["r1"])
    leader(ax, *entry.pt(model["step"]["z0"], step_r), entry.zx_of(20.95), entry.ry_of(1.7),
           f"STEP TO Ø {f3(diam(step_r))}", ha="left")
    ent = model["entry"]
    leader(ax, *entry.pt(0.5 * (ent["z0"] + ent["z1"]), ent["r0"]),
           entry.zx_of(ent["z1"]) + 0.15, entry.ry_of(3.55), f"ENTRY FLAT {f3(ent['length'])}", ha="left")
    T(ax, 16, 96, "THREAD ENTRY   SCALE 18 : 1", size=5.4, color="#333")

    tip = View(zx=150, y0=78, scale=22, z0=40.55, r0=0.0)
    draw_profile(ax, tip, segs, both=False, lw=0.9, z0=40.5, z1=model["end_z"] + 0.02)
    centerline(ax, tip, 40.5, model["end_z"] + 0.25)
    run = find_ang(segs, 35.45)
    last = find_ang(segs, 37.55)
    leader(ax, *tip.pt(*midpt(run)[::-1]), tip.zx_of(40.85), tip.ry_of(3.4), ang_txt(run), ha="left")
    leader(ax, *tip.pt(model["tip_land"]["z0"] + 0.08, model["tip_land"]["r0"]),
           tip.zx_of(41.15), tip.ry_of(3.55), f"Ø {f3(diam(model['tip_land']['r0']))}", ha="left")
    leader(ax, *tip.pt(*midpt(last)[::-1]), tip.zx_of(41.85), tip.ry_of(1.4), ang_txt(last), ha="right")
    end_r = max(model["end_face"]["r0"], model["end_face"]["r1"])
    leader(ax, *tip.pt(model["end_z"], end_r * 0.7), tip.zx_of(model["end_z"]) + 0.2, tip.ry_of(1.9),
           f"END Ø {f3(diam(end_r))}", ha="left")
    T(ax, 148, 66, "TIP   SCALE 22 : 1", size=5.4, color="#333")

    rows = []
    prev = None
    for i, c in enumerate(model["form_b"], start=1):
        pitch = "—" if prev is None else f5(c["z0"] - prev)
        rows.append([str(i), f5(c["z0"]), f5(c["z1"]), pitch])
        prev = c["z0"]
    left, right = rows[:8], rows[8:]
    while len(right) < len(left):
        right.append(["", "", "", ""])
    paired = [a + b for a, b in zip(left, right)]
    panel(ax, 228, 228, 176, 44, "FORM B  —  15 CRESTS, FLAT 0.250, Ø 6.100")
    draw_table(ax, 232, 264, [8, 28, 28, 22, 8, 28, 28, 22],
               ["#", "Z0", "Z1", "PITCH", "#", "Z0", "Z1", "PITCH"], paired, row_h=3.5, size=4.7)

    # Root-flat groups, longest first, so the repeating ones lead the note.
    groups = sorted(model["root_groups"], key=lambda g: (-len(g), g[0]["z0"]))
    spec = [
        ["CREST Ø / FLAT", "6.100 / 0.250", "15 CRESTS AFTER THE ENTRY"],
        ["ENTRY CREST", f"Z {f5(model['entry']['z0'])}", f"FLAT {f5(model['entry']['length'])}  NOT IN THE 15"],
        ["ROOT Ø", "5.200", "AXIAL FLAT, SEE GROUPS"],
        ["FALL / RISE", "47.15° / 46.95°", "FROM THE AXIS"],
        ["PITCH CHANGE", "AFTER CREST 8", "1.26250 THEN 1.25000"],
    ]
    for g in groups:
        spec.append([
            f"ROOT ×{len(g)}",
            f5(g[0]["length"]),
            f"Z {f5(g[0]['z0'])} … {f5(g[-1]['z1'])}",
        ])
    T(ax, 232, 218, "DEFINITION", size=6.0, weight="bold", color=DIM)
    draw_table(ax, 232, 215, [36, 40, 92],
               ["ITEM", "VALUE", "NOTE"], spec, row_h=3.45, size=4.8)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_schedule(pdf, model):
    fig, ax = new_page()
    frame(ax, 5, "SHEET 5   DIMENSION SCHEDULE")
    T(ax, 14, 274, "CONTROLLING SIZES.  Z FROM DATUM A, POSITIVE TOWARD THE SHANK.  ANGLES FROM THE AXIS.  MODEL VALUES TO 0.00001 mm.",
      size=6.0, weight="medium", color=DIM)
    m = model["mass"]
    sh = model["shoulder"]
    head_r = max(model["head_face"]["r0"], model["head_face"]["r1"])
    end_r = max(model["end_face"]["r0"], model["end_face"]["r1"])
    crown_r, crown_z = model["crown"]
    axinfo = model["axis"]
    direc = axinfo["direction"]

    overall = [
        ["OVERALL", f5(model["overall"]), "HEAD FACE TO END FACE"],
        ["HEAD FACE Z", f5(model["tip_z"]), f"Ø {f5(diam(head_r))}"],
        ["DATUM A", "0.00000", f"Ø {f5(diam(min(sh['r0'], sh['r1'])))} TO {f5(diam(max(sh['r0'], sh['r1'])))}"],
        ["LARGEST Ø", f5(diam(crown_r)), f"AT Z {f5(crown_z)}"],
        ["END FACE Z", f5(model["end_z"]), f"Ø {f5(diam(end_r))}"],
        ["VOLUME", f"{m['volume']:.5f}", "mm³   SOLID, NO BORE"],
        ["SURFACE", f"{m['area']:.5f}", "mm²"],
        ["COM Z", f5(m["com_z"]), "ON THE AXIS"],
    ]
    T(ax, 12, 266, "1   OVERALL", size=6.4, weight="bold", color=DIM)
    y_left = draw_table(ax, 12, 263, [32, 36, 78],
                        ["ITEM", "MODEL", "NOTE"], overall, row_h=3.5, size=5.0)

    arc_rows = []
    for name, arc in (("NOSE", model["nose"]), ("HEAD", model["head"])):
        c = arc["center"]
        arc_rows.append([name, f5(arc["radius"]), f5(c[0]), f5(c[1]), f5(arc["z0"]), f5(arc["z1"])])
    T(ax, 12, y_left - 5, "2   ARCS", size=6.4, weight="bold", color=DIM)
    y_left = draw_table(ax, 12, y_left - 8, [18, 24, 26, 26, 26, 26],
                        ["ARC", "R", "CTR R", "CTR Z", "Z0", "Z1"], arc_rows, row_h=3.7, size=5.0)

    neck = model["neck"]
    T(ax, 12, y_left - 5, "3   NECK", size=6.4, weight="bold", color=DIM)
    y_left = draw_table(ax, 12, y_left - 8, [28, 32, 32, 54],
                        ["ITEM", "Z0", "Z1", "SIZE"],
                        [["CYLINDER", f5(neck["z0"]), f5(neck["z1"]), f"Ø {f5(diam(neck['r0']))}   L {f5(neck['length'])}"]],
                        row_h=3.8, size=5.0)

    T(ax, 12, y_left - 5, "4   AXIS IN THE STEP FILE", size=6.4, weight="bold", color=DIM)
    origin = axinfo["origin"]
    axis_rows = [
        ["DIRECTION", f"{direc[0]:.6f}", f"{direc[1]:.6f}", f"{direc[2]:.6f}"],
        ["POINT ON AXIS", f5(origin[0]), f5(origin[1]), f5(origin[2])],
    ]
    y_left = draw_table(ax, 12, y_left - 8, [36, 36, 36, 38],
                        ["ITEM", "X", "Y", "Z"], axis_rows, row_h=3.8, size=5.0)
    T(ax, 12, y_left - 4, "The point lies in the shank end plane. Z above is the STEP frame, not datum A.",
      size=4.8, color="#444")

    # Right column: lead-in through tip, one row per controlling feature.
    lead = [s for s in seg_between(model["segs"], neck["z1"] - 1e-9, model["form_a"][0]["z0"] + 1e-9) if s["kind"] == "TAPER"]
    right = []
    for s in lead:
        right.append([f"LEAD {s['ang']:.2f}°", f5(s["z0"]), f5(s["z1"]), f"Ø {f5(diam(s['r0']))} → {f5(diam(s['r1']))}"])
    for i, c in enumerate(model["form_a"], start=1):
        right.append([f"A CREST {i}", f5(c["z0"]), f5(c["z1"]), "Ø 6.10000  FLAT 0.35000"])
    flank = find_ang(model["segs"], 25.05)
    g_down = find_ang(model["segs"], 50.75)
    g_up = find_ang(model["segs"], 49.45)
    right.append([f"FLANK {flank['ang']:.2f}°", f5(flank["z0"]), f5(flank["z1"]), f"R {f5(flank['r0'])} → {f5(flank['r1'])}"])
    right.append(["LAND", f5(model["land"]["z0"]), f5(model["land"]["z1"]), f"Ø {f5(diam(model['land']['r0']))}"])
    right.append([f"GROOVE {g_down['ang']:.2f}°", f5(g_down["z0"]), f5(g_down["z1"]), f"TO R {f5(g_down['r1'])}"])
    right.append([f"GROOVE {g_up['ang']:.2f}°", f5(g_up["z0"]), f5(g_up["z1"]), f"TO Ø {f5(diam(g_up['r1']))}"])
    right.append(["BARREL", f5(model["barrel"]["z0"]), f5(model["barrel"]["z1"]), f"Ø {f5(diam(model['barrel']['r0']))}"])
    right.append(["STEP", f5(model["step"]["z0"]), f5(model["step"]["z0"]),
                  f"Ø {f5(diam(max(model['step']['r0'], model['step']['r1'])))} → {f5(diam(min(model['step']['r0'], model['step']['r1'])))}"])
    right.append(["B ENTRY", f5(model["entry"]["z0"]), f5(model["entry"]["z1"]), "FLAT 0.35000"])
    right.append(["B CREST 1", f5(model["form_b"][0]["z0"]), f5(model["form_b"][0]["z1"]), "PITCH GROUP 1.26250"])
    right.append(["B CREST 8", f5(model["form_b"][7]["z0"]), f5(model["form_b"][7]["z1"]), "LAST OF PITCH 1.26250"])
    right.append(["B CREST 9", f5(model["form_b"][8]["z0"]), f5(model["form_b"][8]["z1"]), "FIRST OF PITCH 1.25000"])
    right.append(["B CREST 15", f5(model["form_b"][-1]["z0"]), f5(model["form_b"][-1]["z1"]), "LAST 0.250 FLAT"])
    for g in sorted(model["root_groups"], key=lambda g: g[0]["z0"]):
        right.append([f"ROOT ×{len(g)}", f5(g[0]["z0"]), f5(g[-1]["z1"]), f"FLAT {f5(g[0]['length'])}  Ø 5.20000"])
    run = find_ang(model["segs"], 35.45)
    last = find_ang(model["segs"], 37.55)
    right.append([f"TIP {run['ang']:.2f}°", f5(run["z0"]), f5(run["z1"]), f"Ø {f5(diam(run['r0']))} → {f5(diam(run['r1']))}"])
    right.append(["TIP LAND", f5(model["tip_land"]["z0"]), f5(model["tip_land"]["z1"]), f"Ø {f5(diam(model['tip_land']['r0']))}"])
    right.append([f"TIP {last['ang']:.2f}°", f5(last["z0"]), f5(last["z1"]), f"TO Ø {f5(diam(end_r))}"])
    right.append(["END FACE", f5(model["end_z"]), f5(model["end_z"]), f"Ø {f5(diam(end_r))} TO THE AXIS"])

    T(ax, 175, 266, "5   EVERY CONTROLLING FEATURE", size=6.4, weight="bold", color=DIM)
    y_right = draw_table(ax, 175, 263, [32, 32, 32, 78],
                         ["ITEM", "Z0", "Z1", "SIZE"], right, row_h=3.35, size=4.55)
    T(ax, 12, 18, "SHEET 6 LISTS ALL 97 PROFILE SEGMENTS, INCLUDING EVERY FLANK BETWEEN THE CRESTS ABOVE.",
      size=5.2, color="#444")
    if y_right < 48:
        # Still inside the title-block band on the right; the table is the record, so keep it,
        # but flag the collision for the visual check by raising.
        raise RuntimeError(f"Schedule table collides with the title block at y={y_right:.1f}")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def sheet_segments(pdf, model):
    fig, ax = new_page()
    frame(ax, 6, "SHEET 6   EVERY PROFILE SEGMENT")
    T(ax, 12, 274,
      "ORDERED +R PROFILE, HEAD FACE TO SHANK END. REVOLVE ABOUT DATUM B. MILLIMETRES TO 0.00001. ANGLE IS FROM THE AXIS.",
      size=5.8, weight="medium", color=DIM)
    rows = []
    for seg in model["segs"]:
        if seg["kind"] == "ARC":
            c = seg["center"]
            note = f"R{f5(seg['radius'])}"
            ang = ""
        else:
            note = ""
            ang = f"{seg['ang']:.2f}"
        rows.append([
            f"{seg['n']:03d}",
            seg["kind"],
            f5(seg["z0"]),
            f5(seg["r0"]),
            f5(seg["z1"]),
            f5(seg["r1"]),
            f5(seg["length"]),
            ang,
            note,
        ])
    headers = ["#", "KIND", "Z1", "R1", "Z2", "R2", "LENGTH", "ANGLE", "ARC"]
    widths = [11, 16, 26, 24, 26, 24, 24, 16, 22]
    per_col = 49
    left, right = rows[:per_col], rows[per_col:]
    draw_table(ax, 12, 266, widths, headers, left, row_h=3.42, size=4.45)
    if right:
        draw_table(ax, 214, 266, widths, headers, right, row_h=3.42, size=4.45)
    nose, head = model["nose"], model["head"]
    T(ax, 12, 78,
      f"ARC 002  R {f5(nose['radius'])}   CENTER R {f5(nose['center'][0])}   CENTER Z {f5(nose['center'][1])}"
      f"      ARC 003  R {f5(head['radius'])}   CENTER R {f5(head['center'][0])}   CENTER Z {f5(head['center'][1])}",
      size=5.0, color="#333")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def build():
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
    iso = render_iso(solid, axis)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "1 mid L — all nominal dimensions"
        info["Subject"] = "Dimensioned drawing extracted from 1 mid L.STEP"
        info["Keywords"] = "nominal, millimetres, datum A shoulder, datum B axis"
        sheet_general(pdf, model, iso)
        sheet_head(pdf, model)
        sheet_form_a(pdf, model)
        sheet_form_b(pdf, model)
        sheet_schedule(pdf, model)
        sheet_segments(pdf, model)
    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    build()
