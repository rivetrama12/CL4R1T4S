#!/usr/bin/env python3
"""Nominal dimensions of the MCF000 OP1 forward extrusion die, revision 0.

The STEP file is geometry only: no PMI, tolerances, or surface finish.
Every size on the sheets is measured from that solid.

Datum A is the die-insert entry face (STEP Z = 0), coincident with the
container die-side face. Datum B is the common axis. Positive STEP Z points
toward the punch. On every view the extrusion direction (−Z) runs left to
right, from the punch toward the kickout pin.

Graphic callouts are in millimetres. Lengths that are not an exact 0.001 mm
are given to 0.00001 mm in the tables, which is coarser than the model's
stated distance accuracy of 1e-7 mm.
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
STEP_PATH = ROOT / "FERR-CAD-MCF000-OP1-EXTRUSION-DIE-R0.step"
OUT_PATH = ROOT / "FERR-CAD-MCF000-OP1-EXTRUSION-DIE-R0_DIMENSIONS.pdf"
PREVIEW = Path("/tmp/die-sheets")

PAGE_W, PAGE_H = 420.0, 297.0
TOTAL = 5

GEO = "#161616"
DIM = "#0e3a5d"
EXT = "#8aa0b4"
CENTER = "#a32020"
RULE = "#1c1c1c"
INK = "#1a1a1a"
MUTED = "#4a5160"
PANEL = "#f4f7fb"
LINE = "#c5d2e0"

DIE_FC = "#d5e0ea"
CONT_FC = "#dce8d8"
RING_FC = "#f3ead6"
PIN_FC = "#f3d7c4"
PUNCH_FC = "#d9e2f2"

FP = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Regular.ttf")
FP_MED = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-Medium.ttf")
FP_BOLD = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf")
FP_MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/macos/JetBrainsMono-Regular.ttf")
FONTS = {"regular": FP, "medium": FP_MED, "bold": FP_BOLD, "mono": FP_MONO}

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["hatch.linewidth"] = 0.3
plt.rcParams["hatch.color"] = "#7f93a8"


def T(ax, x, y, text, size=7.0, weight="regular", color=INK, ha="left", va="center", z=5, **kw):
    ax.text(
        x, y, text,
        fontsize=size, fontproperties=FONTS[weight], color=color,
        ha=ha, va=va, zorder=z, **kw,
    )


def fmm(v: float) -> str:
    """Three decimals when the value sits on a 0.001 mm grid, otherwise five."""
    if abs(v) < 5e-7:
        v = 0.0
    if abs(v - round(v, 3)) <= 5e-7:
        return f"{v:.3f}"
    return f"{v:.5f}"


def f3(v: float) -> str:
    return f"{v:.3f}"


def fang(v: float) -> str:
    if abs(v) < 5e-4:
        return "0°"
    if abs(v - round(v)) < 5e-4:
        return f"{int(round(v))}°"
    text = f"{v:.3f}".rstrip("0").rstrip(".")
    return text + "°"


def near_pt(a, b, tol=1e-6) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---------------------------------------------------------------------------
# STEP
# ---------------------------------------------------------------------------

def parse_step(path: Path):
    raw = path.read_text()
    header = raw.split("DATA;", 1)[0]
    data = raw.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    name = re.search(r"FILE_NAME\('([^']*)','([^']*)'", header)
    entities = {}
    buf = []
    for line in data.splitlines():
        buf.append(line.strip())
        if not buf[-1].endswith(";"):
            continue
        chunk = " ".join(buf)[:-1].strip()
        buf = []
        match = re.match(r"#(\d+)\s*=\s*(.*)$", chunk)
        if match:
            entities[int(match.group(1))] = re.sub(r"\s+", " ", match.group(2))
    if buf:
        raise SystemExit("Unterminated STEP entity")
    return {
        "filename": name.group(1) if name else "",
        "time": name.group(2) if name else "",
        "entities": entities,
    }


def refs(body: str) -> list[int]:
    return [int(n) for n in re.findall(r"#(\d+)", body)]


def cartesian(entities, eid: int) -> tuple[float, float, float]:
    match = re.search(r"CARTESIAN_POINT\('[^']*',\(([^)]*)\)", entities[eid])
    if not match:
        raise SystemExit(f"Not a point: #{eid}")
    return tuple(float(tok) for tok in match.group(1).split(","))  # type: ignore[return-value]


SHELLS = {
    "die": [51, 119, 174, 229, 284, 339, 394, 449, 504, 559, 614, 669, 724, 779, 834, 889, 944, 999, 1054, 1089, 1144, 1194],
    "container": [1242, 1310, 1365, 1420, 1455, 1510, 1560],
    "shrink": [1608, 1676, 1731, 1766, 1821, 1871],
    "pin": [1919, 1956, 2011, 2066],
    "punch": [2091, 2128, 2183, 2238, 2293, 2348],
}

PARTS = {
    "die": {
        "item": "1",
        "title": "DIE INSERT MCF000",
        "material": "M2/ASP23  62 HRC",
        "fc": DIE_FC,
        "hatch": "///",
    },
    "container": {
        "item": "2",
        "title": "CONTAINER INSERT",
        "material": "M2  62 HRC",
        "fc": CONT_FC,
        "hatch": "\\\\\\",
    },
    "shrink": {
        "item": "3",
        "title": "SHRINK RING",
        "material": "H13  46–48 HRC",
        "fc": RING_FC,
        "hatch": "...",
    },
    "pin": {
        "item": "4",
        "title": "KICKOUT PIN",
        "material": "M2  60 HRC",
        "fc": PIN_FC,
        "hatch": "||",
    },
    "punch": {
        "item": "5",
        "title": "EXTRUSION PUNCH",
        "material": "M2  60–62 HRC",
        "fc": PUNCH_FC,
        "hatch": "--",
    },
}


def _vertex_xyz(entities, vertex_id: int):
    return cartesian(entities, refs(entities[vertex_id])[0])


def _edge_ends(entities, edge_id: int):
    match = re.search(r"EDGE_CURVE\('[^']*',#(\d+),#(\d+),#(\d+)", entities[edge_id])
    if not match:
        raise SystemExit(f"Not an edge: #{edge_id}")
    return _vertex_xyz(entities, int(match.group(1))), _vertex_xyz(entities, int(match.group(2)))


def _zr(xyz) -> tuple[float, float]:
    return (xyz[2], math.hypot(xyz[0], xyz[1]))


def face_segment(entities, fid: int):
    """One meridional segment for a face of revolution, as ((z, r), (z, r), kind)."""
    body = entities[fid]
    match = re.search(r"ADVANCED_FACE\('[^']*',\(([^)]*)\),#(\d+)", body)
    bound_ids = [int(n) for n in re.findall(r"#(\d+)", match.group(1))]
    surf = entities[int(match.group(2))]
    kind = surf.split("(", 1)[0]
    corners = []
    for bid in bound_ids:
        loop = refs(entities[bid])[0]
        for oriented in refs(entities[loop]):
            edge_refs = refs(entities[oriented])
            if not edge_refs:
                continue
            p0, p1 = _edge_ends(entities, edge_refs[-1])
            corners.append(_zr(p0))
            corners.append(_zr(p1))
    unique = []
    for pt in corners:
        if not any(near_pt(pt, kept) for kept in unique):
            unique.append(pt)
    if kind == "PLANE":
        z = unique[0][0]
        radii = sorted(pt[1] for pt in unique)
        if len(bound_ids) == 1:
            return ((z, 0.0), (z, radii[-1]), "disk")
        return ((z, radii[0]), (z, radii[-1]), "face")
    if len(unique) != 2:
        raise SystemExit(f"Face #{fid} ({kind}) has {len(unique)} corners")
    tag = "cylinder" if kind.startswith("CYLINDRICAL") else "cone"
    return (unique[0], unique[1], tag)


def chain_loop(segments):
    remaining = list(segments)
    start, end, kind = remaining.pop(0)
    loop = [(start, end, kind)]
    while remaining:
        cursor = loop[-1][1]
        found = None
        for i, (a, b, k) in enumerate(remaining):
            if near_pt(a, cursor):
                found = (i, a, b, k)
                break
            if near_pt(b, cursor):
                found = (i, b, a, k)
                break
        if found is None:
            raise SystemExit("Section profile did not chain")
        i, a, b, k = found
        remaining.pop(i)
        loop.append((a, b, k))
    if not near_pt(loop[-1][1], loop[0][0]):
        end, start = loop[-1][1], loop[0][0]
        # Solids are closed along the axis; the STEP file has no edge there.
        if abs(end[1]) < 1e-9 and abs(start[1]) < 1e-9:
            loop.append((end, start, "axis"))
        else:
            raise SystemExit("Section profile did not close")
    return loop


def _angle(z0, r0, z1, r1) -> float:
    return math.degrees(math.atan2(abs(r1 - r0), abs(z1 - z0)))


def _on_fillet(z, r) -> bool:
    return abs(math.hypot(z - (-1.5), r - 12.875) - 1.5) < 1e-5


def bore_features(loop):
    """Die bore, entry to exit, with the R1.500 facets collapsed into one blend."""
    edges = []
    for a, b, kind in loop:
        if max(a[1], b[1]) > 20.0 or kind in ("face", "disk"):
            continue
        z0, r0, z1, r1 = (a[0], a[1], b[0], b[1]) if a[0] >= b[0] else (b[0], b[1], a[0], a[1])
        edges.append({"z0": z0, "r0": r0, "z1": z1, "r1": r1, "kind": kind})
    edges.sort(key=lambda e: -e["z0"])

    features = []
    blend = [e for e in edges if _on_fillet(e["z0"], e["r0"]) and _on_fillet(e["z1"], e["r1"])]
    rest = [e for e in edges if e not in blend]
    if blend:
        blend.sort(key=lambda e: -e["z0"])
        features.append({
            "name": "Entry blend",
            "z0": blend[0]["z0"],
            "r0": blend[0]["r0"],
            "z1": blend[-1]["z1"],
            "r1": blend[-1]["r1"],
            "kind": "blend",
            "edges": blend,
            "note": "8 conical facets on R1.500",
        })
    for e in rest:
        ang = _angle(e["z0"], e["r0"], e["z1"], e["r1"])
        length = abs(e["z0"] - e["z1"])
        opening = e["r1"] > e["r0"] + 1e-9
        if ang < 0.02:
            if length > 40 and abs(2 * e["r0"] - 22.75) < 1e-6:
                name, note = "Guide bore", "parallel"
            elif abs(length - 3.0) < 1e-6:
                name, note = "Bearing land", "parallel"
            elif abs(e["z1"] + 200.0) < 1e-6:
                name, note = "Exit bore", "kickout pin runs in this bore"
            else:
                name, note = "Relief bore", "parallel, after the 0.060 relief"
        elif opening:
            name, note = "Relief taper", "ΔØ +0.060 over 0.150, 1:5"
        else:
            drop = e["r0"] - e["r1"]
            name, note = "Reduction taper", f"ΔR {fmm(drop)} × √3"
        features.append({**e, "name": name, "kind": e["kind"], "edges": [e], "note": note})
    return features


def od_features(loop, r_min: float):
    rows = []
    for a, b, kind in loop:
        if kind not in ("cylinder", "cone"):
            continue
        if min(a[1], b[1]) < r_min:
            continue
        z0, r0, z1, r1 = (a[0], a[1], b[0], b[1]) if a[0] >= b[0] else (b[0], b[1], a[0], a[1])
        ang = _angle(z0, r0, z1, r1)
        if ang < 0.02:
            name = "Outside diameter"
        elif abs(ang - 45.0) < 0.02:
            name = "OD chamfer"
        else:
            name = "OD taper"
        rows.append({
            "name": name,
            "z0": z0, "r0": r0, "z1": z1, "r1": r1,
            "kind": kind,
            "note": "both ends identical" if name == "OD chamfer" and abs(abs(z0 - z1) - 1.0) < 1e-6 else "",
        })
    rows.sort(key=lambda e: -e["z0"])
    return rows


def finish_feature(row):
    row["length"] = abs(row["z0"] - row["z1"])
    row["angle"] = 0.0 if row.get("kind") == "blend" else _angle(row["z0"], row["r0"], row["z1"], row["r1"])
    row["slant"] = math.hypot(row["z1"] - row["z0"], row["r1"] - row["r0"])
    row["d0"] = 2.0 * row["r0"]
    row["d1"] = 2.0 * row["r1"]
    row["depth0"] = -row["z0"]
    row["depth1"] = -row["z1"]
    return row


def polygon_of(loop):
    pts = [loop[0][0]]
    for _, b, _ in loop:
        pts.append(b)
    return pts[:-1]


def radial_extents(loop):
    radii = [p[1] for edge in loop for p in edge[:2]]
    zs = [p[0] for edge in loop for p in edge[:2]]
    return min(zs), max(zs), min(radii), max(radii)


def build_model(step):
    entities = step["entities"]
    model = {"step": step, "parts": {}}
    for key, faces in SHELLS.items():
        segments = [face_segment(entities, fid) for fid in faces]
        loop = chain_loop(segments)
        z0, z1, r0, r1 = radial_extents(loop)
        model["parts"][key] = {
            **PARTS[key],
            "key": key,
            "loop": loop,
            "poly": polygon_of(loop),
            "zmin": z0,
            "zmax": z1,
            "rmin": r0,
            "rmax": r1,
            "length": z1 - z0,
        }
    die = model["parts"]["die"]
    die["bore"] = [finish_feature(row) for row in bore_features(die["loop"])]
    die["od"] = [finish_feature(row) for row in od_features(die["loop"], 20.0)]
    die["facets"] = _facets(die["bore"][0])
    _check(model)
    return model


def _facets(blend):
    edges = blend["edges"]
    stations = [(edges[0]["z0"], edges[0]["r0"])]
    for edge in edges:
        stations.append((edge["z1"], edge["r1"]))
    rows = []
    for i, (z, r) in enumerate(stations):
        angle = None if i == len(stations) - 1 else _angle(z, r, stations[i + 1][0], stations[i + 1][1])
        step = None if i == len(stations) - 1 else abs(stations[i + 1][0] - z)
        rows.append({"i": i + 1, "z": z, "r": r, "d": 2 * r, "depth": -z, "angle": angle, "step": step})
    return rows


def _check(model):
    die = model["parts"]["die"]
    bore_len = sum(row["length"] for row in die["bore"])
    if abs(bore_len - 200.0) > 1e-6:
        raise SystemExit(f"Die bore does not sum to 200 mm ({bore_len})")
    if abs(die["length"] - 200.0) > 1e-9:
        raise SystemExit("Die overall length is not 200 mm")
    if abs(model["parts"]["container"]["length"] - 120.0) > 1e-9:
        raise SystemExit("Container length is not 120 mm")
    if abs(model["parts"]["shrink"]["length"] - 320.0) > 1e-9:
        raise SystemExit("Shrink ring length is not 320 mm")
    if abs(model["parts"]["pin"]["length"] - 75.0) > 1e-9:
        raise SystemExit("Pin length is not 75 mm")
    if abs(model["parts"]["punch"]["length"] - 230.0) > 1e-9:
        raise SystemExit("Punch length is not 230 mm")
    if abs(die["rmax"] - 30.0) > 1e-9 or abs(model["parts"]["shrink"]["rmax"] - 75.0) > 1e-9:
        raise SystemExit("Unexpected outside radius")
    facets = die["facets"]
    if len(facets) != 9:
        raise SystemExit(f"Expected 9 fillet stations, found {len(facets)}")
    expected = [90.0 - 11.25 * (i + 0.5) for i in range(8)]
    for row, ang in zip(facets, expected):
        if row["angle"] is None or abs(row["angle"] - ang) > 1e-4:
            raise SystemExit(f"Fillet facet angle {row['angle']} != {ang}")
    if any(not _on_fillet(row["z"], row["r"]) for row in facets):
        raise SystemExit("A fillet station is not on the R1.500 arc")


def sagitta() -> float:
    return 1.5 * (1.0 - math.cos(math.radians(5.625)))


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

class View:
    """Maps STEP (z, r) onto the sheet. Decreasing Z moves to the right.

    r0 shifts a detail so a local band of radii sits near y0. The axis itself
    stays at r = 0.
    """

    def __init__(self, x_at_z0: float, y0: float, scale: float, r0: float = 0.0):
        self.x0 = x_at_z0
        self.y0 = y0
        self.s = scale
        self.r0 = r0

    def pt(self, z, r=0.0):
        return (self.x0 - z * self.s, self.y0 + (r - self.r0) * self.s)

    def label(self) -> str:
        if self.s >= 1.0 - 1e-9:
            ratio = self.s
            text = f"{ratio:g}" if abs(ratio - round(ratio)) > 1e-6 else f"{ratio:g}"
            return f"SCALE {text} : 1"
        inv = 1.0 / self.s
        return f"SCALE 1 : {inv:g}"


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
    ax.add_patch(Rectangle((6, 6), 408, 285, fill=False, lw=1.1, ec=RULE, zorder=6))
    ax.add_patch(Rectangle((8, 8), 404, 281, fill=False, lw=0.35, ec=RULE, zorder=6))
    T(ax, 12, 286.4, "FERR-CAD-MCF000", size=10.2, weight="bold", va="top")
    T(ax, 62, 286.4, "OP1 FORWARD EXTRUSION DIE   ·   REV 0   ·   NOMINAL", size=7.6, weight="medium", color=DIM, va="top")
    T(ax, 406, 286.4, view_title, size=7.0, weight="medium", ha="right", va="top", color="#333")
    x, y, w, h = 248, 8, 164, 40
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=RULE, lw=0.7, zorder=6))
    ax.plot([x, x + w], [y + 28, y + 28], color=RULE, lw=0.35, zorder=6)
    ax.plot([x, x + w], [y + 16, y + 16], color=RULE, lw=0.35, zorder=6)
    ax.plot([x + 82, x + 82], [y, y + h], color=RULE, lw=0.35, zorder=6)
    T(ax, x + 3, y + 34, "DRAWING", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 24, y + 34, "MCF000-OP1-R0", size=7.2, weight="bold", va="center", z=7)
    T(ax, x + 85, y + 34, "SHEET", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 108, y + 34, f"{page}  /  {TOTAL}", size=7.4, weight="bold", va="center", z=7)
    T(ax, x + 3, y + 22, "PART", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 24, y + 22, "FORWARD EXTRUSION DIE", size=6.4, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 22, "UNITS", size=5.0, color="#666", va="center", z=7)
    T(ax, x + 108, y + 22, "MILLIMETRES", size=6.6, weight="medium", va="center", z=7)
    T(ax, x + 3, y + 8, "DATUM A   ENTRY FACE Z = 0", size=5.2, weight="medium", va="center", z=7)
    T(ax, x + 85, y + 8, "DATUM B   AXIS", size=5.2, weight="medium", va="center", z=7)
    cx, cy = 214, 22
    ax.add_patch(Circle((cx, cy + 8), 3.1, fill=False, lw=0.55, ec=RULE, zorder=6))
    ax.plot([cx + 5.0, cx + 12.2], [cy + 5.4, cy + 6.6], color=RULE, lw=0.5, zorder=6)
    ax.plot([cx + 5.0, cx + 12.2], [cy + 10.6, cy + 9.4], color=RULE, lw=0.5, zorder=6)
    ax.plot([cx + 12.2, cx + 12.2], [cy + 6.6, cy + 9.4], color=RULE, lw=0.5, zorder=6)
    T(ax, cx + 6.2, cy + 1.0, "3RD ANGLE", size=4.3, ha="center", color="#444", z=7)


def footer_note(ax, text):
    T(ax, 12, 14, text, size=5.3, color=MUTED, va="center", z=7)


def fill_profile(ax, view, pts, fc, hatch):
    upper = [view.pt(z, r) for z, r in pts]
    lower = [view.pt(z, -r) for z, r in pts]
    common = dict(closed=True, fc=fc, ec=GEO, lw=0.75, hatch=hatch, joinstyle="round", zorder=2)
    ax.add_patch(Polygon(upper, **common))
    ax.add_patch(Polygon(lower, **common))


def centerline(ax, view, z0, z1, pad=4.0):
    x0, y = view.pt(z0, 0)
    x1, _ = view.pt(z1, 0)
    if x0 > x1:
        x0, x1 = x1, x0
    ax.plot([x0 - pad, x1 + pad], [y, y], color=CENTER, lw=0.45, linestyle=(0, (7, 1.5, 0.9, 1.5)), zorder=3)


def hdim(ax, x1, x2, y, text, ey1=None, ey2=None, text_side="up", min_in=14.0):
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
    if gap >= min_in:
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.2, shrinkA=0, shrinkB=0),
            zorder=4,
        )
        dy = 0.8 if text_side == "up" else -0.7
        va = "bottom" if text_side == "up" else "top"
        T(ax, (x1 + x2) / 2, y + dy, text, size=6.1, weight="medium", color=DIM, ha="center", va=va, z=5, bbox=bbox)
    else:
        wing = 7.0
        ax.plot([x1 - wing, x2 + wing], [y, y], color=DIM, lw=0.45, zorder=4)
        ax.annotate("", xy=(x1, y), xytext=(x1 - wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=4)
        ax.annotate("", xy=(x2, y), xytext=(x2 + wing, y),
                    arrowprops=dict(arrowstyle="->", color=DIM, lw=0.45, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=4)
        T(ax, (x1 + x2) / 2, y + (1.6 if text_side == "up" else -1.6), text, size=5.8, weight="medium",
          color=DIM, ha="center", va="bottom" if text_side == "up" else "top", z=5, bbox=bbox)


def vdim(ax, x, y1, y2, text, ex1=None, ex2=None, text_side="right"):
    if y1 > y2:
        y1, y2 = y2, y1
        ex1, ex2 = ex2, ex1
    over = 1.2
    for y, ex in ((y1, ex1), (y2, ex2)):
        if ex is None:
            continue
        x_end = x + over if x >= ex else x - over
        ax.plot([ex, x_end], [y, y], color=EXT, lw=0.28, zorder=3)
    gap = y2 - y1
    bbox = dict(fc="white", ec="none", pad=0.15, alpha=0.94)
    if gap >= 12:
        ax.annotate(
            "", xy=(x, y2), xytext=(x, y1),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.5, mutation_scale=6.2, shrinkA=0, shrinkB=0),
            zorder=4,
        )
        dx = 1.2 if text_side == "right" else -1.2
        T(ax, x + dx, (y1 + y2) / 2, text, size=6.1, weight="medium", color=DIM,
          ha="left" if text_side == "right" else "right", va="center", z=5, bbox=bbox)
    else:
        T(ax, x + 1.4, (y1 + y2) / 2, text, size=5.8, weight="medium", color=DIM, ha="left", va="center", z=5, bbox=bbox)


def leader(ax, x, y, tx, ty, text, ha="left", size=6.2):
    ax.annotate(
        "", xy=(x, y), xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.4, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.plot([x], [y], marker="o", ms=1.7, color=DIM, zorder=5)
    dx = 1.2 if ha == "left" else (-1.2 if ha == "right" else 0.0)
    T(ax, tx + dx, ty, text, size=size, weight="medium", color=DIM, ha=ha, va="center", z=5,
      bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.94))


def balloon(ax, x, y, label, size=3.2):
    ax.add_patch(Circle((x, y), size, fc="white", ec=DIM, lw=0.7, zorder=6))
    T(ax, x, y, str(label), size=6.6, weight="bold", color=DIM, ha="center", va="center", z=7)


def scale_tag(ax, view, x, y):
    T(ax, x, y, view.label() + "    ·    SECTION THROUGH AXIS", size=5.6, weight="medium", color=MUTED, ha="left", va="center")


def draw_table(ax, x, y_top, widths, headers, rows, row_h=4.6, size=5.5, title=None):
    if title:
        T(ax, x, y_top + 1.2, title, size=6.5, weight="bold", color=DIM, va="bottom")
    total_w = sum(widths)
    head_h = row_h + 0.8
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
    # vertical rules
    cx = x
    for w in widths[:-1]:
        cx += w
        ax.plot([cx, cx], [y, y_top], color=LINE, lw=0.25, zorder=3)
    return y


def panel(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fc=PANEL, ec=LINE, lw=0.5, zorder=1))
    T(ax, x + 2.2, y + h - 3.2, title, size=6.4, weight="bold", color=DIM, va="top", z=2)


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def _dia_pair(d0, d1) -> str:
    if abs(d0 - d1) < 1e-6:
        return fmm(d0)
    return f"{fmm(d0)} / {fmm(d1)}"


def sheet_assembly(pdf, model):
    fig, ax = new_page()
    frame(ax, 1, "SHEET 1    GENERAL ARRANGEMENT")
    parts = model["parts"]
    # 1 : 3. Punch head (Z 262.63) sits at x = 14.
    view = View(14 + 262.63 / 3.0, 232.0, 1.0 / 3.0)
    for key in ("shrink", "die", "container", "pin", "punch"):
        part = parts[key]
        fill_profile(ax, view, part["poly"], part["fc"], part["hatch"])
    centerline(ax, view, 280, -255, pad=2)
    scale_tag(ax, view, 26, 266)
    T(ax, 52, 246, "FLOW  →", size=5.6, weight="bold", color=CENTER)

    x_head, y_head = view.pt(262.63, 20)
    x_back, y_pin = view.pt(-240.0, 10.85)
    hdim(ax, x_head, x_back, 274, "502.630 OVERALL", ey1=y_head, ey2=y_pin, text_side="up")

    def zdim(z0, z1, y, text):
        a, _ = view.pt(z0, 0)
        b, _ = view.pt(z1, 0)
        hdim(ax, a, b, y, text, ey1=204, ey2=204, text_side="down", min_in=16)

    zdim(262.63, 32.63, 196, "230.000 PUNCH")
    zdim(0.0, -200.0, 196, "200.000 DIE")
    zdim(120.0, 0.0, 186, "120.000 CONTAINER")
    zdim(-165.0, -240.0, 186, "75.000 PIN")
    zdim(120.0, -200.0, 176, "320.000 SHRINK RING")

    balloon(ax, *view.pt(-110, 21), "1", size=2.7)
    balloon(ax, *view.pt(55, 21), "2", size=2.7)
    balloon(ax, *view.pt(-50, 52), "3", size=2.7)
    balloon(ax, view.pt(-215, 10.85)[0], view.pt(-215, 10.85)[1] + 9, "4", size=2.7)
    balloon(ax, view.pt(230, 20)[0], view.pt(230, 20)[1] + 9, "5", size=2.7)
    T(ax, *view.pt(-70, 24), "Ø60", size=5.4, weight="bold", color=DIM, ha="center", va="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.9))
    T(ax, *view.pt(10, 52), "Ø150", size=5.4, weight="bold", color=DIM, ha="center", va="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.9))

    # BOM, upper right. Assembly ends near x = 182.
    panel(ax, 196, 176, 210, 96, "PARTS")
    headers = ["", "PART", "MATERIAL", "L", "OD", "ID"]
    widths = [8, 42, 40, 16, 32, 36]
    bom = [
        ["1", "Die insert", "M2/ASP23 62 HRC", "200", "60.000", "22.750–21.860"],
        ["2", "Container", "M2  62 HRC", "120", "60.000", "22.750 / 27.350"],
        ["3", "Shrink ring", "H13  46–48 HRC", "320", "150.000", "60.000"],
        ["4", "Kickout pin", "M2  60 HRC", "75", "21.700 / 21.100", "solid"],
        ["5", "Punch", "M2  60–62 HRC", "230", "40 / 22.700 / 21.700", "solid"],
    ]
    draw_table(ax, 200, 258, widths, headers, bom, row_h=6.0, size=5.15)
    T(ax, 200, 180, "Principal diameters. Every station is on sheets 2–5.", size=5.0, color=MUTED)

    # Clearance table — the pin / insert ID–OD record.
    pin_od = 21.700
    pin_nose = 21.100
    exit_id = 21.860
    cont_id = 22.750
    punch_stem = 22.700
    punch_nose = 21.700
    headers = ["INTERFACE", "MALE Ø", "FEMALE Ø", "Ø CLR.", "RADIAL", "WHERE IT RUNS"]
    widths = [78, 28, 30, 24, 24, 78]
    rows = [
        ["Pin body in die exit bore", fmm(pin_od), fmm(exit_id), fmm(exit_id - pin_od), fmm((exit_id - pin_od) / 2), "Z −165.3 to −200"],
        ["Pin nose in die exit bore", fmm(pin_nose), fmm(exit_id), fmm(exit_id - pin_nose), fmm((exit_id - pin_nose) / 2), "nose face at Z −165"],
        ["Punch stem in container", fmm(punch_stem), fmm(cont_id), fmm(cont_id - punch_stem), fmm((cont_id - punch_stem) / 2), "Z +33.13 to +116"],
        ["Punch nose in container", fmm(punch_nose), fmm(cont_id), fmm(cont_id - punch_nose), fmm((cont_id - punch_nose) / 2), "nose face at Z +32.63"],
        ["Die OD in shrink ring", "60.000", "60.000", "0.000", "0.000", "nominal, line to line"],
        ["Container OD in shrink", "60.000", "60.000", "0.000", "0.000", "nominal, line to line"],
        ["Die guide / container bore", "22.750", "22.750", "0.000", "0.000", "matched at datum A"],
    ]
    draw_table(ax, 12, 158, widths, headers, rows, row_h=5.2, size=5.35,
               title="PIN, INSERT AND PUNCH  —  ID / OD / DIAMETRAL CLEARANCE   (nominal, modelled assembly)")

    headers = ["STATION", "STEP Z", "FROM DATUM A"]
    widths = [62, 28, 36]
    stations = [
        ["Punch head face", "262.630", "+262.630"],
        ["Punch nose face", "32.630", "+32.630"],
        ["Container mouth", "120.000", "+120.000"],
        ["Datum A, entry face", "0.000", "0"],
        ["Die exit face", "−200.000", "depth 200"],
        ["Pin nose face", "−165.000", "depth 165"],
        ["Pin back face", "−240.000", "depth 240"],
    ]
    # The clearance table is full width, so the station list sits in the note band
    # only if it fits. It is placed to the right of a shortened clearance table
    # instead — redraw is avoided by keeping stations in the notes below.
    T(ax, 12, 108, "AXIAL STATIONS", size=6.2, weight="bold", color=DIM, va="center")
    draw_table(ax, 12, 104, widths, headers, stations, row_h=4.35, size=5.3)

    T(ax, 150, 100,
      "Pin insertion past the die exit face is 35.000 (nose) and the stickout is 40.000.\n"
      "The pin stays in the Ø21.860 exit bore. It does not enter the Ø21.800 land.\n"
      "Punch nose is 32.630 inside the container from datum A. Stem clearance on Ø22.750 is 0.050.\n"
      "At datum A the container face continues in to Ø22.750. The die opening there is Ø25.750,\n"
      "so the container leaves a 1.500 radial lip, equal to the entry radius, inside the die mouth.\n"
      "Both inserts carry a 1 × 45° OD chamfer, so the joint under the shrink ring is a 2 mm vee.",
      size=5.5, color=INK, va="top", linespacing=1.35)

    footer_note(ax, "NOMINAL GEOMETRY FROM THE STEP FILE   ·   NO TOLERANCES OR FINISH ARE STORED IN THE MODEL")
    _save(pdf, fig, 1)


def _feature_rows(features, depth=True):
    rows = []
    for i, row in enumerate(features, start=1):
        angle = "R1.500" if row["kind"] == "blend" else ("—" if row["angle"] < 0.02 else fang(row["angle"]))
        slant = "—" if row["angle"] < 0.02 or row["kind"] == "blend" else fmm(row["slant"])
        if depth:
            rows.append([
                str(i), row["name"], fmm(row["depth0"]), fmm(row["depth1"]), fmm(row["length"]),
                fmm(row["d0"]), fmm(row["d1"]), angle, slant, row["note"],
            ])
        else:
            rows.append([
                str(i), row["name"], fmm(row["z0"]), fmm(row["z1"]), fmm(row["length"]),
                fmm(row["d0"]), fmm(row["d1"]), angle, slant, row["note"],
            ])
    return rows


def sheet_die(pdf, model):
    fig, ax = new_page()
    frame(ax, 2, "SHEET 2    DIE INSERT MCF000")
    die = model["parts"]["die"]
    view = View(22.0, 198.0, 1.5)
    fill_profile(ax, view, die["poly"], die["fc"], die["hatch"])
    centerline(ax, view, 8, -208, pad=2)
    scale_tag(ax, view, 36, 256)

    x0, y0 = view.pt(0, 0)
    x1, _ = view.pt(-200, 0)
    hdim(ax, x0, x1, 264, "200.000", ey1=view.pt(0, 30)[1], ey2=view.pt(-200, 30)[1])
    y_od_hi = view.pt(-200, 30)[1]
    y_od_lo = view.pt(-200, -30)[1]
    vdim(ax, x1 + 12, y_od_lo, y_od_hi, "Ø60.000", ex1=x1, ex2=x1, text_side="right")

    leader(ax, *view.pt(-0.5, 29.5), x0 + 18, 252, "1 × 45°  BOTH ENDS", ha="left", size=5.7)
    leader(ax, *view.pt(0, 12.875), x0 + 16, view.pt(0, 8)[1], "Ø25.750 OPENING", ha="left", size=5.7)
    leader(ax, *view.pt(-200, 10.93), x1 - 18, view.pt(-200, 6)[1], "Ø21.860 EXIT", ha="right", size=5.7)
    leader(ax, *view.pt(-0.75, 12.1), x0 + 26, view.pt(-4, 16)[1], "R1.500  → SHEET 3", ha="left", size=5.7)
    T(ax, view.pt(-100, 0)[0], y0 + 7, "M2 / ASP23   62 HRC", size=6.3, weight="medium", color=DIM, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.85))
    T(ax, view.pt(-100, 0)[0], y0 - 7, "FULL BORE SCHEDULE ON SHEET 3", size=5.4, weight="medium", color=MUTED, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.85))

    # Entry blend, 14 : 1, radii measured from r = 11 so the detail stays on the sheet.
    det = View(18.0, 64.0, 14.0, r0=11.0)
    facets = die["facets"]
    outline = [(0.0, 12.875), (0.0, 13.6)]
    outline += [(row["z"], row["r"]) for row in reversed(facets)]
    # facets run entry (z=0) to guide (z=-1.5). reversed() would draw guide first.
    outline = [(0.15, 13.15), (0.0, 13.15), (0.0, 12.875)]
    outline += [(row["z"], row["r"]) for row in facets]
    outline += [(-2.3, 11.375)]
    xs, ys = zip(*(det.pt(z, r) for z, r in outline))
    ax.plot(xs, ys, color=GEO, lw=1.05, solid_capstyle="round", zorder=3)
    arc = []
    for i in range(49):
        th = math.radians(90.0 * i / 48.0)
        z = -1.5 + 1.5 * math.cos(th)
        r = 12.875 - 1.5 * math.sin(th)
        arc.append(det.pt(z, r))
    ax.plot([p[0] for p in arc], [p[1] for p in arc], color=CENTER, lw=0.45, linestyle=(0, (3, 1.4)), zorder=2)
    x_a, y_a = det.pt(0, 11.375)
    x_b, y_b = det.pt(-1.5, 11.375)
    hdim(ax, x_a, x_b, 58, "1.500", ey1=y_a, ey2=y_b, text_side="down", min_in=12)
    leader(ax, *det.pt(0, 12.875), det.pt(0.15, 13.15)[0], det.pt(0, 13.15)[1] + 2, "Ø25.750", ha="left", size=5.6)
    leader(ax, *det.pt(-1.6, 11.375), det.pt(-2.1, 11.375)[0], det.pt(-1.6, 11.8)[1], "Ø22.750", ha="left", size=5.6)
    T(ax, det.pt(-0.55, 13.05)[0], det.pt(-0.55, 13.05)[1], "R1.500", size=6.0, weight="bold", color=CENTER, ha="center")
    T(ax, 16, 112, "ENTRY BLEND    14 : 1", size=6.2, weight="bold", color=DIM)
    T(ax, 16, 107, "Dashed arc is the true R1.500. Solid line is the 8 facets.", size=5.0, color=MUTED)

    # Chamfer detail, 12 : 1, local to the OD corner.
    ch = View(155.0, 68.0, 12.0, r0=28.6)
    chamfer = [(0.45, 29.0), (0.0, 29.0), (-1.0, 30.0), (-1.55, 30.0)]
    xs, ys = zip(*(ch.pt(z, r) for z, r in chamfer))
    ax.plot(xs, ys, color=GEO, lw=1.05, zorder=3)
    x_face, y_face = ch.pt(0, 29)
    x_od, y_od = ch.pt(-1, 30)
    hdim(ax, x_face, x_od, 60, "1.000", ey1=y_od, ey2=y_od, text_side="down", min_in=8)
    vdim(ax, ch.pt(0.45, 29)[0] - 1, y_face, ch.pt(0, 30)[1], "1.000",
         ex1=x_face, ex2=ch.pt(0, 30)[0], text_side="left")
    leader(ax, *ch.pt(-0.5, 29.5), ch.pt(-0.5, 29.5)[0] + 6, ch.pt(-0.5, 30.2)[1], "45°", ha="left", size=5.8)
    T(ax, 148, 112, "OD CHAMFER    12 : 1    BOTH ENDS", size=6.2, weight="bold", color=DIM)
    T(ax, 148, 107, "Face Ø58.000     OD Ø60.000", size=5.3, color=INK)

    headers = ["#", "FEATURE", "Ø FROM", "Ø TO", "LENGTH"]
    widths = [8, 36, 22, 20, 22]
    bore_rows = []
    for i, row in enumerate(die["bore"], start=1):
        bore_rows.append([str(i), row["name"], fmm(row["d0"]), fmm(row["d1"]), fmm(row["length"])])
    draw_table(ax, 268, 140, widths, headers, bore_rows, row_h=4.5, size=5.15,
               title="BORE, ENTRY TO EXIT")
    T(ax, 268, 62, "Angles, slant lengths and the eight facets are on sheet 3.", size=5.1, color=MUTED)

    footer_note(ax, "DEPTH INTO THE DIE IS −Z FROM DATUM A.   FLOW IS LEFT TO RIGHT ON THIS SHEET.")
    _save(pdf, fig, 2)


def _map_widths(features):
    widths = []
    for row in features:
        if row["kind"] == "blend":
            widths.append(34)
        elif row["length"] < 0.3:
            widths.append(22)
        elif row["length"] < 1.0:
            widths.append(26)
        elif row["length"] < 5:
            widths.append(24)
        else:
            widths.append(38)
    return widths


def draw_bore_map(ax, features, x_left, y_cl, width):
    """Not-to-scale bore profile. Long lands are shortened; lengths are true."""
    raw = _map_widths(features)
    scale = width / sum(raw)
    widths = [w * scale for w in raw]
    radii = sorted({round(row["r0"], 5) for row in features} | {round(row["r1"], 5) for row in features})
    gap = 5.4

    def lane(r):
        if r <= radii[0]:
            return y_cl
        if r >= radii[-1]:
            return y_cl + (len(radii) - 1) * gap
        for a, b in zip(radii, radii[1:]):
            if a - 1e-9 <= r <= b + 1e-9:
                span = b - a
                t = 0.0 if span < 1e-12 else (r - a) / span
                ia = radii.index(a)
                return y_cl + (ia + t) * gap
        return y_cl

    # Profile polyline in display space.
    cursor = x_left
    pts = [(cursor, lane(features[0]["r0"]))]
    ticks = [cursor]
    mids = []
    for row, w in zip(features, widths):
        if row["kind"] == "blend":
            samples = row["edges"]
            # edges run entry → exit; sample both ends of each
            zs = [samples[0]["z0"]] + [e["z1"] for e in samples]
            rs = [samples[0]["r0"]] + [e["r1"] for e in samples]
            z_span = zs[0] - zs[-1]
            for z, r in zip(zs[1:], rs[1:]):
                cursor_x = x_left + (zs[0] - z) / z_span * w
                pts.append((cursor_x, lane(r)))
            cursor = x_left + w
        else:
            cursor += w
            pts.append((cursor, lane(row["r1"])))
        ticks.append(cursor)
        mids.append((ticks[-2] + ticks[-1]) / 2)
        # x_left advances only for the blend's local origin; keep a running left.
        if row["kind"] == "blend":
            x_left = cursor
    xs, ys = zip(*pts)
    ax.plot([ticks[0], ticks[-1]], [y_cl - 3.2, y_cl - 3.2], color=CENTER, lw=0.4, linestyle=(0, (6, 1.4, 0.8, 1.4)), zorder=2)
    ax.plot(xs, ys, color=GEO, lw=1.15, solid_joinstyle="round", zorder=3)
    for x in ticks:
        ax.plot([x, x], [y_cl - 3.2, max(ys) + 1.2], color=EXT, lw=0.25, zorder=1)
    for row, mid, w in zip(features, mids, widths):
        if row["kind"] == "blend":
            label = "R1.500"
        elif row["angle"] < 0.02:
            label = "Ø" + fmm(row["d0"])
        else:
            label = fang(row["angle"])
        T(ax, mid, lane(max(row["r0"], row["r1"])) + 3.3, label, size=5.5, weight="bold", color=DIM, ha="center", va="bottom",
          bbox=dict(fc="white", ec="none", pad=0.08, alpha=0.9))
        T(ax, mid, y_cl - 6.4, fmm(row["length"]), size=5.3, weight="mono", color=DIM, ha="center", va="top")
        T(ax, mid, y_cl - 10.2, str(features.index(row) + 1), size=5.0, weight="bold", color=MUTED, ha="center", va="top")
    T(ax, ticks[0], y_cl - 14.5, "ENTRY  DEPTH 0", size=5.2, weight="medium", color=MUTED, ha="left")
    T(ax, ticks[-1], y_cl - 14.5, "EXIT  DEPTH 200", size=5.2, weight="medium", color=MUTED, ha="right")
    return y_cl - 16


def sheet_bore(pdf, model):
    fig, ax = new_page()
    frame(ax, 3, "SHEET 3    DIE INSERT BORE — ALL STATIONS")
    die = model["parts"]["die"]
    T(ax, 14, 276, "BORE MAP    NOT TO SCALE    ·    LONG PARALLELS SHORTENED    ·    LENGTHS AND DIAMETERS ARE TRUE", size=6.0, weight="medium", color=DIM)
    T(ax, 14, 270.5, "Numbers under the map match the station table. Depth is measured from datum A toward the exit.", size=5.3, color=MUTED)
    draw_bore_map(ax, die["bore"], 16, 214, 384)

    headers = ["#", "FEATURE", "DEPTH FROM", "DEPTH TO", "LENGTH", "Ø FROM", "Ø TO", "ANGLE", "SLANT", "NOTE"]
    widths = [8, 36, 24, 22, 22, 20, 18, 18, 18, 78]
    bottom = draw_table(ax, 12, 186, widths, headers, _feature_rows(die["bore"]), row_h=5.05, size=5.35,
                        title="BORE STATIONS    DEPTH = −Z    DATUM A AT THE ENTRY FACE")

    headers = ["#", "FEATURE", "DEPTH FROM", "DEPTH TO", "LENGTH", "Ø FROM", "Ø TO", "ANGLE"]
    widths = [8, 36, 24, 22, 20, 20, 18, 18]
    od_rows = []
    for i, row in enumerate(die["od"], start=1):
        angle = "—" if row["angle"] < 0.02 else fang(row["angle"])
        od_rows.append([str(i), row["name"], fmm(row["depth0"]), fmm(row["depth1"]), fmm(row["length"]),
                        fmm(row["d0"]), fmm(row["d1"]), angle])
    draw_table(ax, 12, bottom - 8, widths, headers, od_rows, row_h=4.8, size=5.35,
               title="OUTSIDE DIAMETER")

    headers = ["STA", "DEPTH", "Ø", "STEP", "FACET ANGLE", "STEP Z"]
    widths = [10, 20, 22, 18, 26, 24]
    facet_rows = []
    for row in die["facets"]:
        facet_rows.append([
            str(row["i"]),
            fmm(row["depth"]),
            fmm(row["d"]),
            "—" if row["step"] is None else fmm(row["step"]),
            "—" if row["angle"] is None else fang(row["angle"]),
            fmm(row["z"]),
        ])
    draw_table(ax, 250, bottom - 8, widths, headers, facet_rows, row_h=4.15, size=5.2,
               title="ENTRY BLEND FACETS    R1.500")

    T(ax, 12, 58,
      "The nine facet stations lie on a circular arc of R1.500 centred at depth 1.500, radius 12.875.\n"
      "The arc is tangent to the entry face and to the Ø22.750 guide bore. Facet angles step by 11.250°.\n"
      f"Each facet is a straight cone, so the surface is inside that arc by up to {sagitta():.5f} mm.\n"
      "Reduction angles are 30° from the axis (60° included). Axial length = ΔR × √3.\n"
      "Reliefs open the bore 0.060 on diameter over 0.150 axial (1:5, 11.310° from the axis).",
      size=5.5, color=INK, va="bottom", linespacing=1.32)
    footer_note(ax, "ANGLE IS FROM THE AXIS.   SLANT IS THE GENERATOR LENGTH.   Ø CLR. OF THE PIN IS ON SHEET 1.")
    _save(pdf, fig, 3)


def _part_leaders_container(ax, view, part):
    x_mouth, _ = view.pt(120, 0)
    x_face, _ = view.pt(0, 0)
    y_od = view.pt(0, 30)[1]
    y_bot = view.pt(0, -30)[1]
    hdim(ax, x_mouth, x_face, y_od + 12, "120.000", ey1=y_od, ey2=y_od)
    vdim(ax, x_face + 14, view.pt(0, -30)[1], y_od, "Ø60.000", ex1=x_face, ex2=x_face)
    x_taper, _ = view.pt(116, 0)
    hdim(ax, x_face, x_taper, y_bot - 8, "116.000 ID", ey1=y_bot, ey2=y_bot, text_side="down")
    leader(ax, *view.pt(60, 11.375), view.pt(60, 0)[0], view.pt(60, 0)[1] + 4, "Ø22.750", ha="center", size=5.8)
    leader(ax, *view.pt(118, 12.6), x_mouth + 2, view.pt(118, 18)[1], "Ø27.350   4.000 × 29.899°", ha="left", size=5.5)
    leader(ax, *view.pt(1.0, 29.4), view.pt(10, 29.4)[0], y_od + 5, "1 × 45° BOTH ENDS", ha="left", size=5.6)
    leader(ax, *view.pt(0, 29), x_face - 2, y_od - 6, "Ø58 FACE", ha="right", size=5.5)


def _part_leaders_shrink(ax, view):
    x_l, _ = view.pt(120, 0)
    x_r, _ = view.pt(-200, 0)
    y_od = view.pt(0, 75)[1]
    y_bot = view.pt(0, -75)[1]
    hdim(ax, x_l, x_r, y_od + 11, "320.000", ey1=y_od, ey2=y_od)
    vdim(ax, x_r + 14, y_bot, y_od, "Ø150.000", ex1=x_r, ex2=x_r)
    leader(ax, *view.pt(-40, 30), view.pt(-40, 0)[0], view.pt(-40, 8)[1], "Ø60.000 ID", ha="center", size=5.7)
    leader(ax, *view.pt(118, 74), view.pt(108, 74)[0], y_od + 4, "2 × 45° BOTH ENDS", ha="right", size=5.6)
    leader(ax, *view.pt(120, 73), x_l + 8, view.pt(120, 55)[1], "Ø146 FACE", ha="left", size=5.5)
    leader(ax, *view.pt(-200, 73), x_r - 6, view.pt(-200, 55)[1], "Ø146 FACE", ha="right", size=5.5)


def sheet_container_shrink(pdf, model):
    fig, ax = new_page()
    frame(ax, 4, "SHEET 4    CONTAINER INSERT AND SHRINK RING")
    container = model["parts"]["container"]
    shrink = model["parts"]["shrink"]

    # Mouth (Z 120) at the left, datum A at the right. Scale 1 : 1.
    cview = View(24 + 120.0, 224.0, 1.0)
    fill_profile(ax, cview, container["poly"], container["fc"], container["hatch"])
    centerline(ax, cview, 128, -8, pad=2)
    scale_tag(ax, cview, cview.pt(120, 0)[0], 276)
    _part_leaders_container(ax, cview, container)
    T(ax, cview.pt(70, 0)[0], cview.pt(70, 0)[1] - 6, "CONTAINER INSERT   M2  62 HRC", size=6.0, weight="medium", color=DIM, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.88))

    # Scale 1 : 2. Mouth at the left.
    sview = View(24 + 120 * 0.5, 104.0, 0.5)
    fill_profile(ax, sview, shrink["poly"], shrink["fc"], shrink["hatch"])
    centerline(ax, sview, 132, -212, pad=2)
    scale_tag(ax, sview, sview.pt(120, 0)[0], 164)
    _part_leaders_shrink(ax, sview)
    T(ax, sview.pt(-40, 0)[0], sview.pt(-40, 0)[1] - 8, "SHRINK RING   H13  46–48 HRC", size=6.0, weight="medium", color=DIM, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.88))

    # Dimension tables
    def simple_rows(loop, r_split):
        rows = []
        items = []
        for a, b, kind in loop:
            if kind == "face":
                z = a[0]
                items.append((z, "Face", fmm(z), "0", fmm(2 * min(a[1], b[1])), fmm(2 * max(a[1], b[1])), "—"))
                continue
            if kind == "disk":
                continue
            z0, r0, z1, r1 = (a[0], a[1], b[0], b[1]) if a[0] >= b[0] else (b[0], b[1], a[0], a[1])
            if min(r0, r1) < r_split and max(r0, r1) < r_split + 5 and kind != "cone":
                side = "ID"
            elif min(r0, r1) >= r_split:
                side = "OD"
            else:
                side = "ID" if max(r0, r1) < 40 else "OD"
            ang = _angle(z0, r0, z1, r1)
            name = side + (" chamfer" if abs(ang - 45) < 0.05 else (" taper" if ang >= 0.05 else ""))
            items.append((z0, name.strip(), fmm(z0), fmm(abs(z0 - z1)), fmm(2 * r0), fmm(2 * r1), "—" if ang < 0.02 else fang(ang)))
        items.sort(key=lambda item: -item[0])
        for i, item in enumerate(items, start=1):
            rows.append([str(i), item[1], item[2], item[3], item[4], item[5], item[6]])
        return rows

    headers = ["#", "FEATURE", "Z", "LENGTH", "Ø FROM", "Ø TO", "ANGLE"]
    widths = [8, 28, 22, 18, 20, 20, 22]
    draw_table(ax, 250, 250, widths, headers, simple_rows(container["loop"], 20), row_h=4.5, size=5.15,
               title="CONTAINER — ALL SEGMENTS")
    draw_table(ax, 250, 150, widths, headers, simple_rows(shrink["loop"], 50), row_h=4.5, size=5.15,
               title="SHRINK RING — ALL SEGMENTS")

    footer_note(ax, "SHRINK RING IS FLUSH WITH THE CONTAINER MOUTH AND THE DIE EXIT.   NOMINAL Ø60 IS LINE-TO-LINE WITH BOTH INSERTS.")
    _save(pdf, fig, 4)


def sheet_pin_punch(pdf, model):
    fig, ax = new_page()
    frame(ax, 5, "SHEET 5    KICKOUT PIN AND EXTRUSION PUNCH")
    pin = model["parts"]["pin"]
    punch = model["parts"]["punch"]

    # Nose on the left (toward the die), back on the right. Scale 5 : 2.
    pview = View(22 + (-165.0) * 2.5, 226.0, 2.5)
    fill_profile(ax, pview, pin["poly"], pin["fc"], pin["hatch"])
    centerline(ax, pview, -158, -248, pad=2)
    scale_tag(ax, pview, 22, 276)
    x_nose, _ = pview.pt(-165, 0)
    x_back, _ = pview.pt(-240, 0)
    y_od = pview.pt(-200, 10.85)[1]
    y_bot = pview.pt(-200, -10.85)[1]
    hdim(ax, x_nose, x_back, y_od + 8, "75.000", ey1=y_od, ey2=y_od)
    x_body, _ = pview.pt(-165.3, 0)
    hdim(ax, x_body, x_back, y_bot - 8, "74.700 BODY", ey1=y_bot, ey2=y_bot, text_side="down")
    vdim(ax, x_back + 12, y_bot, y_od, "Ø21.700", ex1=x_back, ex2=x_back)
    leader(ax, *pview.pt(-165, 10.55), x_nose + 8, y_od - 4, "Ø21.100 NOSE", ha="left", size=5.6)
    leader(ax, *pview.pt(-165.15, 10.7), x_nose + 14, y_od + 3, "0.300 × 45°", ha="left", size=5.5)
    T(ax, pview.pt(-202, 0)[0], pview.pt(-202, 0)[1] - 5, "KICKOUT PIN    M2  60 HRC    SOLID", size=6.0, weight="medium", color=DIM, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.9))

    # Punch at 1 : 1. Head on the left, nose on the right.
    uview = View(28 + 262.63, 102.0, 1.0)
    fill_profile(ax, uview, punch["poly"], punch["fc"], punch["hatch"])
    centerline(ax, uview, 272, 24, pad=2)
    scale_tag(ax, uview, 28, 146)
    x_head, _ = uview.pt(262.63, 0)
    x_pnose, _ = uview.pt(32.63, 0)
    y_head = uview.pt(200, 20)[1]
    y_head_bot = uview.pt(200, -20)[1]
    hdim(ax, x_head, x_pnose, y_head + 12, "230.000", ey1=y_head, ey2=uview.pt(32.63, 10.85)[1])
    x_t0, _ = uview.pt(168.63, 0)
    x_t1, _ = uview.pt(162.63, 0)
    x_stem, _ = uview.pt(33.13, 0)
    hdim(ax, x_head, x_t0, y_head_bot - 8, "94.000 HEAD", ey1=y_head_bot, ey2=uview.pt(168.63, -20)[1], text_side="down")
    hdim(ax, x_t1, x_stem, y_head_bot - 18, "129.500 STEM", ey1=uview.pt(162.63, -11.35)[1], ey2=uview.pt(33.13, -11.35)[1], text_side="down")
    leader(ax, *uview.pt(165.6, 16), uview.pt(150, 16)[0], y_head + 4, "6.000 × 55.253°", ha="right", size=5.5)
    T(ax, uview.pt(220, 0)[0], uview.pt(220, 0)[1], "Ø40.000", size=6.0, weight="bold", color=DIM, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.9))
    vdim(ax, uview.pt(90, 0)[0], uview.pt(90, -11.35)[1], uview.pt(90, 11.35)[1], "Ø22.700",
         ex1=uview.pt(90, 11.35)[0], ex2=uview.pt(90, 11.35)[0])
    leader(ax, *uview.pt(32.63, 10.85), x_pnose + 3, uview.pt(32.63, 10.85)[1] + 6, "Ø21.700 NOSE", ha="left", size=5.6)
    leader(ax, *uview.pt(33.0, 11.15), uview.pt(42, 14)[0], uview.pt(42, 14)[1], "0.500 × 45°", ha="left", size=5.5)
    T(ax, uview.pt(130, 0)[0], uview.pt(130, 0)[1] - 5, "PUNCH  M2  60–62 HRC", size=5.6, weight="medium", color=DIM, ha="center",
      bbox=dict(fc="white", ec="none", pad=0.12, alpha=0.9))

    panel(ax, 248, 188, 158, 78, "PIN AGAINST DIE EXIT BORE")
    T(ax, 254, 254,
      "Die exit bore ID        Ø21.860\n"
      "Pin body OD             Ø21.700\n"
      "Diametral clearance     0.160\n"
      "Radial clearance        0.080\n"
      "Nose face OD            Ø21.100\n"
      "Nose diametral clr.     0.760\n"
      "In the die / stickout   35.000 / 40.000\n"
      "Does not enter Ø21.800 land",
      size=5.5, weight="mono", color=INK, va="top", linespacing=1.28)

    panel(ax, 300, 56, 106, 78, "PUNCH AGAINST CONTAINER")
    T(ax, 304, 124,
      "Container ID       Ø22.750\n"
      "Stem OD            Ø22.700\n"
      "Stem Ø clearance   0.050\n"
      "Radial             0.025\n"
      "Nose OD            Ø21.700\n"
      "Nose Ø clearance   1.050\n"
      "Nose from datum A  32.630\n"
      "Taper              6.000 long\n"
      "Ø22.700 → Ø40.000\n"
      "55.253° from axis\n"
      "110.506° included",
      size=5.15, weight="mono", color=INK, va="top", linespacing=1.22)

    footer_note(ax, "PIN NOSE POINTS TOWARD THE DIE (LEFT ON THIS SHEET).   PUNCH NOSE POINTS TOWARD THE DIE (RIGHT).")
    _save(pdf, fig, 5)


def _save(pdf, fig, page):
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW / f"sheet-{page}.png", dpi=110)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    step = parse_step(STEP_PATH)
    if "MILLI" not in step["entities"].get(36, "SI_UNIT(.MILLI.,.METRE.)"):
        # Unit record is entity 36 in this file; also accept the token anywhere.
        blob = " ".join(step["entities"].values())
        if "SI_UNIT(.MILLI.,.METRE.)" not in blob:
            raise SystemExit("STEP length unit is not millimetres")
    model = build_model(step)
    with PdfPages(OUT_PATH) as pdf:
        pdf.infodict()["Title"] = "FERR-CAD-MCF000 OP1 extrusion die R0 — nominal dimensions"
        pdf.infodict()["Subject"] = f"{step['filename']}  {step['time']}"
        sheet_assembly(pdf, model)
        sheet_die(pdf, model)
        sheet_bore(pdf, model)
        sheet_container_shrink(pdf, model)
        sheet_pin_punch(pdf, model)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
