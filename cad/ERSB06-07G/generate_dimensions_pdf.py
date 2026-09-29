#!/usr/bin/env python3
"""Build a millimetre dimension-point PDF from the ERSB06-07G blank STEP model."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
)

ROOT = Path(__file__).resolve().parent
STEP_PATH = ROOT / "BLANK_ERSB06-07G_PAN_EXTRUDED.step"
PDF_PATH = ROOT / "BLANK_ERSB06-07G_PAN_EXTRUDED_DIMENSIONS.pdf"

NAVY = colors.Color(0.09, 0.16, 0.32)
RED = colors.Color(0.70, 0.12, 0.12)
BLUE = colors.Color(0.08, 0.28, 0.55)
TEAL = colors.Color(0.05, 0.38, 0.38)
RULE = colors.Color(0.75, 0.78, 0.82)
ZEBRA = colors.Color(0.94, 0.95, 0.97)
FILL = colors.Color(0.90, 0.92, 0.94)
INK = colors.Color(0.10, 0.12, 0.16)
MUTED = colors.Color(0.35, 0.38, 0.44)

PAGE_W, PAGE_H = A4


def parse_step(path: Path):
    raw = path.read_text()
    header = raw.split("DATA;", 1)[0]
    data = raw.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    file_name = re.search(r"FILE_NAME\('([^']*)'", header)
    file_time = re.search(r"FILE_NAME\('[^']*','([^']*)'", header)
    entities = {}
    buf = []
    for line in data.splitlines():
        buf.append(line.strip())
        if buf[-1].endswith(";"):
            chunk = " ".join(buf)[:-1].strip()
            buf = []
            m = re.match(r"#(\d+)\s*=\s*(.*)$", chunk)
            if m:
                entities[int(m.group(1))] = m.group(2)
    if buf:
        raise SystemExit("Unterminated STEP entity")
    return {
        "name": file_name.group(1) if file_name else "",
        "time": file_time.group(1) if file_time else "",
        "entities": entities,
    }


def cartesian_points(entities):
    points = {}
    for eid, body in entities.items():
        if not body.startswith("CARTESIAN_POINT"):
            continue
        m = re.search(r"CARTESIAN_POINT\('([^']*)',\(([^)]*)\)\s*\)", body)
        if not m:
            raise SystemExit(f"Unparsed point #{eid}: {body}")
        nums = tuple(tok.strip() for tok in m.group(2).split(","))
        coords = tuple(float(tok) for tok in nums)
        points[eid] = {"raw": nums, "xyz": coords}
    return points


def first_ref(body):
    m = re.search(r"#(\d+)", body)
    return int(m.group(1)) if m else None


def refs(body):
    return [int(n) for n in re.findall(r"#(\d+)", body)]


def near(a, b, tol=1e-9):
    return abs(a - b) <= tol


def clean(v, tol=1e-9):
    if abs(v) <= tol:
        return 0.0
    return v


def fmt(v, digits=6):
    v = clean(v)
    s = f"{v:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if "." not in s:
        s += ".000"
    else:
        frac = s.split(".", 1)[1]
        if len(frac) < 3:
            s += "0" * (3 - len(frac))
    return s


def fmt_raw(raw_tokens):
    parts = []
    for tok in raw_tokens:
        val = float(tok)
        if abs(val) < 1e-9:
            parts.append("0")
        else:
            parts.append(tok.replace(" ", ""))
    return ", ".join(parts)


def hypot2(x, y):
    return math.hypot(x, y)


class Model:
    def __init__(self, step):
        self.step = step
        self.entities = step["entities"]
        self.points = cartesian_points(self.entities)
        self.roles = defaultdict(set)
        self.feature_of_point = defaultdict(list)
        self._index()
        self.vertices = self._vertices()
        self.circles = self._circles()
        self.surfaces = self._surfaces()
        self.profile = self._profile()
        self._check()

    def _index(self):
        for eid, body in self.entities.items():
            kind = body.split("(", 1)[0].strip()
            if kind == "VERTEX_POINT":
                self.roles[first_ref(body)].add("vertex")
            elif kind == "LINE":
                self.roles[first_ref(body)].add("line point")
            elif kind == "AXIS2_PLACEMENT_3D":
                self.roles[first_ref(body)].add("origin")
            elif kind.startswith("B_SPLINE") or "B_SPLINE_CURVE" in kind or "B_SPLINE_SURFACE" in body[:40]:
                pass
            if "B_SPLINE" in body:
                for rid in refs(body):
                    if rid in self.points and len(self.points[rid]["xyz"]) == 3:
                        self.roles[rid].add("spline control")

    def _placement(self, pid):
        body = self.entities[pid]
        r = refs(body)
        origin = self.points[r[0]]["xyz"]
        axis = self.points[r[1]]["xyz"] if False else None
        # directions are DIRECTION entities, not points
        dirs = []
        for rid in r[1:]:
            db = self.entities[rid]
            m = re.search(r"DIRECTION\('([^']*)',\(([^)]*)\)\)", db)
            if m:
                dirs.append(tuple(float(t) for t in m.group(2).split(",")))
        return origin, dirs

    def _vertices(self):
        rows = []
        for eid, body in self.entities.items():
            if not body.startswith("VERTEX_POINT"):
                continue
            pid = first_ref(body)
            x, y, z = self.points[pid]["xyz"]
            rows.append(
                {
                    "vertex": eid,
                    "point": pid,
                    "x": clean(x),
                    "y": clean(y),
                    "z": clean(z),
                    "r": hypot2(clean(x), clean(y)),
                    "raw": self.points[pid]["raw"],
                }
            )
        rows.sort(key=lambda p: (p["z"], -p["r"]))
        return rows

    def _circles(self):
        rows = []
        for eid, body in self.entities.items():
            if not body.startswith("CIRCLE"):
                continue
            m = re.search(r"CIRCLE\('([^']*)',#(\d+),([^)]+)\)", body)
            place = int(m.group(2))
            radius = float(m.group(3))
            pbody = self.entities[place]
            if not pbody.startswith("AXIS2_PLACEMENT_3D"):
                continue
            origin, dirs = self._placement(place)
            rows.append(
                {
                    "id": eid,
                    "radius": radius,
                    "origin": tuple(clean(v) for v in origin),
                    "axis": tuple(clean(v) for v in dirs[0]),
                    "point": refs(pbody)[0],
                }
            )
        return rows

    def _surfaces(self):
        rows = []
        specs = {
            "SPHERICAL_SURFACE": 1,
            "CYLINDRICAL_SURFACE": 1,
            "CONICAL_SURFACE": 2,
            "TOROIDAL_SURFACE": 2,
            "PLANE": 0,
        }
        for eid, body in self.entities.items():
            kind = body.split("(", 1)[0].strip()
            if kind not in specs:
                continue
            place = first_ref(body)
            origin, dirs = self._placement(place)
            nums = [float(t) for t in re.findall(r",\s*([+-]?\d+(?:\.\d+)?(?:E[+-]?\d+)?)\s*\)", body)]
            # last numeric args sit before the final paren; grab trailing numbers instead
            tail = re.search(r"#\d+,(.*)\)$", body)
            params = []
            if tail:
                params = [float(t.strip()) for t in tail.group(1).split(",") if t.strip()]
            rows.append(
                {
                    "id": eid,
                    "kind": kind,
                    "origin": tuple(clean(v) for v in origin),
                    "axis": tuple(clean(v) for v in dirs[0]) if dirs else None,
                    "params": params,
                }
            )
        return rows

    def _profile(self):
        """Seam vertices on +X, ordered along the boundary from the bottom pole."""
        pts = []
        for v in self.vertices:
            if abs(v["y"]) > 1e-6 or v["x"] < -1e-6:
                continue
            pts.append(v)
        # Boundary order is not pure Z order: at Z=0 the outer point precedes the inner.
        by_key = {(round(p["x"], 6), round(p["z"], 6)): p for p in pts}
        order = [
            (0.0, -3.7),
            (4.578947, -2.259973),
            (4.75, -1.931974),
            (4.75, 0.0),
            (2.75, 0.0),
            (2.45, 0.3),
            (2.45, 8.579338),
            (2.189810, 9.03),
            (2.189810, 43.84),
            (1.689810, 44.34),
        ]
        ordered = []
        for key in order:
            match = None
            for p in pts:
                if abs(p["x"] - key[0]) < 5e-4 and abs(p["z"] - key[1]) < 5e-4:
                    match = p
                    break
            if match is None:
                raise SystemExit(f"Profile vertex not found for {key}")
            ordered.append(match)
        labels = [
            ("P01", "Bottom pole", "Single point on the axis. South pole of the SR8 sphere."),
            ("P02", "Sphere / R0.4 blend", "Tangency of the spherical base and the R0.400 blend."),
            ("P03", "Blend / outside cylinder", "Tangency of the R0.400 blend and the Ø9.500 cylinder."),
            ("P04", "Flange, outside", "Top of the outside cylinder. Outer corner of the Z0 flange face."),
            ("P05", "Flange, inside", "Inner corner of the Z0 face, start of the R0.300 stem round."),
            ("P06", "Round / stem cylinder", "End of the R0.300 round. Start of the Ø4.900 stem."),
            ("P07", "Stem / 30° taper", "End of the straight Ø4.900 stem. Start of the 30° taper."),
            ("P08", "Taper / neck", "End of the 30° taper. Start of the neck cylinder."),
            ("P09", "Neck / 45° chamfer", "End of the neck. Start of the 0.500 × 45° chamfer."),
            ("P10", "Chamfer / top face", "Outer edge of the closed top face."),
        ]
        for p, (pid, name, note) in zip(ordered, labels):
            p["pid"] = pid
            p["name"] = name
            p["note"] = note
        return ordered

    def _check(self):
        p = {row["pid"]: row for row in self.profile}
        sphere = next(s for s in self.surfaces if s["kind"] == "SPHERICAL_SURFACE")
        sx, sy, sz = sphere["origin"]
        sr = sphere["params"][0]
        for pid in ("P01", "P02"):
            d = math.dist((p[pid]["x"], p[pid]["y"], p[pid]["z"]), (sx, sy, sz))
            if abs(d - sr) > 1e-6:
                raise SystemExit(f"{pid} is not on the sphere: {d}")
        # R0.4 blend center lies on the sphere-center / P02 ray, 0.4 mm inside P02.
        blend_c = (4.35, 0.0, p["P03"]["z"])
        for pid in ("P02", "P03"):
            d = math.dist((p[pid]["x"], p[pid]["y"], p[pid]["z"]), blend_c)
            if abs(d - 0.4) > 1e-6:
                raise SystemExit(f"{pid} is not on the R0.4 blend: {d} center {blend_c}")
        torus_c = (2.75, 0.0, 0.3)
        for pid in ("P05", "P06"):
            d = math.dist((p[pid]["x"], p[pid]["y"], p[pid]["z"]), torus_c)
            if abs(d - 0.3) > 1e-6:
                raise SystemExit(f"{pid} is not on the R0.3 round: {d}")
        taper = math.degrees(math.atan2(p["P07"]["r"] - p["P08"]["r"], p["P08"]["z"] - p["P07"]["z"]))
        chamfer = math.degrees(math.atan2(p["P09"]["r"] - p["P10"]["r"], p["P10"]["z"] - p["P09"]["z"]))
        if abs(taper - 30) > 1e-4 or abs(chamfer - 45) > 1e-4:
            raise SystemExit(f"Unexpected cone angles {taper}, {chamfer}")


def right_profile(model, n=48):
    p = {row["pid"]: row for row in model.profile}
    pts = []

    def arc(cx, cz, radius, a0, a1, steps):
        # angle is atan2(dz, dr)
        for i in range(steps + 1):
            a = a0 + (a1 - a0) * i / steps
            pts.append((cx + radius * math.cos(a), cz + radius * math.sin(a)))

    zf = p["P03"]["z"]
    a_pole = -math.pi / 2
    a_join = math.atan2(p["P02"]["z"] - 4.3, p["P02"]["r"] - 0.0)
    arc(0.0, 4.3, 8.0, a_pole, a_join, n)
    a0 = math.atan2(p["P02"]["z"] - zf, p["P02"]["r"] - 4.35)
    a1 = math.atan2(p["P03"]["z"] - zf, p["P03"]["r"] - 4.35)
    arc(4.35, zf, 0.4, a0, a1, max(8, n // 4))
    pts.append((p["P04"]["r"], p["P04"]["z"]))
    pts.append((p["P05"]["r"], p["P05"]["z"]))
    arc(2.75, 0.3, 0.3, -math.pi / 2, -math.pi, max(8, n // 4))
    for pid in ("P07", "P08", "P09", "P10"):
        pts.append((p[pid]["r"], p[pid]["z"]))
    pts.append((0.0, p["P10"]["z"]))
    # drop consecutive duplicates
    cleaned = [pts[0]]
    for pt in pts[1:]:
        if math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(pt)
    return cleaned


def closed_outline(model):
    right = right_profile(model)
    left = [(-r, z) for r, z in reversed(right[:-1])]
    return right + left


class SectionView(Flowable):
    def __init__(self, model, width, height, window, title, subtitle, markers, dims):
        super().__init__()
        self.model = model
        self.width = width
        self.height = height
        self.window = window  # r0, r1, z0, z1
        self.title = title
        self.subtitle = subtitle
        self.markers = markers
        self.dims = dims

    def wrap(self, aw, ah):
        return self.width, self.height

    def _mapper(self):
        r0, r1, z0, z1 = self.window
        pad_l, pad_r, pad_b, pad_t = 16 * mm, 8 * mm, 8 * mm, 12 * mm
        span_r = r1 - r0
        span_z = z1 - z0
        scale = min((self.width - pad_l - pad_r) / span_r, (self.height - pad_b - pad_t) / span_z)
        used_w = scale * span_r
        used_h = scale * span_z
        ox = pad_l + (self.width - pad_l - pad_r - used_w) / 2
        oy = pad_b + (self.height - pad_b - pad_t - used_h) / 2

        def xy(r, z):
            return ox + (r - r0) * scale, oy + (z - z0) * scale

        return xy, scale

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(colors.white)
        c.roundRect(0, 0, self.width, self.height, 3 * mm, fill=1, stroke=0)

        xy, scale = self._mapper()
        r0, r1, z0, z1 = self.window
        x_left, y_bot = xy(r0, z0)
        x_right, y_top = xy(r1, z1)

        c.saveState()
        clip = c.beginPath()
        clip.rect(x_left, y_bot, x_right - x_left, y_top - y_bot)
        c.clipPath(clip, stroke=0, fill=0)

        outline = closed_outline(self.model)
        path = c.beginPath()
        x0, y0 = xy(*outline[0])
        path.moveTo(x0, y0)
        for r, z in outline[1:]:
            path.lineTo(*xy(r, z))
        path.close()
        c.setFillColor(FILL)
        c.setStrokeColor(INK)
        c.setLineWidth(1.05)
        c.drawPath(path, fill=1, stroke=1)

        axis0 = xy(0, z0)
        axis1 = xy(0, z1)
        c.setStrokeColor(TEAL)
        c.setDash(3, 2)
        c.setLineWidth(0.5)
        c.line(axis0[0], axis0[1], axis1[0], axis1[1])
        c.setDash()
        c.restoreState()

        c.saveState()
        band = c.beginPath()
        band.rect(1.5, 1.5, self.width - 3, self.height - 10 * mm)
        c.clipPath(band, stroke=0, fill=0)
        by_id = {row["pid"]: row for row in self.model.profile}
        for pid, dx, dy, anchor in self.markers:
            row = by_id[pid]
            self._tag(c, *xy(row["r"], row["z"]), dx, dy, anchor, pid, BLUE)
        for dim in self.dims:
            self._dim(c, xy, scale, dim)
        c.restoreState()

        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.roundRect(0, 0, self.width, self.height, 3 * mm, fill=0, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(4 * mm, self.height - 6.2 * mm, self.title)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7)
        c.drawRightString(self.width - 4 * mm, self.height - 6.0 * mm, self.subtitle)
        c.restoreState()

    def _tag(self, c, px, py, dx, dy, anchor, text, color):
        lx, ly = px + dx, py + dy
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(0.5)
        c.line(px, py, lx, ly)
        c.circle(px, py, 1.35, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.circle(px, py, 0.45, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 6.5)
        if anchor == "e":
            c.drawString(lx + 1.6, ly - 2, text)
        elif anchor == "w":
            c.drawRightString(lx - 1.6, ly - 2, text)
        else:
            c.drawCentredString(lx, ly - 2, text)

    def _arrow(self, c, x, y, ang):
        size = 3.1
        path = c.beginPath()
        path.moveTo(x, y)
        path.lineTo(x - size * math.cos(ang - 0.38), y - size * math.sin(ang - 0.38))
        path.lineTo(x - size * math.cos(ang + 0.38), y - size * math.sin(ang + 0.38))
        path.close()
        c.drawPath(path, fill=1, stroke=0)

    def _dim(self, c, xy, scale, dim):
        kind = dim[0]
        c.setStrokeColor(RED)
        c.setFillColor(RED)
        c.setLineWidth(0.45)
        if kind == "dia":
            _k, z, r, text, z_off = dim
            x1, y1 = xy(-r, z)
            x2, y2 = xy(r, z)
            y = y1 + z_off
            c.setStrokeColor(colors.Color(0.55, 0.58, 0.62))
            c.line(x1, y1, x1, y)
            c.line(x2, y2, x2, y)
            c.setStrokeColor(RED)
            c.line(x1, y, x2, y)
            ang = 0 if x2 > x1 else math.pi
            self._arrow(c, x1, y, ang + math.pi)
            self._arrow(c, x2, y, ang)
            c.setFont("Helvetica", 6.5)
            c.drawCentredString((x1 + x2) / 2, y + 1.6, text)
        elif kind == "v":
            _k, z_a, z_b, r_ext, text, r_off = dim
            x1, y1 = xy(r_ext, z_a)
            x2, y2 = xy(r_ext, z_b)
            x = xy(r_ext + r_off, z_a)[0]
            c.setStrokeColor(colors.Color(0.55, 0.58, 0.62))
            c.line(x1, y1, x, y1)
            c.line(x2, y2, x, y2)
            c.setStrokeColor(RED)
            c.line(x, y1, x, y2)
            self._arrow(c, x, y1, math.pi / 2 if y2 > y1 else -math.pi / 2)
            self._arrow(c, x, y2, -math.pi / 2 if y2 > y1 else math.pi / 2)
            c.setFont("Helvetica", 6.5)
            c.saveState()
            c.translate(x + (4 if r_off >= 0 else -4), (y1 + y2) / 2)
            c.rotate(90)
            c.drawCentredString(0, 0, text)
            c.restoreState()
        elif kind == "leader":
            _k, r, z, text, dx, dy, anchor = dim
            x, y = xy(r, z)
            self._tag(c, x, y, dx, dy, anchor, text, RED)
        elif kind == "note":
            _k, r, z, text, dx, dy = dim
            x, y = xy(r, z)
            c.setFont("Helvetica", 6.5)
            c.drawString(x + dx, y + dy, text)


def styles():
    return {
        "h1": ParagraphStyle(
            "h1", fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=NAVY, spaceAfter=2
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=3
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=8.5, leading=11.4, textColor=INK, spaceAfter=3
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=7.5, leading=9.6, textColor=MUTED, spaceAfter=2
        ),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7, leading=9, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=NAVY),
        "head": ParagraphStyle(
            "head", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=colors.white, alignment=TA_CENTER
        ),
        "center": ParagraphStyle(
            "center", fontName="Helvetica", fontSize=7, leading=9, textColor=INK, alignment=TA_CENTER
        ),
    }


def P(text, style):
    return Paragraph(str(text), style)


def table(data, col_widths):
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.Color(0.82, 0.84, 0.87)),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 11 * mm, PAGE_W, 11 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(12 * mm, PAGE_H - 7 * mm, "BLANK_ERSB06-07G_PAN_EXTRUDED")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - 12 * mm, PAGE_H - 7 * mm, "DIMENSION POINTS  ·  MILLIMETRES")
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 9 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(12 * mm, 3.4 * mm, "STEP AP214  ·  Z axis datum at the flange face  ·  uncertainty 0.00001 mm")
    canvas.drawRightString(PAGE_W - 12 * mm, 3.4 * mm, f"Page {doc.page}")
    canvas.restoreState()


def unique_3d(model):
    groups = []
    for pid, rec in sorted(model.points.items()):
        if len(rec["xyz"]) != 3:
            continue
        x, y, z = rec["xyz"]
        placed = False
        for g in groups:
            if math.dist((x, y, z), g["xyz"]) <= 1e-8:
                g["ids"].append(pid)
                placed = True
                break
        if not placed:
            groups.append({"ids": [pid], "xyz": (x, y, z), "raw": rec["raw"], "roles": set()})
    for g in groups:
        for pid in g["ids"]:
            g["roles"].update(model.roles.get(pid, set()))
        if not g["roles"]:
            g["roles"].add("coordinate")
    role_rank = {"vertex": 0, "origin": 1, "line point": 2, "spline control": 3, "coordinate": 4}
    groups.sort(
        key=lambda g: (
            min(role_rank.get(r, 9) for r in g["roles"]),
            round(g["xyz"][2], 6),
            round(g["xyz"][0], 6),
            round(g["xyz"][1], 6),
        )
    )
    return groups


def build(model):
    s = styles()
    p = {row["pid"]: row for row in model.profile}
    story = []
    usable = PAGE_W - 24 * mm

    story.append(P("Dimension points", s["h1"]))
    story.append(
        P(
            "Part BLANK_ERSB06-07G_PAN_EXTRUDED. Solid of revolution about the Z axis. "
            "Source file BLANK_ERSB06-07G_PAN_EXTRUDED.step, written "
            f"{model.step['time']} by build123d / Open CASCADE. "
            "Every model-space coordinate below is taken from that file. Units are millimetres.",
            s["body"],
        )
    )
    story.append(P("How to read the points", s["h2"]))
    story.append(
        P(
            "The boundary is stored as one seam on the +X side of each circular edge, plus the bottom pole. "
            "P01 through P10 are those points, in boundary order from the bottom pole to the top face. "
            "A point with radius R is the +X sample of a full circle: centre (0, 0, Z), diameter 2R. "
            "That circle is the complete set of dimension points at the station. "
            "The spline-control register at the end lists every other 3D coordinate stored in the file.",
            s["body"],
        )
    )

    story.append(P("Overall size", s["h2"]))
    overall = [
        [P(h, s["head"]) for h in ["Dimension", "Value (mm)", "From", "To"]],
        [P("Overall length, along Z", s["cellb"]), P("48.040", s["center"]), P("P01  Z −3.700", s["center"]), P("P10  Z 44.340", s["center"])],
        [P("Overall diameter", s["cellb"]), P("Ø 9.500", s["center"]), P("P03", s["center"]), P("P04", s["center"])],
        [P("Top-face diameter", s["cellb"]), P(f"Ø {fmt(2 * p['P10']['r'])}", s["center"]), P("P10", s["center"]), P("closed disk at Z 44.340", s["center"])],
        [P("Datum", s["cellb"]), P("Z = 0", s["center"]), P("flange face", s["center"]), P("axis through (0, 0)", s["center"])],
    ]
    story.append(table(overall, [52 * mm, 32 * mm, 48 * mm, 54 * mm]))
    story.append(Spacer(1, 2 * mm))
    story.append(
        P(
            "Z = 0 is the flat flange face. Positive Z runs from that face toward the tip. "
            "The base sphere hangs below the datum, so the bottom pole is at Z −3.700.",
            s["small"],
        )
    )

    story.append(PageBreak())
    story.append(P("Profile drawing — full length", s["h2"]))
    story.append(
        SectionView(
            model,
            usable,
            236 * mm,
            (-8.6, 8.6, -7.4, 48.6),
            "Section through the Z axis",
            "true proportion  ·  +X seam points",
            [
                ("P01", -4, -12, "c"),
                ("P04", 22, 8, "w"),
                ("P08", 24, 6, "w"),
                ("P10", 22, 8, "w"),
            ],
            [
                ("leader", 4.75, -1.2, "Ø 9.500", 22, -6, "e"),
                ("leader", p["P08"]["r"], 26.0, f"Ø {fmt(2 * p['P08']['r'], 3)}", 24, 0, "e"),
                ("leader", p["P10"]["r"], 44.34, f"Ø {fmt(2 * p['P10']['r'], 3)}", 22, 2, "e"),
                ("v", -3.7, 44.34, 6.2, "48.040", 1.35),
            ],
        )
    )

    story.append(PageBreak())
    story.append(P("Base — points P01 to P06", s["h2"]))
    story.append(
        P(
            "Spherical base SR 8.000 centred at (0, 0, 4.300), blended by an external R0.400 arc "
            "into the Ø9.500 cylinder. The cylinder ends on the datum face. "
            "From P04 the face runs in to P05, then an external R0.300 round drops onto the Ø4.900 stem.",
            s["body"],
        )
    )
    story.append(
        SectionView(
            model,
            usable,
            158 * mm,
            (-7.6, 7.6, -5.5, 1.7),
            "Base detail",
            "Z -5.50 to 1.70",
            [
                ("P01", 0, -14, "c"),
                ("P02", 40, -30, "e"),
                ("P03", 42, 10, "e"),
                ("P04", 40, 2, "e"),
                ("P05", 12, 20, "e"),
                ("P06", 58, 22, "e"),
            ],
            [
                ("leader", 2.6, -2.85, "SR 8.000", -48, -6, "w"),
                ("leader", 4.55, -2.05, "R 0.400", 52, -2, "e"),
                ("leader", 4.75, 0.0, "Ø 9.500", 46, -20, "e"),
                ("leader", 2.75, 0.0, "Ø 5.500", 92, 28, "e"),
                ("leader", 2.54, 0.09, "R 0.300", -188, 38, "w"),
                ("leader", 2.45, 1.2, "Ø 4.900", 96, 4, "e"),
                ("v", -3.7, 0.0, -4.75, "3.700", -1.5),
            ],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        P(
            "The R0.400 centre is (4.350, 0, Z of P03). It is tangent to the sphere at P02 and tangent "
            "to the Ø9.500 cylinder at P03. The R0.300 centre is (2.750, 0, 0.300): a quarter-circle "
            "from P05 to P06.",
            s["small"],
        )
    )

    story.append(PageBreak())
    story.append(P("Stem taper — points P07 and P08", s["h2"]))
    story.append(
        P(
            "Straight Ø4.900 stem ends at P07. A 30° semi-angle taper runs to the neck at Z 9.030 (P08). "
            "P07's station and the neck radius are the intersection of that taper with the stem and with Z 9.030; "
            "both are driven values and are quoted from the STEP literals.",
            s["body"],
        )
    )
    story.append(
        SectionView(
            model,
            usable,
            92 * mm,
            (-4.3, 4.3, 8.05, 9.75),
            "30 degree taper",
            "Z 8.05 to 9.75",
            [("P07", 18, -10, "w"), ("P08", 22, 8, "w")],
            [
                ("dia", p["P07"]["z"], p["P07"]["r"], "Ø 4.900", -11),
                ("dia", 9.03, p["P08"]["r"], f"Ø {fmt(2 * p['P08']['r'])}", 9),
                ("leader", 2.32, 8.78, "30°", 22, 2, "e"),
            ],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(P("Tip chamfer — points P09 and P10", s["h2"]))
    story.append(
        P(
            "The neck is a cylinder at the P08 radius from Z 9.030 to Z 43.840. "
            "P09 to P10 is a 45° chamfer, 0.500 mm on the radius and 0.500 mm on Z. "
            "The top face is a closed disk at Z 44.340.",
            s["body"],
        )
    )
    story.append(
        SectionView(
            model,
            usable,
            92 * mm,
            (-3.8, 3.8, 43.25, 45.05),
            "45 degree tip chamfer",
            "Z 43.25 to 45.05",
            [("P09", 16, -10, "w"), ("P10", 20, 8, "w")],
            [
                ("dia", 43.84, p["P09"]["r"], f"Ø {fmt(2 * p['P09']['r'])}", -10),
                ("dia", 44.34, p["P10"]["r"], f"Ø {fmt(2 * p['P10']['r'])}", 8),
                ("v", 43.84, 44.34, 2.15, "0.500", 0.85),
                ("leader", 1.94, 44.09, "45°", -24, 4, "w"),
            ],
        )
    )

    story.append(PageBreak())
    story.append(P("Seam dimension points P01–P10", s["h2"]))
    story.append(
        P(
            "Coordinates are the STEP vertex coordinates. Radius is the distance to the Z axis. "
            "Diameter is the full circle through that point. Values below 0.000001 mm are written as 0.",
            s["small"],
        )
    )
    header = [P(h, s["head"]) for h in ["Point", "Feature", "X", "Y", "Z", "Radius", "Diameter"]]
    data = [header]
    for row in model.profile:
        data.append(
            [
                P(row["pid"], s["cellb"]),
                P(row["name"], s["cell"]),
                P(fmt(row["x"]), s["center"]),
                P(fmt(row["y"]), s["center"]),
                P(fmt(row["z"]), s["center"]),
                P(fmt(row["r"]), s["center"]),
                P("—" if row["r"] < 1e-9 else f"Ø {fmt(2 * row['r'])}", s["center"]),
            ]
        )
    story.append(table(data, [16 * mm, 48 * mm, 24 * mm, 18 * mm, 28 * mm, 26 * mm, 26 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(P("Stations along the boundary", s["h2"]))
    station_header = [P(h, s["head"]) for h in ["From", "To", "Δ radius", "Δ Z", "What it is"]]
    stations = [station_header]
    descriptions = [
        "Sphere, from the pole to the blend tangency",
        "R0.400 blend",
        "Ø9.500 cylinder, up to the datum face",
        "Flange face, radially inward",
        "R0.300 external round",
        "Ø4.900 stem cylinder",
        "30° taper",
        "Neck cylinder",
        "45° chamfer, 0.500 × 0.500",
    ]
    for a, b, desc in zip(model.profile, model.profile[1:], descriptions):
        stations.append(
            [
                P(a["pid"], s["cellb"]),
                P(b["pid"], s["cellb"]),
                P(fmt(b["r"] - a["r"]), s["center"]),
                P(fmt(b["z"] - a["z"]), s["center"]),
                P(desc, s["cell"]),
            ]
        )
    story.append(table(stations, [16 * mm, 16 * mm, 24 * mm, 24 * mm, 106 * mm]))

    story.append(PageBreak())
    story.append(P("Feature dimensions", s["h2"]))
    story.append(
        P(
            "Analytic surfaces and the circles that bound them. A semi-angle is the angle between the cone "
            "axis and the cone generator. Design values that land on a clean millimetre are marked exact; "
            "the remaining figures are the driven STEP values.",
            s["small"],
        )
    )
    feat_header = [P(h, s["head"]) for h in ["Feature", "Size", "Location", "Status"]]
    neck_r = p["P08"]["r"]
    tip_r = p["P10"]["r"]
    features = [
        ["Spherical base", "SR 8.000", "Centre (0, 0, 4.300)", "Exact"],
        ["Bottom pole P01", "Z −3.700", "(0, 0, −3.700)", "Exact"],
        ["Base blend", "R 0.400", "Centre (4.350, 0, Z of P03), tangent at P02 and P03", "Exact radius"],
        ["Blend tangency P03", f"Z {fmt(p['P03']['z'])}", "X 4.750, Y 0", "Driven"],
        ["Outside cylinder", "Ø 9.500", f"From Z {fmt(p['P03']['z'])} to Z 0.000", "Exact diameter"],
        ["Flange face", "Z 0.000", "Annulus from Ø5.500 to Ø9.500", "Exact"],
        ["Flange radial width", "2.000", "P04 to P05", "Exact"],
        ["Stem round", "R 0.300", "Centre (2.750, 0, 0.300), quarter-circle P05 to P06", "Exact"],
        ["Stem cylinder", "Ø 4.900", f"From Z 0.300 to Z {fmt(p['P07']['z'])}", "Exact diameter"],
        ["Stem length", f"{fmt(p['P07']['z'] - 0.3)}", "P06 to P07, parallel to Z", "Driven"],
        ["Taper", "30° semi-angle", f"Ø4.900 at P07 to Ø{fmt(2 * neck_r)} at Z 9.030", "Exact angle"],
        ["Taper end station", "Z 9.030", "P08", "Exact station"],
        ["Neck cylinder", f"Ø {fmt(2 * neck_r)}", "From Z 9.030 to Z 43.840", "Driven diameter"],
        ["Neck length", "34.810", "P08 to P09", "Exact"],
        ["Tip chamfer", "0.500 × 45°", "Radial drop 0.500 and axial rise 0.500", "Exact"],
        ["Top face", "Z 44.340", f"Closed disk Ø {fmt(2 * tip_r)}", "Exact station"],
        ["Overall length", "48.040", "Z −3.700 to Z 44.340", "Exact"],
    ]
    feat_data = [feat_header]
    for name, size, loc, status in features:
        feat_data.append(
            [P(name, s["cellb"]), P(size, s["cell"]), P(loc, s["cell"]), P(status, s["center"])]
        )
    story.append(table(feat_data, [38 * mm, 42 * mm, 78 * mm, 28 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(
        P(
            "Neck radius is 2.450 − (9.030 − Z<sub>P07</sub>) · tan 30°. "
            f"Z<sub>P07</sub> = {p['P07']['raw'][2]} mm and the neck radius is {p['P08']['raw'][0]} mm. "
            "Tip radius is the neck radius minus 0.500 mm.",
            s["small"],
        )
    )

    story.append(P("Circular edges", s["h2"]))
    circ_header = [P(h, s["head"]) for h in ["STEP", "Diameter", "Centre X", "Centre Y", "Centre Z", "Axis"]]
    circ = [circ_header]
    for row in model.circles:
        ox, oy, oz = row["origin"]
        ax, ay, az = row["axis"]
        # Meridian construction circles (axis not along Z) are called out as radii, not part diameters.
        if abs(az) < 0.5:
            label = f"R {fmt(row['radius'])} meridian"
            dia = label
        else:
            dia = f"Ø {fmt(2 * row['radius'])}"
        circ.append(
            [
                P(f"#{row['id']}", s["cellb"]),
                P(dia, s["center"]),
                P(fmt(ox), s["center"]),
                P(fmt(oy), s["center"]),
                P(fmt(oz), s["center"]),
                P(f"({fmt(ax, 3)}, {fmt(ay, 3)}, {fmt(az, 3)})", s["center"]),
            ]
        )
    story.append(table(circ, [18 * mm, 36 * mm, 28 * mm, 28 * mm, 32 * mm, 44 * mm]))
    story.append(Spacer(1, 2 * mm))
    story.append(
        P(
            "Circles whose axis is Z are the part diameters. The two meridian circles are the sphere "
            "great-circle seam (R 8.000, lying in XZ) and the R0.300 round seam.",
            s["small"],
        )
    )

    story.append(PageBreak())
    story.append(P("Every 3D coordinate in the STEP file", s["h2"]))
    groups = unique_3d(model)
    n_3d = sum(1 for rec in model.points.values() if len(rec["xyz"]) == 3)
    n_2d = sum(1 for rec in model.points.values() if len(rec["xyz"]) == 2)
    story.append(
        P(
            f"The file contains {n_3d} three-dimensional CARTESIAN_POINT entities. "
            f"Points within 0.00000001 mm of each other are one row ({len(groups)} unique positions). "
            f"{n_2d} further points are two-dimensional (u, v) parameters of pcurves. "
            "Those are not positions on the part and are not listed.",
            s["body"],
        )
    )
    story.append(
        P(
            "Role “vertex” is a boundary dimension point (the P-points). "
            "“origin” is a circle, surface, or axis origin. "
            "“line point” lies on an unbounded line used as an edge. "
            "“spline control” is a B-spline control point of the base blend; "
            "except where it is also a vertex, it is not a point on the finished surface.",
            s["small"],
        )
    )
    reg_header = [P(h, s["head"]) for h in ["#", "STEP ids", "Role", "X", "Y", "Z"]]
    reg = [reg_header]
    for i, g in enumerate(groups, start=1):
        ids = ", ".join(f"#{n}" for n in g["ids"])
        x, y, z = g["xyz"]
        pids = [
            row["pid"]
            for row in model.profile
            if math.dist((x, y, z), (row["x"], row["y"], row["z"])) < 1e-6
        ]
        role = ", ".join(pids + sorted(g["roles"]))
        reg.append(
            [
                P(str(i), s["center"]),
                P(ids, s["cell"]),
                P(role, s["cell"]),
                P(fmt_raw([g["raw"][0]]), s["cell"]),
                P(fmt_raw([g["raw"][1]]), s["cell"]),
                P(fmt_raw([g["raw"][2]]), s["cell"]),
            ]
        )
    story.append(
        table(reg, [10 * mm, 42 * mm, 32 * mm, 36 * mm, 36 * mm, 30 * mm])
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        P(
            "Raw STEP literals are shown so driven values are not rounded away. "
            "A coordinate whose magnitude is below 1e-9 is numerical noise and is written as 0. "
            "Model uncertainty, entity #594, is 0.00001 mm.",
            s["small"],
        )
    )
    return story


def main():
    step = parse_step(STEP_PATH)
    model = Model(step)
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        title="BLANK_ERSB06-07G_PAN_EXTRUDED dimension points",
        author="Dimension extraction from STEP AP214",
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body", showBoundary=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
    doc.build(build(model))
    print(f"Wrote {PDF_PATH}")
    print(f"Profile points: {len(model.profile)}")
    print(f"3D points: {sum(1 for rec in model.points.values() if len(rec['xyz']) == 3)}")
    print(f"Unique 3D positions: {len(unique_3d(model))}")


if __name__ == "__main__":
    main()
