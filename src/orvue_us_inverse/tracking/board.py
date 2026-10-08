"""
orvue_us_inverse.tracking.board - Generate docs/print/tracking_board.pdf (A4) from the layout in tracking.markers.
    python -m orvue_us_inverse board            # optional: logo.png in the current folder
Page 1: reference board (IDs 1-4 around a 100 x 100 mm region) + 100 mm scale check.
Page 2: probe markers to cut out (ID 0 for the platform, ID 5 for the arm end) + spares.
Markers are drawn as vector cells, so they stay sharp at any printer resolution.
"""
import os
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import black, white, Color
from reportlab.pdfgen import canvas

from orvue_us_inverse.paths import BOARD_PDF
from orvue_us_inverse.tracking import markers as tm

PW, PH = A4[0] / mm, A4[1] / mm          # 210 x 297 mm
GREY = Color(0.6, 0.6, 0.6)
DARK = Color(0.2, 0.2, 0.2)
TEAL = Color(0.03, 0.31, 0.25)
OX, OY = 55.0, 50.0                       # page position (mm from left, mm from top) of the phantom origin


def Y(y_top):                             # mm from top -> reportlab points from bottom
    return (PH - y_top) * mm


def X(x):
    return x * mm


def marker(c, mid, x, y_top, size):
    """Vector ArUco marker: black square, white data cells. (x, y_top) = top-left of black square."""
    bits = cv2.aruco.generateImageMarker(tm.dictionary(), mid, 6, borderBits=1)
    cell = size / 6.0
    c.setFillColor(black)
    c.rect(X(x), Y(y_top + size), size * mm, size * mm, stroke=0, fill=1)
    c.setFillColor(white)
    for r in range(6):
        for k in range(6):
            if bits[r, k] > 127:
                c.rect(X(x + k * cell), Y(y_top + (r + 1) * cell), cell * mm, cell * mm, stroke=0, fill=1)


def arrow(c, x0, y0, x1, y1, col=DARK, head=2.2):
    import math
    c.setStrokeColor(col); c.setFillColor(col); c.setLineWidth(0.8)
    c.line(X(x0), Y(y0), X(x1), Y(y1))
    a = math.atan2(y1 - y0, x1 - x0)
    p = c.beginPath()
    p.moveTo(X(x1), Y(y1))
    for s in (+1, -1):
        p.lineTo(X(x1 - head * math.cos(a + s * 0.45)), Y(y1 - head * math.sin(a + s * 0.45)))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def text(c, x, y_top, s, size=8, col=DARK, font="Helvetica", align="left"):
    c.setFillColor(col); c.setFont(font, size)
    {"left": c.drawString, "center": c.drawCentredString, "right": c.drawRightString}[align](X(x), Y(y_top), s)


def scale_bar(c, x, y_top):
    c.setStrokeColor(black); c.setLineWidth(0.6)
    c.line(X(x), Y(y_top), X(x + 100), Y(y_top))
    for i in range(101):
        h = 4 if i % 10 == 0 else (2.5 if i % 5 == 0 else 1.5)
        c.line(X(x + i), Y(y_top), X(x + i), Y(y_top - h))
    for i in range(0, 101, 10):
        text(c, x + i, y_top + 4, str(i), 6, align="center")
    text(c, x + 50, y_top + 9, "SCALE CHECK: this bar must measure exactly 100.0 mm. If not, reprint at 100% / Actual size.",
         7.5, black, "Helvetica-Bold", "center")


def page_reference(c):
    text(c, PW / 2, 18, "Ultrasound Imaging Simulation Prototype  -  Reference board (170 x 110 mm)", 12, black,
         "Helvetica-Bold", "center")
    text(c, PW / 2, 25, "Orvue Surgical  |  ArUco DICT_4X4_50  |  reference IDs 1-4  |  print at 100% (Actual size), matte paper",
         8, DARK, align="center")
    text(c, PW / 2, 31, "Cut along the dashed line. Everything inside it goes on the flat surface.", 8, DARK, align="center")
    # cut line of the 170 x 110 mm sheet + corner crop marks
    sx, sy = tm.SHEET_TL; sw, sh = tm.SHEET_SIZE
    c.setStrokeColor(GREY); c.setLineWidth(0.4); c.setDash(3, 2)
    c.rect(X(OX + sx), Y(OY + sy + sh), sw * mm, sh * mm, stroke=1, fill=0)
    c.setDash()
    for (cx, cy) in [(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh)]:
        dx = -1 if cx == sx else 1; dy = -1 if cy == sy else 1
        c.line(X(OX + cx + dx * 2), Y(OY + cy), X(OX + cx + dx * 7), Y(OY + cy))
        c.line(X(OX + cx), Y(OY + cy + dy * 2), X(OX + cx), Y(OY + cy + dy * 7))
    s = tm.REF_MARKER_SIZE_MM
    for mid, (mx, my) in tm.REF_MARKERS_TL.items():
        marker(c, mid, OX + mx, OY + my, s)
        ly = OY + my + s + tm.REF_QUIET_ZONE_MM + 4 if my < 50 else OY + my - tm.REF_QUIET_ZONE_MM - 2
        text(c, OX + mx + s / 2, ly, f"ID {mid}", 8, DARK, "Helvetica-Bold", "center")
    # active region outline and ticks (thin grey, kept clear of the quiet zones)
    R = tm.REGION_MM
    c.setStrokeColor(GREY); c.setLineWidth(0.5)
    c.rect(X(OX), Y(OY + R), R * mm, R * mm, stroke=1, fill=0)
    for i in range(10, int(R), 10):
        L = 4 if i == 50 else 2
        for (ax, ay, bx, by) in [(i, 0, i, L), (i, R, i, R - L), (0, i, L, i), (R, i, R - L, i)]:
            c.line(X(OX + ax), Y(OY + ay), X(OX + bx), Y(OY + by))
    text(c, OX + 1.5, OY + 4.5, "0 mm", 6.5, GREY)
    text(c, OX + R - 1.5, OY + 4.5, "100 mm", 6.5, GREY, align="right")
    text(c, OX + 1.5, OY + R - 1.5, "100 mm", 6.5, GREY)
    # axes inside the region (no room outside on a 110 mm sheet)
    arrow(c, OX + 8, OY + 9, OX + 30, OY + 9)
    text(c, OX + 32, OY + 10, "+x", 8, DARK, "Helvetica-Bold")
    arrow(c, OX + 8, OY + 9, OX + 8, OY + 31)
    text(c, OX + 8, OY + 36, "+y", 8, DARK, "Helvetica-Bold", "center")
    # logo and title in the region
    if os.path.exists("logo.png"):
        c.drawImage("logo.png", X(OX + 39), Y(OY + 22 + 22), 22 * mm, 22 * mm)
    text(c, OX + 50, OY + 56, "Ultrasound Imaging", 11, TEAL, "Helvetica-Bold", "center")
    text(c, OX + 50, OY + 62, "Simulation Prototype", 11, TEAL, "Helvetica-Bold", "center")
    text(c, OX + 50, OY + 69, "Orvue Surgical", 8, DARK, align="center")
    scale_bar(c, OX, 185)
    lines = [
        "Setup",
        "1. Print at 100% / Actual size (disable 'fit to page'). The scale bar and the 100 mm region must both measure 100.0 mm.",
        "2. Cut along the dashed line (170 x 110 mm). Glue fully flat on the rigid surface. Matte finish only; gloss causes glare.",
        "3. Keep the white margins around each marker clear: no tape, pen marks or overlapping edges.",
        "4. The gel (including its container walls) must stay inside the 100 mm region and never cover a marker.",
        "5. Measure the gel surface height above this sheet and set H_MM in tracking/markers.py (markers sit at z = +H).",
        "Frame: origin = top-left corner of the region, +x right, +y down, +z into the table; z = 0 is the gel surface.",
        "Marker black-square top-left corners (mm): ID1 (-30,0)  ID2 (106,0)  ID3 (106,76)  ID4 (-30,76); size 24 mm.",
        "The camera does not need to see the region outline: its position follows from the marker positions above.",
    ]
    for i, s_ in enumerate(lines):
        text(c, 20, 208 + i * 6, s_, 8 if i else 9, black if i == 0 else DARK, "Helvetica-Bold" if i == 0 else "Helvetica")


def cut_tile(c, mid, x, y_top, size, qz):
    """Marker with dashed cut line at the outer edge of its quiet zone."""
    t = size + 2 * qz
    tw, th = 7.0, 10.0                          # orientation tab on the +x side, outside the quiet zone
    c.setStrokeColor(GREY); c.setLineWidth(0.4); c.setDash(2, 2)
    p = c.beginPath()
    for (px, py) in [(x, y_top), (x + t, y_top), (x + t, y_top + t / 2 - th / 2), (x + t + tw, y_top + t / 2 - th / 2),
                     (x + t + tw, y_top + t / 2 + th / 2), (x + t, y_top + t / 2 + th / 2), (x + t, y_top + t), (x, y_top + t)]:
        (p.moveTo if (px, py) == (x, y_top) else p.lineTo)(X(px), Y(py))
    p.close(); c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    text(c, x + t + tw / 2, y_top + t / 2 + 1.2, "+x", 7, DARK, "Helvetica-Bold", "center")
    marker(c, mid, x + qz, y_top + qz, size)
    arrow(c, x + t + tw + 3, y_top + t / 2, x + t + tw + 15, y_top + t / 2)
    text(c, x + t + tw + 17, y_top + t / 2 + 1.2, "+x", 8, DARK, "Helvetica-Bold")
    arrow(c, x + t / 2, y_top + t + 3, x + t / 2, y_top + t + 12)
    text(c, x + t / 2 + 2, y_top + t + 12, "+y", 8, DARK, "Helvetica-Bold")
    # probe orientation mark on the -x side: the screen dot shows this end on the image's left
    arrow(c, x - 3, y_top + t / 2, x - 13, y_top + t / 2)
    text(c, x - 8, y_top + t / 2 - 2.5, "mark", 6.5, DARK, "Helvetica-Bold", "center")
    text(c, x, y_top - 2.5, f"ID {mid}  ({size:.0f} mm marker, cut {t:.0f} x {t:.0f} mm + tab)", 8, DARK, "Helvetica-Bold")


def page_probe(c):
    text(c, PW / 2, 18, "Probe markers  -  cut out along the dashed lines", 13, black, "Helvetica-Bold", "center")
    text(c, PW / 2, 25, "ArUco DICT_4X4_50  |  ID 0 = probe platform, ID 5 = second marker on the arm  |  spares in the lower row",
         8, DARK, align="center")
    for row, y in enumerate((40, 120)):
        cut_tile(c, tm.PROBE_MARKER_ID, 25, y, tm.PROBE_MARKER_SIZE_MM, 5.0)
        cut_tile(c, tm.SECOND_MARKER_ID, 120, y + 6, tm.SECOND_MARKER_SIZE_MM, 4.0)
        if row == 1:
            text(c, 25, y - 9, "Spares", 9, GREY, "Helvetica-Bold")
    scale_bar(c, 55, 213)
    lines = [
        "Mounting",
        "1. Cut along the dashed lines, keeping the small +x tab. Stick ID 0 centred on the 40 x 40 mm platform, flat.",
        "2. Orientation: the printed +x arrow points along the array, AWAY from the probe's orientation mark: the mark",
        "    is at the -x end (\"mark\" side of the tile), shown by the dot on the image's left. +y points across the face.",
        "3. Stick ID 5 the same way round on the arm end, centred on the probe x axis.",
        "4. Measure and set in tracking/markers.py: PLATFORM_HEIGHT_MM (face to marker surface) and",
        "    SECOND_MARKER_OFFSET_X_MM (face centre to ID 5 centre, along +x). Set USE_SECOND_MARKER = False if unused.",
        "5. Trim the +x tabs off after mounting (keep them clear of the white margin until then).",
        "6. Refine the face offset with the calibration cradle before relying on depth or contact detection.",
    ]
    for i, s_ in enumerate(lines):
        text(c, 20, 236 + i * 6, s_, 8 if i else 9, black if i == 0 else DARK, "Helvetica-Bold" if i == 0 else "Helvetica")


def main(path=BOARD_PDF):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("Ultrasound simulator tracking board")
    c.setAuthor("Orvue Surgical")
    page_reference(c); c.showPage()
    page_probe(c); c.showPage()
    c.save()
    print("wrote", path)


if __name__ == "__main__":
    main()
