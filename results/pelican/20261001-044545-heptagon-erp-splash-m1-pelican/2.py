import tkinter as tk
import math
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

WIDTH = 900
HEIGHT = 900
CENTER = (WIDTH / 2.0, HEIGHT / 2.0)
CX, CY = CENTER

HEPT_RADIUS = 360.0
BALL_RADIUS = 12.0

GRAVITY = 1600.0                     # pixels / s^2, downward in canvas coords
OMEGA = 2.0 * math.pi / 5.0          # 360 degrees / 5 seconds

SUBSTEPS = 4
COLLISION_ITER = 3

FRAME_FPS = 60
FRAME_MS = 16

WALL_RESTITUTION = 0.68
BALL_RESTITUTION = 0.90

WALL_FRICTION = 0.22
BALL_FRICTION = 0.16

AIR_DRAG = 0.06
SPIN_DRAG = 0.25

# The ball material is tuned so that the rebound height stays below the
# heptagon radius, while typical rebounds are larger than one ball radius.
MAX_BOUNCE_HEIGHT = HEPT_RADIUS * 0.95
MAX_SPEED = math.sqrt(2.0 * GRAVITY * MAX_BOUNCE_HEIGHT)
MAX_REBOUND_SPEED = math.sqrt(2.0 * GRAVITY * MAX_BOUNCE_HEIGHT)

MIN_BOUNCE_HEIGHT = BALL_RADIUS * 1.15
MIN_REBOUND_SPEED = math.sqrt(2.0 * GRAVITY * MIN_BOUNCE_HEIGHT)

MAX_ANG_VEL = 12.0

COLORS = [
    "#f8b862", "#f6ad49", "#f39800", "#f08300",
    "#ec6d51", "#ee7948", "#ed6d3d", "#ec6800",
    "#ec6800", "#ee7800", "#eb6238", "#ea5506",
    "#ea5506", "#eb6101", "#e49e61", "#e45e32",
    "#e17b34", "#dd7a56", "#db8449", "#d66a35",
]

HEPT_FILL = "#101010"
HEPT_OUTLINE = "#3a3a3a"
BALL_OUTLINE = "#33180b"
NUMBER_COLOR = "#2b1000"


# ---------------------------------------------------------------------------
# Small vector helper
# ---------------------------------------------------------------------------

def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Ball:
    number: int
    color: str
    pos: List[float]
    vel: List[float]
    angle: float
    ang_vel: float


@dataclass(frozen=True)
class Edge:
    x0: float
    y0: float
    x1: float
    y1: float
    out_x: float
    out_y: float
    a: float
    seg_x: float
    seg_y: float
    len2: float


# ---------------------------------------------------------------------------
# Regular heptagon geometry
# ---------------------------------------------------------------------------

# In canvas coordinates y grows downward.  Choose the orientation so that one
# flat side is near the bottom.
BASE_ANGLE = math.pi / 2.0 - math.pi / 7.0

LOCAL_VERTS: List[tuple] = []
for _i in range(7):
    _ang = BASE_ANGLE + _i * 2.0 * math.pi / 7.0
    LOCAL_VERTS.append((
        HEPT_RADIUS * math.cos(_ang),
        HEPT_RADIUS * math.sin(_ang)
    ))


def build_edges() -> List[Edge]:
    edges: List[Edge] = []
    for i in range(7):
        x0, y0 = LOCAL_VERTS[i]
        x1, y1 = LOCAL_VERTS[(i + 1) % 7]

        mx = (x0 + x1) * 0.5
        my = (y0 + y1) * 0.5
        ml = math.hypot(mx, my)

        if ml > 1e-9:
            ox = mx / ml
            oy = my / ml
        else:
            ox = oy = 0.0

        a = ox * x0 + oy * y0
        sx = x1 - x0
        sy = y1 - y0
        len2 = sx * sx + sy * sy

        edges.append(Edge(x0, y0, x1, y1, ox, oy, a, sx, sy, len2))

    return edges


EDGES = build_edges()


# ---------------------------------------------------------------------------
# Ball creation
# ---------------------------------------------------------------------------

def make_balls() -> List[Ball]:
    """
    The balls effectively start from the center.  They are placed in a tiny
    non-overlapping cluster around the exact center to avoid a singular 20-fold
    coincident overlap at one point.
    """
    balls: List[Ball] = []
    r = BALL_RADIUS

    # Compact non-overlapping starting cluster: 1 + 6 + 13 = 20
    specs = [
        (1, 0.0),
        (6, 2.2 * r),
        (13, 4.5 * r),
    ]

    num = 1
    for count, radius in specs:
        for i in range(count):
            if count == 1:
                ox = oy = 0.0
                ang = 0.0
            else:
                phase = (
                    2.0 * math.pi * i / count
                    + (math.pi / count if count == 13 else 0.0)
                )
                ox = radius * math.cos(phase)
                oy = radius * math.sin(phase)
                ang = phase

            pos = [CX + ox, CY + oy]
            vel = [0.0, 0.0]

            balls.append(
                Ball(
                    number=num,
                    color=COLORS[num - 1],
                    pos=pos,
                    vel=vel,
                    angle=ang,
                    ang_vel=0.0
                )
            )
            num += 1

    return balls


# ---------------------------------------------------------------------------
# Ball-ball collisions
# ---------------------------------------------------------------------------

def resolve_ball_balls(balls: List[Ball]) -> None:
    n = len(balls)
    two_r = 2.0 * BALL_RADIUS
    two_r2 = two_r * two_r

    for i in range(n):
        bi = balls[i]
        xi = bi.pos[0]
        yi = bi.pos[1]

        for j in range(i + 1, n):
            bj = balls[j]

            dx = bj.pos[0] - xi
            dy = bj.pos[1] - yi
            d2 = dx * dx + dy * dy

            if d2 < two_r2:
                dist = math.hypot(dx, dy)

                if dist < 1e-9:
                    # Deterministic fallback for exactly coincident centers.
                    ang = (i * 1.13 + j * 2.399) % (2.0 * math.pi)
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                else:
                    nx = dx / dist
                    ny = dy / dist

                # Position correction.
                overlap = two_r - dist
                corr = 0.5 * overlap * 0.8

                bi.pos[0] -= corr * nx
                bi.pos[1] -= corr * ny
                bj.pos[0] += corr * nx
                bj.pos[1] += corr * ny

                xi = bi.pos[0]
                yi = bi.pos[1]

                rvx = bi.vel[0] - bj.vel[0]
                rvy = bi.vel[1] - bj.vel[1]

                # Normal impulse.  n points from i to j.
                vn = rvx * nx + rvy * ny

                if vn > 1e-6:
                    imp = 0.5 * (1.0 + BALL_RESTITUTION) * vn

                    bi.vel[0] -= imp * nx
                    bi.vel[1] -= imp * ny
                    bj.vel[0] += imp * nx
                    bj.vel[1] += imp * ny

                    # Tangential friction and spin coupling.
                    tx = -ny
                    ty = nx

                    ut = (
                        rvx * tx + rvy * ty
                        + (bi.ang_vel + bj.ang_vel) * BALL_RADIUS
                    )

                    max_f = BALL_FRICTION * imp
                    J = clip(-ut / 6.0, -max_f, max_f)

                    bi.vel[0] += J * tx
                    bi.vel[1] += J * ty
                    bj.vel[0] -= J * tx
                    bj.vel[1] -= J * ty

                    spin_imp = 2.0 * J / BALL_RADIUS
                    bi.ang_vel += spin_imp
                    bj.ang_vel += spin_imp


# ---------------------------------------------------------------------------
# Rotating heptagon collisions
# ---------------------------------------------------------------------------

def resolve_wall(ball: Ball, c: float, s: float) -> None:
    r = BALL_RADIUS

    # World to local rotating frame.
    px = ball.pos[0] - CX
    py = ball.pos[1] - CY

    plx = c * px + s * py
    ply = -s * px + c * py

    # Find closest boundary point among the seven segments.
    best_d2 = float("inf")
    best_qx = best_qy = best_dx = best_dy = best_ox = best_oy = 0.0

    for e in EDGES:
        t = ((plx - e.x0) * e.seg_x + (ply - e.y0) * e.seg_y) / e.len2
        t = clip(t, 0.0, 1.0)

        qx = e.x0 + t * e.seg_x
        qy = e.y0 + t * e.seg_y

        dx = plx - qx
        dy = ply - qy
        d2 = dx * dx + dy * dy

        if d2 < best_d2:
            best_d2 = d2
            best_qx = qx
            best_qy = qy
            best_dx = dx
            best_dy = dy
            best_ox = e.out_x
            best_oy = e.out_y

    if best_d2 < r * r:
        dist = math.sqrt(best_d2)

        # Is the ball center outside the heptagon?
        outside = False
        for e in EDGES:
            if e.out_x * plx + e.out_y * ply > e.a + 1e-5:
                outside = True
                break

        if dist < 1e-9:
            nlx = -best_ox
            nly = -best_oy
        else:
            nlx = best_dx / dist
            nly = best_dy / dist

            if outside:
                nlx = -nlx
                nly = -nly

        nl_len = math.hypot(nlx, nly)
        if nl_len < 1e-9:
            nlx = nly = 0.0
        else:
            nlx /= nl_len
            nly /= nl_len

        # Contact point in world coordinates.
        qwx = CX + c * best_qx - s * best_qy
        qwy = CY + s * best_qx + c * best_qy

        # Normal in world coordinates.
        nwx = c * nlx - s * nly
        nwy = s * nlx + c * nly

        # Reposition the ball so it is exactly one radius from contact.
        plx = best_qx + nlx * r
        ply = best_qy + nly * r

        ball.pos[0] = CX + c * plx - s * ply
        ball.pos[1] = CY + s * plx + c * ply

        # Velocity of the rotating wall at the contact point.
        vwx = -OMEGA * (qwx - CX)
        vwy = OMEGA * (qwy - CY)

        uvx = ball.vel[0] - vwx
        uvy = ball.vel[1] - vwy
        vn = uvx * nwx + uvy * nwy

        if vn < -1e-6:
            impact = -vn

            rebound = WALL_RESTITUTION * impact
            rebound = min(rebound, MAX_REBOUND_SPEED)

            # Keep characteristic rebounds above one ball radius when the
            # impact is strong enough for that to be meaningful.
            if impact > MIN_REBOUND_SPEED and rebound < MIN_REBOUND_SPEED:
                rebound = MIN_REBOUND_SPEED

            e_eff = rebound / impact if impact > 1e-9 else 0.0

            # Normal velocity response.
            dvx = (1.0 + e_eff) * (-vn) * nwx
            dvy = (1.0 + e_eff) * (-vn) * nwy

            ball.vel[0] += dvx
            ball.vel[1] += dvy

            # Tangential friction and spin coupling.
            tx = -nwy
            ty = nwx

            # Ball surface velocity at contact point:
            # r_vec = -n * r, so surface velocity = v - ang_vel * r * t
            ct_vx = ball.vel[0] - ball.ang_vel * r * tx
            ct_vy = ball.vel[1] - ball.ang_vel * r * ty

            ut = (ct_vx - vwx) * tx + (ct_vy - vwy) * ty

            normal_impulse = (1.0 + e_eff) * (-vn)
            max_f = WALL_FRICTION * normal_impulse

            J = clip(-ut / 3.0, -max_f, max_f)

            ball.vel[0] += J * tx
            ball.vel[1] += J * ty

            ball.ang_vel += -2.0 * J / r

    # Final half-plane containment check in the local rotating frame.
    # This prevents any residual tunneling, especially near vertices.
    for e in EDGES:
        old_plx = plx
        old_ply = ply

        val = e.out_x * old_plx + e.out_y * old_ply
        limit = e.a - r

        if val > limit:
            pen = val - limit
            plx -= e.out_x * pen
            ply -= e.out_y * pen

            # If the center is leaving / has left the wall, remove outward
            # normal relative velocity.
            if val > limit + 1e-4:
                # Contact point on the edge line, clamped to the segment.
                qx = old_plx + (e.a - val) * e.out_x
                qy = old_ply + (e.a - val) * e.out_y

                t = ((qx - e.x0) * e.seg_x + (qy - e.y0) * e.seg_y) / e.len2
                t = clip(t, 0.0, 1.0)

                qx = e.x0 + t * e.seg_x
                qy = e.y0 + t * e.seg_y

                qwx = CX + c * qx - s * qy
                qwy = CY + s * qx + c * qy

                vwx = -OMEGA * (qwx - CX)
                vwy = OMEGA * (qwy - CY)

                owx = c * e.out_x - s * e.out_y
                owy = s * e.out_x + c * e.out_y

                rel_out = (ball.vel[0] - vwx) * owx + (ball.vel[1] - vwy) * owy

                if rel_out > 0.0:
                    ball.vel[0] -= (1.0 + WALL_RESTITUTION) * rel_out * owx
                    ball.vel[1] -= (1.0 + WALL_RESTITUTION) * rel_out * owy

    ball.pos[0] = CX + c * plx - s * ply
    ball.pos[1] = CY + s * plx + c * ply


# ---------------------------------------------------------------------------
# Rotated number drawing using seven-segment style digits
# ---------------------------------------------------------------------------

SEGMENT_COORDS = {
    "a": ((-1.0, -1.0), (1.0, -1.0)),
    "b": ((1.0, -1.0), (1.0, 0.0)),
    "c": ((1.0, 0.0), (1.0, 1.0)),
    "d": ((-1.0, 1.0), (1.0, 1.0)),
    "e": ((-1.0, 1.0), (-1.0, 0.0)),
    "f": ((-1.0, 0.0), (-1.0, -1.0)),
    "g": ((-1.0, 0.0), (1.0, 0.0)),
}

DIGIT_SEGMENTS = {
    "0": ("a", "b", "c", "d", "e", "f"),
    "1": ("b", "c"),
    "2": ("a", "b", "g", "e", "d"),
    "3": ("a", "b", "g", "c", "d"),
    "4": ("f", "g", "b", "c"),
    "5": ("a", "f", "g", "c", "d"),
    "6": ("a", "f", "g", "e", "c", "d"),
    "7": ("a", "b", "c"),
    "8": ("a", "b", "c", "d", "e", "f", "g"),
    "9": ("a", "b", "c", "d", "f", "g"),
}

DIGIT_SHIFT = {}
for _ch, _segs in DIGIT_SEGMENTS.items():
    _xs = []
    for _s in _segs:
        _x1, _y1 = SEGMENT_COORDS[_s][0]
        _x2, _y2 = SEGMENT_COORDS[_s][1]
        _xs.extend((_x1, _x2))
    DIGIT_SHIFT[_ch] = -(min(_xs) + max(_xs)) * 0.5


def draw_number(canvas: tk.Canvas, x: float, y: float, angle: float, number: int) -> None:
    text = str(number)

    if len(text) == 1:
        scale = BALL_RADIUS * 0.48
        centers = [0.0]
    else:
        scale = BALL_RADIUS * 0.32
        centers = [-1.25, 1.25]

    ca = math.cos(angle)
    sa = math.sin(angle)

    line_width = max(1, int(BALL_RADIUS * 0.18))

    for ch, cx in zip(text, centers):
        dx = DIGIT_SHIFT[ch]

        for seg in DIGIT_SEGMENTS[ch]:
            x1, y1 = SEGMENT_COORDS[seg][0]
            x2, y2 = SEGMENT_COORDS[seg][1]

            ax = x1 + cx + dx
            ay = y1
            bx = x2 + cx + dx
            by = y2

            axr = ca * ax - sa * ay
            ayr = sa * ax + ca * ay
            bxr = ca * bx - sa * by
            byr = sa * bx + ca * by

            canvas.create_line(
                x + scale * axr,
                y + scale * ayr,
                x + scale * bxr,
                y + scale * byr,
                fill=NUMBER_COLOR,
                width=line_width,
                caps="round"
            )


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.title("20 Balls in a Spinning Heptagon")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, background="#000000")
    canvas.pack()

    balls = make_balls()

    theta = 0.0
    dt = 1.0 / (FRAME_FPS * SUBSTEPS)

    def animate() -> None:
        nonlocal theta

        # Physics substeps for stability.
        for _ in range(SUBSTEPS):
            theta += OMEGA * dt
            c = math.cos(theta)
            s = math.sin(theta)

            drag = math.exp(-AIR_DRAG * dt)
            spin_drag = math.exp(-SPIN_DRAG * dt)

            for b in balls:
                b.vel[1] += GRAVITY * dt

                b.vel[0] *= drag
                b.vel[1] *= drag
                b.ang_vel *= spin_drag

                b.pos[0] += b.vel[0] * dt
                b.pos[1] += b.vel[1] * dt

                b.angle += b.ang_vel * dt

            for _ in range(COLLISION_ITER):
                resolve_ball_balls(balls)

                for b in balls:
                    resolve_wall(b, c, s)

            for b in balls:
                sp = math.hypot(b.vel[0], b.vel[1])
                if sp > MAX_SPEED:
                    f = MAX_SPEED / sp
                    b.vel[0] *= f
                    b.vel[1] *= f

                if b.ang_vel > MAX_ANG_VEL:
                    b.ang_vel = MAX_ANG_VEL
                elif b.ang_vel < -MAX_ANG_VEL:
                    b.ang_vel = -MAX_ANG_VEL

                if b.angle > 2.0 * math.pi:
                    b.angle -= 2.0 * math.pi
                elif b.angle < -2.0 * math.pi:
                    b.angle += 2.0 * math.pi

        # Drawing
        c = math.cos(theta)
        s = math.sin(theta)

        canvas.delete("all")

        # Spinning heptagon
        coords = []
        for x, y in LOCAL_VERTS:
            wx = CX + c * x - s * y
            wy = CY + s * x + c * y
            coords.append(wx)
            coords.append(wy)

        canvas.create_polygon(
            *coords,
            fill=HEPT_FILL,
            outline=HEPT_OUTLINE,
            width=4,
            smooth=False
        )

        # Balls
        r = BALL_RADIUS
        for b in balls:
            x = b.pos[0]
            y = b.pos[1]

            canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                fill=b.color,
                outline=BALL_OUTLINE,
                width=1
            )

            draw_number(canvas, x, y, b.angle, b.number)

        root.after(FRAME_MS, animate)

    animate()
    root.mainloop()


if __name__ == "__main__":
    main()
