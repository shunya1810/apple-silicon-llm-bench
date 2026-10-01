# heptagon_balls.py
# Balls bouncing inside a spinning heptagon.
# Uses only: tkinter, math, dataclasses, typing.

import math
import tkinter as tk
from dataclasses import dataclass
from typing import List, Tuple

Vector = Tuple[float, float]

# ----------------------------------------------------------------------
# Basic scene / physics constants
# ----------------------------------------------------------------------

CANVAS_SIZE = 900
CENTER_X = CANVAS_SIZE // 2
CENTER_Y = CANVAS_SIZE // 2

HEPTAGON_RADIUS = 400.0
BALL_RADIUS = 16.0
BALL_COUNT = 20

BALL_MASS = 1.0
BALL_I = 0.5 * BALL_MASS * BALL_RADIUS * BALL_RADIUS  # solid disk moment of inertia

GRAVITY = 1200.0
GRAVITY_VEC: Vector = (0.0, GRAVITY)

# 360 degrees per 5 seconds
HEPTA_OMEGA = 2.0 * math.pi / 5.0

# Material bounce height:
# Maximum possible internal drop height is approximately 2*(R - r).
# Choose a coefficient of restitution so the maximum rebound height is
# 0.8 * heptagon_radius, i.e. below the heptagon radius and above ball radius.
MAX_BOUNCE_HEIGHT = 0.8 * HEPTAGON_RADIUS
if MAX_BOUNCE_HEIGHT <= BALL_RADIUS:
    MAX_BOUNCE_HEIGHT = BALL_RADIUS * 1.25

MAX_DROP_HEIGHT = 2.0 * max(HEPTAGON_RADIUS - BALL_RADIUS, 1e-6)
WALL_RESTITUTION = math.sqrt(max(0.0, MAX_BOUNCE_HEIGHT / MAX_DROP_HEIGHT))
WALL_RESTITUTION = min(0.95, max(0.0, WALL_RESTITUTION))
BALL_RESTITUTION = min(0.75, 0.85 * WALL_RESTITUTION)

MAX_UP_SPEED = math.sqrt(2.0 * GRAVITY * MAX_BOUNCE_HEIGHT)

# Friction / damping
WALL_FRICTION = 0.32
BALL_FRICTION = 0.22
LINEAR_DAMPING = 0.05
SPIN_DAMPING = 0.25
CONTACT_SPIN_DAMPING = 0.5

REST_THRESHOLD = 10.0
POSITION_PERCENT = 0.9
POSITION_SLOP = 0.01

# Simulation stepping
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 4
SOLVER_ITERATIONS = 4

COLORS: List[str] = [
    "#f8b862", "#f6ad49", "#f39800", "#f08300", "#ec6d51",
    "#ee7948", "#ed6d3d", "#ec6800", "#ec6800", "#ee7800",
    "#eb6238", "#ea5506", "#ea5506", "#eb6101", "#e49e61",
    "#e45e32", "#e17b34", "#dd7a56", "#db8449", "#d66a35",
]

# ----------------------------------------------------------------------
# Small vector helpers
# ----------------------------------------------------------------------

def rotate_vec(v: Vector, angle: float) -> Vector:
    """Rotate a 2D vector by angle radians."""
    c = math.cos(angle)
    s = math.sin(angle)
    return (v[0] * c - v[1] * s,
            v[0] * s + v[1] * c)


def spin_cross(spin: float, rc: Vector) -> Vector:
    """
    In 2D, scalar angular velocity crossed with radius vector:
        spin x rc = (-spin * rc.y, spin * rc.x)
    """
    return (-spin * rc[1], spin * rc[0])


def omega_cross(r: Vector) -> Vector:
    """
    Velocity of a point on a rigid body rotating about origin:
        omega x r = (-omega * r.y, omega * r.x)
    """
    return (-HEPTA_OMEGA * r[1], HEPTA_OMEGA * r[0])


def cross2(a: Vector, b: Vector) -> float:
    """2D scalar cross product: a x b."""
    return a[0] * b[1] - a[1] * b[0]

# ----------------------------------------------------------------------
# Heptagon geometry
# ----------------------------------------------------------------------

# Regular heptagon centered at local origin.
# One vertex points "up" in screen coordinates.
HEPTA_VERTS: List[Vector] = []
for _k in range(7):
    _a = -math.pi / 2.0 + _k * 2.0 * math.pi / 7.0
    HEPTA_VERTS.append((HEPTAGON_RADIUS * math.cos(_a),
                        HEPTAGON_RADIUS * math.sin(_a)))

HEPTA_EDGES: List[Tuple[Vector, Vector]] = [
    (HEPTA_VERTS[i], HEPTA_VERTS[(i + 1) % 7]) for i in range(7)
]

# Inward unit normals for the fixed local heptagon.
HEPTA_INWARD_NORMALS: List[Vector] = []
for _a, _b in HEPTA_EDGES:
    _tx = _b[0] - _a[0]
    _ty = _b[1] - _a[1]
    _L = math.hypot(_tx, _ty)
    # The vertex order is algebraically counter-clockwise; left normal is inward.
    HEPTA_INWARD_NORMALS.append((-_ty / _L, _tx / _L))


def point_inside_local(q: Vector) -> bool:
    """Return True if q is inside the fixed local heptagon."""
    for a, n in zip(HEPTA_EDGES, HEPTA_INWARD_NORMALS):
        if (q[0] - a[0]) * n[0] + (q[1] - a[1]) * n[1] < -1e-7:
            return False
    return True


def closest_point_on_segment(p: Vector, a: Vector, b: Vector) -> Vector:
    """Closest point on segment AB to point P."""
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    denom = abx * abx + aby * aby

    if denom < 1e-12:
        return a

    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / denom
    t = max(0.0, min(1.0, t))
    return (a[0] + t * abx, a[1] + t * aby)


def polygon_sdf_local(q: Vector) -> Tuple[float, Vector, Vector, bool]:
    """
    Signed distance from q to the fixed local heptagon boundary.

    Returns:
        s:      signed distance. Positive inside, negative outside.
        n:      unit normal pointing into the interior (increases s).
        p:      closest boundary point to q.
        inside: boolean interior test.
    """
    inside = point_inside_local(q)

    best_p = (0.0, 0.0)
    best_d2 = float("inf")

    for a, b in HEPTA_EDGES:
        p = closest_point_on_segment(q, a, b)
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_p = p

    dist = math.sqrt(best_d2)

    if dist < 1e-9:
        # q is exactly on the boundary. Use direction toward center as fallback.
        L = math.hypot(best_p[0], best_p[1])
        if L < 1e-9:
            n = (1.0, 0.0)
        else:
            n = (-best_p[0] / L, -best_p[1] / L)
        s = 0.0
    elif inside:
        n = ((q[0] - best_p[0]) / dist,
             (q[1] - best_p[1]) / dist)
        s = dist
    else:
        n = ((best_p[0] - q[0]) / dist,
             (best_p[1] - q[1]) / dist)
        s = -dist

    return s, n, best_p, inside

# ----------------------------------------------------------------------
# Ball data
# ----------------------------------------------------------------------

@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    spin: float      # angular velocity, rad/s
    angle: float     # visual number rotation, rad
    radius: float
    color: str
    label: int

# ----------------------------------------------------------------------
# Application / simulation
# ----------------------------------------------------------------------

class HeptagonBallsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("20 Balls in a Spinning Heptagon")
        root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="#181818"
        )
        self.canvas.pack()

        self.hept_angle = 0.0

        self.polygon_item = self.canvas.create_polygon(
            [],
            fill="#101010",
            outline="#777777",
            width=3
        )

        self.ball_items: List[int] = []
        self.text_items: List[int] = []

        # Create ovals first, then all numbers, so numbers stay on top.
        for color in COLORS:
            self.ball_items.append(self.canvas.create_oval(
                -9999, -9999, 0, 0,
                fill=color,
                outline="black",
                width=1,
                tags="ball"
            ))

        for _ in COLORS:
            self.text_items.append(self.canvas.create_text(
                0, 0,
                text="",
                fill="black",
                font=("Helvetica", 9, "bold"),
                anchor="center",
                angle=0,
                tags="number"
            ))

        self.canvas.tag_raise("number")

        self.balls: List[Ball] = []
        self.after_id = None

        self.reset()

        root.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.after_id = root.after(16, self.tick)

    def cleanup(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.root.destroy()

    def reset(self) -> None:
        self.balls = []

        # Nominal release point is the heptagon center.
        # A tiny deterministic spread is used before position-only relaxation
        # because exact coincident centers are non-physical and numerically degenerate.
        for i in range(BALL_COUNT):
            a = i * 2.0 * math.pi / BALL_COUNT
            off = BALL_RADIUS * 0.3
            x = CENTER_X + off * math.cos(a)
            y = CENTER_Y + off * math.sin(a)

            self.balls.append(
                Ball(
                    x=x,
                    y=y,
                    vx=0.0,
                    vy=0.0,
                    spin=0.0,
                    angle=0.0,
                    radius=BALL_RADIUS,
                    color=COLORS[i],
                    label=i + 1
                )
            )

        self.hept_angle = 0.0
        self.relax_initial()

        # Re-center the relaxed cluster exactly on the heptagon center.
        com_x = sum(b.x for b in self.balls) / BALL_COUNT
        com_y = sum(b.y for b in self.balls) / BALL_COUNT
        for b in self.balls:
            b.x += CENTER_X - com_x
            b.y += CENTER_Y - com_y
            b.vx = 0.0
            b.vy = 0.0
            b.spin = 0.0
            b.angle = 0.0

        self.draw()

    def relax_initial(self) -> None:
        """
        Position-only relaxation of the initial central cluster.
        This keeps the start centered on the heptagon center while removing
        the impossible exact overlap of all 20 balls at one point.
        """
        for _ in range(120):
            any_overlap = False

            for i in range(BALL_COUNT):
                bi = self.balls[i]
                for j in range(i + 1, BALL_COUNT):
                    bj = self.balls[j]

                    dx = bj.x - bi.x
                    dy = bj.y - bi.y
                    d = math.hypot(dx, dy)

                    if d < 2.0 * BALL_RADIUS:
                        if d < 1e-9:
                            ang = (i * 2.399963 + j * 1.7) % (2.0 * math.pi)
                            nx = math.cos(ang)
                            ny = math.sin(ang)
                        else:
                            nx = dx / d
                            ny = dy / d

                        corr = (2.0 * BALL_RADIUS - d) * 0.5

                        bi.x -= nx * corr
                        bi.y -= ny * corr
                        bj.x += nx * corr
                        bj.y += ny * corr

                        any_overlap = True

            if not any_overlap:
                break

    def tick(self) -> None:
        if self.root.winfo_exists():
            h = FRAME_DT / SUBSTEPS
            for _ in range(SUBSTEPS):
                self.step(h)

            self.draw()
            self.after_id = self.root.after(16, self.tick)
        else:
            self.after_id = None

    def step(self, h: float) -> None:
        # Advance rotating heptagon
        self.hept_angle = (self.hept_angle + HEPTA_OMEGA * h) % (2.0 * math.pi)

        # Integrate balls
        for b in self.balls:
            b.vy += GRAVITY * h

            ld = max(0.0, 1.0 - LINEAR_DAMPING * h)
            b.vx *= ld
            b.vy *= ld

            b.x += b.vx * h
            b.y += b.vy * h

            b.spin *= max(0.0, 1.0 - SPIN_DAMPING * h)
            b.angle = (b.angle + b.spin * h) % (2.0 * math.pi)

            # Enforce material maximum vertical rebound height.
            if b.vy < -MAX_UP_SPEED:
                b.vy = -MAX_UP_SPEED

        # Iterative constraint solver
        for _ in range(SOLVER_ITERATIONS):
            self.resolve_ball_pairs()
            self.resolve_walls(h)

        self.enforce_containment()

        # Final safety clamp for vertical bounce height
        for b in self.balls:
            if b.vy < -MAX_UP_SPEED:
                b.vy = -MAX_UP_SPEED

    def resolve_ball_pairs(self) -> None:
        balls = self.balls

        for i in range(BALL_COUNT):
            bi = balls[i]
            for j in range(i + 1, BALL_COUNT):
                bj = balls[j]

                dx = bj.x - bi.x
                dy = bj.y - bi.y
                d = math.hypot(dx, dy)

                if d < 2.0 * BALL_RADIUS + 1e-9:
                    if d < 1e-9:
                        ang = (i * 2.399963 + j * 1.31) % (2.0 * math.pi)
                        nx = math.cos(ang)
                        ny = math.sin(ang)
                    else:
                        nx = dx / d
                        ny = dy / d

                    pen = 2.0 * BALL_RADIUS - d

                    # Position correction
                    corr = max(pen - POSITION_SLOP, 0.0) * POSITION_PERCENT * 0.5
                    bi.x -= nx * corr
                    bi.y -= ny * corr
                    bj.x += nx * corr
                    bj.y += ny * corr

                    # Normal impulse
                    rvx = bj.vx - bi.vx
                    rvy = bj.vy - bi.vy
                    vn = rvx * nx + rvy * ny

                    jn = 0.0
                    if vn < 0.0:
                        e = BALL_RESTITUTION if -vn > REST_THRESHOLD else 0.0
                        inv_mass = 2.0 / BALL_MASS
                        jn = -(1.0 + e) * vn / inv_mass

                        bi.vx -= jn * nx
                        bi.vy -= jn * ny
                        bj.vx += jn * nx
                        bj.vy += jn * ny

                    # Friction + spin
                    if pen > 0.0 and vn < REST_THRESHOLD:
                        r = BALL_RADIUS
                        rci = (r * nx, r * ny)
                        rcj = (-r * nx, -r * ny)

                        si = spin_cross(bi.spin, rci)
                        sj = spin_cross(bj.spin, rcj)

                        # Contact surface relative velocity
                        rvx = (bj.vx + sj[0]) - (bi.vx + si[0])
                        rvy = (bj.vy + sj[1]) - (bi.vy + si[1])

                        vn_s = rvx * nx + rvy * ny
                        tx = rvx - vn_s * nx
                        ty = rvy - vn_s * ny
                        tmag = math.hypot(tx, ty)

                        if tmag > 1e-7:
                            cap = BALL_FRICTION * (abs(jn) if vn < 0.0 else 0.0)

                            if cap > 0.0:
                                # Effective inverse mass for tangential contact:
                                # 1/m1 + 1/m2 + r1^2/I1 + r2^2/I2
                                inv_eff = (2.0 / BALL_MASS) + 2.0 * (r * r) / BALL_I

                                jt = -tmag / inv_eff

                                if jt > cap:
                                    jt = cap
                                elif jt < -cap:
                                    jt = -cap

                                imp_jx = tx * (jt / tmag)
                                imp_jy = ty * (jt / tmag)

                                bi.vx -= imp_jx
                                bi.vy -= imp_jy
                                bj.vx += imp_jx
                                bj.vy += imp_jy

                                torque_i = cross2(rci, (-imp_jx, -imp_jy))
                                torque_j = cross2(rcj, (imp_jx, imp_jy))

                                bi.spin += torque_i / BALL_I
                                bj.spin += torque_j / BALL_I

    def resolve_walls(self, h: float) -> None:
        for b in self.balls:
            q_world = (b.x - CENTER_X, b.y - CENTER_Y)
            q_local = rotate_vec(q_world, -self.hept_angle)

            s, n_local, p_local, inside = polygon_sdf_local(q_local)

            if s < BALL_RADIUS:
                pen = BALL_RADIUS - s

                n_world = rotate_vec(n_local, self.hept_angle)
                p_rel_world = rotate_vec(p_local, self.hept_angle)

                wall_v = omega_cross(p_rel_world)

                # Relative center velocity at contact
                vx = b.vx - wall_v[0]
                vy = b.vy - wall_v[1]
                vn = vx * n_world[0] + vy * n_world[1]

                jn = 0.0
                if vn < 0.0:
                    e = WALL_RESTITUTION if -vn > REST_THRESHOLD else 0.0
                    jn = -(1.0 + e) * vn / (1.0 / BALL_MASS)

                    b.vx += jn * n_world[0]
                    b.vy += jn * n_world[1]

                # Friction, spin, and rolling contact
                if pen > 0.0 and vn < REST_THRESHOLD:
                    # Direction from ball center toward the actual boundary
                    if inside:
                        bdir = (-n_world[0], -n_world[1])
                    else:
                        bdir = (n_world[0], n_world[1])

                    rc = (BALL_RADIUS * bdir[0], BALL_RADIUS * bdir[1])
                    sc = spin_cross(b.spin, rc)

                    surf_x = b.vx + sc[0] - wall_v[0]
                    surf_y = b.vy + sc[1] - wall_v[1]

                    vn_s = surf_x * n_world[0] + surf_y * n_world[1]
                    tx = surf_x - vn_s * n_world[0]
                    ty = surf_y - vn_s * n_world[1]
                    tmag = math.hypot(tx, ty)

                    if tmag > 1e-7:
                        # Estimate normal support impulse from gravity component.
                        support = max(
                            0.0,
                            -(GRAVITY_VEC[0] * n_world[0] + GRAVITY_VEC[1] * n_world[1])
                        ) * BALL_MASS * h

                        cap = WALL_FRICTION * max(abs(jn), support)

                        if cap > 0.0:
                            # Effective inverse mass for ball against infinite wall:
                            # 1/m + r^2/I
                            inv_eff = 1.0 / BALL_MASS + (BALL_RADIUS * BALL_RADIUS) / BALL_I

                            jt = -tmag / inv_eff

                            if jt > cap:
                                jt = cap
                            elif jt < -cap:
                                jt = -cap

                            imp_x = tx * (jt / tmag)
                            imp_y = ty * (jt / tmag)

                            b.vx += imp_x
                            b.vy += imp_y

                            torque = cross2(rc, (imp_x, imp_y))
                            b.spin += torque / BALL_I

                            b.spin *= max(0.0, 1.0 - CONTACT_SPIN_DAMPING * h)

                # Position correction into the interior
                corr = max(pen - POSITION_SLOP, 0.0) * POSITION_PERCENT
                b.x += n_world[0] * corr
                b.y += n_world[1] * corr

    def enforce_containment(self) -> None:
        """
        Final safety containment pass.
        The main solver usually keeps balls inside; this prevents rare escapes.
        """
        for b in self.balls:
            q_world = (b.x - CENTER_X, b.y - CENTER_Y)
            q_local = rotate_vec(q_world, -self.hept_angle)

            s, n_local, _p, _inside = polygon_sdf_local(q_local)

            if s < 0.0:
                pen = BALL_RADIUS - s
                n_world = rotate_vec(n_local, self.hept_angle)
                b.x += n_world[0] * pen
                b.y += n_world[1] * pen
            elif s < BALL_RADIUS * 0.5:
                pen = BALL_RADIUS - s
                n_world = rotate_vec(n_local, self.hept_angle)
                b.x += n_world[0] * pen * 0.35
                b.y += n_world[1] * pen * 0.35

    def draw(self) -> None:
        c = math.cos(self.hept_angle)
        s = math.sin(self.hept_angle)

        pts = []
        for vx, vy in HEPTA_VERTS:
            pts.append(CENTER_X + vx * c - vy * s)
            pts.append(CENTER_Y + vx * s + vy * c)

        self.canvas.coords(self.polygon_item, *pts)

        for i, b in enumerate(self.balls):
            self.canvas.coords(
                self.ball_items[i],
                b.x - b.radius,
                b.y - b.radius,
                b.x + b.radius,
                b.y + b.radius
            )

            deg = math.degrees(b.angle % (2.0 * math.pi))
            self.canvas.itemconfig(
                self.text_items[i],
                text=str(b.label),
                x=b.x,
                y=b.y,
                angle=deg
            )


def main() -> None:
    root = tk.Tk()
    HeptagonBallsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
