"""
Fake-3D raycaster shooter.

Optional: `pip install numpy` for textured, lit floors and ceilings with lamps.
Without numpy the game still runs, using a simple gradient floor and ceiling.
Put a picture named scary.png / scary.jpg next to this file to replace the hidden face.
"""
import array
import math
import os
import random
from collections import deque

import pygame

try:
    import numpy as np
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

W, H = 800, 450              # lower render resolution for weak GPUs
FOV = math.radians(75)
RAYS = 160                      # fewer, wider columns = much faster raycasting
COL_W = W // RAYS
MAX_DEPTH = 20
MOVE_SPEED = 3.0
SPRINT_MULT = 1.65
TURN_SPEED = 2.2
MOUSE_SENS = 0.0025             # radians per pixel (left/right)
MOUSE_SENS_Y = 1.0              # screen pixels of tilt per pixel of mouse (up/down)
MAX_PITCH = int(H * 0.75)
SCARE_DIST = 1.3
SCARE_TIME = 1.8
TEX = 128                       # wall texture size
FT = 128                        # floor / ceiling tile size
LV = 32                         # light levels for pre-shaded wall textures
SPR_LV = 12                     # light levels for pre-shaded sprites
LVF = 48                        # light levels baked into the floor/ceiling tables
FLAT_SCALE = 4                  # floor/ceiling render at quarter resolution
ENABLE_BLOOM = False             # expensive full-screen post effect; enable for screenshots
ENABLE_GRAIN = False             # expensive animated full-screen noise

# look
AMB = (0.085, 0.105, 0.19)      # cold ambient light
WARM = (1.0, 0.92, 0.78)        # warm flashlight colour
FOG = (9, 12, 22)               # distance fog colour
TORCH = 1.4                     # flashlight strength

PLAYER_HP = 100
MAG_SIZE = 12
START_RESERVE = 24
MAX_RESERVE = 108
PICKUP_AMMO = 12
PICKUP_COUNT = 7
FIRE_DELAY = 0.28
RELOAD_TIME = 1.1
GUN_DAMAGE = 34
ENEMY_HP = 100
ENEMY_DAMAGE = 12
ENEMY_AGGRO = 11.0
ENEMY_HSCALE = 0.85

MAP = [
    "####################",
    "#..................#",
    "#..##....###....##.#",
    "#..#......#.....#..#",
    "#.........#........#",
    "#....####.....###..#",
    "#....#..........#..#",
    "#....#...##.....#..#",
    "#........##........#",
    "#..###..........##.#",
    "#..#.....####......#",
    "#..#.........#.....#",
    "#.......##...#..##.#",
    "#..##...##.........#",
    "#..................#",
    "####################",
]
MAP_H, MAP_W = len(MAP), len(MAP[0])
OPEN_CELLS = [(x + 0.5, y + 0.5) for y in range(MAP_H) for x in range(MAP_W) if MAP[y][x] != "#"]

START = (2.5, 2.5, 0.0)
FACE_POS = (17.5, 13.5)         # where the scary picture is hidden
BG_HORIZON = int(H * 1.5)

# per-column constants
OFFS = [(i / RAYS - 0.5) * FOV for i in range(RAYS)]
COS = [math.cos(o) for o in OFFS]
BEAM = [math.exp(-(o / 0.55) ** 2) for o in OFFS]


def light_t(d, off, flick, flash):
    """Flashlight cone + distance falloff + muzzle flash. ~0 dark, 1 fully lit."""
    return ((0.30 + 0.70 * math.exp(-(off / 0.55) ** 2)) * TORCH * flick / (1 + 0.07 * d * d)
            + flash * 0.9 / (1 + 0.03 * d * d))


# ------------------------------------------------------------------ map / rays
def is_wall(x, y):
    mx, my = int(x), int(y)
    if mx < 0 or my < 0 or mx >= MAP_W or my >= MAP_H:
        return True
    return MAP[my][mx] == "#"


def has_los(x0, y0, x1, y1):
    n = int(math.hypot(x1 - x0, y1 - y0) / 0.2) + 1
    for i in range(1, n):
        t = i / n
        if is_wall(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
            return False
    return True


def compute_flow(cx, cy):
    """BFS distance from every open cell to the player's cell (enemy pathfinding)."""
    dist = [[-1] * MAP_W for _ in range(MAP_H)]
    dist[cy][cx] = 0
    q = deque([(cx, cy)])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H and MAP[ny][nx] != "#" and dist[ny][nx] < 0:
                dist[ny][nx] = dist[y][x] + 1
                q.append((nx, ny))
    return dist


def cast_ray(px, py, angle):
    """DDA. Returns (distance, side, wall_x, (cell_x, cell_y))."""
    dx, dy = math.cos(angle), math.sin(angle)
    map_x, map_y = int(px), int(py)
    delta_x = abs(1 / dx) if dx else 1e30
    delta_y = abs(1 / dy) if dy else 1e30

    if dx < 0:
        step_x, side_x = -1, (px - map_x) * delta_x
    else:
        step_x, side_x = 1, (map_x + 1 - px) * delta_x
    if dy < 0:
        step_y, side_y = -1, (py - map_y) * delta_y
    else:
        step_y, side_y = 1, (map_y + 1 - py) * delta_y

    side = 0
    while True:
        if side_x < side_y:
            side_x += delta_x
            map_x += step_x
            side = 0
        else:
            side_y += delta_y
            map_y += step_y
            side = 1
        if map_x < 0 or map_y < 0 or map_x >= MAP_W or map_y >= MAP_H:
            break
        if MAP[map_y][map_x] == "#":
            break

    raw = (side_x - delta_x) if side == 0 else (side_y - delta_y)
    wx = (py + raw * dy) if side == 0 else (px + raw * dx)
    wx -= math.floor(wx)
    if (side == 0 and dx > 0) or (side == 1 and dy < 0):
        wx = 1 - wx
    return min(raw, MAX_DEPTH), side, wx, (map_x, map_y)


def wall_kind(cx, cy):
    if cx in (0, MAP_W - 1) or cy in (0, MAP_H - 1):
        return "s"                                   # stone outer wall
    return "m" if (cx * 7 + cy * 13) % 6 == 0 else "b"   # some metal panels, rest brick


# ---------------------------------------------------------------- textures
def _clip(c, v=0):
    return tuple(max(0, min(255, x + v)) for x in c)


def _noise(s, amount, density=5):
    w, h = s.get_size()
    for _ in range(w * h // density):
        x, y = random.randrange(w), random.randrange(h)
        s.set_at((x, y), _clip(s.get_at((x, y))[:3], random.randint(-amount, amount)))


def _decorate_wall(s, size=TEX, moss=True):
    """Baked ambient occlusion, drips, cracks and moss."""
    ov = pygame.Surface((size, size), pygame.SRCALPHA)
    top = int(size * 0.14)
    for y in range(top):
        ov.fill((0, 0, 0, int(120 * (1 - y / top) ** 1.5)), (0, y, size, 1))
    bot = int(size * 0.80)
    for y in range(bot, size):
        t = (y - bot) / (size - bot)
        ov.fill((8, 6, 4, int(165 * t ** 1.3)), (0, y, size, 1))
    for _ in range(7):                                                   # drip stains
        x, y0 = random.randint(6, size - 10), random.randint(0, int(size * 0.4))
        ln, w = random.randint(20, 70), random.randint(2, 5)
        for k in range(ln):
            pygame.draw.line(ov, (10, 8, 6, int(60 * (1 - k / ln))), (x, y0 + k), (x + w, y0 + k))
    for _ in range(3):                                                   # cracks
        x, y = random.randint(10, size - 10), random.randint(10, size // 2)
        for _ in range(random.randint(8, 16)):
            nx, ny = x + random.randint(-4, 4), y + random.randint(3, 8)
            pygame.draw.line(ov, (6, 5, 5, 190), (x, y), (nx, ny), 1)
            x, y = max(4, min(size - 4, nx)), ny
    if moss:
        for _ in range(9):
            x, y = random.randint(8, size - 8), random.randint(int(size * 0.8), size - 2)
            for _ in range(5):
                pygame.draw.circle(ov, (34, 72, 36, 80), (x + random.randint(-7, 7), y + random.randint(-5, 3)),
                                   random.randint(3, 8))
    s.blit(ov, (0, 0))
    return s


def make_blocks(base, mortar, rows, cols, stagger=True, size=TEX):
    s = pygame.Surface((size, size))
    s.fill(mortar)
    bh, bw = size // rows, size // cols
    for r in range(rows):
        off = bw // 2 if (stagger and r % 2) else 0
        for c in range(-1, cols + 1):
            x, y = c * bw - off, r * bh
            col = _clip(base, random.randint(-18, 18))
            rect = pygame.Rect(x + 2, y + 2, bw - 4, bh - 4)
            pygame.draw.rect(s, col, rect)
            pygame.draw.line(s, _clip(col, 28), rect.topleft, rect.topright)
            pygame.draw.line(s, _clip(col, -34), rect.bottomleft, rect.bottomright)
    _noise(s, 14)
    _decorate_wall(s)
    return s.convert()


def make_metal(size=TEX):
    s = pygame.Surface((size, size))
    s.fill((60, 66, 72))
    half = size // 2
    for r in range(2):
        for c in range(2):
            rect = pygame.Rect(c * half + 3, r * half + 3, half - 6, half - 6)
            col = _clip((92, 98, 106), random.randint(-10, 10))
            pygame.draw.rect(s, col, rect)
            pygame.draw.line(s, _clip(col, 34), rect.topleft, rect.topright, 2)
            pygame.draw.line(s, _clip(col, -40), rect.bottomleft, rect.bottomright, 2)
            pygame.draw.line(s, _clip(col, 20), rect.topleft, rect.bottomleft, 1)
            for cx, cy in ((rect.left + 7, rect.top + 7), (rect.right - 8, rect.top + 7),
                           (rect.left + 7, rect.bottom - 8), (rect.right - 8, rect.bottom - 8)):
                pygame.draw.circle(s, (44, 48, 54), (cx, cy), 4)
                pygame.draw.circle(s, (150, 156, 166), (cx - 1, cy - 1), 2)
    _noise(s, 16)
    y0, y1 = size - 34, size - 18                                    # hazard stripe
    pygame.draw.rect(s, (22, 22, 20), (0, y0, size, y1 - y0))
    for i in range(-2, size // 16 + 2):
        x = i * 16
        pygame.draw.polygon(s, (205, 172, 34), [(x, y1), (x + 8, y1), (x + 20, y0), (x + 12, y0)])
    ov = pygame.Surface((size, size), pygame.SRCALPHA)
    for _ in range(9):                                               # rust streaks
        x, y0r = random.randint(6, size - 10), random.randint(0, int(size * 0.5))
        ln = random.randint(20, 60)
        for k in range(ln):
            pygame.draw.line(ov, (130, 62, 24, int(85 * (1 - k / ln))), (x, y0r + k), (x + 3, y0r + k))
    s.blit(ov, (0, 0))
    _decorate_wall(s, moss=False)
    return s.convert()


def make_shaded(tex):
    """Pre-lit copies of a texture (cold ambient + warm light, fading into fog)."""
    out = []
    for i in range(LV):
        t = i / (LV - 1)
        c = tex.copy()
        c.fill(tuple(int(255 * min(1.0, a + w * t)) for a, w in zip(AMB, WARM)), special_flags=pygame.BLEND_MULT)
        c.fill(tuple(int(f * (1 - t)) for f in FOG), special_flags=pygame.BLEND_ADD)
        out.append(c)
    return out


def make_sprite_levels(img):
    out = []
    for i in range(SPR_LV):
        t = i / (SPR_LV - 1)
        c = img.copy()
        c.fill(tuple(int(255 * min(1.0, a + w * t)) for a, w in zip(AMB, WARM)) + (255,),
               special_flags=pygame.BLEND_RGBA_MULT)
        c.fill(tuple(int(f * (1 - t)) for f in FOG), special_flags=pygame.BLEND_RGB_ADD)
        out.append(c)
    return out


def make_floor_tile(base):
    s = pygame.Surface((FT, FT))
    s.fill(base)
    _noise(s, 14, 3)
    ov = pygame.Surface((FT, FT), pygame.SRCALPHA)
    for _ in range(6):
        pygame.draw.ellipse(ov, (12, 10, 8, random.randint(24, 55)),
                            (random.randint(0, FT - 40), random.randint(0, FT - 40),
                             random.randint(20, 60), random.randint(14, 40)))
    for _ in range(2):
        x, y = random.randint(10, FT - 10), random.randint(10, FT - 10)
        for _ in range(10):
            nx, ny = x + random.randint(-9, 9), y + random.randint(-9, 9)
            pygame.draw.line(ov, (10, 9, 9, 170), (x, y), (nx, ny))
            x, y = max(2, min(FT - 3, nx)), max(2, min(FT - 3, ny))
    s.blit(ov, (0, 0))
    pygame.draw.rect(s, (22, 21, 20), (0, 0, FT, FT), 3)
    pygame.draw.line(s, _clip(base, 22), (3, 3), (FT - 4, 3))
    pygame.draw.line(s, _clip(base, 22), (3, 3), (3, FT - 4))
    return s


def make_ceiling_tile(lamp):
    s = pygame.Surface((FT, FT))
    s.fill((46, 50, 57))
    _noise(s, 10, 4)
    pygame.draw.rect(s, (20, 22, 26), (0, 0, FT, FT), 3)
    for cx, cy in ((12, 12), (FT - 13, 12), (12, FT - 13), (FT - 13, FT - 13)):
        pygame.draw.circle(s, (30, 33, 38), (cx, cy), 4)
        pygame.draw.circle(s, (90, 96, 106), (cx - 1, cy - 1), 2)
    if lamp:
        pygame.draw.rect(s, (18, 20, 24), (20, 44, 88, 40))
        pygame.draw.rect(s, (255, 255, 255), (28, 52, 72, 24))      # emissive lamp (pure white)
    else:
        pygame.draw.rect(s, (18, 20, 24), (30, 30, 68, 68))
        for i in range(8):
            pygame.draw.rect(s, (58, 62, 70), (34, 34 + i * 8, 60, 4))
    return s


def make_background():
    """Fallback gradient floor/ceiling (used when numpy is missing)."""
    bg = pygame.Surface((W, H * 3))
    ceil_top, floor_bot = (2, 3, 6), (44, 40, 40)
    for y in range(H * 3):
        if y < BG_HORIZON:
            t = (y / BG_HORIZON) ** 0.7
            col = [int(a + (b - a) * t) for a, b in zip(ceil_top, FOG)]
        else:
            t = ((y - BG_HORIZON) / (H * 3 - BG_HORIZON)) ** 0.8
            col = [int(a + (b - a) * t) for a, b in zip(FOG, floor_bot)]
        pygame.draw.line(bg, col, (0, y), (W, y))
    return bg.convert()


def _radial_small(color, power, scale):
    small = pygame.Surface((64, 36), pygame.SRCALPHA)
    for y in range(36):
        for x in range(64):
            nx, ny = (x / 63 - 0.5) * 2, (y / 35 - 0.5) * 2
            r = min(1.0, math.hypot(nx, ny) / 1.35)
            small.set_at((x, y), color + (int(scale * r ** power),))
    return pygame.transform.smoothscale(small, (W, H)).convert_alpha()


def make_vignette():
    return _radial_small((0, 0, 0), 2.2, 225)


def make_hurt_vignette():
    return _radial_small((190, 0, 0), 1.6, 255)


def make_puff(color, size=64):
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    for y in range(size):
        for x in range(size):
            r = math.hypot(x - c, y - c) / c
            if r < 1:
                a = (1 - r) ** 1.6 * 255 * random.uniform(0.75, 1.0)
                s.set_at((x, y), color + (int(a),))
    return s


def make_glow(size, color):
    s = pygame.Surface((size, size))
    c = size // 2
    for r in range(c, 0, -2):
        f = (1 - r / c) ** 2.2
        pygame.draw.circle(s, tuple(int(ch * f) for ch in color), (c, c), r)
    return s


def make_soft_ellipse(w, h, alpha):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        for x in range(w):
            nx, ny = (x / (w - 1) - 0.5) * 2, (y / (h - 1) - 0.5) * 2
            a = max(0.0, 1 - math.hypot(nx, ny)) ** 1.3 * alpha
            s.set_at((x, y), (0, 0, 0, int(a)))
    return s


def make_grain():
    out = []
    for _ in range(5):
        small = pygame.Surface((W // 3, H // 3))
        for y in range(H // 3):
            for x in range(W // 3):
                v = random.randint(0, 9)
                small.set_at((x, y), (v, v, v))
        out.append(pygame.transform.scale(small, (W, H)).convert())
    return out


def make_face(size=256):
    """scary.png/jpg next to this script is used if present (aspect ratio kept)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("scary.png", "scary.jpg", "scary.jpeg"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            iw, ih = img.get_size()
            scale = 512 / max(iw, ih)
            return pygame.transform.smoothscale(img, (max(1, int(iw * scale)), max(1, int(ih * scale))))

    s = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (205, 200, 185), (30, 8, 196, 240))
    pygame.draw.ellipse(s, (90, 80, 75), (30, 8, 196, 240), 5)
    pygame.draw.ellipse(s, (10, 0, 0), (45, 70, 72, 62))
    pygame.draw.ellipse(s, (10, 0, 0), (139, 70, 72, 62))
    pygame.draw.circle(s, (255, 20, 20), (82, 102), 9)
    pygame.draw.circle(s, (255, 20, 20), (174, 102), 9)
    pygame.draw.circle(s, (255, 255, 255), (80, 99), 3)
    pygame.draw.circle(s, (255, 255, 255), (172, 99), 3)
    pygame.draw.line(s, (20, 10, 10), (40, 60), (120, 85), 8)
    pygame.draw.line(s, (20, 10, 10), (216, 60), (136, 85), 8)
    pygame.draw.polygon(s, (60, 20, 20), [(128, 120), (112, 160), (124, 156)])
    pygame.draw.polygon(s, (60, 20, 20), [(128, 120), (144, 160), (132, 156)])
    mouth = [(55, 172), (90, 165), (128, 175), (166, 165), (201, 172),
             (185, 215), (150, 240), (106, 240), (71, 215)]
    pygame.draw.polygon(s, (15, 0, 0), mouth)
    for i in range(8):
        x = 62 + i * 18
        pygame.draw.polygon(s, (235, 230, 200), [(x, 170), (x + 16, 170), (x + 8, 196)])
    for i in range(6):
        x = 80 + i * 18
        pygame.draw.polygon(s, (235, 230, 200), [(x, 238), (x + 16, 238), (x + 8, 214)])
    for x, ln in ((70, 120), (92, 90), (160, 95), (182, 125)):
        pygame.draw.line(s, (150, 0, 0), (x, 125), (x + random.randint(-4, 4), 125 + ln), 4)
    return s


def make_enemy(frame):
    """A shambling ghoul; two frames give a walk cycle."""
    s = pygame.Surface((128, 192), pygame.SRCALPHA)
    skin, dark, cloth = (138, 160, 118), (36, 42, 34), (70, 78, 62)
    a = frame
    ly = (134, 140) if a == 0 else (140, 134)
    pygame.draw.rect(s, dark, (42, ly[0], 20, 192 - ly[0]))
    pygame.draw.rect(s, dark, (68, ly[1], 20, 192 - ly[1]))
    left_hand, right_hand = (12, 150 - a * 14), (116, 136 + a * 14)
    pygame.draw.line(s, skin, (34, 82), left_hand, 14)
    pygame.draw.line(s, skin, (94, 82), right_hand, 14)
    pygame.draw.line(s, _clip(skin, -40), (34, 82), left_hand, 4)
    pygame.draw.line(s, _clip(skin, -40), (94, 82), right_hand, 4)
    for hx, hy in (left_hand, right_hand):
        for d in (-6, 0, 6):
            pygame.draw.polygon(s, (225, 220, 190), [(hx + d - 3, hy), (hx + d + 3, hy), (hx + d, hy + 18)])
    torso = [(30, 70), (98, 70), (106, 120), (100, 140), (90, 132), (80, 146), (68, 134),
             (56, 148), (44, 134), (34, 142), (24, 120)]
    pygame.draw.polygon(s, cloth, torso)
    pygame.draw.polygon(s, dark, torso, 3)
    for x0 in (40, 56, 72, 88):
        pygame.draw.line(s, _clip(cloth, -26), (x0, 76), (x0 + random.randint(-4, 4), 130), 2)
    for bx, by, bw, bh in ((44, 90, 14, 10), (70, 104, 18, 8), (56, 120, 10, 14)):
        pygame.draw.ellipse(s, (130, 12, 12), (bx, by, bw, bh))
    pygame.draw.ellipse(s, skin, (36, 6, 56, 66))
    pygame.draw.ellipse(s, _clip(skin, -50), (36, 6, 56, 66), 3)
    pygame.draw.ellipse(s, _clip(skin, -28), (42, 46, 14, 16))          # hollow cheeks
    pygame.draw.ellipse(s, _clip(skin, -28), (72, 46, 14, 16))
    for hx in (44, 54, 64, 74, 84):
        pygame.draw.line(s, dark, (hx, 10), (hx + random.randint(-6, 6), 0), 3)
    for ex in (47, 69):
        pygame.draw.ellipse(s, (10, 6, 6), (ex, 30, 16, 14))
        pygame.draw.circle(s, (255, 120, 20), (ex + 8, 37), 5)
        pygame.draw.circle(s, (255, 230, 150), (ex + 8, 37), 2)
    pygame.draw.polygon(s, (18, 0, 0), [(48, 54), (80, 54), (76, 68), (52, 68)])
    for tx in range(50, 78, 6):
        pygame.draw.polygon(s, (235, 230, 200), [(tx, 54), (tx + 5, 54), (tx + 2, 61)])
    pygame.draw.line(s, (130, 12, 12), (64, 68), (66, 84), 3)            # drool
    return s


def make_gun():
    """Beretta M9, first-person view from behind/above-left, drawn with a tiny 3D projection."""
    SS = 2                                   # supersample for smooth edges
    OW, OH = 560, 500                        # output size
    big = pygame.Surface((OW * SS, OH * SS), pygame.SRCALPHA)

    cam = (-9.0, 14.0, -42.0)
    tgt = (0.5, -2.0, 12.0)
    F = 1500.0
    CX, CY = 300.0, 200.0

    def norm(v):
        l = math.sqrt(sum(c * c for c in v))
        return tuple(c / l for c in v)

    def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    fwd = norm(sub(tgt, cam))
    right = norm(cross((0, 1, 0), fwd))
    upv = cross(fwd, right)

    def proj(p):
        v = sub(p, cam)
        zc = dot(v, fwd)
        return ((CX + F * dot(v, right) / zc) * SS, (CY - F * dot(v, upv) / zc) * SS), zc

    def P(p): return proj(p)[0]

    def poly(pts3, col, outline=None, w=2):
        pts = [P(p) for p in pts3]
        pygame.draw.polygon(big, col, pts)
        if outline:
            pygame.draw.polygon(big, outline, pts, w * SS)

    def line(a, b, col, w=1):
        pygame.draw.line(big, col, P(a), P(b), max(1, int(w * SS)))

    def blob(p, r, col):
        (x, y), zc = proj(p)
        pygame.draw.circle(big, col, (int(x), int(y)), max(1, int(r * F * SS / zc)))

    def lerp(a, b, t): return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

    # ---- dimensions (cm) -------------------------------------------------
    L, SW, SH = 21.7, 1.2, 3.0               # slide length, half width, height
    black, black_hi = (34, 36, 42), (96, 100, 112)
    side = (24, 26, 30)
    frame_c = (46, 48, 54)
    steel, steel_hi = (170, 176, 188), (236, 240, 248)

    # ---- grip left face + checkering ------------------------------------
    gx = -1.75
    TF, TR, BR, BF = (gx, -1.8, 8.5), (gx, -1.8, 1.0), (gx, -12.5, -2.6), (gx, -12.5, 5.2)
    poly([TF, TR, BR, BF], (30, 31, 35), (12, 12, 14))
    def gp(u, v):                               # u: front->rear, v: top->bottom
        a, b = lerp(TF, TR, u), lerp(BF, BR, u)
        return lerp(a, b, v)
    for i in range(1, 10):
        t = i / 10
        pygame.draw.line(big, (52, 54, 60), P(gp(t, 0.12)), P(gp(min(1, t + 0.35), 0.92)), SS)
        pygame.draw.line(big, (52, 54, 60), P(gp(t, 0.92)), P(gp(max(0, t - 0.35), 0.12)), SS)

    # ---- frame (dust cover) left face + trigger guard ---------------------
    fx = -SW
    poly([(fx, -1.8, 7.0), (fx, 0, 7.0), (fx, 0, L), (fx, -1.8, L)], frame_c, (14, 14, 16))
    line((fx, -0.9, 9), (fx, -0.9, L - 1.5), (70, 72, 80), 1)
    outer = [(fx, -1.8, 7.4), (fx, -2.6, 8.2), (fx, -5.8, 8.6), (fx, -6.7, 10.2), (fx, -6.7, 13.0),
             (fx, -5.6, 14.6), (fx, -1.8, 14.6)]
    inner = [(fx, -1.8, 9.3), (fx, -5.0, 9.8), (fx, -5.6, 11.0), (fx, -5.3, 12.6), (fx, -4.6, 13.2), (fx, -1.8, 13.2)]
    poly(outer, frame_c, (14, 14, 16))
    poly(inner, (14, 14, 18))
    poly([(fx + 0.1, -1.8, 10.2), (fx + 0.1, -4.4, 10.6), (fx + 0.1, -4.5, 11.2), (fx + 0.1, -1.8, 10.9)], (170, 172, 180))  # trigger

    # ---- slide left face + serrations + safety lever ----------------------
    poly([(fx, 0, 0), (fx, SH, 0), (fx, SH, L), (fx, 0, L)], side, (10, 10, 12))
    poly([(fx, SH - 0.5, 0.5), (fx, SH - 0.1, 0.5), (fx, SH - 0.1, L - 0.5), (fx, SH - 0.5, L - 0.5)], (52, 54, 62))
    for i in range(7):
        z = 0.9 + i * 0.55
        line((fx, 0.4, z), (fx, SH - 0.7, z), (8, 8, 10), 1.2)
    for i in range(5):
        z = L - 3.2 + i * 0.55
        line((fx, 0.4, z), (fx, SH - 0.7, z), (8, 8, 10), 1.2)
    blob((fx - 0.05, 1.5, 4.6), 0.95, (60, 62, 70))      # decocker / safety
    blob((fx - 0.05, 1.5, 4.6), 0.55, (24, 24, 28))

    # ---- slide rear face ---------------------------------------------------
    poly([(-SW, 0, 0), (SW, 0, 0), (SW, SH, 0), (-SW, SH, 0)], (28, 30, 34), (10, 10, 12))

    # ---- slide top face with open-top cutout ------------------------------
    poly([(-SW, SH, 0), (SW, SH, 0), (SW, SH, L), (-SW, SH, L)], black)
    line((-SW, SH, 0), (-SW, SH, L), black_hi, 1.2)
    line((SW, SH, 0), (SW, SH, L), (60, 62, 70), 1)
    line((-SW, SH, 0), (SW, SH, 0), black_hi, 1)
    c0, c1, rw = 6.8, 17.8, 0.45
    poly([(-SW + rw, SH, c0), (SW - rw, SH, c0), (SW - rw, SH, c1), (-SW + rw, SH, c1)], (8, 8, 10))
    # barrel + locking block seen through the cutout
    poly([(-0.62, 2.1, c0 + 0.4), (0.62, 2.1, c0 + 0.4), (0.62, 2.1, c1 - 0.2), (-0.62, 2.1, c1 - 0.2)], steel)
    poly([(-0.2, 2.1, c0 + 0.4), (0.15, 2.1, c0 + 0.4), (0.15, 2.1, c1 - 0.2), (-0.2, 2.1, c1 - 0.2)], steel_hi)
    poly([(-0.62, 2.1, c0 + 0.4), (0.62, 2.1, c0 + 0.4), (0.62, 2.1, c0 + 3.6), (-0.62, 2.1, c0 + 3.6)], (120, 124, 136))  # chamber hood
    poly([(-0.7, 1.4, c0 + 4.6), (0.7, 1.4, c0 + 4.6), (0.7, 1.4, c0 + 6.4), (-0.7, 1.4, c0 + 6.4)], (58, 60, 68))        # locking block
    # ejection port edge, right side
    line((SW - rw, SH, c0), (SW - rw, SH, c1), (70, 72, 80), 1)
    # slide front cap + muzzle
    line((-SW, SH, c1), (SW, SH, c1), black_hi, 1)

    # ---- sights (3-dot) ----------------------------------------------------
    # rear sight: two posts with a notch between
    for x0, x1 in ((-1.05, -0.32), (0.32, 1.05)):
        poly([(x0, SH, 1.0), (x1, SH, 1.0), (x1, SH + 0.8, 1.0), (x0, SH + 0.8, 1.0)], (22, 24, 28), (8, 8, 10), 1)
        poly([(x0, SH + 0.8, 1.0), (x1, SH + 0.8, 1.0), (x1, SH + 0.8, 2.2), (x0, SH + 0.8, 2.2)], (44, 46, 52))
        blob(((x0 + x1) / 2, SH + 0.4, 0.95), 0.24, (240, 240, 235))
    # front sight
    poly([(-0.32, SH, L - 1.2), (0.32, SH, L - 1.2), (0.32, SH + 0.95, L - 1.2), (-0.32, SH + 0.95, L - 1.2)], (22, 24, 28), (8, 8, 10), 1)
    poly([(-0.32, SH + 0.95, L - 1.2), (0.32, SH + 0.95, L - 1.2), (0.32, SH + 0.95, L - 0.4), (-0.32, SH + 0.95, L - 0.4)], (46, 48, 54))
    blob((0, SH + 0.5, L - 1.25), 0.2, (240, 240, 235))

    # ---- hammer ---------------------------------------------------------------
    poly([(-0.55, 0.3, -0.9), (0.55, 0.3, -0.9), (0.5, 1.9, -0.7), (-0.5, 1.9, -0.7)], (62, 64, 72), (14, 14, 16), 1)
    for i in range(4):
        y = 0.7 + i * 0.35
        line((-0.45, y, -0.95), (0.45, y, -0.95), (22, 22, 26), 1)

    # ---- hand (tactical glove) -------------------------------------------------
    glove, glove_hi, glove_lo = (38, 40, 44), (78, 82, 90), (20, 21, 24)
    back = [(-2.2, 0.5, -0.6), (2.4, 0.5, -0.6), (3.4, -12.0, -5.0), (-3.2, -12.0, -5.0)]
    pts = [P(p) for p in back] + [(P((4.0, -30, -9))[0], OH * SS + 40), (P((-3.8, -30, -9))[0], OH * SS + 40)]
    # order the polygon properly: top-left, top-right, bottom-right..., bottom-left
    pygame.draw.polygon(big, glove, [pts[0], pts[1], pts[2], pts[4], pts[5], pts[3]])
    pygame.draw.polygon(big, glove_lo, [pts[0], pts[1], pts[2], pts[4], pts[5], pts[3]], 3 * SS)
    # knuckle / seam highlights
    for t in (0.25, 0.5, 0.75):
        a = lerp((-2.0, -0.4, -0.8), (2.2, -0.4, -0.8), t)
        b = lerp((-3.8, -11.0, -4.2), (4.4, -11.0, -4.2), t)
        pygame.draw.line(big, glove_lo, P(a), P(b), SS)
    pygame.draw.line(big, glove_hi, P((-2.1, 0.4, -0.6)), P((2.3, 0.4, -0.6)), 3 * SS)
    (kx, ky), kz = proj((0.1, -0.3, -0.8))
    pygame.draw.ellipse(big, (50, 52, 58), (int(kx - 2.3 * F * SS / kz), int(ky - 0.7 * F * SS / kz),
                                            int(4.6 * F * SS / kz), int(1.4 * F * SS / kz)))
    # thumb along the left of the frame
    A, B = (-2.0, -1.6, 9.0), (-2.5, -1.4, 0.5)
    for i in range(0, 41):
        t = i / 40
        blob(lerp(A, B, t), 1.05 - 0.1 * (1 - t), glove if i % 2 else (44, 46, 50))
    blob(lerp(A, B, 0.0), 1.0, (50, 52, 58))
    for t in (0.2, 0.45):
        (x, y), zc = proj(lerp(A, B, t))
        pygame.draw.circle(big, glove_hi, (int(x), int(y - 1.0 * F * SS / zc)), max(1, int(0.55 * F * SS / zc)), SS)
    # sleeve cuff
    cuff = [P((-4.4, -19.0, -7.2)), P((5.3, -19.0, -7.2)), P((5.6, -22.5, -8.2)), P((-4.7, -22.5, -8.2))]
    pygame.draw.polygon(big, (62, 68, 50), cuff)

    out = pygame.transform.smoothscale(big, (OW, OH))
    tip, _ = proj((0, 2.1, L))
    return out, (int(tip[0] / SS), int(tip[1] / SS))



def make_sound(dur, vol, decay, tone=0.0):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(22050, -16, 1)
        rate = pygame.mixer.get_init()[0]
        n = int(rate * dur)
        buf = array.array("h")
        for i in range(n):
            env = (1 - i / n) ** decay
            wob = 1.0 if not tone else 0.5 + 0.5 * math.sin(i * tone * (1 + 3 * i / n))
            buf.append(int(random.uniform(-1, 1) * 32000 * vol * env * wob))
        return pygame.mixer.Sound(buffer=buf.tobytes())
    except Exception:
        return None


def play(snd):
    if snd:
        snd.play()


# ------------------------------------------------------------ floor + ceiling
class Flat:
    """Textured, lit floor and ceiling (numpy). Lighting is pre-baked into lookup tables, so each
    frame is just index maths plus one gather. Drawn at reduced resolution, then scaled up."""

    def __init__(self):
        S = FLAT_SCALE
        self.RW, self.RH = W // S, H // S
        xs = (np.arange(self.RW) + 0.5) * S
        self.th = ((xs / W) - 0.5) * FOV
        self.invcos = (1 / np.cos(self.th)).astype(np.float32)
        self.invcos2 = self.invcos ** 2
        self.beam = (0.30 + 0.70 * np.exp(-(self.th / 0.55) ** 2)).astype(np.float32)
        self.ys = (np.arange(self.RH) + 0.5) * S
        self.surf = pygame.Surface((self.RW, self.RH))
        self.amb = np.array(AMB, np.float32)
        self.warm = np.array(WARM, np.float32)
        self.fog = np.array(FOG, np.float32)
        sa = pygame.surfarray.array3d
        self.floor_tbl = self._bake(np.stack([sa(make_floor_tile((80, 78, 76))), sa(make_floor_tile((70, 68, 67)))]), False)
        self.ceil_tbl = self._bake(np.stack([sa(make_ceiling_tile(True)), sa(make_ceiling_tile(False))]), True)

    def _bake(self, tex, ceil):
        flat = tex.reshape(-1, 3).astype(np.float32)                 # index = par*FT*FT + u*FT + v
        lamp = flat.min(axis=1) >= 252
        tbl = np.empty((LVF, flat.shape[0], 3), np.uint8)
        lamp_col = np.array([255, 244, 214], np.float32)
        for i in range(LVF):
            t = i / (LVF - 1)
            lit = flat * (self.amb + self.warm * t) + self.fog * (1 - t)
            if ceil:                                                  # emissive lamps, fading into the fog
                k = 0.3 + 0.7 * t
                lit[lamp] = lamp_col * k + self.fog * (1 - k)
            tbl[i] = np.clip(lit, 0, 255)
        return tbl.reshape(-1, 3)

    def render(self, px, py, pa, horizon, flick, flash):
        RH, RW = self.RH, self.RW
        k = int(np.searchsorted(self.ys, horizon))          # rows [0:k] ceiling, [k:] floor
        ang = pa + self.th
        dx = (np.cos(ang) * self.invcos).astype(np.float32)
        dy = (np.sin(ang) * self.invcos).astype(np.float32)
        out = np.empty((RH, RW, 3), np.uint8)
        if k > 0:
            out[:k] = self._shade(self.ceil_tbl, 0.5, horizon - self.ys[:k], px, py, dx, dy, flick, flash)
        if k < RH:
            out[k:] = self._shade(self.floor_tbl, 1.0, self.ys[k:] - horizon, px, py, dx, dy, flick, flash)
        pygame.surfarray.blit_array(self.surf, np.ascontiguousarray(out.transpose(1, 0, 2)))
        return pygame.transform.scale(self.surf, (W, H))

    def _shade(self, tbl, scale, p, px, py, dx, dy, flick, flash):
        drow = np.minimum(H / (2 * np.maximum(p, 0.6)), 60.0).astype(np.float32)   # perpendicular distance
        X = (px + drow[:, None] * dx[None, :]) * scale
        Y = (py + drow[:, None] * dy[None, :]) * scale
        ix, iy = np.floor(X), np.floor(Y)
        par = (ix + iy).astype(np.int32) & 1
        u = np.minimum(((X - ix) * FT).astype(np.int32), FT - 1)
        v = np.minimum(((Y - iy) * FT).astype(np.int32), FT - 1)
        d2 = (drow * drow)[:, None] * self.invcos2[None, :]
        t = self.beam[None, :] * (TORCH * flick) / (1 + 0.07 * d2)
        if flash:
            t = t + flash * 0.9 / (1 + 0.03 * d2)
        lvl = (np.minimum(t, 1.0) * (LVF - 1)).astype(np.int32)
        return np.take(tbl, lvl * (2 * FT * FT) + par * (FT * FT) + u * FT + v, axis=0)


# ---------------------------------------------------------------- sprites
def sprite_geom(sx, sy, px, py, pa, aspect, hscale, anchor, horizon, squash=1.0):
    dx, dy = sx - px, sy - py
    ang = (math.atan2(dy, dx) - pa + math.pi) % (2 * math.pi) - math.pi
    perp = math.hypot(dx, dy) * math.cos(ang)
    if abs(ang) > FOV / 2 + 0.7 or perp < 0.2:
        return None
    wall_h = H / perp
    base_h = wall_h * hscale
    w, h = int(base_h * aspect), max(1, int(base_h * squash))
    left = int((0.5 + ang / FOV) * W) - w // 2
    top = horizon + int(wall_h / 2) - h if anchor == "floor" else horizon - h // 2
    return perp, left, w, top, h


def draw_billboard(screen, img, geom, zbuf):
    """Scale only the on-screen part of a sprite, then clip it against wall depth."""
    perp, left, w, top, h = geom
    if w <= 0 or h <= 0:
        return
    clip_l, clip_r = max(0, left), min(W, left + w)
    vt, vb = max(top, 0), min(top + h, H)
    if clip_r <= clip_l or vb <= vt:
        return

    iw, ih = img.get_size()
    u0 = max(0, min(iw - 1, int((clip_l - left) / w * iw)))
    u1 = max(u0 + 1, min(iw, math.ceil((clip_r - left) / w * iw)))
    v0 = max(0, min(ih - 1, int((vt - top) / h * ih)))
    v1 = max(v0 + 1, min(ih, math.ceil((vb - top) / h * ih)))
    visible = img.subsurface((u0, v0, u1 - u0, v1 - v0))
    scaled = pygame.transform.scale(visible, (clip_r - clip_l, vb - vt))

    x = clip_l - (clip_l % COL_W)
    while x < clip_r:
        if perp < zbuf[min(x // COL_W, RAYS - 1)]:
            d0, d1 = max(x, clip_l), min(x + COL_W, clip_r)
            if d1 > d0:
                src = scaled.subsurface((d0 - clip_l, 0, d1 - d0, vb - vt))
                screen.blit(src, (d0, vt))
        x += COL_W


def sprite_idx(perp, ang, flick, flash):
    t = min(1.0, max(0.14, light_t(perp, ang, flick, flash)))
    return int(t * (SPR_LV - 1))


# ------------------------------------------------------------------- world
def draw_world(screen, px, py, pa, horizon, tex, flick, flash):
    zbuf = [MAX_DEPTH] * RAYS
    for i in range(RAYS):
        raw, side, wx, (cx, cy) = cast_ray(px, py, pa + OFFS[i])
        dist = max(raw * COS[i], 0.0001)
        zbuf[i] = dist

        full_h = max(1, int(H / dist))
        top = horizon - full_h // 2
        vt, vb = max(top, 0), min(top + full_h, H)
        if vb <= vt:
            continue

        t = (0.30 + 0.70 * BEAM[i]) * TORCH * flick / (1 + 0.07 * dist * dist) + flash * 0.9 / (1 + 0.03 * dist * dist)
        t = min(1.0, t) * (0.78 if side else 1.0)
        img = tex[wall_kind(cx, cy)][int(t * (LV - 1))]
        tx = min(int(wx * TEX), TEX - 1)
        t0 = int((vt - top) / full_h * TEX)
        t1 = min(TEX, max(t0 + 1, math.ceil((vb - top) / full_h * TEX)))
        # scale straight into the screen: no temp surface, no tint pass
        pygame.transform.scale(img.subsurface((tx, t0, 1, t1 - t0)), (COL_W, vb - vt),
                               screen.subsurface((i * COL_W, vt, COL_W, vb - vt)))
    return zbuf


def make_ammo_pickup():
    """Small glowing ammo crate billboard."""
    s = pygame.Surface((64, 48), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0, 0, 0, 105), (7, 37, 50, 9))
    pygame.draw.rect(s, (24, 48, 38), (8, 8, 48, 32), border_radius=5)
    pygame.draw.rect(s, (70, 118, 78), (10, 7, 44, 29), border_radius=4)
    pygame.draw.rect(s, (145, 176, 104), (10, 7, 44, 5), border_radius=2)
    pygame.draw.rect(s, (25, 50, 38), (27, 7, 7, 29))
    pygame.draw.rect(s, (205, 190, 116), (18, 14, 5, 17), border_radius=2)
    pygame.draw.rect(s, (205, 190, 116), (40, 14, 5, 17), border_radius=2)
    pygame.draw.line(s, (225, 225, 180), (14, 36), (51, 36), 2)
    return s.convert_alpha()


# --------------------------------------------------------------- particles
class Particles:
    def __init__(self):
        self.items = []
        self.puff_cache = {}

    def add(self, x, y, vx, vy, life, color, size, grav, kind="dot"):
        self.items.append([x, y, vx, vy, life, life, color, size, grav, kind])

    def update(self, dt):
        out = []
        for p in self.items:
            p[4] -= dt
            if p[4] > 0:
                p[0] += p[2] * dt
                p[1] += p[3] * dt
                p[3] += p[8] * dt
                if p[9] in ("smoke", "mist"):
                    p[2] *= 0.97
                    p[3] *= 0.97
                out.append(p)
        self.items = out

    def draw(self, screen, puffs):
        for x, y, vx, vy, life, maxl, color, size, grav, kind in self.items:
            t = life / maxl
            if kind == "dot":
                pygame.draw.circle(screen, color, (int(x), int(y)), max(1, int(size * (0.4 + 0.6 * t))))
            elif kind == "spark":
                pygame.draw.line(screen, color, (int(x), int(y)), (int(x - vx * 0.03), int(y - vy * 0.03)), 2)
            elif kind == "shell":
                pygame.draw.rect(screen, color, (int(x), int(y), 7, 3))
                pygame.draw.line(screen, (255, 240, 170), (int(x), int(y)), (int(x) + 6, int(y)))
            else:                                                  # smoke / mist: soft puffs
                sz = max(4, int(size * (2.3 - 1.3 * t)))
                key = (kind, sz)
                img = self.puff_cache.get(key)
                if img is None:
                    img = pygame.transform.scale(puffs[kind], (sz, sz))
                    self.puff_cache[key] = img
                img.set_alpha(int((95 if kind == "smoke" else 120) * t))
                screen.blit(img, (int(x - sz / 2), int(y - sz / 2)))


_MINIMAP_BASE = None

def draw_minimap(screen, px, py, pa, enemies):
    """Cache static wall geometry; only draw actors each frame."""
    global _MINIMAP_BASE
    s = 7
    if _MINIMAP_BASE is None:
        _MINIMAP_BASE = pygame.Surface((MAP_W * s + 8, MAP_H * s + 8), pygame.SRCALPHA)
        _MINIMAP_BASE.fill((8, 10, 16, 170))
        pygame.draw.rect(_MINIMAP_BASE, (140, 130, 120, 160), _MINIMAP_BASE.get_rect(), 1)
        for y, row in enumerate(MAP):
            for x, c in enumerate(row):
                if c == "#":
                    pygame.draw.rect(_MINIMAP_BASE, (120, 124, 140, 220), (4 + x * s, 4 + y * s, s - 1, s - 1))
    screen.blit(_MINIMAP_BASE, (12, 12))
    for e in enemies:
        if not e.dead:
            pygame.draw.circle(screen, (230, 40, 40), (12 + int(4 + e.x * s), 12 + int(4 + e.y * s)), 3)
    cx, cy = 12 + 4 + px * s, 12 + 4 + py * s
    pygame.draw.line(screen, (255, 220, 80), (cx, cy), (cx + math.cos(pa) * 12, cy + math.sin(pa) * 12), 2)
    pygame.draw.circle(screen, (255, 220, 80), (int(cx), int(cy)), 3)


def draw_text(screen, font, text, pos, color=(225, 220, 210), center=False, shadow=True):
    img = font.render(text, True, color)
    if center:
        pos = (pos[0] - img.get_width() // 2, pos[1] - img.get_height() // 2)
    if shadow:
        screen.blit(font.render(text, True, (0, 0, 0)), (pos[0] + 2, pos[1] + 2))
    screen.blit(img, pos)
    return img


def make_panel(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (6, 8, 12, 150), (0, 0, w, h), border_radius=8)
    pygame.draw.rect(s, (255, 255, 255, 38), (0, 0, w, h), 1, border_radius=8)
    return s


# -------------------------------------------------------------------- game
class Enemy:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.hp = ENEMY_HP
        self.dead = False
        self.die_t = 0.0
        self.hit_t = 0.0
        self.anim = random.random() * 4
        self.cd = 0.4
        self.alerted = False
        self.speed = random.uniform(1.4, 1.9)
        self.groan_t = random.uniform(1.8, 4.0)


class Game:
    def __init__(self, screen):
        self.screen = screen
        names = "bahnschrift,segoeui,helveticaneue,arial"
        self.f_s = pygame.font.SysFont(names, 16, bold=True)
        self.f_m = pygame.font.SysFont(names, 26, bold=True)
        self.f_l = pygame.font.SysFont(names, 54, bold=True)
        self.f_xl = pygame.font.SysFont(names, 76, bold=True)
        self.fps = 0.0

        self.tex = {
            "b": make_shaded(make_blocks((150, 66, 50), (52, 48, 46), rows=8, cols=4)),
            "s": make_shaded(make_blocks((92, 98, 112), (30, 32, 38), rows=4, cols=2)),
            "m": make_shaded(make_metal()),
        }
        self.flat = Flat() if HAVE_NP else None
        self.bg = None if self.flat else make_background()
        self.vignette = make_vignette()
        self.hurt_vig = make_hurt_vignette()
        self.grain = make_grain() if ENABLE_GRAIN else []
        self.grain_i = 0

        self.face_raw = make_face()
        self.face_aspect = self.face_raw.get_width() / self.face_raw.get_height()
        self.face_lv = make_sprite_levels(self.face_raw)

        frames = [make_enemy(0), make_enemy(1)]
        self.enemy_aspect = frames[0].get_width() / frames[0].get_height()
        self.enemy_lv = [make_sprite_levels(f) for f in frames]
        self.enemy_flash = []
        for f in frames:
            t = f.copy()
            t.fill((105, 28, 22, 0), special_flags=pygame.BLEND_RGB_ADD)
            self.enemy_flash.append(t)
        self.shadow = make_soft_ellipse(64, 20, 150)
        self.pickup_img = make_ammo_pickup()
        self.pickup_aspect = self.pickup_img.get_width() / self.pickup_img.get_height()

        gun_raw, tip = make_gun()
        gs = 0.71                 # keep the weapon proportional at 800x450
        base = pygame.transform.smoothscale(gun_raw, (int(gun_raw.get_width() * gs), int(gun_raw.get_height() * gs)))
        self.gun_dim = base.copy()
        self.gun_dim.fill((200, 206, 222, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.gun_lit = base.copy()
        self.gun_lit.fill((55, 44, 26, 0), special_flags=pygame.BLEND_RGB_ADD)
        self.gun_tip = (int(tip[0] * gs), int(tip[1] * gs))
        self.gun_xy = (W // 2 + 25 - self.gun_tip[0], H // 2 + 55 - self.gun_tip[1])

        self.glow = make_glow(260, (255, 150, 50))
        self.puffs = {"smoke": make_puff((150, 150, 155)), "mist": make_puff((150, 10, 10))}
        self.flash_overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        self.flash_overlay.fill((255, 215, 140, 26))
        self.dead_overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        self.dead_overlay.fill((50, 0, 0, 175))
        self.panel_hp = make_panel(280, 62)
        self.panel_ammo = make_panel(268, 74)
        self.panel_top = make_panel(190, 60)
        self.panel_compass = make_panel(174, 48)

        self.snd_scream = make_sound(SCARE_TIME, 0.95, 0.4, 0.05)
        self.snd_shot = make_sound(0.2, 1.0, 2.5)
        self.snd_hit = make_sound(0.09, 0.6, 1.5, 0.3)
        self.snd_hurt = make_sound(0.3, 0.8, 1.2, 0.02)
        self.snd_die = make_sound(0.4, 0.7, 1.0, 0.03)
        self.snd_pickup = make_sound(0.16, 0.52, 2.0, 0.42)
        self.snd_reload = make_sound(0.20, 0.38, 2.8, 0.22)
        self.snd_reload_done = make_sound(0.09, 0.42, 2.0, 0.72)
        self.snd_empty = make_sound(0.075, 0.30, 2.5, 0.9)
        self.snd_step = make_sound(0.055, 0.22, 2.3, 0.55)
        self.snd_groan = make_sound(0.42, 0.34, 1.0, 0.035)
        self.reset()

    # -------------------------------------------------------------- setup
    def reset(self):
        self.px, self.py, self.pa = START
        self.pitch = 0.0
        self.hp = PLAYER_HP
        self.ammo = MAG_SIZE
        self.reserve_ammo = START_RESERVE
        self.reload_t = self.fire_cd = self.kick = self.flash_t = 0.0
        self.hitmark_t = self.hurt_t = self.scare_left = self.banner_t = 0.0
        self.damage_dir_t = 0.0
        self.damage_dir = 0.0
        self.pickup_msg_t = 0.0
        self.pickup_msg = ""
        self.step_t = 0.0
        self.sprinting = False
        self.bob_phase = self.time_s = self.shake = self.dip_t = 0.0
        self.flick = 1.0
        self.moving = False
        self.kills = self.wave = 0
        self.enemies = []
        self.pickups = []
        self.particles = Particles()
        self.state = "play"
        self.fire_req = self.reload_req = False
        self.flow, self.flow_cell = None, None
        self.zbuf = [MAX_DEPTH] * RAYS
        self.horizon = H // 2
        self.next_wave_t = None
        self.spawn_pickups()
        self.start_wave()

    def spawn_pickups(self):
        candidates = [c for c in OPEN_CELLS
                      if math.hypot(c[0] - self.px, c[1] - self.py) > 4.5
                      and math.hypot(c[0] - FACE_POS[0], c[1] - FACE_POS[1]) > 2.5]
        random.shuffle(candidates)
        chosen = []
        for x, y in candidates:
            if all(math.hypot(x - px, y - py) > 3.0 for px, py in chosen):
                chosen.append((x, y))
                if len(chosen) >= PICKUP_COUNT:
                    break
        self.pickups = [{"x": x, "y": y, "amount": PICKUP_AMMO} for x, y in chosen]

    def start_wave(self):
        self.wave += 1
        n = min(1 + self.wave * 2, 14)
        spots = [c for c in OPEN_CELLS
                 if math.hypot(c[0] - self.px, c[1] - self.py) > 7
                 and math.hypot(c[0] - FACE_POS[0], c[1] - FACE_POS[1]) > 2.5]
        random.shuffle(spots)
        self.enemies += [Enemy(x, y) for x, y in spots[:n]]
        self.banner_t = 2.5

    # ------------------------------------------------------------ gameplay
    def start_reload(self):
        if self.reload_t <= 0 and self.ammo < MAG_SIZE and self.reserve_ammo > 0:
            self.reload_t = RELOAD_TIME
            play(self.snd_reload)

    def hit_test(self):
        best, best_perp = None, 1e9
        for e in self.enemies:
            if e.dead:
                continue
            g = sprite_geom(e.x, e.y, self.px, self.py, self.pa, self.enemy_aspect,
                            ENEMY_HSCALE, "floor", self.horizon)
            if not g:
                continue
            perp, left, w, top, h = g
            if perp >= self.zbuf[RAYS // 2] or perp >= best_perp:
                continue
            if abs(W // 2 - (left + w // 2)) < w * 0.28 and top <= H // 2 <= top + h:
                best, best_perp = e, perp
        return best

    def shoot(self):
        self.ammo -= 1
        self.fire_cd = FIRE_DELAY
        self.kick = 1.0
        self.flash_t = 0.07
        self.shake = max(self.shake, 0.4)
        play(self.snd_shot)
        for e in self.enemies:
            e.alerted = True

        gx, gy = self.gun_xy
        tx, ty = self.gun_tip
        P = self.particles
        for _ in range(3):                                           # muzzle smoke
            P.add(gx + tx + random.uniform(-6, 6), gy + ty + random.uniform(-6, 6), random.uniform(-20, 20),
                  random.uniform(-70, -30), random.uniform(0.5, 0.9), None, 26, -15, "smoke")
        P.add(gx + tx + 115, gy + ty + 85, random.uniform(120, 220), random.uniform(-260, -150), 0.7,   # shell casing
              (214, 172, 62), 3, 950, "shell")

        cx, cy = W // 2, H // 2
        e = self.hit_test()
        if e:
            e.hp -= GUN_DAMAGE
            e.hit_t = 0.12
            self.hitmark_t = 0.15
            for _ in range(30):
                a = random.uniform(0, math.tau)
                sp = random.uniform(60, 320)
                P.add(cx, cy, math.cos(a) * sp, math.sin(a) * sp - 80, random.uniform(0.3, 0.7),
                      random.choice([(150, 10, 10), (200, 20, 20), (110, 6, 6)]), random.randint(2, 5), 750)
            for _ in range(3):
                P.add(cx + random.uniform(-10, 10), cy + random.uniform(-10, 10), random.uniform(-40, 40),
                      random.uniform(-40, 10), 0.6, None, 30, 0, "mist")
            if e.hp <= 0:
                e.dead = True
                self.kills += 1
                play(self.snd_die)
            else:
                play(self.snd_hit)
        elif self.zbuf[RAYS // 2] < MAX_DEPTH - 0.5:                 # bullet hit a wall
            for _ in range(12):
                a = random.uniform(0, math.tau)
                sp = random.uniform(100, 380)
                P.add(cx, cy, math.cos(a) * sp, math.sin(a) * sp - 60, random.uniform(0.15, 0.4),
                      random.choice([(255, 220, 120), (255, 255, 220), (255, 170, 60)]), 2, 520, "spark")
            for _ in range(4):
                P.add(cx + random.uniform(-6, 6), cy + random.uniform(-6, 6), random.uniform(-30, 30),
                      random.uniform(-45, -10), 0.55, None, 26, 0, "smoke")
        if self.ammo == 0:
            self.start_reload()

    def move_enemy(self, e, tx, ty, step):
        dx, dy = tx - e.x, ty - e.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        vx, vy = dx / d * step, dy / d * step
        m = 0.25
        if not is_wall(e.x + vx + math.copysign(m, vx), e.y):
            e.x += vx
        if not is_wall(e.x, e.y + vy + math.copysign(m, vy)):
            e.y += vy

    def update_enemies(self, dt):
        cell = (int(self.px), int(self.py))
        if cell != self.flow_cell:
            self.flow, self.flow_cell = compute_flow(*cell), cell

        for e in self.enemies:
            if e.dead:
                e.die_t += dt
                continue
            e.hit_t = max(0.0, e.hit_t - dt)
            dist = math.hypot(self.px - e.x, self.py - e.y)
            if dist < 6.0:
                e.groan_t -= dt
                if e.groan_t <= 0:
                    play(self.snd_groan)
                    e.groan_t = random.uniform(2.6, 5.0)
            if dist > ENEMY_AGGRO and not e.alerted:
                continue
            if dist < 0.85:
                e.cd -= dt
                if e.cd <= 0:
                    self.hp -= ENEMY_DAMAGE
                    self.hurt_t = 0.45
                    self.damage_dir = math.atan2(e.y - self.py, e.x - self.px) - self.pa
                    self.damage_dir_t = 0.95
                    self.shake = 1.0
                    e.cd = 0.9
                    play(self.snd_hurt)
                    if self.hp <= 0:
                        self.hp, self.state = 0, "dead"
                continue
            e.cd = 0.4
            e.anim += dt * 5
            if dist < 3 and has_los(e.x, e.y, self.px, self.py):
                tx, ty = self.px, self.py
            else:
                ex, ey = int(e.x), int(e.y)
                best, tx, ty = self.flow[ey][ex], e.x, e.y
                for nx, ny in ((ex + 1, ey), (ex - 1, ey), (ex, ey + 1), (ex, ey - 1)):
                    if 0 <= nx < MAP_W and 0 <= ny < MAP_H and 0 <= self.flow[ny][nx] < best:
                        best, tx, ty = self.flow[ny][nx], nx + 0.5, ny + 0.5
            self.move_enemy(e, tx, ty, e.speed * dt)

        alive = [e for e in self.enemies if not e.dead]
        for i, a in enumerate(alive):                       # keep them from stacking
            for b in alive[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if 0 < d < 0.5:
                    push = (0.5 - d) / 2 / d
                    for o, sgn in ((a, -1), (b, 1)):
                        nx, ny = o.x + dx * push * sgn, o.y + dy * push * sgn
                        if not is_wall(nx, o.y):
                            o.x = nx
                        if not is_wall(o.x, ny):
                            o.y = ny

        self.enemies = [e for e in self.enemies if not (e.dead and e.die_t > 0.6)]
        if not alive:
            if self.next_wave_t is None:
                self.next_wave_t = 2.0
            else:
                self.next_wave_t -= dt
                if self.next_wave_t <= 0:
                    self.start_wave()
                    self.next_wave_t = None

    def update(self, dt, mouse_dx, mouse_dy, keys):
        self.time_s += dt
        self.hurt_t = max(0.0, self.hurt_t - dt)
        self.damage_dir_t = max(0.0, self.damage_dir_t - dt)
        self.pickup_msg_t = max(0.0, self.pickup_msg_t - dt)
        self.banner_t = max(0.0, self.banner_t - dt)
        self.shake = max(0.0, self.shake - dt * 4.5)
        self.particles.update(dt)

        # flashlight flicker, with the occasional spooky dip
        if self.dip_t > 0:
            self.dip_t -= dt
        elif random.random() < 0.0025:
            self.dip_t = 0.14
        self.flick = (0.95 + 0.05 * math.sin(self.time_s * 11) * math.sin(self.time_s * 5.3)) * (0.55 if self.dip_t > 0 else 1.0)

        if self.state == "dead":
            return

        if self.scare_left > 0:
            self.scare_left -= dt
            if self.scare_left <= 0:
                self.px, self.py, self.pa = START
                self.pitch = 0.0
            return

        self.fire_cd = max(0.0, self.fire_cd - dt)
        self.kick = max(0.0, self.kick - dt * 6)
        self.flash_t = max(0.0, self.flash_t - dt)
        self.hitmark_t = max(0.0, self.hitmark_t - dt)
        if self.reload_t > 0:
            self.reload_t -= dt
            if self.reload_t <= 0:
                loaded = min(MAG_SIZE - self.ammo, self.reserve_ammo)
                self.ammo += loaded
                self.reserve_ammo -= loaded
                play(self.snd_reload_done)

        self.pa += mouse_dx * MOUSE_SENS
        self.pa += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * TURN_SPEED * dt
        self.pitch = max(-MAX_PITCH, min(MAX_PITCH, self.pitch - mouse_dy * MOUSE_SENS_Y))

        fwd = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
        strafe = keys[pygame.K_d] - keys[pygame.K_a]
        vx = math.cos(self.pa) * fwd - math.sin(self.pa) * strafe
        vy = math.sin(self.pa) * fwd + math.cos(self.pa) * strafe
        moving_input = bool(fwd or strafe)
        if moving_input:
            length = math.hypot(vx, vy)
            vx, vy = vx / length, vy / length
        self.sprinting = bool(moving_input and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]))
        speed = MOVE_SPEED * (SPRINT_MULT if self.sprinting else 1.0)
        nx, ny = self.px + vx * speed * dt, self.py + vy * speed * dt
        m = 0.2
        if not is_wall(nx + math.copysign(m, vx), self.py):
            self.px = nx
        if not is_wall(self.px, ny + math.copysign(m, vy)):
            self.py = ny
        self.moving = moving_input
        if self.moving:
            self.bob_phase += dt * (12 if self.sprinting else 9)
            self.step_t -= dt
            if self.step_t <= 0:
                play(self.snd_step)
                self.step_t = 0.31 if self.sprinting else 0.43
        else:
            self.step_t = min(self.step_t, 0.08)

        # Collect one-use ammo crates. The total ammunition supply is finite.
        for pickup in self.pickups[:]:
            if math.hypot(pickup["x"] - self.px, pickup["y"] - self.py) < 0.68:
                gained = min(pickup["amount"], MAX_RESERVE - self.reserve_ammo)
                if gained > 0:
                    self.reserve_ammo += gained
                    self.pickup_msg = f"+{gained} AMMO"
                    self.pickup_msg_t = 1.5
                    play(self.snd_pickup)
                self.pickups.remove(pickup)

        if self.reload_req:
            self.reload_req = False
            self.start_reload()
        if self.fire_req:
            self.fire_req = False
            if self.reload_t <= 0 and self.fire_cd <= 0:
                if self.ammo > 0:
                    self.shoot()
                elif self.reserve_ammo > 0:
                    self.start_reload()
                else:
                    play(self.snd_empty)

        self.update_enemies(dt)

        if math.hypot(FACE_POS[0] - self.px, FACE_POS[1] - self.py) < SCARE_DIST:
            self.scare_left = SCARE_TIME
            play(self.snd_scream)

    # --------------------------------------------------------------- draw
    def draw_jumpscare(self):
        t = 1 - self.scare_left / SCARE_TIME
        screen = self.screen
        screen.fill(random.choice([(0, 0, 0), (110, 0, 0), (0, 0, 0)]))
        fh = int(H * (1.1 + 0.5 * t))
        fw = int(fh * self.face_aspect)
        if fw > W * 1.2:
            fw = int(W * 1.2)
            fh = int(fw / self.face_aspect)
        img = pygame.transform.scale(self.face_raw, (fw, fh))
        shake = int(25 * (1 - t) + 4)
        screen.blit(img, (W // 2 - fw // 2 + random.randint(-shake, shake),
                          H // 2 - fh // 2 + random.randint(-shake, shake)))

    def draw_gun(self):
        offset = self.kick * 30
        sway_x = math.sin(self.time_s * 1.6) * 3                      # breathing
        sway_y = math.sin(self.time_s * 2.1) * 2
        if self.moving:
            offset += abs(math.cos(self.bob_phase)) * 6
            sway_x += math.sin(self.bob_phase) * 8
        if self.reload_t > 0:
            p = 1 - self.reload_t / RELOAD_TIME
            offset += math.sin(p * math.pi) * 260
        tx, ty = self.gun_tip
        gx = W // 2 + 25 - tx + int(sway_x)
        gy = H // 2 + 55 - ty + int(offset + sway_y)
        self.gun_xy = (gx - int(sway_x), gy - int(offset + sway_y))   # steady anchor for particles
        self.screen.blit(self.gun_lit if self.flash_t > 0 else self.gun_dim, (gx, gy))
        if self.flash_t > 0:
            tip = (gx + tx, gy + ty)
            self.screen.blit(self.glow, (tip[0] - 130, tip[1] - 130), special_flags=pygame.BLEND_RGB_ADD)
            pts = []
            for i in range(16):
                r = random.randint(40, 74) if i % 2 == 0 else 14
                a = i / 16 * 2 * math.pi
                pts.append((tip[0] + math.cos(a) * r, tip[1] + math.sin(a) * r))
            pygame.draw.polygon(self.screen, (255, 205, 80), pts)
            pygame.draw.circle(self.screen, (255, 250, 225), tip, 17)

    def post(self):
        s = self.screen
        if ENABLE_BLOOM:
            small = pygame.transform.scale(s, (W // 4, H // 4))
            small.fill((105, 105, 105), special_flags=pygame.BLEND_RGB_SUB)
            small = pygame.transform.scale(small, (W // 16, H // 16))
            small = pygame.transform.scale(small, (W, H))
            s.blit(small, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            s.blit(small, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if self.grain:
            self.grain_i = (self.grain_i + 1) % len(self.grain)
            s.blit(self.grain[self.grain_i], (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        s.blit(self.vignette, (0, 0))

    def draw_hud(self):
        screen = self.screen
        cx, cy = W // 2, H // 2
        col = (255, 70, 60) if self.hitmark_t > 0 else (235, 235, 235)
        gap = 6 + int(self.kick * 6)
        for a, b in (((cx - gap - 8, cy), (cx - gap, cy)), ((cx + gap, cy), (cx + gap + 8, cy)),
                     ((cx, cy - gap - 8), (cx, cy - gap)), ((cx, cy + gap), (cx, cy + gap + 8))):
            pygame.draw.line(screen, (0, 0, 0), (a[0] + 1, a[1] + 1), (b[0] + 1, b[1] + 1), 3)
            pygame.draw.line(screen, col, a, b, 2)
        if self.hitmark_t > 0:
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                pygame.draw.line(screen, (255, 70, 60), (cx + sx * 6, cy + sy * 6), (cx + sx * 13, cy + sy * 13), 2)

        # Compass: bearing uses conventional north-at-zero degrees.
        screen.blit(self.panel_compass, (W // 2 - 87, 12))
        bearing = (90 + math.degrees(self.pa)) % 360
        compass_dirs = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        direction = compass_dirs[int((bearing + 22.5) // 45) % 8]
        draw_text(screen, self.f_s, "COMPASS", (W // 2 - 73, 17), (145, 155, 168), shadow=False)
        draw_text(screen, self.f_m, f"{direction}  {bearing:03.0f}°", (W // 2 - 73, 31), (225, 230, 220))

        # Directional damage arrow; it stays on the edge nearest the attacker.
        if self.damage_dir_t > 0:
            a = self.damage_dir
            ux, uy = math.sin(a), -math.cos(a)
            px, py = W // 2 + ux * 205, H // 2 + uy * 175
            bx, by = px - ux * 18, py - uy * 18
            vx, vy = math.cos(a), math.sin(a)
            alpha = min(1.0, self.damage_dir_t / 0.95)
            red = int(90 + 165 * alpha)
            pts = [(int(px), int(py)), (int(bx + vx * 12), int(by + vy * 12)),
                   (int(bx - vx * 12), int(by - vy * 12))]
            pygame.draw.polygon(screen, (35, 0, 0), pts)
            pygame.draw.polygon(screen, (red, 32, 28), pts, 3)

        # health
        screen.blit(self.panel_hp, (14, H - 76))
        draw_text(screen, self.f_s, "HEALTH", (30, H - 70), (140, 150, 165), shadow=False)
        frac = self.hp / PLAYER_HP
        bar = (int(60 + 190 * (1 - frac)), int(40 + 190 * frac), 60)
        lit = math.ceil(frac * 10)
        for i in range(10):
            r = pygame.Rect(30 + i * 17, H - 48, 14, 16)
            pygame.draw.rect(screen, bar if i < lit else (32, 34, 40), r, border_radius=3)
        draw_text(screen, self.f_m, f"{self.hp}", (218, H - 56), (235, 235, 235))

        # ammo
        screen.blit(self.panel_ammo, (W - 282, H - 88))
        for i in range(MAG_SIZE):
            r = pygame.Rect(W - 268 + i * 11, H - 80, 7, 20)
            pygame.draw.rect(screen, (232, 192, 84) if i < self.ammo else (46, 44, 42), r, border_radius=2)
            if i < self.ammo:
                pygame.draw.rect(screen, (150, 112, 40), (r.x, r.y + 12, 7, 8), border_radius=2)
        if self.reload_t > 0:
            draw_text(screen, self.f_m, "RELOADING", (W - 268, H - 52), (255, 160, 60))
            draw_text(screen, self.f_s, f"RES {self.reserve_ammo:03d}", (W - 104, H - 49),
                      (170, 170, 160), shadow=False)
            p = 1 - self.reload_t / RELOAD_TIME
            pygame.draw.rect(screen, (40, 38, 36), (W - 268, H - 20, 130, 5), border_radius=2)
            pygame.draw.rect(screen, (255, 160, 60), (W - 268, H - 20, int(130 * p), 5), border_radius=2)
        else:
            draw_text(screen, self.f_l, f"{self.ammo}", (W - 268, H - 66), (240, 226, 172))
            draw_text(screen, self.f_s, f"/ {MAG_SIZE}   RES {self.reserve_ammo:02d}",
                      (W - 268 + 80 + (20 if self.ammo >= 10 else 0), H - 30),
                      (150, 150, 150), shadow=False)

        # wave / kills
        screen.blit(self.panel_top, (W - 204, 12))
        draw_text(screen, self.f_s, "WAVE", (W - 192, 19), (140, 150, 165), shadow=False)
        draw_text(screen, self.f_m, f"{self.wave}", (W - 192, 38), (235, 235, 235))
        draw_text(screen, self.f_s, "KILLS", (W - 110, 19), (140, 150, 165), shadow=False)
        draw_text(screen, self.f_m, f"{self.kills}", (W - 110, 38), (235, 90, 80))

        draw_text(screen, self.f_s, f"{self.fps:.0f} FPS", (14, MAP_H * 7 + 24), (130, 135, 145), shadow=False)
        if self.pickup_msg_t > 0:
            img = self.f_m.render(self.pickup_msg, True, (170, 230, 150))
            img.set_alpha(int(255 * min(1.0, self.pickup_msg_t)))
            screen.blit(img, (W // 2 - img.get_width() // 2, H - 116))
        if self.time_s < 14:
            msg = "Mouse look   Click shoot   R reload   WASD move   Shift sprint   Collect ammo crates   Esc quit"
            img = self.f_s.render(msg, True, (200, 200, 200))
            img.set_alpha(int(255 * min(1.0, (14 - self.time_s) / 3)))
            screen.blit(img, (W // 2 - img.get_width() // 2, H - 22))

        if self.banner_t > 0:
            img = self.f_xl.render(f"WAVE {self.wave}", True, (230, 60, 50))
            img.set_alpha(int(255 * min(1.0, self.banner_t / 0.8)))
            screen.blit(img, (W // 2 - img.get_width() // 2, 70))

    def draw(self):
        screen = self.screen
        if self.scare_left > 0:
            self.draw_jumpscare()
            return

        self.horizon = H // 2 + int(self.pitch + (math.sin(self.bob_phase) * 4 if self.moving else 0)
                                    + self.kick * 8)
        sh = self.shake
        horizon = self.horizon + (int(random.uniform(-1, 1) * 10 * sh) if sh > 0.02 else 0)
        pa = self.pa + (random.uniform(-1, 1) * 0.006 * sh if sh > 0.02 else 0.0)
        flash = 0.55 if self.flash_t > 0 else 0.0
        flick = self.flick

        if self.flat:
            screen.blit(self.flat.render(self.px, self.py, pa, horizon, flick, flash), (0, 0))
        else:
            screen.blit(self.bg, (0, horizon - BG_HORIZON))
        self.zbuf = draw_world(screen, self.px, self.py, pa, horizon, self.tex, flick, flash)

        items = []
        g = sprite_geom(*FACE_POS, self.px, self.py, pa, self.face_aspect, 0.85, "center", horizon)
        if g:
            ang = ((g[1] + g[2] // 2) / W - 0.5) * FOV
            items.append((g[0], g, self.face_lv[sprite_idx(g[0], ang, flick, flash)], False))
        for pickup in self.pickups:
            g = sprite_geom(pickup["x"], pickup["y"], self.px, self.py, pa,
                            self.pickup_aspect, 0.48, "floor", horizon)
            if g:
                items.append((g[0], g, self.pickup_img, False))
        for e in self.enemies:
            squash = 1.0 if not e.dead else max(0.08, 1 - e.die_t / 0.45 * 0.92)
            g = sprite_geom(e.x, e.y, self.px, self.py, pa, self.enemy_aspect,
                            ENEMY_HSCALE, "floor", horizon, squash)
            if g:
                ang = ((g[1] + g[2] // 2) / W - 0.5) * FOV
                frame = int(e.anim) % 2
                img = self.enemy_flash[frame] if e.hit_t > 0 else self.enemy_lv[frame][sprite_idx(g[0], ang, flick, flash)]
                items.append((g[0], g, img, True))
        for _, g, img, shadow in sorted(items, key=lambda it: -it[0]):
            if shadow:
                perp, left, w, top, h = g
                sw, shh = max(4, int(w * 0.95)), max(3, int(w * 0.16))
                cx = left + w // 2
                if 0 <= cx < W and perp < self.zbuf[min(cx // COL_W, RAYS - 1)]:
                    screen.blit(pygame.transform.scale(self.shadow, (sw, shh)), (cx - sw // 2, top + h - shh // 2))
            draw_billboard(screen, img, g, self.zbuf)

        self.particles.draw(screen, self.puffs)
        self.draw_gun()
        self.post()
        if self.flash_t > 0:
            screen.blit(self.flash_overlay, (0, 0))
        if self.hurt_t > 0:
            self.hurt_vig.set_alpha(int(210 * self.hurt_t / 0.45))
            screen.blit(self.hurt_vig, (0, 0))

        draw_minimap(screen, self.px, self.py, self.pa, self.enemies)
        self.draw_hud()

        if self.state == "dead":
            screen.blit(self.dead_overlay, (0, 0))
            draw_text(screen, self.f_xl, "YOU DIED", (W // 2, H // 2 - 34), (220, 40, 40), center=True)
            draw_text(screen, self.f_m, f"Wave {self.wave}   Kills {self.kills}", (W // 2, H // 2 + 28),
                      (225, 220, 210), center=True)
            draw_text(screen, self.f_s, "press Enter to restart", (W // 2, H // 2 + 62), (170, 170, 170), center=True)


def main():
    pygame.mixer.pre_init(22050, -16, 1)
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Fake 3D Raycaster")
    clock = pygame.time.Clock()

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()

    game = Game(screen)
    running = True
    while running:
        dt = min(clock.tick(60) / 1000, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                game.fire_req = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    game.fire_req = True
                elif e.key == pygame.K_r:
                    game.reload_req = True
                elif e.key == pygame.K_RETURN and game.state == "dead":
                    game.reset()
                    pygame.mouse.get_rel()

        mdx, mdy = pygame.mouse.get_rel()
        game.fps = clock.get_fps()
        game.update(dt, mdx, mdy, pygame.key.get_pressed())
        game.draw()
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()