import os, pygame, random, math

ANDROID = "ANDROID_ARGUMENT" in os.environ

pygame.init()
W, H = 960, 540              # logical resolution (everything is drawn at this size, then scaled)
WW, WH = 3600, 3300          # world size
KILL_Y = WH + 200
MAX_CY = WH - H

if ANDROID:
    window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    # we use finger events on Android, so ignore the emulated mouse events (avoids double input)
    pygame.event.set_blocked([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])
else:
    window = pygame.display.set_mode((W, H), pygame.RESIZABLE)
screen = pygame.Surface((W, H)).convert()   # the game draws here; scaled to the window at the end
pygame.display.set_caption("Box Climber")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
hint_font = pygame.font.Font(None, 26)
big = pygame.font.Font(None, 80)
btn_font = pygame.font.Font(None, 34)
arrow_font = pygame.font.Font(None, 80)

speed, gravity, jump_v, max_fall = 340, 1500, 560, 700

# ================= TOUCH UI =================
btn = {
    "left":   pygame.Rect(24, 404, 112, 112),
    "right":  pygame.Rect(152, 404, 112, 112),
    "jump":   pygame.Rect(824, 404, 112, 112),
    "recall": pygame.Rect(W - 208, 12, 92, 52),
    "reset":  pygame.Rect(W - 108, 12, 92, 52),
}
btn_labels = {"left": ("<", arrow_font), "right": (">", arrow_font), "jump": ("JUMP", btn_font),
              "recall": ("BOX", btn_font), "reset": ("RESET", btn_font)}


def make_btn(r, label, fnt, pressed):
    s = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (255, 255, 255, 95 if pressed else 40), s.get_rect(), border_radius=18)
    pygame.draw.rect(s, (255, 255, 255, 170), s.get_rect(), 3, border_radius=18)
    img = fnt.render(label, True, (255, 255, 255))
    s.blit(img, img.get_rect(center=s.get_rect().center))
    return s


btn_imgs = {n: (make_btn(btn[n], *btn_labels[n], False), make_btn(btn[n], *btn_labels[n], True))
            for n in btn}

touches = {}          # finger id -> (x, y) in logical coords
drag_id = None        # which finger is dragging the box
show_ui = ANDROID     # on desktop the buttons appear the first time a touch is seen


def to_logical(px, py):
    """window pixel -> logical (W x H) coordinate, accounting for letterboxing"""
    ww, wh = window.get_size()
    sc = min(ww / W, wh / H)
    return (int((px - (ww - W * sc) / 2) / sc), int((py - (wh - H * sc) / 2) / sc))


def finger_pos(e):
    ww, wh = window.get_size()
    return to_logical(e.x * ww, e.y * wh)   # finger x/y are normalized 0..1


# ================= LEVEL =================
statics = [pygame.Rect(*r) for r in [
    (0, 3100, 1000, 200),    # start ground
    (600, 2990, 400, 110),   # step: build a staircase with the box
    (1060, 2740, 260, 30),   # launch target
    (1660, 2740, 260, 30),   # 340px gap: use the box as a bridge
    (1920, 2820, 360, 60),   # spike pit floor
    (2280, 2740, 300, 30),
    (2700, 2480, 240, 30),   # zig-zag tower
    (3060, 2220, 240, 30),
    (2700, 1960, 240, 30),
    (2300, 1700, 260, 30),
    (1900, 1810, 400, 60),   # second spike pit floor
    (1400, 1700, 500, 30),
    (1440, 1480, 40, 220),   # wall: too tall to jump
    (1000, 1700, 400, 30),
    (700, 1440, 240, 30),    # final tower
    (1060, 1180, 260, 30),
    (700, 920, 240, 30),
    (1060, 660, 260, 30),
    (700, 400, 300, 30),     # goal platform
]]
spikes = [pygame.Rect(*r) for r in [
    (1920, 2802, 360, 18),
    (1900, 1792, 400, 18),
    (1220, 1162, 100, 18),
    (700, 902, 80, 18),
]]


def make_cp(x, ground_y):
    r = pygame.Rect(x, ground_y - 60, 16, 60)
    return {"rect": r, "spawn": (r.centerx - 20, r.bottom - 40), "on": False}


cps = [make_cp(1250, 2740), make_cp(2500, 2740), make_cp(3200, 2220),
       make_cp(2330, 1700), make_cp(1100, 1700), make_cp(740, 1440),
       make_cp(840, 920)]
goal = pygame.Rect(850, 340, 16, 60)
hints = [
    ("Stand on the box to climb the wall", 30, 3040),
    ("Q = bring the box back   R = respawn", 30, 3000),
    ("Flick the box UP under you to launch (it can't push sideways)", 560, 2930),
    ("Hold the box in mid-air and use it as a bridge", 1090, 2680),
    ("Drop the box into the pit, then hop across", 1930, 2690),
    ("Zig-zag up: launch, recall the box, repeat", 2600, 2400),
    ("This wall is too tall to jump", 1500, 1640),
    ("Don't overshoot the spikes!", 1050, 1120),
]
if ANDROID:
    hints = [(tx.replace("Q = bring the box back   R = respawn",
                         "BOX button = bring the box back   RESET = respawn"), x, y)
             for tx, x, y in hints]

# ================= STATE =================
spawn = (60, 3060)
player = pygame.Rect(spawn[0], spawn[1], 40, 40)
player_x, player_y = float(player.x), float(player.y)
vel_y, onfloor, facing = 0, False, 1
coyote, jumping = 0.0, False

ky = pygame.Rect(300, 3070, 30, 30)
ky_y, ky_vel = float(ky.y), 0
dragging, drag_offset = False, (0, 0)
push_speed = 0.0

# built once: the rect objects are mutated in place, so these stay valid
solid_p = statics + [ky]       # what the player collides with
solid_k = statics + [player]   # what the box collides with

squash, shake, t = 0.0, 0.0, 0.0
particles = []
run_time, won, deaths = 0.0, False, 0
hud_key, hud_img = None, None

# ================= CACHED VISUALS =================
def gradient(top, bottom):
    s = pygame.Surface((W, H))
    for y in range(H):
        k = y / H
        c = tuple(int(top[i] + (bottom[i] - top[i]) * k) for i in range(3))
        pygame.draw.line(s, c, (0, y), (W, y))
    return s.convert()


bg_low = gradient((14, 10, 36), (40, 26, 80))
bg_high = gradient((50, 60, 130), (250, 170, 120))
stars = [(random.randint(0, W), random.randint(0, H), random.randint(1, 2),
          random.random() * 6.28) for _ in range(60)]


def make_hills(fx, color, amp, freq):
    ws = W + int(WW * fx) + 40
    s = pygame.Surface((ws, 240), pygame.SRCALPHA)
    pts = [(0, 240)]
    for x in range(0, ws + 20, 20):
        h = amp * (math.sin(x * freq) + 0.5 * math.sin(x * freq * 2.3 + 1.0))
        pts.append((x, 120 - h))
    pts.append((ws + 20, 240))
    pygame.draw.polygon(s, color, pts)
    return s


hills = [(make_hills(0.15, (30, 24, 66), 70, 0.004), 0.15, 0.10, (30, 24, 66)),
         (make_hills(0.30, (44, 34, 88), 50, 0.007), 0.30, 0.22, (44, 34, 88))]


def render_block(r):
    s = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (58, 48, 82), (0, 0, r.w, r.h), border_radius=4)
    pygame.draw.rect(s, (78, 66, 108), (2, 2, r.w - 4, max(1, r.h - 6)), border_radius=3)
    for x in range(10, r.w - 10, 44):
        for y in range(16, r.h - 8, 34):
            pygame.draw.rect(s, (62, 52, 90), (x, y, 14, 4), border_radius=2)
    pygame.draw.rect(s, (90, 200, 120), (0, 0, r.w, 7), border_radius=3)
    pygame.draw.rect(s, (140, 240, 160), (0, 0, r.w, 3))
    return s


def render_spikes(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    n = max(1, w // 18)
    step = w / n
    for i in range(n):
        x0 = i * step
        pts = [(x0, h), (x0 + step / 2, 0), (x0 + step, h)]
        pygame.draw.polygon(s, (235, 90, 110), pts)
        pygame.draw.polygon(s, (255, 170, 180), pts, 1)
    return s


block_imgs = [render_block(r) for r in statics]
spike_imgs = [render_spikes(r.w, r.h) for r in spikes]
dim = pygame.Surface((W, H), pygame.SRCALPHA)
dim.fill((0, 0, 0, 140))
glow_cache = {}
if ANDROID:
    hud_ctrl = font.render("Hold the buttons to move/jump   Drag the box with a finger",
                           True, (170, 170, 200))
else:
    hud_ctrl = font.render("A/D move   Space jump   Drag the box   Q bring box   R respawn",
                           True, (170, 170, 200))


def glow(pos, radius, color):
    radius = max(4, int(radius) // 4 * 4)
    key = (radius, color)
    g = glow_cache.get(key)
    if g is None:
        g = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(4):
            r = radius - i * radius // 5
            pygame.draw.circle(g, (*color, 25 + i * 15), (radius, radius), r)
        glow_cache[key] = g
    screen.blit(g, (pos[0] - radius, pos[1] - radius))


hint_cache = {}


def burst(x, y, n, color, spread=200, up=0, size=4, life=0.5):
    if len(particles) > 350:
        return
    for _ in range(n):
        a = random.uniform(0, 6.283)
        s = random.uniform(0.2, 1) * spread
        particles.append([x, y, math.cos(a) * s, math.sin(a) * s - up,
                          life * random.uniform(0.6, 1), life, color, size])


# ================= PHYSICS =================
def apply_gravity(rect, y, vel, solids, dt):
    feet = pygame.Rect(rect.left, rect.bottom, rect.width, 24)
    hit = feet.collidelist(solids)
    if hit != -1 and vel >= 0:
        vel = 0
        rect.bottom = solids[hit].top
        return rect.y, vel, True
    vel = min(vel + gravity * dt, max_fall)
    y += vel * dt
    rect.y = round(y)
    return y, vel, False


def sign(n):
    return (n > 0) - (n < 0)


def move_ky(ky, player, statics, dx, dy):
    """Moves the box 1px at a time. Blocked by statics. It can push the player
    up/down but never sideways. Returns how many px it pushed the player up."""
    pushed = 0
    for sx, sy, count in ((sign(dx), 0, abs(dx)), (0, sign(dy), abs(dy))):
        for _ in range(count):
            ky.move_ip(sx, sy)
            blocked = ky.collidelist(statics) != -1
            if not blocked and ky.colliderect(player):
                if sx != 0:
                    blocked = True
                else:
                    player.move_ip(0, sy)
                    if player.collidelist(statics) != -1:
                        player.move_ip(0, -sy)
                        blocked = True
                    elif sy < 0:
                        pushed += 1
            if blocked:
                ky.move_ip(-sx, -sy)
                break
    return pushed


def find_spot():
    """Free spot beside the player with ground under it (so the box doesn't fall away)."""
    for ox in (facing * 50, -facing * 50, facing * 85, -facing * 85, facing * 120, -facing * 120):
        r = pygame.Rect(0, 0, 30, 30)
        r.midbottom = (player.centerx + ox, player.bottom)
        if (r.collidelist(statics) == -1 and not r.colliderect(player)
                and 0 <= r.x <= WW - 30
                and pygame.Rect(r.x, r.bottom, r.w, 6).collidelist(statics) != -1):
            return r.topleft
    return (player.centerx - 15, player.top - 40)


def cam_target():
    tx = player.centerx - W / 2 + facing * 60
    ty = player.centery - H / 2 - 40 + max(-60, min(120, vel_y * 0.12))
    return (max(0, min(WW - W, tx)), max(0, min(MAX_CY, ty)))


cam_x, cam_y = cam_target()

# ================= MAIN LOOP =================
running = True
while running:
    dt = min(clock.tick(60) / 1000, 0.033)
    t += dt
    if not won:
        run_time += dt
    do_recall = do_respawn = False
    cx, cy = int(cam_x), int(cam_y)
    mpos = (-9999, -9999) if ANDROID else to_logical(*pygame.mouse.get_pos())

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                do_recall = True
            elif event.key == pygame.K_r:
                do_respawn = True
            elif event.key == pygame.K_AC_BACK:
                running = False

        # unify finger + mouse into (phase, id, logical pos, grab padding)
        ev = None
        if event.type == pygame.FINGERDOWN:
            ev = ("down", event.finger_id, finger_pos(event), 70)
        elif event.type == pygame.FINGERMOTION:
            ev = ("move", event.finger_id, finger_pos(event), 70)
        elif event.type == pygame.FINGERUP:
            ev = ("up", event.finger_id, finger_pos(event), 70)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            ev = ("down", "m", to_logical(*event.pos), 16)
        elif event.type == pygame.MOUSEMOTION:
            ev = ("move", "m", to_logical(*event.pos), 16)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            ev = ("up", "m", to_logical(*event.pos), 16)

        if ev:
            ph, fid, pos, pad = ev
            if ph == "down":
                if fid != "m":
                    show_ui = True
                touches[fid] = pos
                if show_ui and btn["recall"].collidepoint(pos):
                    do_recall = True
                elif show_ui and btn["reset"].collidepoint(pos):
                    do_respawn = True
                elif show_ui and any(r.collidepoint(pos) for r in btn.values()):
                    pass                       # pressed a hold-button, not the box
                elif not dragging:
                    wx, wy = pos[0] + cx, pos[1] + cy
                    if ky.inflate(pad, pad).collidepoint(wx, wy):
                        dragging, drag_id = True, fid
                        drag_offset = (ky.x - wx, ky.y - wy)
                        burst(ky.centerx, ky.centery, 10, (255, 230, 90), 150, size=3)
            elif ph == "move":
                if fid in touches:
                    touches[fid] = pos
            else:
                touches.pop(fid, None)
                if fid == drag_id:
                    dragging = False

    if dragging and drag_id not in touches:
        dragging = False

    # which on-screen buttons are held (ignoring the finger that's dragging the box)
    free = [p for fid, p in touches.items() if not (dragging and fid == drag_id)]
    t_left = any(btn["left"].collidepoint(p) for p in free)
    t_right = any(btn["right"].collidepoint(p) for p in free)
    t_jump = any(btn["jump"].collidepoint(p) for p in free)

    keys = pygame.key.get_pressed()
    jump_held = keys[pygame.K_SPACE] or t_jump

    # ----- player horizontal -----
    move_dir = max(-1, min(1, keys[pygame.K_d] - keys[pygame.K_a] + t_right - t_left))
    if move_dir:
        facing = move_dir
    player_x += move_dir * speed * dt
    player.x = round(player_x)
    for i in player.collidelistall(solid_p):
        s = solid_p[i]
        if player.colliderect(s):
            if move_dir > 0:
                player.right = s.left
            elif move_dir < 0:
                player.left = s.right
            player_x = float(player.x)

    # ----- box: drag or fall -----
    pushed = 0
    if dragging:
        mp = touches[drag_id]
        dx = mp[0] + cx + drag_offset[0] - ky.x
        dy = mp[1] + cy + drag_offset[1] - ky.y
        pushed = move_ky(ky, player, statics, dx, dy)
        ky_y, ky_vel = float(ky.y), 0
        player_x, player_y = float(player.x), float(player.y)
        if random.random() < 0.6:
            burst(ky.centerx, ky.centery, 1, (255, 240, 130), 90, size=3, life=0.4)
    else:
        ky_y, ky_vel, _ = apply_gravity(ky, ky_y, ky_vel, solid_k, dt)

    # ----- launch: flick the box up under you, then stop -----
    if pushed:
        push_speed = max(pushed / dt, push_speed * 0.8)
    else:
        if push_speed > 260:
            vel_y = -min(push_speed, 1100)
            jumping = False
            squash, shake = -0.35, max(shake, 5)
            burst(player.centerx, player.bottom, 14, (255, 230, 90), 220, up=80, size=4)
        push_speed = 0

    # ----- player vertical -----
    prev_vel, was_on = vel_y, onfloor
    player_y, vel_y, onfloor = apply_gravity(player, player_y, vel_y, solid_p, dt)

    if vel_y < 0:  # head bump
        hit = player.collidelist(solid_p)
        if hit != -1:
            player.top = solid_p[hit].bottom
            player_y = float(player.y)
            vel_y = 0

    if onfloor:
        jumping = False
    coyote = 0.1 if onfloor else coyote - dt
    if jump_held and coyote > 0:
        vel_y = -jump_v
        coyote = 0
        jumping = True
        squash = -0.3
        burst(player.centerx, player.bottom, 8, (200, 200, 220), 150, up=40, size=3)
    if jumping and vel_y < -220 and not jump_held:
        vel_y = -220

    if onfloor and not was_on and prev_vel > 250:
        squash = min(0.4, prev_vel / 2500)
        shake = max(shake, min(8, prev_vel / 150))
        burst(player.centerx, player.bottom, 12, (200, 200, 220), 180, up=60, size=4)
    if onfloor and move_dir and random.random() < 0.3:
        burst(player.centerx - facing * 14, player.bottom, 1, (170, 170, 190), 60, up=40, size=3, life=0.35)
    if not onfloor and random.random() < 0.5:
        burst(player.centerx, player.centery, 1, (90, 215, 140), 30, size=4, life=0.3)

    # ----- checkpoints, goal, death -----
    for c in cps:
        if not c["on"] and player.colliderect(c["rect"]):
            for o in cps:
                o["on"] = False
            c["on"] = True
            spawn = c["spawn"]
            burst(c["rect"].centerx, c["rect"].top, 18, (90, 230, 140), 220, up=60)
    if not won and player.colliderect(goal):
        won = True
        burst(goal.centerx, goal.top, 80, (255, 220, 90), 450, up=200, size=5, life=1.2)

    died = player.top > KILL_Y or player.inflate(-12, -6).collidelist(spikes) != -1
    if died or do_respawn:
        if died:
            deaths += 1
        burst(player.centerx, min(player.centery, KILL_Y - 20), 20, (255, 120, 120), 250)
        player.topleft = spawn
        player_x, player_y = float(player.x), float(player.y)
        vel_y, push_speed, dragging = 0, 0, False
        shake = 10
        do_recall = True
    if ky.top > KILL_Y:
        do_recall = True
    if do_recall:
        ky.topleft = find_spot()
        ky_y, ky_vel, dragging = float(ky.y), 0, False
        burst(ky.centerx, ky.centery, 14, (255, 230, 90), 180, size=3)

    # ----- effects update -----
    squash += (0 - squash) * min(1, 12 * dt)
    shake = max(0, shake - 40 * dt)
    for p in particles:
        p[0] += p[2] * dt
        p[1] += p[3] * dt
        p[3] += 600 * dt
        p[4] -= dt
    particles[:] = [p for p in particles if p[4] > 0]

    # ----- camera (smooth) -----
    tx, ty = cam_target()
    cam_x += (tx - cam_x) * min(1, 5 * dt)
    cam_y += (ty - cam_y) * min(1, 6 * dt)
    cam_x = max(0, min(WW - W, cam_x))
    cam_y = max(0, min(MAX_CY, cam_y))
    cx, cy = int(cam_x), int(cam_y)
    ox = int(random.uniform(-shake, shake)) if shake > 0.3 else 0
    oy = int(random.uniform(-shake, shake)) if shake > 0.3 else 0
    rx, ry = cx + ox, cy + oy                      # draw offset (includes shake)
    view = pygame.Rect(cx - 20, cy - 20, W + 40, H + 40)
    alt = 1 - cam_y / MAX_CY

    # ================= DRAW =================
    screen.blit(bg_low, (0, 0))
    a = int(255 * alt)
    if a > 3:
        bg_high.set_alpha(a)
        screen.blit(bg_high, (0, 0))
    for sx, sy, size, phase in stars:
        b = int((130 + 110 * math.sin(t * 2 + phase)) * (1 - 0.7 * alt))
        pygame.draw.circle(screen, (b, b, min(255, b + 20)),
                           (int((sx - cam_x * 0.08) % W), int((sy - cam_y * 0.08) % H)), size)
    for surf, fx, fy, col in hills:
        base = int(H - 20 + (MAX_CY - cam_y) * fy)
        screen.blit(surf, (-int(cam_x * fx), base - 120))
        if base + 120 < H:
            pygame.draw.rect(screen, col, (0, base + 120, W, H - base - 120))

    for i, r in enumerate(statics):
        if r.colliderect(view):
            screen.blit(block_imgs[i], (r.x - rx, r.y - ry))
    for i, r in enumerate(spikes):
        if r.colliderect(view):
            screen.blit(spike_imgs[i], (r.x - rx, r.y - ry))

    for text, hx, hy in hints:
        if pygame.Rect(hx, hy, 600, 24).colliderect(view):
            img = hint_cache.get(text)
            if img is None:
                img = hint_cache[text] = hint_font.render(text, True, (200, 200, 230))
            screen.blit(img, (hx - rx, hy - ry))

    for c in cps:
        if c["rect"].colliderect(view):
            r = c["rect"].move(-rx, -ry)
            pygame.draw.rect(screen, (150, 150, 170), (r.x, r.y, 3, r.h))
            col = (90, 230, 140) if c["on"] else (120, 120, 150)
            pygame.draw.polygon(screen, col, [(r.x + 3, r.y), (r.x + 3 + r.w + 8, r.y + 10), (r.x + 3, r.y + 20)])

    if goal.colliderect(view):
        g = goal.move(-rx, -ry)
        glow((g.x + 8, g.y + 10), 50 + int(5 * math.sin(t * 4)), (255, 220, 90))
        pygame.draw.rect(screen, (200, 200, 210), (g.x, g.y, 3, g.h))
        pygame.draw.polygon(screen, (255, 210, 60), [(g.x + 3, g.y), (g.x + 30, g.y + 12), (g.x + 3, g.y + 24)])

    # player shadow
    best = None
    for s in solid_p:
        if s.left < player.centerx < s.right and s.top >= player.bottom:
            if best is None or s.top < best.top:
                best = s
    if best is not None:
        gap = best.top - player.bottom
        if gap < 250:
            sw = max(10, int(40 - gap * 0.15))
            sh = pygame.Rect(0, 0, sw, 6)
            sh.center = (player.centerx - rx, best.top + 3 - ry)
            pygame.draw.ellipse(screen, (30, 20, 50), sh)

    # box
    kr = ky.move(-rx, -ry)
    if dragging:
        glow(kr.center, 40 + int(4 * math.sin(t * 10)), (255, 230, 90))
    elif ky.inflate(16, 16).collidepoint(mpos[0] + cx, mpos[1] + cy):
        glow(kr.center, 28, (255, 230, 90))
    pygame.draw.rect(screen, (190, 160, 20), kr, border_radius=6)
    pygame.draw.rect(screen, (255, 225, 60), kr.inflate(-6, -6), border_radius=4)

    # player
    pw, ph = int(40 * (1 + squash)), int(40 * (1 - squash))
    body = pygame.Rect(0, 0, pw, ph)
    body.midbottom = (player.midbottom[0] - rx, player.midbottom[1] - ry)
    pygame.draw.rect(screen, (50, 150, 90), body, border_radius=10)
    pygame.draw.rect(screen, (90, 215, 140), body.inflate(-6, -6), border_radius=8)
    for ex in (-0.2, 0.2):
        eye = (int(body.centerx + pw * ex), int(body.top + ph * 0.38))
        pygame.draw.circle(screen, (255, 255, 255), eye, 5)
        pygame.draw.circle(screen, (20, 20, 30), (eye[0] + facing * 2, eye[1]), 2)

    for p in particles:
        px, py = int(p[0]) - rx, int(p[1]) - ry
        if -10 < px < W + 10 and -10 < py < H + 10:
            pygame.draw.circle(screen, p[6], (px, py), max(1, int(p[7] * p[4] / p[5])))

    # HUD
    screen.blit(hud_ctrl, (12, 10))
    key = (int(run_time * 10), deaths)
    if key != hud_key:
        hud_key = key
        hud_img = font.render(f"Time {run_time:0.1f}s   Falls {deaths}", True, (170, 170, 200))
    screen.blit(hud_img, (12, 32))

    # touch buttons
    if show_ui:
        held = {"left": t_left, "right": t_right, "jump": t_jump}
        for name, r in btn.items():
            screen.blit(btn_imgs[name][1 if held.get(name) else 0], r)

    if won:
        screen.blit(dim, (0, 0))
        msg = big.render("YOU MADE IT!", True, (255, 220, 90))
        screen.blit(msg, msg.get_rect(center=(W // 2, H // 2 - 20)))
        sub = font.render(f"Time {run_time:0.1f}s   Falls {deaths}", True, (230, 230, 240))
        screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 + 30)))

    # scale the logical screen to the real window (letterboxed, keeps aspect ratio)
    ww, wh = window.get_size()
    sc = min(ww / W, wh / H)
    dw, dh = int(W * sc), int(H * sc)
    window.fill((0, 0, 0))
    if (dw, dh) == (W, H):
        window.blit(screen, ((ww - W) // 2, (wh - H) // 2))
    else:
        window.blit(pygame.transform.scale(screen, (dw, dh)), ((ww - dw) // 2, (wh - dh) // 2))
    pygame.display.flip()

pygame.quit()
