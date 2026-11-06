import sys
import time
import math
import numpy as np

# Global Constants & Constants Setup
TAU = 2 * math.pi
TURN = 1.0

# Geometry metrics
GIRDLE_R = 0.5
GIRDLE_Y = 0.15
TABLE_R = 0.3
TABLE_Y = 0.35
CULET_Y = -0.2

R_BAND = 0.6
R_BAND_THICK = 0.05

P, N, F = [], [], []
k = 0

def facet(corners, m=16):
    global k
    u, v = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    u, v = u.ravel(), v.ravel()
    keep = (u + v) <= 1
    u, v = u[keep], v[keep]
    
    a, b, c = corners
    pts = a + np.outer(u, (b - a)) + np.outer(v, (c - a))
    
    normal = np.cross(b - a, c - a)
    normal = normal / np.linalg.norm(normal)
    
    P.append(pts)
    N.append(np.tile(normal, (len(pts), 1)))
    F.append(np.full(len(pts), k))
    k += 1

def diamond():
    m = 16
    for i in range(m):
        a0, a1 = TAU * i / m, TAU * (i + 1) / m
        
        t0 = np.array([TABLE_R * np.cos(a0), TABLE_Y, TABLE_R * np.sin(a0)])
        t1 = np.array([TABLE_R * np.cos(a1), TABLE_Y, TABLE_R * np.sin(a1)])
        g0 = np.array([GIRDLE_R * np.cos(a0), GIRDLE_Y, GIRDLE_R * np.sin(a0)])
        g1 = np.array([GIRDLE_R * np.cos(a1), GIRDLE_Y, GIRDLE_R * np.sin(a1)])
        culet = np.array([0, CULET_Y, 0])

        crown = np.cross(g1 - g0, t0 - g0)
        crown = crown / np.linalg.norm(crown)
        
        # Facets setup
        facet([g0, g1, t0], crown[1] > 0)
        facet([g1, t1, t0], crown[1] > 0)
        
        pav = np.cross(culet - g1, g0 - g1)
        pav = pav / np.linalg.norm(pav)
        
        facet([g0, culet, g1], pav[1] > 0)
        facet([g0, g1, t0], crown[1] > 0)

    # Table facet (top)
    facet([np.array([0, TABLE_Y, 0]), t0, t1], True)

    return np.concatenate(P), np.concatenate(N), np.concatenate(F)

def prongs():
    P_prongs, N_prongs = [], []
    for i in range(4):
        a = TAU * (i + 0.5) / 4
        base = np.array([0.8 * np.cos(a), R_BAND + 0.02, 0.8 * np.sin(a)])
        tip = np.array([GIRDLE_R * 0.95 * np.cos(a), TABLE_Y - 0.03, GIRDLE_R * 0.95 * np.sin(a)])
        
        d = (tip - base) / np.linalg.norm(tip - base)
        s = np.cross(d, [0, 1, 0])
        s = s / np.linalg.norm(s)
        w = np.cross(s, d)
        
        u, phi = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, TAU, 10))
        u, phi = u.ravel(), phi.ravel()
        
        pts = base + np.outer(u, (tip - base)) + 0.02 * (np.outer(np.cos(phi), s) + np.outer(np.sin(phi), w))
        norms = np.outer(np.cos(phi), s) + np.outer(np.sin(phi), w)
        norms = norms / np.linalg.norm(norms, axis=1, keepdims=True)
        
        P_prongs.append(pts)
        N_prongs.append(norms)
        
    return np.concatenate(P_prongs), np.concatenate(N_prongs)

def band():
    phi = np.linspace(0, TAU, 300)
    theta = np.linspace(0, TAU, 30)
    
    PHI, THETA = np.meshgrid(phi, theta)
    PHI, THETA = PHI.ravel(), THETA.ravel()
    
    r = R_BAND + R_BAND_THICK * np.cos(THETA)
    y = R_BAND_THICK * np.sin(THETA)
    x = r * np.cos(PHI)
    z = r * np.sin(PHI)
    
    pts = np.column_stack([x, y, z])
    
    nx = np.cos(THETA) * np.cos(PHI)
    ny = np.sin(THETA)
    nz = np.cos(THETA) * np.sin(PHI)
    norms = np.column_stack([nx, ny, nz])
    
    return pts, norms

# Assembly
GEM_P, GEM_N, GEM_F = diamond()
PROMO_P, PROMO_N = prongs()
BAND_P, BAND_N = band()

RING_P = np.concatenate([BAND_P, PROMO_P, GEM_P])
RING_N = np.concatenate([BAND_N, PROMO_N, GEM_N])
RING_KIND = np.concatenate([
    np.zeros(len(BAND_P)), 
    np.ones(len(PROMO_P)), 
    np.full(len(GEM_P), 2)
])

SEED = np.random.default_rng(1).random(len(RING_P))

def lights(t):
    key = np.array([0.0, 0.8, 0.6])
    rim = np.array([np.cos(t * 0.9), 0.5, np.sin(t * 0.9)])
    return key / np.linalg.norm(key), rim / np.linalg.norm(rim)

def spin(t):
    return TAU * (t - 0.6) / TURN

def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] + s * p[:, 2], p[:, 1], -s * p[:, 0] + c * p[:, 2]], axis=-1)

def draw_order():
    """
    when each point appears: the band draws itself from the bottom up,
    then the prongs rise, then the diamond fills in from its tip
    """
    p = RING_P
    up_band = np.abs(np.angle(np.exp(1j * (np.arctan2(p[:, 1], p[:, 0])))))
    prong = 1.0 + (p[:, 1] - R_BAND) / (TABLE_Y - R_BAND) * 0.25
    gem = 1.25 + (p[:, 1] - CULET_Y) / (TABLE_Y - CULET_Y) * 0.45
    return np.select([RING_KIND == 0, RING_KIND == 1, RING_KIND == 2], [up_band, prong, gem])

ORDER = draw_order()
DRAW = (0.3, 5.6)

def gather(p, t):
    """
    sweep: points appear where the drawing light has passed
    """
    head = (t - DRAW[0]) / (DRAW[1] - DRAW[0]) * ORDER.max()
    k = np.maximum(0, head - ORDER) / 0.035
    return p, k, np.clip(head - ORDER, 0, 1)

# ASCII Rendering Loop setup
CHARS = " .':-`^=;+x*%#@$"
WIDTH, HEIGHT = 80, 40

def render_frame(t):
    rot_angle = spin(t)
    rotated_pts = rot_y(RING_P, rot_angle)
    key_dir, rim_dir = lights(t)
    
    # Calculate lighting intensity
    dots_key = np.maximum(0, np.sum(RING_N * key_dir, axis=1))
    dots_rim = np.maximum(0, np.sum(RING_N * rim_dir, axis=1))
    intensity = dots_key * 0.7 + dots_rim * 0.3
    
    # Projection to 2D screen
    x_proj = (rotated_pts[:, 0] * 35 + WIDTH / 2).astype(int)
    y_proj = (-rotated_pts[:, 1] * 20 + HEIGHT / 2).astype(int)
    
    grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
    z_buffer = np.full((HEIGHT, WIDTH), -np.inf)
    
    for i in range(len(RING_P)):
        x, y = x_proj[i], y_proj[i]
        z = rotated_pts[i, 2]
        
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            if z > z_buffer[y, x]:
                z_buffer[y, x] = z
                char_idx = int(intensity[i] * (len(CHARS) - 1))
                char_idx = min(max(char_idx, 0), len(CHARS) - 1)
                grid[y][x] = CHARS[char_idx]

    output = "\x1b[H" + "\n".join("".join(row) for row in grid)
    sys.stdout.write(output)
    sys.stdout.flush()

if __name__ == "__main__":
    print("\x1b[2J")  # Clear terminal screen
    start_time = time.time()
    try:
        while True:
            t = time.time() - start_time
            render_frame(t)
            time.sleep(1 / 30)  # 30 fps target
    except KeyboardInterrupt:
        print("\x1b[0m\nExiting...")