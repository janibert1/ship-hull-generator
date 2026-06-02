import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import config as cfg

def cubic_hermite_spline(t, P0, P1, T0, T1):
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    y = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
    z = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
    return y, z

def get_stern_curve(N=100):
    """Haalt de opgeslagen spiegel spline op en genereert N punten"""
    B_half = cfg.BOA / 2.0
    D = cfg.DOA
    
    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "Stern_Spline_Points" in data and "Stern_Spline_Tangents" in data:
        pts = [np.array(p) for p in data["Stern_Spline_Points"]]
        ctrls = [np.array(c) for c in data["Stern_Spline_Tangents"]]
    else:
        # Default als er nog niet is opgeslagen
        pts = [
            np.array([0.0, 1.0]),
            np.array([B_half * 0.4, 3.5]),
            np.array([B_half * 0.7, 6.5]),
            np.array([B_half * 0.9, D])
        ]
        ctrls = [
            np.array([B_half * 0.3, 1.0]),
            np.array([B_half * 0.5, 5.0]),
            np.array([B_half * 0.8, 8.0]),
            np.array([B_half * 0.9, D - 2.0])
        ]

    # Genereer de 3 segmenten en combineer ze tot N punten
    t_vals = np.linspace(0, 1, N // 3)
    y_all, z_all = [], []
    
    for i in range(3):
        P0 = pts[i]
        P1 = pts[i+1]
        C0 = ctrls[i]
        C1 = ctrls[i+1]
        
        T0 = (C0 - P0) * 3
        if i == 2:
            T1 = (P1 - C1) * 3
        else:
            T1 = (C1 - P1) * 3
            
        y, z = cubic_hermite_spline(t_vals, P0, P1, T0, T1)
        # Voorkom dubbele punten op de naden
        if i > 0:
            y, z = y[1:], z[1:]
        y_all.extend(y)
        z_all.extend(z)
        
    # Resample naar exact N punten via lineaire interpolatie voor een nette grid
    y_all = np.array(y_all)
    z_all = np.array(z_all)
    
    # Maak een oplopende parameter op basis van lengte
    distances = np.sqrt(np.diff(y_all)**2 + np.diff(z_all)**2)
    cumulative_length = np.insert(np.cumsum(distances), 0, 0)
    normalized_length = cumulative_length / cumulative_length[-1]
    
    uniform_t = np.linspace(0, 1, N)
    y_uniform = np.interp(uniform_t, normalized_length, y_all)
    z_uniform = np.interp(uniform_t, normalized_length, z_all)
    
    return y_uniform, z_uniform

def get_midship_curve(N=100):
    """Genereert N punten voor de klassieke midship vorm (platte bodem, kimradius, zijwand)"""
    B_half = cfg.BOA / 2.0
    D = cfg.DOA
    R = cfg.BILGE_RADIUS
    
    y_flat = np.linspace(0, max(0, B_half - R), N // 3)
    z_flat = np.zeros_like(y_flat)
    
    theta = np.linspace(1.5 * np.pi, 2.0 * np.pi, N // 3)
    y_bilge = (B_half - R) + R * np.cos(theta)
    z_bilge = R + R * np.sin(theta)
    
    y_side = np.full(N - 2*(N//3), B_half)
    z_side = np.linspace(R, D, N - 2*(N//3))
    
    y_all = np.concatenate([y_flat, y_bilge[1:], y_side[1:]])
    z_all = np.concatenate([z_flat, z_bilge[1:], z_side[1:]])
    
    # Resample net als bij de stern voor gelijke N distributie
    distances = np.sqrt(np.diff(y_all)**2 + np.diff(z_all)**2)
    cumulative_length = np.insert(np.cumsum(distances), 0, 0)
    normalized_length = cumulative_length / cumulative_length[-1]
    
    uniform_t = np.linspace(0, 1, N)
    y_uniform = np.interp(uniform_t, normalized_length, y_all)
    z_uniform = np.interp(uniform_t, normalized_length, z_all)
    
    return y_uniform, z_uniform

def plot_stern_half():
    LPP = cfg.LPP
    center = LPP * (cfg.MIDSHIP_LOC_PCT / 100.0)
    l_mid = LPP * (cfg.MIDSHIP_LENGTH_PCT / 100.0)
    x_aft = center - (l_mid / 2.0)
    
    # Lpp begint bij 0 (AP). De spiegel ligt dus op een negatieve X-waarde.
    # Laten we aannemen dat de overhang 50% achter en 50% voor is.
    overhang_total = cfg.LOA - LPP
    x_stern = -(overhang_total / 2.0)
    
    N_u = 50 # Punten over de dwarsdoorsnede
    N_t = 50 # Punten in de lengterichting
    
    # 1. Haal de Y en Z waarden op voor de twee extremen
    y_mid, z_mid = get_midship_curve(N_u)
    y_stern, z_stern = get_stern_curve(N_u)
    
    # 2. Definieer de 3 Control Curves voor de Bezier Surface
    # Curve 0 (Vast): Midden van de midship
    C0_x = np.full(N_u, center)
    C0_y = y_mid
    C0_z = z_mid
    
    # Curve 1 (Attractor / Bezier Control): Uiteinde van de midship
    C1_x = np.full(N_u, x_aft)
    C1_y = y_mid
    C1_z = z_mid
    
    # Curve 2 (Vast): Achtersteven / Spiegel
    C2_x = np.full(N_u, x_stern)
    C2_y = y_stern
    C2_z = z_stern
    
    # 3. Genereer de Quadratic Bezier Surface (Achterschip)
    # S(t) = (1-t)^2 * C0 + 2t(1-t) * C1 + t^2 * C2
    # t loopt van 0 (midship center) naar 1 (spiegel)
    T = np.linspace(0, 1, N_t)
    
    X_surf = np.zeros((N_t, N_u))
    Y_surf = np.zeros((N_t, N_u))
    Z_surf = np.zeros((N_t, N_u))
    
    for i, t in enumerate(T):
        w0 = (1 - t)**2
        w1 = 2 * (1 - t) * t
        w2 = t**2
        
        X_surf[i, :] = w0 * C0_x + w1 * C1_x + w2 * C2_x
        Y_surf[i, :] = w0 * C0_y + w1 * C1_y + w2 * C2_y
        Z_surf[i, :] = w0 * C0_z + w1 * C1_z + w2 * C2_z

    # 3.5 Genereer de Parallel Midbody (Voorschip / Midden)
    # Dit is simpelweg een extrusie van de midship curve (C0) van center naar x_fwd
    x_fwd = center + (l_mid / 2.0)
    T_mid = np.linspace(center, x_fwd, N_t // 2)
    X_mid_surf, Y_mid_surf = np.meshgrid(T_mid, y_mid)
    _, Z_mid_surf = np.meshgrid(T_mid, z_mid)
    
    # Transponeer om aan te sluiten bij de (N_t, N_u) orientatie van X_surf
    X_mid_surf = X_mid_surf.T
    Y_mid_surf = Y_mid_surf.T
    Z_mid_surf = Z_mid_surf.T

    # 4. Plotten
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Achterschip oppervlak (Bezier)
    ax.plot_surface(X_surf, Y_surf, Z_surf, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.2)
    ax.plot_surface(X_surf, -Y_surf, Z_surf, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.2)
    
    # Midship oppervlak (Parallel)
    ax.plot_surface(X_mid_surf, Y_mid_surf, Z_mid_surf, color='lightblue', alpha=0.8, edgecolor='k', linewidth=0.2)
    ax.plot_surface(X_mid_surf, -Y_mid_surf, Z_mid_surf, color='lightblue', alpha=0.8, edgecolor='k', linewidth=0.2)
    
    # Teken de 3 control curves dikgedrukt om het concept te tonen
    ax.plot(C0_x, C0_y, C0_z, 'r-', lw=4, label='Curve 0: Midship Center (Vast)')
    ax.plot(C1_x, C1_y, C1_z, 'g--', lw=3, label='Curve 1: Midship Aft (Bezier Attractor)')
    ax.plot(C2_x, C2_y, C2_z, 'm-', lw=4, label='Curve 2: Stern Spline (Vast)')
    
    # Voorste begrenzing midship ter referentie
    C_fwd_x = np.full(N_u, x_fwd)
    ax.plot(C_fwd_x, y_mid, z_mid, 'r--', lw=2, label='Midship Forward (Einde Middenromp)')
    ax.plot(C_fwd_x, -y_mid, z_mid, 'r--', lw=2)

    # Teken ook aan bakboord voor de volledigheid
    ax.plot(C0_x, -C0_y, C0_z, 'r-', lw=4)
    ax.plot(C1_x, -C1_y, C1_z, 'g--', lw=3)
    ax.plot(C2_x, -C2_y, C2_z, 'm-', lw=4)

    # Assen en titels
    ax.set_xlabel('X [m] (AP -> Voorschip)')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Quadratic Bezier Surface Lofting + Parallel Midship\n(Stern -> Midship -> Forward)')
    
    # Pas as ratio aan om 1:1:1 ware schaal te hebben
    x_range = x_fwd - x_stern
    y_range = cfg.BOA
    z_range = cfg.DOA
    
    # We forceren matplotlib om een 1:1:1 verhouding in 3D te tekenen
    ax.set_box_aspect((x_range, y_range, z_range))
    
    # Zorg ook dat de limits exact kloppen zodat er geen interne scaling gebeurt
    ax.set_xlim(x_stern, x_fwd)
    ax.set_ylim(-cfg.BOA/2, cfg.BOA/2)
    ax.set_zlim(0, cfg.DOA)
    
    ax.view_init(elev=20, azim=230)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_stern_half()
