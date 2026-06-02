import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import config as cfg

def cubic_hermite_spline(t, P0, P1, T0, T1):
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    x = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
    z = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
    return x, z

def rotate_and_lock_x(pts, ctrls, angle_deg):
    """
    Roteert de control points rond (0,0) met 'angle_deg'.
    Daarna schaalt het de X-as terug zodat het laatste punt (B) zijn originele X behoudt.
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Originele X van punt B (index 3)
    orig_x_b = pts[3][0]
    
    # 1. Roteer alle punten en tangents rond (0,0)
    new_pts = []
    for p in pts:
        x_rot = p[0]*cos_a - p[1]*sin_a
        z_rot = p[0]*sin_a + p[1]*cos_a
        new_pts.append(np.array([x_rot, z_rot]))
        
    new_ctrls = []
    for c in ctrls:
        x_rot = c[0]*cos_a - c[1]*sin_a
        z_rot = c[0]*sin_a + c[1]*cos_a
        new_ctrls.append(np.array([x_rot, z_rot]))
        
    # 2. Bereken de benodigde schaalfactor om Punt B weer op originele X te krijgen
    rot_x_b = new_pts[3][0]
    scale_x = orig_x_b / rot_x_b if rot_x_b != 0 else 1.0
    
    # 3. Schaal alle X-coördinaten (en de X van de tangents)
    for i in range(4):
        new_pts[i][0] *= scale_x
        new_ctrls[i][0] *= scale_x
        
    return new_pts, new_ctrls

def main():
    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pts_orig = [np.array(p) for p in data["Bow_Centerline_Points"]]
    ctrls_orig = [np.array(c) for c in data["Bow_Centerline_Tangents"]]

    # We testen 3 verschillende rounding angles
    angles = [0, 25, 50]
    colors = ['darkgreen', 'orange', 'red']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    t = np.linspace(0, 1, 50)
    
    for idx, angle in enumerate(angles):
        if angle == 0:
            pts, ctrls = pts_orig, ctrls_orig
            label = "Originele Bow Centre Line (0°)"
        else:
            pts, ctrls = rotate_and_lock_x(pts_orig, ctrls_orig, angle)
            label = f"Nieuwe Bezier Curve ({angle}° Rounding)"
            
        y_all, z_all = [], []
        for i in range(3):
            P0, P1 = pts[i], pts[i+1]
            C0, C1 = ctrls[i], ctrls[i+1]
            
            T0 = (C0 - P0) * 3
            if i == 2:
                T1 = (P1 - C1) * 3
            else:
                T1 = (C1 - P1) * 3
                
            y, z = cubic_hermite_spline(t, P0, P1, T0, T1)
            y_all.extend(y)
            z_all.extend(z)
            
        ax.plot(y_all, z_all, color=colors[idx], lw=3, label=label)
        
        # Plot de punten ter verificatie
        for i in range(4):
            ax.plot(pts[i][0], pts[i][1], color=colors[idx], marker='o', markersize=6)
            
    ax.axhline(0, color='k', ls='--', lw=1, label='Kiel (z=0)')
    ax.axhline(cfg.DOA, color='k', ls='--', lw=1, label=f'Dek (z={cfg.DOA})')
    ax.axvline(pts_orig[3][0], color='gray', ls=':', label=f'Vaste X voor Punt B ({pts_orig[3][0]:.2f}m)')

    ax.set_aspect('equal')
    ax.set_title('Concept: Bow Rounding (Rotatie rond (0,0) + X-Lock op Punt B)')
    ax.set_xlabel('Lengte X [m]')
    ax.set_ylabel('Hoogte Z [m]')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.savefig("C:/users/janal/scheepsgenerator/v2/concept_bow_rounding.png", dpi=150)
    print("Opgeslagen als concept_bow_rounding.png")

if __name__ == "__main__":
    main()