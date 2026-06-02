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
    x = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
    z = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
    return x, z

def main():
    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pts = [np.array(p) for p in data["Bow_Centerline_Points"]]
    ctrls = [np.array(c) for c in data["Bow_Centerline_Tangents"]]

    # Genereer de originele Bow Centerline (in het lokale X-Z vlak, Y=0)
    t_vals = np.linspace(0, 1, 30)
    x_cl_all, z_all = [], []
    
    for i in range(3):
        P0, P1 = pts[i], pts[i+1]
        C0, C1 = ctrls[i], ctrls[i+1]
        
        T0 = (C0 - P0) * 3
        if i == 2: T1 = (P1 - C1) * 3
        else:      T1 = (C1 - P1) * 3
            
        x, z = cubic_hermite_spline(t_vals, P0, P1, T0, T1)
        if i > 0: x, z = x[1:], z[1:]
        x_cl_all.extend(x)
        z_all.extend(z)
        
    x_cl_all = np.array(x_cl_all)
    z_all = np.array(z_all)
    y_cl_all = np.zeros_like(x_cl_all) # Centerline ligt op Y=0

    # Bepaal de 'Bow Rounding' curve
    # De curve zwaait uit in de Y-richting (breedte) met hoek theta, 
    # maar het uiteinde B blijft op dezelfde X.
    # Dit is mathematisch een 'Shear' of schaling: Y = X * tan(theta)
    angle_deg = 50.0
    angle_rad = np.radians(angle_deg)
    
    y_rounding_all = x_cl_all * np.tan(angle_rad)
    x_rounding_all = x_cl_all # X blijft hetzelfde!
    z_rounding_all = z_all    # Z blijft hetzelfde!

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot de Centerline (Donkergroen)
    ax.plot(x_cl_all, y_cl_all, z_all, 'darkgreen', lw=4, label='Originele Bow Centerline (Y=0)')
    
    # Plot de Rounding Curve aan Stuurboord (Oranje)
    ax.plot(x_rounding_all, y_rounding_all, z_rounding_all, 'orange', lw=4, label=f'Nieuwe Bow Rounding Curve ({angle_deg}° SB)')
    
    # Plot de Rounding Curve aan Bakboord voor de context (Rood)
    ax.plot(x_rounding_all, -y_rounding_all, z_rounding_all, 'r', lw=4, label=f'Nieuwe Bow Rounding Curve ({angle_deg}° BB)')
    
    # Grijze stippellijnen om aan te tonen dat X gelijk is gebleven
    for i in range(0, len(x_cl_all), 10):
        ax.plot([x_cl_all[i], x_rounding_all[i]], [y_cl_all[i], y_rounding_all[i]], [z_all[i], z_rounding_all[i]], 'gray', ls=':')
        ax.plot([x_cl_all[i], x_rounding_all[i]], [y_cl_all[i], -y_rounding_all[i]], [z_all[i], z_rounding_all[i]], 'gray', ls=':')

    # Omkadering ter referentie
    max_x = np.max(x_cl_all)
    max_y = np.max(y_rounding_all)
    D = cfg.DOA
    
    # Teken de randen van het schip ter referentie
    ax.plot([0, max_x], [0, 0], [0, 0], 'k--', lw=1) # Kiel Centerline
    ax.plot([max_x, max_x], [-max_y, max_y], [D, D], 'k--', lw=1, label='Vaste X-grens (Punt B)')
    
    # Forceer 1:1:1 aspect ratio voor ware weergave
    ax.set_box_aspect((max_x, max_y*2, D))
    ax.set_xlim(0, max_x)
    ax.set_ylim(-max_y, max_y)
    ax.set_zlim(0, D)
    
    ax.set_xlabel('Lokale Lengte X [m]')
    ax.set_ylabel('Breedte Y [m]')
    ax.set_zlabel('Hoogte Z [m]')
    ax.set_title(f'3D Concept: Bow Rounding (Hoek = {angle_deg}°)\nDe boeglijn zwaait uit naar de zijkant, met vaste X op het uiteinde')
    ax.view_init(elev=30, azim=-45)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("C:/users/janal/scheepsgenerator/v2/concept_bow_rounding_3d.png", dpi=150)
    print("Opgeslagen als concept_bow_rounding_3d.png")

if __name__ == "__main__":
    main()