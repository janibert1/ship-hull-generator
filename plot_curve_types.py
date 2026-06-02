import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Functie voor een Bezier Curve
def cubic_bezier(t, P0, P1, P2, P3):
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Definieer de 4 punten (P0 is uiteinde 1, P3 is uiteinde 2)
    # Punten liggen tussen 0 en 1
    P0 = np.array([1.0, 0.2])  # Rechtermuur
    P1 = np.array([0.8, 0.4])  # Tussenpunt 1
    P2 = np.array([0.4, 0.8])  # Tussenpunt 2
    P3 = np.array([0.3, 1.0])  # Bovenkant

    pts = np.vstack((P0, P1, P2, P3))

    # --- Optie A: Bezier Curve (Controlepunten) ---
    t_vals = np.linspace(0, 1, 100)
    bezier_pts = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_vals])
    
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=1)
    ax1.plot(bezier_pts[:,0], bezier_pts[:,1], 'b-', lw=3, label='Bezier Curve')
    ax1.plot(pts[:,0], pts[:,1], 'r--o', label='Controlepunten / Hulplijn')
    ax1.plot([P0[0], P3[0]], [P0[1], P3[1]], 'ko', markersize=8, label='Uiteindes')
    ax1.set_title('Optie 1: Controlepunten (Bezier)\nCurve wordt "aangetrokken" door tussenpunten')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.1, 1.1); ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc='lower left', fontsize=8)

    # --- Optie B: Geïnterpoleerde Spline (Gaat DOOR de punten) ---
    # Voor CubicSpline in scipy moeten de x-waarden strikt oplopend of aflopend zijn
    # We parametriseren over een variabele t
    t_nodes = np.linspace(0, 1, 4)
    cs_x = CubicSpline(t_nodes, pts[:,0])
    cs_y = CubicSpline(t_nodes, pts[:,1])
    
    interp_x = cs_x(t_vals)
    interp_y = cs_y(t_vals)

    ax2.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=1)
    ax2.plot(interp_x, interp_y, 'g-', lw=3, label='Interpolated Spline')
    ax2.plot(pts[:,0], pts[:,1], 'ro', markersize=6, label='Tussenpunten')
    ax2.plot([P0[0], P3[0]], [P0[1], P3[1]], 'ko', markersize=8, label='Uiteindes')
    ax2.set_title('Optie 2: Interpolatie\nCurve gaat EXACT door alle 4 de punten')
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.1, 1.1); ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='lower left', fontsize=8)

    plt.suptitle('Welk type "4-punten spline" gebruik je?', fontsize=14)
    plt.savefig("C:/users/janal/scheepsgenerator/v2/curve_keuze.png", dpi=150)
    print("Plot opgeslagen als C:/users/janal/scheepsgenerator/v2/curve_keuze.png")

if __name__ == "__main__":
    main()