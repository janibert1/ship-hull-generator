import numpy as np
import matplotlib.pyplot as plt
import config as cfg

def cubic_hermite_spline(t, P0, P1, T0, T1):
    """Berekent 1 segment van de Cubic Hermite Spline"""
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    y = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
    z = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
    
    return y, z

def main():
    B_half = cfg.BOA / 2.0
    D = cfg.DOA

    fig, ax = plt.subplots(figsize=(10, 8))

    # Bounding box
    ax.plot([B_half, 0, 0, B_half, B_half], [0, 0, D, D, 0], 'k--', lw=2, label='Bounding Box')

    # 4 Punten OP de spline
    P0 = np.array([0.0, 1.0])           # Uiteinde 1 (Midden/Centerline)
    P1 = np.array([B_half * 0.4, 3.5])  # Tussenpunt 1
    P2 = np.array([B_half * 0.7, 6.5])  # Tussenpunt 2
    P3 = np.array([B_half * 0.9, D])    # Uiteinde 2 (Bovenkant/Dek)

    # 4 Punten voor de tangents (Hoek en Gewicht)
    # De vector van P naar Ctrl bepaalt de curve richting *vanaf* dat punt
    C0 = np.array([B_half * 0.3, 1.0])  # Trekt horizontaal vanaf P0
    C1 = np.array([B_half * 0.5, 5.0])  # Trekt schuin omhoog vanaf P1
    C2 = np.array([B_half * 0.8, 8.0])  # Trekt schuin omhoog vanaf P2
    C3 = np.array([B_half * 0.9, D - 2.0]) # Trekt recht naar beneden vanaf dek

    # Tangent vectoren (geschaald voor visualisatie)
    T0 = (C0 - P0) * 3
    T1 = (C1 - P1) * 3
    T2 = (C2 - P2) * 3
    T3 = (P3 - C3) * 3 # Let op de richting: de curve komt *aan* bij P3 vanuit onder

    # Plot de 3 segmenten van de piecewise spline
    t = np.linspace(0, 1, 50)
    
    # Segment 1: P0 -> P1
    seg1_y, seg1_z = cubic_hermite_spline(t, P0, P1, T0, T1)
    ax.plot(seg1_y, seg1_z, 'b-', lw=3, label='Spline Segmenten')
    
    # Segment 2: P1 -> P2
    seg2_y, seg2_z = cubic_hermite_spline(t, P1, P2, T1, T2)
    ax.plot(seg2_y, seg2_z, 'b-', lw=3)
    
    # Segment 3: P2 -> P3
    seg3_y, seg3_z = cubic_hermite_spline(t, P2, P3, T2, T3)
    ax.plot(seg3_y, seg3_z, 'b-', lw=3)

    # Visualiseer de punten en tangents
    pts = [P0, P1, P2, P3]
    ctrls = [C0, C1, C2, C3]
    colors = ['r', 'orange', 'm', 'g']
    
    for i in range(4):
        ax.plot([pts[i][0], ctrls[i][0]], [pts[i][1], ctrls[i][1]], color=colors[i], ls=':', marker='o', lw=2)
        ax.plot(pts[i][0], pts[i][1], 'ko', markersize=8) # Zwarte stip voor punten op de curve

    # Assen configureren
    ax.set_xlim(B_half + 1, -1)
    ax.set_ylim(-1, D + 1)
    ax.set_aspect('equal')
    
    ax.set_title('Piecewise Cubic Hermite Spline (8 Punten)\n4 punten op de curve (Zwart), 4 tangent/gewicht stuurpunten (Gekleurd)')
    ax.set_xlabel('Breedte (y) - Midden zit rechts')
    ax.set_ylabel('Hoogte (z)')
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='b', lw=3),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8),
                    Line2D([0], [0], color='gray', ls=':', marker='o')]
    ax.legend(custom_lines, ['De uiteindelijke Curve', '4 Punten OP de curve', '4 Tangent "Stuurpunten" (Hoek/Gewicht)'], loc='lower left')
    
    ax.grid(True, alpha=0.3)

    plt.savefig("C:/users/janal/scheepsgenerator/v2/stern_8_points.png", dpi=150)
    print("Plot opgeslagen als C:/users/janal/scheepsgenerator/v2/stern_8_points.png")

if __name__ == "__main__":
    main()