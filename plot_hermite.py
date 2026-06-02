import numpy as np
import matplotlib.pyplot as plt

def cubic_hermite_spline(t, P0, P1, T0, T1):
    """
    Evalueert een Cubic Hermite Spline.
    t: array van parameterwaarden tussen 0 en 1
    P0: startpunt (x, y)
    P1: eindpunt (x, y)
    T0: start tangent vector (dx, dy)
    T1: eind tangent vector (dx, dy)
    """
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    x = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
    y = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
    
    return x, y

def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bounding box (Rechthoek)
    # Linkermuur = 0 (Uiterste buitenkant)
    # Rechtermuur = 1 (Midden / Centerline)
    # Onderkant = 0 (Laagste punt / Kiel)
    # Bovenkant = 1 (Hoogste punt / Dek)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=2, label='Bounding Box')
    
    # 4 Punten voor de Hermite Spline (2 uiteindes, 2 voor de raaklijnen/tangents)
    # Uiteinde 1 op de rechtermuur (Midden)
    P0 = np.array([1.0, 0.2]) 
    # Uiteinde 2 op de bovenkant (Hoogste punt)
    P1 = np.array([0.3, 1.0]) 

    # Tangent controlepunten (zoals in veel CAD software)
    Ctrl0 = np.array([0.7, 0.2]) # Bepaalt de richting/sterkte vanaf P0
    Ctrl1 = np.array([0.3, 0.6]) # Bepaalt de richting/sterkte vanaf P1
    
    # Tangent vectoren berekend uit de controlepunten
    # (Vermenigvuldigd met 3 voor een standaard Bezier/Hermite conversie, 
    # maar we houden het puur vectorieel voor de visualisatie)
    T0 = (Ctrl0 - P0) * 3
    T1 = (P1 - Ctrl1) * 3

    # Genereer de curve
    t = np.linspace(0, 1, 100)
    curve_x, curve_y = cubic_hermite_spline(t, P0, P1, T0, T1)

    # Plot de curve
    ax.plot(curve_x, curve_y, 'b-', lw=3, label='Cubic Hermite Spline')
    
    # Plot de 4 punten
    ax.plot([P0[0], Ctrl0[0]], [P0[1], Ctrl0[1]], 'r:o', label='Tangent vector T0')
    ax.plot([P1[0], Ctrl1[0]], [P1[1], Ctrl1[1]], 'g:o', label='Tangent vector T1')
    
    ax.plot(P0[0], P0[1], 'bo', markersize=8, label='Uiteinde 1 (Rechtermuur)')
    ax.plot(P1[0], P1[1], 'bo', markersize=8, label='Uiteinde 2 (Bovenkant)')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Concept: Cubic Hermite Spline in Bounding Box')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Labels voor de muren
    ax.text(-0.05, 0.5, 'Linkermuur\n(Buitenkant)', rotation=90, va='center', ha='center')
    ax.text(1.05, 0.5, 'Rechtermuur\n(Midden)', rotation=270, va='center', ha='center')
    ax.text(0.5, 1.05, 'Bovenkant (Hoogste punt)', va='center', ha='center')
    ax.text(0.5, -0.05, 'Onderkant (Laagste punt)', va='center', ha='center')

    plt.savefig("hermite_concept.png", dpi=150)
    print("Plot opgeslagen als hermite_concept.png")

if __name__ == "__main__":
    main()