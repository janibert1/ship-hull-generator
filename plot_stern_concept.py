import numpy as np
import matplotlib.pyplot as plt
import config as cfg

def cubic_hermite_spline(t, P0, P1, T0, T1):
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

    fig, ax = plt.subplots(figsize=(8, 8))

    # Bounding box (Rechthoek) op ware schaal
    # Rechtermuur = Centerline (y = 0)
    # Linkermuur = Buitenkant (y = B_half)
    # Onderkant = Kiel (z = 0)
    # Bovenkant = Dek (z = D)
    
    # We tekenen hem zo dat 0 (midden) rechts zit, en B_half (buitenkant) links.
    ax.plot([B_half, 0, 0, B_half, B_half], [0, 0, D, D, 0], 'k--', lw=2, label='Bounding Box')

    # Uiteinde 1: Op de rechtermuur (Centerline, y=0). 
    # Laten we aannemen dat de spiegel hier op z=0 (kiel) of iets hoger begint.
    # Ik zet hem voor het concept op y=0, z=2.0 (een lichte oplopende spiegel)
    P0 = np.array([0.0, 2.0])  

    # Uiteinde 2: Op de bovenkant (Dek, z=D).
    # Laten we aannemen dat het dek hier de volle breedte heeft, of iets smaller.
    P1 = np.array([B_half * 0.8, D]) 

    # Tangent controlepunten (hoeken en gewichten)
    # Ctrl0 trekt de curve vanuit het midden naar links en omhoog
    Ctrl0 = np.array([B_half * 0.5, 2.0]) 
    # Ctrl1 trekt de curve vanuit het dek recht naar beneden
    Ctrl1 = np.array([B_half * 0.8, D * 0.6]) 

    # Vectoren berekend uit de weging
    T0 = (Ctrl0 - P0) * 3
    T1 = (P1 - Ctrl1) * 3

    # Genereer de curve
    t = np.linspace(0, 1, 100)
    curve_y, curve_z = cubic_hermite_spline(t, P0, P1, T0, T1)

    # Plot de curve
    ax.plot(curve_y, curve_z, 'b-', lw=3, label='Achtersteven (Stern) Spline')
    
    # Plot de tangens (hoeken + gewicht)
    ax.plot([P0[0], Ctrl0[0]], [P0[1], Ctrl0[1]], 'r:o', lw=2, label='Tangent 1 (Hoek & Gewicht)')
    ax.plot([P1[0], Ctrl1[0]], [P1[1], Ctrl1[1]], 'g:o', lw=2, label='Tangent 2 (Hoek & Gewicht)')
    
    # Plot de uiteindes
    ax.plot(P0[0], P0[1], 'bo', markersize=8, label='Uiteinde 1 (Rechtermuur/Midden)')
    ax.plot(P1[0], P1[1], 'bo', markersize=8, label='Uiteinde 2 (Bovenkant/Dek)')

    # Assen omdraaien zodat 0 (midden) rechts zit en B_half links
    ax.set_xlim(B_half + 1, -1)
    ax.set_ylim(-1, D + 1)
    
    ax.set_aspect('equal')
    ax.set_title('Achtersteven (Stern) Cross-Section\nCubic Hermite Spline met Hoeken & Gewichten')
    ax.set_xlabel('Breedte (y) - Midden zit rechts!')
    ax.set_ylabel('Hoogte (z)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.savefig("C:/users/janal/scheepsgenerator/v2/stern_cross_section.png", dpi=150)
    print("Plot opgeslagen als C:/users/janal/scheepsgenerator/v2/stern_cross_section.png")

if __name__ == "__main__":
    main()