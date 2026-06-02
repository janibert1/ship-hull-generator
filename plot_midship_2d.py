import numpy as np
import matplotlib.pyplot as plt
import config as cfg

def get_midship_curve_2d(N=100):
    B_half = cfg.BOA / 2.0
    D = cfg.DOA
    R = cfg.BILGE_RADIUS
    
    # 1. Platte bodem
    y_flat = np.linspace(0, max(0, B_half - R), N // 3)
    z_flat = np.zeros_like(y_flat)
    
    # 2. Kimradius (Kwart cirkel)
    theta = np.linspace(1.5 * np.pi, 2.0 * np.pi, N // 3)
    y_bilge = (B_half - R) + R * np.cos(theta)
    z_bilge = R + R * np.sin(theta)
    
    # 3. Verticale zijwand
    y_side = np.full(N - 2*(N//3), B_half)
    z_side = np.linspace(R, D, N - 2*(N//3))
    
    y_all = np.concatenate([y_flat, y_bilge[1:], y_side[1:]])
    z_all = np.concatenate([z_flat, z_bilge[1:], z_side[1:]])
    
    return y_all, z_all

def main():
    y, z = get_midship_curve_2d(200)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Teken Stuurboord
    ax.plot(y, z, 'r-', lw=3, label='Midship Doorsnede (Stuurboord)')
    # Teken Bakboord (spiegeling)
    ax.plot(-y, z, 'r-', lw=3)
    
    # Teken een perfecte cirkel ter controle van de radius
    circle = plt.Circle((cfg.BOA/2.0 - cfg.BILGE_RADIUS, cfg.BILGE_RADIUS), cfg.BILGE_RADIUS, 
                        color='green', fill=False, linestyle='--', lw=2, label='Perfecte Cirkel Check')
    ax.add_patch(circle)

    # Lijnen voor de Centerline en Waterlijn/Dek
    ax.axvline(0, color='gray', linestyle=':', label='Centerline')
    ax.axhline(cfg.DOA, color='k', linestyle='--', label=f'Dek (D={cfg.DOA}m)')
    ax.axhline(0, color='k', linestyle='-', lw=1)

    # Belangrijk: aspect='equal' forceert 1:1 weergave
    ax.set_aspect('equal')
    
    # Inzoomen op de kim (bilge) voor detail, maar hou het hele schip in beeld
    ax.set_xlim(-cfg.BOA/2 - 2, cfg.BOA/2 + 2)
    ax.set_ylim(-2, cfg.DOA + 2)
    
    ax.set_title(f'2D Vooraanzicht Midship\nBreedte: {cfg.BOA}m, Holte: {cfg.DOA}m, Kimradius: {cfg.BILGE_RADIUS}m')
    ax.set_xlabel('Breedte (y)')
    ax.set_ylabel('Hoogte (z)')
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)
    
    plt.savefig("C:/users/janal/scheepsgenerator/v2/midship_2d_check.png", dpi=150)
    print("Plot opgeslagen als C:/users/janal/scheepsgenerator/v2/midship_2d_check.png")

if __name__ == "__main__":
    main()