import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import config as cfg

def plot_midship():
    LPP = cfg.LPP
    
    # De positie van de midship is een percentage van Lpp vanaf de AP (Aft Perpendicular)
    # AP = x=0, FP = x=LPP
    center = LPP * (cfg.MIDSHIP_LOC_PCT / 100.0)
    l_mid = LPP * (cfg.MIDSHIP_LENGTH_PCT / 100.0)
    
    x_aft = center - (l_mid / 2.0)
    x_fwd = center + (l_mid / 2.0)
    
    # Afmetingen van de doorsnede
    B_half = cfg.BOA / 2.0
    D = cfg.DOA
    R = cfg.BILGE_RADIUS
    
    # 1. Platte bodem (z=0, y=0 tot B/2 - R)
    y_flat = np.linspace(0, max(0, B_half - R), 10)
    z_flat = np.zeros_like(y_flat)
    
    # 2. Ronde kim (kwart cirkel)
    # Hoek loopt van 270 graden (bodem) naar 360/0 graden (zijkant)
    theta = np.linspace(1.5 * np.pi, 2.0 * np.pi, 20)
    y_bilge = (B_half - R) + R * np.cos(theta)
    z_bilge = R + R * np.sin(theta)
    
    # 3. Verticale zijwand (y=B/2, z=R tot D)
    y_side = np.full(10, B_half)
    z_side = np.linspace(R, D, 10)
    
    # Combineer de delen tot één profiel voor stuurboord
    y_section = np.concatenate([y_flat, y_bilge, y_side])
    z_section = np.concatenate([z_flat, z_bilge, z_side])
    
    # 3D Grid genereren in de x-richting
    N_x = 20
    x_grid = np.linspace(x_aft, x_fwd, N_x)
    
    X, Y = np.meshgrid(x_grid, y_section)
    _, Z = np.meshgrid(x_grid, z_section)
    
    # Plot instellen
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Stuurboord oppervlak
    ax.plot_surface(X, Y, Z, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.3)
    # Bakboord oppervlak (spiegelen over y=0)
    ax.plot_surface(X, -Y, Z, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.3)
    
    # Dek dichtmaken
    X_deck, Y_deck = np.meshgrid(x_grid, np.linspace(-B_half, B_half, 10))
    Z_deck = np.full_like(X_deck, D)
    ax.plot_surface(X_deck, Y_deck, Z_deck, color='gray', alpha=0.5, edgecolor='none')
    
    # Spiegel/wanden dichtmaken voor duidelijker blok
    # Achterwand
    Y_wall, Z_wall = np.meshgrid(np.concatenate([-y_section[::-1], y_section]), z_section)
    # Let op: de Z_wall moet overeenkomen met de profielen.
    # Dit is alleen visueel dus we doen een simpele polygon patch of plot_surface met Y, Z grid
    
    # Assen en titels
    ax.set_xlabel('X [m] (AP -> FP)')
    ax.set_ylabel('Y [m] (Bakboord <-> Stuurboord)')
    ax.set_zlabel('Z [m] (Kiel -> Dek)')
    ax.set_title(f'3D Middenromp (Parallel Midship)\nLengte: {l_mid:.2f}m, Kimradius: {R}m')
    
    # Zorg dat de verhoudingen kloppen
    ax.set_box_aspect([l_mid, cfg.BOA, D])
    
    plt.tight_layout()
    plt.show()