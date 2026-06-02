import numpy as np
from scipy.interpolate import BSpline

def get_rhino_style_spline(t_array, control_points, degree=3):
    """
    Evalueert een B-Spline die de controlepunten gebruikt zoals Rhino's 'Curve' commando.
    - Graad 3 (Cubic)
    - Geen smoothing (exacte wiskunde)
    - Chord-length knot vector
    """
    pts = np.array(control_points)
    n = len(pts)
    k = min(degree, n - 1) # Graad kan niet hoger zijn dan aantal punten - 1

    # 1. Bereken Chord Lengths tussen controlepunten voor de knoopvector
    # Dit zorgt voor de 'snelheid' en 'spanning' die Rhino ook heeft.
    deltas = np.diff(pts, axis=0)
    chords = np.sqrt(np.sum(deltas**2, axis=1))
    
    # Als alle punten op elkaar liggen, val terug op uniform
    if np.sum(chords) < 1e-9:
        knots_internal = np.linspace(0, 1, n - k + 1)
    else:
        # Genereer interne knopen op basis van relatieve chord lengths
        relative_chords = np.cumsum(chords) / np.sum(chords)
        # We hebben n - k interne intervallen nodig (inclusief 0 en 1)
        # Voor k=3 en n=7 hebben we 7-3+1 = 5 knopen nodig voor de intervallen.
        knots_internal = np.zeros(n - k + 1)
        knots_internal[0] = 0.0
        knots_internal[-1] = 1.0
        
        # Verdeel de overige n-k-1 knopen
        for i in range(1, n - k):
            # Pak een gemiddelde van de chord posities om de knoop te plaatsen
            # Dit is een standaard methode voor CP-gebaseerde knoopvectoren.
            knots_internal[i] = np.mean(relative_chords[i-1 : i+k-1])

    # 2. Bouw de 'Clamped' Knot Vector
    # [k+1 keer 0, interne knopen, k+1 keer 1]
    knots = np.concatenate([
        np.zeros(k),
        knots_internal,
        np.ones(k)
    ])
    
    # 3. Maak de B-Spline
    # Let op: scipy BSpline verwacht (knots, coefficients, degree)
    # knots lengte moet n + k + 1 zijn.
    # Onze knots lengte: k + (n-k+1) + k = n + k + 1. Klopt!
    spline = BSpline(knots, pts, k)
    
    return spline(t_array)
