import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import json
from pathlib import Path
import config as cfg
from bezier_math import get_rhino_style_spline

class DraggableBezier:
    def __init__(self, ax, pts, D, B_half, config_key):
        self.ax = ax
        self.pts = pts
        self.D = D
        self.B_half = B_half
        self.config_key = config_key
        
        self.selected_idx = None
        
        # Teken stippellijn tussen de controlepunten (de 'polygon')
        self.polygon_line, = ax.plot([], [], 'r--o', lw=1.5, markersize=8)
        
        # Teken de uiteindelijke vloeiende Bezier curve
        self.curve_line, = ax.plot([], [], 'b-', lw=3)
        
        self.update_plot()
        
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_plot(self):
        pts_arr = np.array(self.pts)
        
        # Update polygon
        self.polygon_line.set_data(pts_arr[:, 0], pts_arr[:, 1])
        
        # Update curve met Rhino-style B-Spline (NURBS)
        t = np.linspace(0, 1, 100)
        curve_pts = get_rhino_style_spline(t, pts_arr, degree=3)
        self.curve_line.set_data(curve_pts[:, 0], curve_pts[:, 1])
        
        self.ax.figure.canvas.draw_idle()

    def get_closest_point(self, event):
        min_dist = float('inf')
        idx = None
        for i, p in enumerate(self.pts):
            dist = np.hypot(p[0] - event.xdata, p[1] - event.ydata)
            if dist < min_dist and dist < 0.5:
                min_dist = dist
                idx = i
        return idx

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.selected_idx = self.get_closest_point(event)

    def on_release(self, event):
        self.selected_idx = None

    def on_motion(self, event):
        if self.selected_idx is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None: 
            return
        
        new_x = np.clip(event.xdata, -0.5, self.B_half + 0.5)
        new_y = np.clip(event.ydata, -0.5, self.D + 0.5)
        
        # Hard lock:
        if self.selected_idx == 0:
            new_x = 0.0 # Uiteinde 1 altijd op centerline (y=0)
        elif self.selected_idx == len(self.pts) - 1:
            new_y = self.D # Laatste punt altijd op dek (z=D)
            
        self.pts[self.selected_idx] = [new_x, new_y]
        self.update_plot()

def save_to_config(dragger):
    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Normalize back to 0-1 before saving
    norm_pts = [[p[0] / dragger.B_half, p[1] / dragger.D] for p in dragger.pts]
    data[dragger.config_key] = norm_pts
    
    with open(pad, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Succesvol opgeslagen in config.json onder {dragger.config_key} (genormaliseerd)!")

def main():
    B_half = cfg.BOA / 2.0
    D = cfg.DOA
    config_key = "Stern_Bezier_Points"

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    ax.plot([B_half, 0, 0, B_half, B_half], [0, 0, D, D, 0], 'k--', lw=2, label='Bounding Box')

    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if config_key in data:
        # Load and scale dimensionless points
        pts = [[p[0] * B_half, p[1] * D] for p in data[config_key]]
    else:
        # Default 6-point bezier (already dimensioned)
        pts = [
            [0.0, 0.0],
            [B_half * 0.2, D * 0.2],
            [B_half * 0.4, D * 0.4],
            [B_half * 0.6, D * 0.6],
            [B_half * 0.8, D * 0.8],
            [B_half * 0.9, D]
        ]

    dragger = DraggableBezier(ax, pts, D, B_half, config_key)

    ax.set_xlim(B_half + 1, -1)
    ax.set_ylim(-1, D + 1)
    ax.set_aspect('equal')
    ax.set_title('Interactieve Rhino-Style B-Spline (6 Punten)\nSleep de stippen!')
    ax.set_xlabel('Breedte (y) - Midden zit rechts')
    ax.set_ylabel('Hoogte (z)')
    ax.grid(True, alpha=0.3)
    
    custom_lines = [Line2D([0], [0], color='b', lw=3),
                    Line2D([0], [0], color='r', ls='--', marker='o')]
    ax.legend(custom_lines, ['Rhino-Style Curve', 'Controle Polygon (6 punten)'], loc='lower left')

    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn_save = Button(ax_button, 'Opslaan naar Config')
    
    def on_save_clicked(event):
        save_to_config(dragger)
        btn_save.label.set_text("Opgeslagen!")
        fig.canvas.draw_idle()
        
    btn_save.on_clicked(on_save_clicked)
    plt.show()

if __name__ == "__main__":
    main()