import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import json
from pathlib import Path
import config as cfg
from bezier_math import get_rhino_style_spline

class DraggableBezierBow:
    def __init__(self, ax, pts, D, x_start, x_end, config_key):
        self.ax = ax
        self.pts = pts
        self.D = D
        self.x_start = x_start
        self.x_end = x_end
        self.config_key = config_key
        
        self.selected_idx = None
        
        self.polygon_line, = ax.plot([], [], 'r--o', lw=1.5, markersize=8)
        self.curve_line, = ax.plot([], [], 'b-', lw=3)
        
        self.update_plot()
        
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_plot(self):
        pts_arr = np.array(self.pts)
        self.polygon_line.set_data(pts_arr[:, 0], pts_arr[:, 1])
        t = np.linspace(0, 1, 100)
        curve_pts = get_rhino_style_spline(t, pts_arr, degree=3)
        self.curve_line.set_data(curve_pts[:, 0], curve_pts[:, 1])
        self.ax.figure.canvas.draw_idle()

    def get_closest_point(self, event):
        min_dist = float('inf')
        idx = None
        for i, p in enumerate(self.pts):
            dist = np.hypot(p[0] - event.xdata, p[1] - event.ydata)
            if dist < min_dist and dist < 1.0:
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
        
        new_x = np.clip(event.xdata, -2.0, self.x_end + 15)
        new_y = np.clip(event.ydata, -1.0, self.D + 1.0)
        
        # Hard lock:
        if self.selected_idx == 0:
            new_x = 0.0
            new_y = 0.0
        elif self.selected_idx in (1, 2):
            new_y = 0.0
        elif self.selected_idx == len(self.pts) - 1:
            new_y = self.D
            
        self.pts[self.selected_idx] = [new_x, new_y]
        self.update_plot()

def save_to_config(dragger):
    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Normalize Z back to 0-1 before saving, X remains absolute
    norm_pts = [[p[0], p[1] / dragger.D] for p in dragger.pts]
    data[dragger.config_key] = norm_pts
    
    with open(pad, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Succesvol opgeslagen in config.json onder {dragger.config_key} (Z genormaliseerd)!")

def main():
    D = cfg.DOA
    config_key = "Bow_Centerline_Points"

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)
    
    ax.axhline(0, color='k', lw=2, label='Kiel (z=0)')
    ax.axhline(D, color='k', ls='--', lw=2, label=f'Dek (z={D}m)')
    ax.axvline(0, color='gray', lw=2, label='Start Boeg (x=0)')

    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if config_key in data:
        # Load and scale dimensionless Z
        pts = [[p[0], p[1] * D] for p in data[config_key]]
    else:
        # Default (already dimensioned)
        pts = [[0.0, 0.0], [2.0, D*0.1], [4.0, D*0.3], [6.0, D*0.5], [8.0, D*0.7], [9.0, D*0.85], [10.0, D]]

    LPP = cfg.LPP
    center = LPP * (cfg.MIDSHIP_LOC_PCT / 100.0)
    l_mid = LPP * (cfg.MIDSHIP_LENGTH_PCT / 100.0)
    x_fwd = center + (l_mid / 2.0)
    target_bow_tip = LPP
    max_bow_length = target_bow_tip - x_fwd

    dragger = DraggableBezierBow(ax, pts, D, 0.0, max_bow_length, config_key)

    ax.set_aspect('equal')
    ax.set_xlim(-2, max_bow_length + 5)
    ax.set_ylim(-2, D + 3)
    ax.axvline(max_bow_length, color='brown', ls='-.', label=f'Maximale Boeg ({max_bow_length:.2f}m)')
    
    ax.set_title('Interactieve Bow Centerline Rhino-Style (7 Punten)\nSleep de stippen!')
    ax.set_xlabel('Lengte X [m]')
    ax.set_ylabel('Hoogte Z [m]')
    ax.grid(True, alpha=0.3)
    
    custom_lines = [Line2D([0], [0], color='b', lw=3),
                    Line2D([0], [0], color='r', ls='--', marker='o')]
    ax.legend(custom_lines + ax.get_legend_handles_labels()[0], 
              ['Rhino-Style Curve', 'Controle Polygon (7 punten)'] + ax.get_legend_handles_labels()[1], 
              loc='upper left')

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