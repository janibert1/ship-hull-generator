import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.path import Path as MplPath
from pathlib import Path
import json
import config as cfg
from plot_full_surface import build_hull_loft

TP_DIAM    = 8.0
TP_RADIUS  = 4.0
TP_MIN_GAP = 0.5
WT_MIN     = 230.0
WT_MAX     = 550.0
WT_DEFAULT = 350.0
CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_pieces():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [dict(p) for p in data.get("Transition_Pieces", [])]


def _save_pieces(pieces):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["Transition_Pieces"] = pieces
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Opgeslagen: {len(pieces)} transitiestuk(ken).")


def _build_deck_outline(geo):
    """Bovenaanzicht-omtrek van het dek: (x_array, y_halfwidth_array)."""
    X, Y = geo['X_surf'], geo['Y_surf']
    x_row = X[:, 0]          # één x-waarde per loft-rij
    y_top = Y[:, -1]         # halve deksbreedde (laatste parametrische punt)
    return x_row, y_top


def _halfwidth_at_x(x_pos, x_row, y_top):
    """Geïnterpoleerde halve deksbreedde op positie x (0 buiten bereik)."""
    if x_pos < x_row[0] or x_pos > x_row[-1]:
        return 0.0
    return float(np.interp(x_pos, x_row, y_top))


def _circle_fits(cx, cy, x_row, y_top, existing):
    """True als een TP met middelpunt (cx, cy) past op het dek."""
    # Controleer 24 punten op de cirkelrand
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        px = cx + TP_RADIUS * np.cos(a)
        py = cy + TP_RADIUS * np.sin(a)
        hw = _halfwidth_at_x(px, x_row, y_top)
        if hw < 0.5 or abs(py) > hw - 0.05:
            return False
    # Geen overlap met reeds geplaatste stukken
    # (middelafstand >= diameter + minimale tussenruimte).
    for p in existing:
        if np.hypot(p['x'] - cx, p['y'] - cy) < (TP_DIAM + TP_MIN_GAP - 0.05):
            return False
    return True


def main():
    print("Loft bouwen…")
    geo = build_hull_loft(N_u=50, N_t=80)
    x_row, y_top = _build_deck_outline(geo)
    pieces = _load_pieces()

    # ── figuur-indeling ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 7))
    fig.patch.set_facecolor('#1e1e2e')

    # Bovenaanzicht-assen: 80 % breedte, vol hoogte
    ax = fig.add_axes([0.01, 0.18, 0.78, 0.78])
    ax.set_facecolor('#2a2a3e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555577')
    ax.tick_params(colors='#ccccdd')
    ax.xaxis.label.set_color('#ccccdd')
    ax.yaxis.label.set_color('#ccccdd')

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]  —  Scheepslengte  (AP = 0)', fontsize=10)
    ax.set_ylabel('Y [m]  —  Breedte', fontsize=10)
    ax.set_title('Transitiestukken op dek  ·  LMB = plaatsen / verwijderen  ·  RMB = verwijderen',
                 color='white', fontsize=11, pad=8)
    ax.grid(True, alpha=0.15, color='white')

    # Deksomtrek-polygoon
    ox = np.concatenate([x_row, x_row[::-1], [x_row[0]]])
    oy = np.concatenate([y_top, -y_top[::-1], [y_top[0]]])
    deck_patch = plt.Polygon(list(zip(ox, oy)), closed=True,
                              facecolor='#3a3a5a', edgecolor='#aaaacc',
                              linewidth=2, zorder=1)
    ax.add_patch(deck_patch)

    # Centerline
    ax.axhline(0, color='#666688', lw=0.8, ls='--', zorder=2)

    # AP / FP markeringen
    for xv, lbl in [(0, 'AP'), (geo['LPP'], 'FP')]:
        ax.axvline(xv, color='#888899', lw=0.6, ls=':')
        ax.text(xv, geo['B_half'] + 1.0, lbl, color='#aaaacc',
                ha='center', fontsize=8, zorder=6)

    ax.set_xlim(x_row[0] - 4, x_row[-1] + 4)
    ax.set_ylim(-geo['B_half'] - 4, geo['B_half'] + 4)

    # ── kleurschaal sidebar ──────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.81, 0.18, 0.015, 0.78])
    norm_cb = plt.Normalize(vmin=WT_MIN, vmax=WT_MAX)
    cmap_tp = plt.cm.RdYlGn_r
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_cb, cmap=cmap_tp),
                      cax=ax_cb)
    cb.set_label('Gewicht [t]', color='white', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # ── widgets ─────────────────────────────────────────────────────────────
    slider_ax = fig.add_axes([0.06, 0.08, 0.50, 0.05])
    slider_ax.set_facecolor('#2a2a3e')
    weight_slider = Slider(slider_ax, 'Gewicht nieuw stuk [t]',
                           WT_MIN, WT_MAX, valinit=WT_DEFAULT, valstep=5.0,
                           color='#5566aa')
    weight_slider.label.set_color('white')
    weight_slider.valtext.set_color('white')

    btn_save_ax  = fig.add_axes([0.62, 0.06, 0.10, 0.07])
    btn_clear_ax = fig.add_axes([0.74, 0.06, 0.10, 0.07])
    btn_save  = Button(btn_save_ax,  'Opslaan',      color='#334466', hovercolor='#445577')
    btn_clear = Button(btn_clear_ax, 'Alles wissen', color='#553333', hovercolor='#664444')
    for btn in (btn_save, btn_clear):
        btn.label.set_color('white')
        btn.label.set_fontsize(10)

    # ── info-tekst ───────────────────────────────────────────────────────────
    info = ax.text(0.01, 0.98, '', transform=ax.transAxes, va='top',
                   color='white', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='#222233', alpha=0.85,
                             edgecolor='#445566'))

    # ── hover-preview cirkel ─────────────────────────────────────────────────
    hover_circ = [None]

    # ── stuk-patches ─────────────────────────────────────────────────────────
    piece_artists = []   # list of (circle_patch, text_artist)

    def _redraw():
        for circ, txt in piece_artists:
            circ.remove(); txt.remove()
        piece_artists.clear()

        total_w = sum(p['weight_t'] for p in pieces)
        for p in pieces:
            color = cmap_tp(norm_cb(p['weight_t']))
            c = plt.Circle((p['x'], p['y']), TP_RADIUS,
                            facecolor=color, edgecolor='white',
                            linewidth=1.2, alpha=0.90, zorder=4)
            ax.add_patch(c)
            t = ax.text(p['x'], p['y'], f"{p['weight_t']:.0f} t",
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='white', zorder=5)
            piece_artists.append((c, t))

        n = len(pieces)
        info.set_text(
            f"Geplaatst: {n}    Totaalgewicht: {total_w:.0f} t\n"
            f"Klik op bestaand stuk om te verwijderen"
        )
        btn_save.label.set_text('Opslaan')
        fig.canvas.draw_idle()

    # ── event-handlers ───────────────────────────────────────────────────────
    def _on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cx, cy = event.xdata, event.ydata
        if hover_circ[0] is not None:
            hover_circ[0].remove()
            hover_circ[0] = None

        valid = _circle_fits(cx, cy, x_row, y_top, pieces)
        ec = '#00ff88' if valid else '#ff4444'
        hc = plt.Circle((cx, cy), TP_RADIUS,
                         facecolor='none', edgecolor=ec,
                         lw=1.5, ls='--', alpha=0.75, zorder=6)
        ax.add_patch(hc)
        hover_circ[0] = hc
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cx, cy = event.xdata, event.ydata

        # Rechts-klik: verwijder dichtstbijzijnde stuk
        if event.button == 3:
            if not pieces:
                return
            idx = int(np.argmin([np.hypot(p['x']-cx, p['y']-cy) for p in pieces]))
            if np.hypot(pieces[idx]['x']-cx, pieces[idx]['y']-cy) <= TP_RADIUS + 0.5:
                pieces.pop(idx)
                _redraw()
            return

        if event.button != 1:
            return

        # Links-klik op bestaand stuk → verwijder
        for i, p in enumerate(pieces):
            if np.hypot(p['x']-cx, p['y']-cy) <= TP_RADIUS:
                pieces.pop(i)
                _redraw()
                return

        # Links-klik op lege dekplek → plaats nieuw stuk
        w = float(weight_slider.val)
        if _circle_fits(cx, cy, x_row, y_top, pieces):
            pieces.append({'x': round(cx, 2), 'y': round(cy, 2),
                           'weight_t': round(w, 1)})
            _redraw()

    def _on_save(event):
        _save_pieces(pieces)
        btn_save.label.set_text('Opgeslagen ✓')
        fig.canvas.draw_idle()

    def _on_clear(event):
        pieces.clear()
        _redraw()

    fig.canvas.mpl_connect('motion_notify_event', _on_move)
    fig.canvas.mpl_connect('button_press_event', _on_click)
    btn_save.on_clicked(_on_save)
    btn_clear.on_clicked(_on_clear)

    _redraw()
    plt.show()


if __name__ == '__main__':
    main()
