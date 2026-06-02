# Ship Hull Generator

A parametric ship hull generator and engineering analysis tool written in Python. Describe a vessel geometry through a JSON configuration file, and the tool lofts the full 3-D hull surface, computes hydrostatics, longitudinal strength, stability, resistance, and tank equilibrium — then exports plots and data files ready for reporting.

An integrated multi-objective optimizer (**Apex Architect**) uses NSGA-II to search the design space and find Pareto-optimal hulls that simultaneously minimise resistance, maximise payload capacity, and minimise lightweight ship mass.

## Features

- **Parametric hull lofting** — stern Bézier polygon, parallel midship section, bow intermediate curve, and bow centerline spline; all interactively editable
- **Side flare** — configurable hull flare angle and rotation point
- **Hydrostatics** — buoyancy, draught, waterplane area, LCB, BM at any fill
- **Tank equilibrium** — three-tank layout (starboard side / centre / port side); automatic heel- and trim-free fill computation
- **Free-surface correction** — hull-integrated second moment of waterplane area for each tank
- **Longitudinal strength** — shear force, bending moment, stress, and deflection along the ship's length (Holtrop & Mennen section properties)
- **Resistance** — Holtrop–Mennen 1984 regression over a configurable speed range
- **Crane analysis** — SWL adequacy, reach, hook height, and heel check during lifting operations
- **Multi-objective optimisation** — NSGA-II via [pymoo](https://pymoo.org/), 44-variable design vector, 12 inequality constraints, live generation plot
- **Interactive editors** — matplotlib-based GUI editors for hull shape curves and deck transition pieces
- **Output** — PNG plots, `antwoordenblad.json`, `info.txt`, and a full CSV/JSON data folder

## Project layout

```
ship-hull-generator/
├── main.py                        # Interactive text menu (entry point)
├── itereer.py                     # Optimizer entry point (calls apex_architect)
├── plot_full_surface.py           # 3-D hull loft + crane rendering
├── config.json                    # Active design configuration
├── config_best.json               # Best design found by the optimizer
├── config_side_flare_example.json # Example config with side flare
├── limits.json                    # Optimizer variable bounds
├── bezier_math.py                 # Bézier / Hermite curve utilities
├── engineering/
│   ├── run.py                     # Main orchestrator — equilibrium → output
│   ├── hydrostatics.py            # Buoyancy, draught-finding, waterplane data
│   ├── tanks.py                   # Tank CSA, fill diagrams, free-surface Ix
│   ├── strength.py                # Longitudinal strength
│   ├── sections.py                # Shell cross-section properties
│   ├── geometry.py                # Low-level hull geometry helpers
│   ├── resistance.py              # Holtrop–Mennen resistance
│   ├── output.py                  # Save PNGs and report files
│   └── data_files.py              # Save CSV/JSON data folder
├── apex_architect/                # Multi-objective optimiser package
│   ├── cli.py                     # Command-line interface
│   ├── design.py                  # Decision-vector encode/decode
│   ├── evaluation.py              # Single-candidate evaluation
│   ├── problem.py                 # pymoo Problem wrapper
│   ├── runner.py                  # NSGA-II run loop + reporting
│   ├── mcdm.py                    # Multi-criteria scoring (MCDM)
│   ├── monitoring.py              # Live generation plot callback
│   ├── packing.py                 # Deck transition-piece packing
│   ├── constants.py               # Physical and algorithmic constants
│   └── io_utils.py                # Config I/O helpers
└── interactive_*.py               # GUI shape editors
```

## Installation

Python 3.11+ is recommended.

```bash
git clone https://github.com/janibert1/ship-hull-generator.git
cd ship-hull-generator
pip install numpy scipy matplotlib
# For the optimizer:
pip install pymoo
```

No additional build steps are required.

## Quick start

### Interactive menu

```bash
python main.py
```

The menu lets you edit hull geometry interactively, run engineering calculations, visualise the 3-D hull with crane, and launch the optimizer.

### Engineering calculations only

```bash
python engineering/run.py
```

Reads `config.json`, finds the equilibrium draught, computes strength and stability, and writes output to `engineering/output/` and `engineering/Data/`.

### Optimizer

```bash
# Quick smoke run (pop=40, gen=20)
python itereer.py --quick

# Standard run (pop=200, gen=500)
python itereer.py

# Custom run
python itereer.py --pop-size 300 --n-gen 1000 --seed 7 --threads 8
```

Results are written to `optim_results.json`. Use option **10** in the main menu to promote the best result into `config.json`.

## Configuration reference

All parameters live in `config.json`.

### User-editable geometry

| Key | Description |
|-----|-------------|
| `Length_Loa_m` | Length overall [m] |
| `Breadth_Boa_m` | Breadth on deck [m] |
| `Depth_Doa_m` | Depth overall [m] |
| `Lpp_Loa_ratio` | Lpp / Loa ratio |
| `MidshipLength_pct_Lpp` | Parallel midship length [% Lpp] |
| `Location_midship_pct_Lpp` | Midship location from AP [% Lpp] |
| `Bilge_Radius_m` | Bilge radius [m] |
| `Aft_Shoulder_pct` | Aft shoulder position |
| `Fwd_Shoulder_pct` | Forward shoulder position |
| `Bow_Rounding_deg` | Bow rounding angle [°] |
| `Side_Flare_deg` | Side flare angle [°, positive = inward/V-shape] |
| `Hull_Thickness_mm` | Plating thickness [mm] |

### Tank and loading

| Key | Description |
|-----|-------------|
| `Target_Draft_m` | Target draught for equilibrium [m] |
| `Tank1_Width_m` | Starboard side-tank width [m] |
| `Tank1_Fill_pct` | Starboard side-tank fill [%] |
| `Tank2_Length_pct_Loa` | Centre tank length [% Loa] |
| `Tank3_Width_m` | Port side-tank width [m] |

### Computed and written back by `run.py`

| Key | Description |
|-----|-------------|
| `Tank2_Fill_pct` | Centre tank fill for trim = 0 |
| `Tank2_Center_from_AP_m` | Centre tank longitudinal centre [m from AP] |
| `Tank3_Fill_pct` | Port side-tank fill for heel = 0 |

### Crane

Nested under `"Crane"`:

| Key | Description |
|-----|-------------|
| `swl_max_t` | Safe working load [t] |
| `pivot_x_m` | Crane pivot position from AP [m] |
| `pivot_height_m` | Pivot height above baseline [m] |
| `boom_length_m` | Boom length [m] |
| `jib_angle_deg` | Jib angle [°] |
| `slewing_angle_deg` | Slewing angle [°] |

### Speed and resistance

| Key | Description |
|-----|-------------|
| `Design_Speed_kn` | Design speed [kn] |
| `Max_Speed_delta_kn` | Range above design speed [kn] |
| `Speed_Steps_per_kn` | Evaluation steps per knot |
| `Cstern` | Stern shape factor (−25 U/pram … +10 V-form) |
| `Method_S_wet` | Wetted-area method (0 = regression, 1 = hull integral) |
| `Method_IE` | Half-entrance angle method (0 = 1984 reg., 1 = 1978, 2 = hull) |

## Output

After `python engineering/run.py`:

- `engineering/output/` — 9 PNG plots, `antwoordenblad.json`, `info.txt`
- `engineering/Data/` — CSV and JSON tables (hydrostatics, tanks, strength, resistance)

After the optimizer:

- `optim_results.json` — Pareto front, selected best, and extreme points

## Equilibrium logic

The calculation is driven by the user-set `Target_Draft_m`:

1. Hydrostatics at target draught → displacement, LCB, BM.
2. Structural mass from hull area + transition pieces + crane.
3. Tank 1 (starboard) fill is user-set.
4. Tank 3 (port) fill is computed so total transverse moment = 0 (no heel).
5. Tank 2 (centre) volume is computed from the remaining displacement budget.
6. Tank 2 longitudinal centre is computed so LCG = LCB (no trim).

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array maths, lofting |
| `scipy` | Integration, interpolation |
| `matplotlib` | Plotting and interactive editors |
| `pymoo` | NSGA-II optimizer (optional, only for `itereer.py`) |

## License

This project is released under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
