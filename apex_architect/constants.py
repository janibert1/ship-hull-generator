from __future__ import annotations

# Design constants
DESIGN_SPEED_KN = 14.0
CSTERN = -25.0
MIN_FREEBOARD_M = 3.0
MIN_GM_M = 1.0
MIN_DRAFT_M = 0.2
MAX_DRAFT_M = 10.0
MAX_FN = 0.80
MIN_ROLL_PERIOD_S = 8.0
MAX_ROLL_PERIOD_S = 16.0

TP_WEIGHT_T = 550.0
TP_RADIUS_M = 4.0
TP_MARGIN_M = 0.5
TP_GAP_M = 0.0
TP_CG_HEIGHT_M = 10.0

# Crane / lifting operation requirements
CRANE_SWL_FIXED_T = 595.1
CRANE_JIB_ANGLE_FIXED_DEG = 60.0
CRANE_SLEW_FIXED_DEG = 90.0
CRANE_PIVOT_HEIGHT_FIXED_M = 15.0
CRANE_BOOM_LENGTH_FIXED_M = 55.0
CRANE_PIVOT_EDGE_INSET_M = 0.75
CRANE_MIN_PIVOT_HEIGHT_M = 1.0
CRANE_MAX_HEEL_DEG = 5.0
CRANE_SWL_FULL_ANGLE_DEG = 60.0
CRANE_SWL_ZERO_ANGLE_FACTOR = 0.50
CRANE_BOOM_MASS_FRAC = 0.17
CRANE_HOUSE_MASS_FRAC = 0.34
CRANE_RIGGING_MASS_FRAC = 0.06
CRANE_RIGGING_HEIGHT_M = 8.0
CRANE_LOAD_HEIGHT_M = 20.0
CRANE_CLEARANCE_M = 1.0

MIN_BALLAST_FRAC = 0.08
MAX_BALLAST_FRAC = 0.40
LOFT_N_U_FAST = 60
LOFT_N_T_FAST = 60

# Robustness buffers for strict equilibrium viability in optimization.
STRICT_EQ_TANK3_BUFFER_M3 = 8.0
STRICT_EQ_TANK2_HEADROOM_BUFFER_M3 = 8.0

# Post-optimization full verification using groep10 strength pipeline
STRENGTH_SIGMA_ALLOW_MPA = 190.0
STRENGTH_FORCE_RESIDUAL_MAX_KN = 1.0
STRENGTH_MOMENT_RESIDUAL_MAX_MNM = 10.0
TRIM_LCG_LCB_TOL_M = 0.10
HEEL_TCG_TOL_M = 0.05

HULL_MASS_FACTOR = 2.1
STRUCTURAL_AREA_FACTOR = 1.15
STEEL_RHO = 7850.0
LSW_PER_PAYLOAD_CAP = 500.0
DEFAULT_EVAL_THREADS = 8

# Hypervolume monitoring / early-stop
HV_REF_F1 = 700.0
HV_REF_F2 = LSW_PER_PAYLOAD_CAP
HV_REF_F3 = 0.0
HV_EARLYSTOP_PATIENCE_GEN = 40
HV_EARLYSTOP_MIN_DELTA = 1e-4

# Midship/section-inspired correction for early LSW estimate
LSW_B_REF_M = 18.0
LSW_D_REF_M = 8.0
LSW_B_EXP = 1.20
LSW_D_EXP = 1.15
LSW_STRENGTH_GAIN = 0.30

# Goal-reference scoring zones
TARGET_RESISTANCE_KN = 250.0
LIMIT_RESISTANCE_KN = 450.0
TARGET_PAYLOAD_TPS = 35.0
LIMIT_PAYLOAD_TPS = 10.0
TARGET_LSW_T = 800.0
LIMIT_LSW_T = 1500.0

# Competition benchmark targets (beat-best strategy from provided leaderboard)
BENCHMARK_RESISTANCE_KN = 268.356
BENCHMARK_PAYLOAD_T = 3680.0
BENCHMARK_CRANE_LOAD_MN = 4.500
BENCHMARK_SHIP_WEIGHT_T = 1780.0

# Decision vector:
#   0..21  main vars (includes crane pivot x + explicit payload control)
#   22     crane boom length (m)
#   23     crane pivot Y fraction (−1..+1; physical = frac × (b_half − 0.75 m))
#   24     crane jib angle (deg; 60–80)
#   25     crane slewing angle (deg; 0–360)
#   26..43 18 morph vars (2 global + 16 bezier perturbations)
MORPH_VAR_COUNT = 18
N_VAR = 44
