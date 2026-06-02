import json
from pathlib import Path

def load_config():
    pad = Path(__file__).parent / "config.json"
    with open(pad, encoding="utf-8") as f:
        return json.load(f)

_c = load_config()

# 1. Length
LOA = _c["Length_Loa_m"]
# 2. Breadth
BOA = _c["Breadth_Boa_m"]
# 3. Depth
DOA = _c["Depth_Doa_m"]
# 4. Lpp/L (position of the rudder / Lpp start)
LPP_RATIO = _c["Lpp_Loa_ratio"]

# 5. Midshiplength
MIDSHIP_LENGTH_PCT = _c["MidshipLength_pct_Lpp"]
# 6. Location of midship
MIDSHIP_LOC_PCT = _c["Location_midship_pct_Lpp"]
# 7. Bilge radius
BILGE_RADIUS = _c["Bilge_Radius_m"]
# 8. Shoulder Percentages
AFT_SHLD_PCT = _c.get("Aft_Shoulder_pct", 50.0)
FWD_SHLD_PCT = _c.get("Fwd_Shoulder_pct", 50.0)
# 9. Bow Intermediate Curve Location
BOW_INT_PCT = _c.get("Location_bow_intermediate_curve_pct", 50.0)
# 9. Bow Rounding Angle (degrees)
BOW_ROUNDING_DEG = _c.get("Bow_Rounding_deg", 50.0)
# 9b. Side flare of midship side wall [deg] and pivot interpolation [0..1]
SIDE_FLARE_DEG = _c.get("Side_Flare_deg", 0.0)
SIDE_FLARE_ROTATION_POINT = _c.get("Side_Flare_Rotation_Point", 0.0)
# 10. Parallel Midship Combinations (0=geen shoulders, 1=alleen shoulders, 2=ook 25%-curves)
PARALLEL_MIDSHIP_COMB = int(_c.get("Parallel_Midship_Combinations", 2))
# 11. Romplaat dikte [mm] — minimaal 8 mm, itereerbaar
HULL_THICKNESS_MM = _c.get("Hull_Thickness_mm", 8.0)

# 11b. Doeldiepgang [m]
TARGET_DRAFT = _c.get("Target_Draft_m", 2.0)

# 12. Tanks
TANK1_WIDTH   = _c.get("Tank1_Width_m",          3.0)   # stuurboord breedte [m]
TANK1_FILL    = _c.get("Tank1_Fill_pct",         50.0)   # vulling [%]
TANK2_LEN_PCT = _c.get("Tank2_Length_pct_Loa",  30.0)   # lengte als % LOA
TANK2_CENTER  = _c.get("Tank2_Center_from_AP_m", 20.0)  # midden t.o.v. AP [m]
TANK2_FILL    = _c.get("Tank2_Fill_pct",         75.0)   # vulling [%]
TANK3_WIDTH   = _c.get("Tank3_Width_m",          3.0)   # bakboord breedte [m]
TANK3_FILL    = _c.get("Tank3_Fill_pct",         50.0)   # vulling [%]

# Afgeleide afmetingen
LPP = LOA * LPP_RATIO
