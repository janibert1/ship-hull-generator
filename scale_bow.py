import json
from pathlib import Path
import config as cfg

def scale_bow():
    pad = Path(__file__).parent / "config.json"
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    LPP = cfg.LPP
    center = LPP * (cfg.MIDSHIP_LOC_PCT / 100.0)
    l_mid = LPP * (cfg.MIDSHIP_LENGTH_PCT / 100.0)
    x_fwd = center + (l_mid / 2.0)
    
    overhang_total = cfg.LOA - LPP
    x_stern = -(overhang_total / 2.0)
    
    # Doel X is x_stern + LOA
    target_bow_tip = x_stern + cfg.LOA
    
    # Lokale lengte nodig voor de boeg:
    target_local_length = target_bow_tip - x_fwd
    
    # Huidige lokale lengte:
    current_length = data["Bow_Centerline_Points"][-1][0]
    
    if current_length > 0:
        scale_factor = target_local_length / current_length
        print(f"Huidige lengte boeg: {current_length}m. Schalen naar {target_local_length}m met factor {scale_factor:.2f}")
        
        for key in ["Bow_Centerline_Points", "Bow_Centerline_Tangents", "Bow_Intermediate_Points", "Bow_Intermediate_Tangents"]:
            if key in data:
                for pt in data[key]:
                    pt[0] *= scale_factor
                    
        with open(pad, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
if __name__ == "__main__":
    scale_bow()