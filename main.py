import sys
import subprocess
import json
from pathlib import Path

def import_optimizer_config_to_current(pad: Path):
    cfg_path = pad / "config.json"
    results_path = pad / "optim_results.json"
    best_cfg_path = pad / "config_best.json"

    print("\n" + "-"*55)
    print("  IMPORTEER OPTIMALISATIE-ONTWERP -> config.json")
    print("-"*55)
    print("1. Selected best (uit optim_results.json)")
    print("2. Extreme: minimum resistance")
    print("3. Extreme: maximum payload")
    print("4. Extreme: minimum empty ship weight")
    print("5. Gebruik config_best.json")
    print("0. Annuleren")
    print("-"*55)

    keuze = input("Kies een optie (0-5): ").strip()
    if keuze == "0":
        return

    selected_cfg = None
    label = ""

    if keuze in {"1", "2", "3", "4"}:
        if not results_path.exists():
            print(">>> FOUT: optim_results.json niet gevonden. Draai eerst optimalisatie.")
            return
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f">>> FOUT: kon optim_results.json niet lezen: {exc}")
            return

        key_map = {
            "1": ("selected_best", "selected_best"),
            "2": ("extremes.min_resistance", "extreme min_resistance"),
            "3": ("extremes.max_payload", "extreme max_payload"),
            "4": ("extremes.min_empty_ship_weight", "extreme min_empty_ship_weight"),
        }
        key, label = key_map[keuze]

        try:
            if key == "selected_best":
                selected_cfg = data.get("selected_best", {}).get("cfg")
            else:
                _, subkey = key.split(".", 1)
                selected_cfg = data.get("extremes", {}).get(subkey, {}).get("cfg")
                if selected_cfg is None and subkey == "min_empty_ship_weight":
                    # Backward compatibility with older result files.
                    selected_cfg = data.get("extremes", {}).get("min_lsw_per_payload", {}).get("cfg")
        except Exception:
            selected_cfg = None

        if not isinstance(selected_cfg, dict):
            print(f">>> FOUT: geen geldige cfg gevonden voor '{label}' in optim_results.json.")
            return

    elif keuze == "5":
        if not best_cfg_path.exists():
            print(">>> FOUT: config_best.json niet gevonden.")
            return
        try:
            with open(best_cfg_path, "r", encoding="utf-8") as f:
                selected_cfg = json.load(f)
            label = "config_best.json"
        except Exception as exc:
            print(f">>> FOUT: kon config_best.json niet lezen: {exc}")
            return
    else:
        print(">>> Ongeldige keuze.")
        return

    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(selected_cfg, f, indent=2, ensure_ascii=False)
        print(f">>> Opgeslagen: {label} is nu actief in config.json")
    except Exception as exc:
        print(f">>> FOUT: kon config.json niet overschrijven: {exc}")

def edit_parameters():
    pad = Path(__file__).parent / "config.json"
    
    keys = [
        ("Length_Loa_m", "Lengte overall (LOA) [m]"),
        ("Breadth_Boa_m", "Breedte op dek (BOA) [m]"),
        ("Depth_Doa_m", "Holte (DOA) [m]"),
        ("Lpp_Loa_ratio", "Lpp/Loa ratio"),
        ("MidshipLength_pct_Lpp", "Parallelle middenromp [% Lpp]"),
        ("Location_midship_pct_Lpp", "Locatie midscheeps [% Lpp]"),
        ("Bilge_Radius_m", "Kimradius [m]"),
        ("Aft_Shoulder_pct", "Aft Shoulder [% v. Mid-Aft naar AP]"),
        ("Fwd_Shoulder_pct", "Fwd Shoulder [% v. Mid-Fwd naar Bow]"),
        ("Location_bow_intermediate_curve_pct", "Locatie Bow Intermediate [% v. Shoulder naar Stem]"),
        ("Bow_Rounding_deg", "Bow Rounding [graden]"),
        ("Side_Flare_deg", "Side flare [graden, positief=inwaarts/V-vorm]"),
        ("Side_Flare_Rotation_Point", "Side flare draaipunt [0=holte, 1=waterlijn]"),
        ("Parallel_Midship_Combinations", "Parallel Midship Comb. [0/1/2]"),
        ("Hull_Thickness_mm",            "Romplaat dikte [mm, min 8]"),
        ("Target_Draft_m",         "Doeldiepgang [m]"),
        ("Tank1_Width_m",          "Tank 1 (SB) breedte [m]"),
        ("Tank1_Fill_pct",         "Tank 1 (SB) vulling [%]"),
        ("Tank2_Length_pct_Loa",   "Tank 2 (Mid) lengte [% LOA]"),
        ("Tank3_Width_m",          "Tank 3 (BB) breedte [m]"),
        ("Crane_slewing_angle_deg","Kraan slewing angle [graden] (in Crane-blok)"),
        ("Crane_pivot_x_m",        "Kraan pivot X [m vanaf AP] (in Crane-blok)"),
        ("Crane_pivot_height_m",   "Kraan pivot hoogte [m] (in Crane-blok)"),
        ("Crane_boom_length_m",    "Kraan gieklengte [m] (in Crane-blok)"),
        ("Design_Speed_kn",              "Ontwerpsnelheid [kn]"),
        ("Max_Speed_delta_kn",           "Max extra snelheid boven ontwerp [kn]"),
        ("Speed_Steps_per_kn",           "Stappen per knoop"),
        ("Cstern",                       "Cstern (-25 U-pram, +10 V-vormig, 0 normaal)"),
        ("Entrance_Angle_Factor_pct_BWL","Ingredehoekfactor [% BWL, bijv. 30]"),
        ("Method_S_wet",                 "S_wet methode (0=HM regressie, 1=romp)"),
        ("Method_IE",                    "IE methode (0=reg. 1984/1978, 2=romp)"),
    ]

    while True:
        with open(pad, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("\n" + "-"*40)
        print("  BEWERK NUMERIEKE PARAMETERS")
        print("-"*40)
        crane = data.get("Crane", {}) if isinstance(data.get("Crane", {}), dict) else {}
        crane_display_map = {
            "Crane_slewing_angle_deg": crane.get("slewing_angle_deg", "—"),
            "Crane_pivot_x_m": crane.get("pivot_x_m", "—"),
            "Crane_pivot_height_m": crane.get("pivot_height_m", "—"),
            "Crane_boom_length_m": crane.get("boom_length_m", "—"),
        }
        for i, (key, label) in enumerate(keys, 1):
            val = crane_display_map.get(key, data.get(key, '—'))
            print(f"{i}. {label:35}: {val}")
        print("0. Terug naar hoofdmenu")
        print("-"*40)

        choice = input("Kies een nummer om te wijzigen: ")
        if choice == "0":
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                key, label = keys[idx]
                crane_key_map = {
                    "Crane_slewing_angle_deg": ("slewing_angle_deg", 90.0),
                    "Crane_pivot_x_m": ("pivot_x_m", data.get("Length_Loa_m", 100.0) * data.get("Lpp_Loa_ratio", 0.93) * 0.8),
                    "Crane_pivot_height_m": ("pivot_height_m", 15.0),
                    "Crane_boom_length_m": ("boom_length_m", 55.0),
                }
                if key in crane_key_map:
                    ckey, default_val = crane_key_map[key]
                    crane = data.get("Crane", {}) if isinstance(data.get("Crane", {}), dict) else {}
                    huidig = crane.get(ckey, default_val)
                else:
                    huidig = data.get(key, '—')
                val = input(f"Nieuwe waarde voor {label} (huidig {huidig}): ")
                if key in crane_key_map:
                    ckey, _ = crane_key_map[key]
                    crane = data.get("Crane", {}) if isinstance(data.get("Crane", {}), dict) else {}
                    crane[ckey] = float(val)
                    data["Crane"] = crane
                else:
                    data[key] = float(val)
                with open(pad, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(">>> Opgeslagen!")
            else:
                print(">>> Ongeldige keuze.")
        except ValueError:
            print(">>> Voer a.u.b. een getal in.")

def menu():
    pad = Path(__file__).parent

    def run_optimizer_menu():
        def ask_min_tps(default: int = 0) -> int:
            raw = input(f"Minimum transition pieces [default {default}]: ").strip()
            if not raw:
                return default
            return max(0, int(raw))

        def ask_ship_mode() -> str:
            print("\n  Scheepsmodus (welke componenten meenemen):")
            print("  1. TPs en kraan (standaard)")
            print("  2. Alleen kraan (geen transition pieces)")
            print("  3. Alleen TPs (geen kraan)")
            raw = input("  Kies modus [default 1]: ").strip()
            mapping = {"1": "both", "2": "crane-only", "3": "tps-only"}
            return mapping.get(raw, "both")

        def ask_plot_mode() -> str:
            print("\n  Visualisatie tijdens optimalisatie:")
            print("  1. Automatisch — toon hull elke generatie (standaard)")
            print("  2. Op aanvraag — druk Enter in terminal om hull te tonen")
            print("  3. Uit — geen live plot")
            raw = input("  Kies modus [default 1]: ").strip()
            mapping = {"1": "auto", "2": "ondemand", "3": "none"}
            return mapping.get(raw, "auto")

        while True:
            print("\n" + "-"*50)
            print("  OPTIMALISATIE MENU (Apex Architect)")
            print("-"*50)
            print("1. Snelle run (--quick)")
            print("2. Standaard run (pop=200, gen=500)")
            print("3. Aangepast (pop/gen/seed/quiet)")
            print("0. Terug")
            print("-"*50)
            keuze_opt = input("Kies een optie (0-3): ").strip()

            if keuze_opt == "0":
                return
            if keuze_opt == "1":
                try:
                    min_tps = ask_min_tps(default=0)
                    ship_mode = ask_ship_mode()
                    plot_mode = ask_plot_mode()
                except ValueError:
                    print(">>> Ongeldige invoer: minimum transition pieces moet een geheel getal zijn.")
                    continue
                subprocess.run([sys.executable, str(pad / "itereer.py"), "--quick", "--min-tps", str(min_tps), "--ship-mode", ship_mode, "--plot-mode", plot_mode])
                return
            if keuze_opt == "2":
                try:
                    min_tps = ask_min_tps(default=0)
                    ship_mode = ask_ship_mode()
                    plot_mode = ask_plot_mode()
                except ValueError:
                    print(">>> Ongeldige invoer: minimum transition pieces moet een geheel getal zijn.")
                    continue
                subprocess.run([sys.executable, str(pad / "itereer.py"), "--min-tps", str(min_tps), "--ship-mode", ship_mode, "--plot-mode", plot_mode])
                return
            if keuze_opt == "3":
                try:
                    pop = input("Populatiegrootte [default 200]: ").strip()
                    gen = input("Generaties [default 500]: ").strip()
                    seed = input("Seed [default 42]: ").strip()
                    threads = input("Threads [default 8]: ").strip()
                    min_tps = ask_min_tps(default=0)
                    ship_mode = ask_ship_mode()
                    plot_mode = ask_plot_mode()

                    cmd = [sys.executable, str(pad / "itereer.py")]
                    if pop:
                        cmd += ["--pop-size", str(int(pop))]
                    if gen:
                        cmd += ["--n-gen", str(int(gen))]
                    if seed:
                        cmd += ["--seed", str(int(seed))]
                    if threads:
                        cmd += ["--threads", str(int(threads))]
                    cmd += ["--min-tps", str(min_tps)]
                    cmd += ["--ship-mode", ship_mode]
                    cmd += ["--plot-mode", plot_mode]

                    quick = input("Quick mode gebruiken? (j/n, default n): ").strip().lower()
                    if quick in ("j", "ja", "y", "yes"):
                        cmd += ["--quick"]

                    subprocess.run(cmd)
                    return
                except ValueError:
                    print(">>> Ongeldige invoer: pop/gen/seed moeten gehele getallen zijn.")
                    continue

            print(">>> Ongeldige keuze.")

    while True:
        print("\n" + "="*50)
        print("  SCHEEPSGENERATOR V2 - HOOFDMENU")
        print("="*50)
        print("1. Bewerk Achtersteven (Stern Bezier Polygon)")
        print("2. Bewerk Boeg Dwarsdoorsnede (Bow Intermediate)")
        print("3. Bewerk Voorsteven Zijaanzicht (Bow Centerline)")
        print("4. Bewerk Numerieke Parameters (LOA, BOA, etc.)")
        print("5. Bekijk 3D Romp — Kraan in LAADPOSITIE")
        print("6. Bewerk Transitiestukken op Dek")
        print("7. Engineering Berekeningen (sterkte, stabiliteit, data-export)")
        print("8. Bekijk Beste Ontwerp (3D, config_best.json)")
        print("9. Start Optimalisatie (NSGA-II / Pareto)")
        print("10. Zet best/extreme ontwerp naar config.json")
        print("11. Bekijk 3D Romp — Kraan in STOWAWAY (gestuwd)")
        print("0. Afsluiten")
        print("="*50)

        keuze = input("Kies een optie (0-11): ")

        if keuze == "1":
            subprocess.run([sys.executable, str(pad / "interactive_spline.py")])
        elif keuze == "2":
            subprocess.run([sys.executable, str(pad / "interactive_bow_int.py")])
        elif keuze == "3":
            subprocess.run([sys.executable, str(pad / "interactive_bow_spline.py")])
        elif keuze == "4":
            edit_parameters()
        elif keuze == "5":
            subprocess.run([sys.executable, str(pad / "plot_full_surface.py"), "--crane-mode", "loading"])
        elif keuze == "6":
            subprocess.run([sys.executable, str(pad / "interactive_transition_pieces.py")])
        elif keuze == "7":
            subprocess.run([sys.executable, str(pad / "engineering" / "run.py")])
            cfg_path = pad / "config.json"
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    _d = json.load(f)
                print("\n" + "-"*40)
                print("  BEREKENDE UITKOMSTEN (na berekening):")
                print(f"  Tank 2 (Mid) vulling : {_d.get('Tank2_Fill_pct', '—')} %")
                print(f"  Tank 2 (Mid) midden  : {_d.get('Tank2_Center_from_AP_m', '—')} m van AP")
                print(f"  Tank 3 (BB)  vulling : {_d.get('Tank3_Fill_pct', '—')} %")
                print("-"*40)
            except Exception:
                pass
        elif keuze == "8":
            best_cfg = pad / "config_best.json"
            if best_cfg.exists():
                subprocess.run([sys.executable, str(pad / "plot_full_surface.py"), "--config", "config_best.json"])
            else:
                print("\n>>> FOUT: config_best.json niet gevonden. Voer eerst de optimalisatie uit.")
        elif keuze == "9":
            run_optimizer_menu()
        elif keuze == "10":
            import_optimizer_config_to_current(pad)
        elif keuze == "11":
            subprocess.run([sys.executable, str(pad / "plot_full_surface.py"), "--crane-mode", "stowaway"])
        elif keuze == "0":
            print("Afsluiten...")
            break
        else:
            print("Ongeldige keuze. Probeer opnieuw.")

if __name__ == "__main__":
    menu()
