#!/usr/bin/env python3
"""
Add a running entry to your Notion database from a JSON file.

Requirements:
  pip install notion-client python-dotenv

Environment (.env file):
  NOTION_TOKEN          -> your internal integration secret
  NOTION_DATABASE_ID    -> database ID (created by create_notion_db.py)

Usage:
  python3 add_run_to_notion.py <json_file> [options]
  
Examples:
  python3 add_run_to_notion.py my_run.json
  python3 add_run_to_notion.py my_run.json --shoes "Nike Pegasus" --category easy --rpe 6
  python3 add_run_to_notion.py my_run.json --notes "Great run today!"
"""

import os, json, math, argparse

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] Loaded .env file")
except ImportError:
    print("[INFO] python-dotenv not installed, using system environment variables")
    pass

from notion_client import Client

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

notion = Client(auth=NOTION_TOKEN)

# ------------- Helpers: Notion property builders -------------
def to_title(text):
    txt = (text or "")[:2000]
    return {"title": [{"type": "text", "text": {"content": txt}}]}

def to_rt(text):
    if not text:
        return {"rich_text": []}
    txt = str(text)[:2000]
    return {"rich_text": [{"type": "text", "text": {"content": txt}}]}

def to_date(iso_str):
    return {"date": {"start": iso_str}} if iso_str else {"date": None}

def to_number(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return {"number": None}
    try:
        return {"number": float(x)}
    except Exception:
        return {"number": None}

def kmh_from_pace_min_per_km(pmk):
    try:
        return 60.0 / float(pmk) if pmk and float(pmk) > 0 else None
    except Exception:
        return None

def select_value(name):
    return {"select": {"name": name}} if name else {"select": None}

def multi_select_values(names):
    opts = [{"name": n} for n in names if n]
    return {"multi_select": opts}

# ------------- JSON -> Notion page mapping -------------
def build_page_props(run_json, shoes=None, run_category=None, rpe=None,
                     inrun_nutrition=None, post_notes=None):
    """
    Map your cleaned JSON to Notion properties.
    """
    file_name = (run_json.get("file") or {}).get("name")
    summary = run_json.get("summary") or {}
    session_std = ((run_json.get("session") or {}).get("std") or {})
    
    # Numeric metrics
    distance_km = (summary.get("distance_m") or 0) / 1000.0 if summary.get("distance_m") else None
    elapsed_min = (summary.get("elapsed_s") or 0) / 60.0 if summary.get("elapsed_s") else None
    moving_min  = (summary.get("timer_s") or 0) / 60.0 if summary.get("timer_s") else None
    pace_min_km = summary.get("avg_pace_min_per_km")
    pace_pretty = summary.get("avg_pace")
    speed_avg   = kmh_from_pace_min_per_km(pace_min_km)

    # Cadence avg (spm)
    cadence_spm = ((run_json.get("session") or {}).get("derived") or {}).get("avg_cadence_total_spm")

    # Mechanics
    step_len_m = ((run_json.get("session") or {}).get("derived") or {}).get("avg_step_length_m")
    vo_mm      = ((run_json.get("session") or {}).get("derived") or {}).get("avg_vertical_oscillation")
    vr_pct     = ((run_json.get("session") or {}).get("derived") or {}).get("avg_vertical_ratio")

    # GCT split at session level
    gct_left = gct_right = gct_imb = None
    if session_std.get("avg_stance_time") is not None and session_std.get("avg_stance_time_balance") is not None:
        try:
            bal = float(session_std["avg_stance_time_balance"]) / 100.0
            gct = float(session_std["avg_stance_time"])
            gct_left  = gct * bal
            gct_right = gct * (1.0 - bal)
            gct_imb   = abs(gct_left - gct_right)
        except Exception:
            pass

    # Devices
    devices = []
    for d in (summary.get("devices_used") or []):
        man = d.get("manufacturer")
        prod = d.get("product")
        # If product is already a mapped name (Forerunner 955, HRM-Pro Plus), use it alone
        if prod in ["Forerunner 955", "HRM-Pro Plus"]:
            devices.append(prod)
        else:
            # Otherwise, combine manufacturer and product as before
            label = " ".join([x for x in [man, prod] if x])
            if label:
                devices.append(label)

    # Convert RPE to "X/10" format
    raw_rpe = rpe if rpe is not None else summary.get("workout_rpe")
    rpe_display = None
    if raw_rpe is not None:
        if rpe is not None:
            # User provided RPE via command line - use as-is but add /10 format
            rpe_display = f"{int(rpe)}/10"
        else:
            # RPE from device - convert from device encoding (20 = 2/10)
            converted_value = int(raw_rpe / 10.0)
            rpe_display = f"{converted_value}/10"

    props = {
        # Metadata
        "Name": to_title(file_name or "Run"),
        "Date & Time (Local)": to_date(summary.get("local_time")),
        "Route / Location": to_rt(summary.get("location")),
        "Run Category": select_value(run_category),
        "Shoes": select_value(shoes),

        # Performance
        "Distance (km)": to_number(distance_km),
        "Elapsed Time (min)": to_number(elapsed_min),
        "Moving Time (min)": to_number(moving_min),
        "Pace Avg (min/km)": to_number(pace_min_km),
        "Pace Avg (pretty)": to_rt(pace_pretty),
        "Speed Avg (km/h)": to_number(speed_avg),

        # Physiology / mechanics
        "HR Avg (bpm)": to_number(summary.get("avg_heart_rate")),
        "HR Max (bpm)": to_number(summary.get("max_heart_rate")),
        "Power Avg (W)": to_number(summary.get("avg_power")),
        "Power NP (W)": to_number(summary.get("normalized_power")),
        "Power Max (W)": to_number(session_std.get("max_power")),
        "Cadence Avg (spm)": to_number(cadence_spm),
        "Step Length Avg (m)": to_number(step_len_m),
        "Vertical Oscillation Avg (mm)": to_number(vo_mm),
        "Vertical Ratio Avg (%)": to_number(vr_pct),

        "GCT Left Avg (ms)": to_number(gct_left),
        "GCT Right Avg (ms)": to_number(gct_right),
        "GCT Imbalance Avg (ms)": to_number(gct_imb),

        # Calories
        "Calories Active (kcal)": to_number(summary.get("total_calories_active_kcal")),
        "Resting Calories (kcal)": to_number(summary.get("resting_calories_kcal")),
        "Gross Calories (kcal)": to_number(summary.get("gross_calories_kcal")),

        # Terrain totals
        "Total Ascent (m)": to_number(summary.get("total_ascent")),
        "Total Descent (m)": to_number(summary.get("total_descent")),

        # Counts
        "Pauses": to_number(summary.get("pause_count")),
        "Laps": to_number(summary.get("lap_count")),

        # Context
        "RPE": to_rt(rpe_display),
        "In-run Nutrition": to_rt(inrun_nutrition),
        "Post-run Notes": to_rt(post_notes),

        # Devices
        "Devices": multi_select_values(devices),
    }
    return props

def add_run_to_notion(json_file, shoes=None, run_category=None, rpe=None,
                     inrun_nutrition=None, post_notes=None):
    """
    Add a single run to the Notion database from a JSON file.
    """
    try:
        print(f"[INFO] Reading JSON file: {json_file}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"[INFO] Building Notion properties...")
        
        props = build_page_props(
            data,
            shoes=shoes,
            run_category=run_category,
            rpe=rpe,
            inrun_nutrition=inrun_nutrition,
            post_notes=post_notes,
        )
        
        print(f"[INFO] Adding run to Notion database...")
        
        notion.pages.create(
            parent={"type": "database_id", "database_id": DATABASE_ID},
            properties=props
        )
        
        run_name = (data.get("file") or {}).get("name") or os.path.basename(json_file)
        print(f"[SUCCESS] Added run '{run_name}' to Notion database!")
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {json_file}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {json_file}: {e}")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to add run to Notion: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check that NOTION_DATABASE_ID is valid in your .env")
        print(f"  2. Run create_notion_db.py first if you haven't")
        print(f"  3. Make sure your integration has access to the database")
        exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Add a running entry to Notion database from JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 add_run_to_notion.py my_run.json
  python3 add_run_to_notion.py my_run.json --shoes "Nike Pegasus" --category easy
  python3 add_run_to_notion.py my_run.json --rpe 6 --notes "Great run today!"
        """
    )
    
    parser.add_argument("json_file", help="Path to the JSON file containing run data")
    parser.add_argument("--shoes", help="Shoes used for the run")
    parser.add_argument("--category", 
                       choices=["easy", "long", "tempo", "intervals", "race", "recovery", "progression", "fartlek"],
                       help="Run category")
    parser.add_argument("--rpe", type=float, help="Rate of Perceived Exertion (1-10)")
    parser.add_argument("--inrun", help="In-run nutrition notes")
    parser.add_argument("--notes", help="Post-run notes")
    
    args = parser.parse_args()
    
    add_run_to_notion(
        args.json_file,
        shoes=args.shoes,
        run_category=args.category,
        rpe=args.rpe,
        inrun_nutrition=args.inrun,
        post_notes=args.notes
    )

if __name__ == "__main__":
    main()
