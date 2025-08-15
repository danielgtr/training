#!/usr/bin/env python3
"""
Create a Notion database for your running log and insert rows from your cleaned JSON.

Requirements:
  pip install notion-client
Environment:
  NOTION_TOKEN          -> your internal integration secret
  NOTION_PARENT_PAGE    -> target parent page ID (share the page with the integration!)

Usage:
  python notion_running_db.py --create-db
  python notion_running_db.py --add one /path/to/<basename>/<basename>.json
  python notion_running_db.py --add-batch /path/to/folder/with/jsons
"""

import os, json, math, argparse, glob
from notion_client import Client

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
PARENT_PAGE_ID = os.environ["NOTION_PARENT_PAGE"]

notion = Client(auth=NOTION_TOKEN)

DB_TITLE = "Running Log"

# --------- Database schema (properties) ----------
# Matches your current JSON; adds manual context fields you wanted.
PROPERTIES = {
    # Metadata
    "Name": {"title": {}},
    "Date & Time (Local)": {"date": {}},
    "Route / Location": {"rich_text": {}},
    "Run Category": {"select": {"options": [
        {"name": "easy"}, {"name": "long"}, {"name": "tempo"}, {"name": "intervals"},
        {"name": "race"}, {"name": "recovery"}, {"name": "progression"}, {"name": "fartlek"}
    ]}},
    "Shoes": {"select": {}},

    # Performance (session-level)
    "Distance (km)": {"number": {"format": "number"}},
    "Elapsed Time (min)": {"number": {"format": "number"}},
    "Moving Time (min)": {"number": {"format": "number"}},
    "Pace Avg (min/km)": {"number": {"format": "number"}},   # numeric, 4 dec in your JSON
    "Pace Avg (pretty)": {"rich_text": {}},                  # "m:ss/km" helper
    "Speed Avg (km/h)": {"number": {"format": "number"}},

    # Physiology / mechanics
    "HR Avg (bpm)": {"number": {"format": "number"}},
    "HR Max (bpm)": {"number": {"format": "number"}},
    "Power Avg (W)": {"number": {"format": "number"}},
    "Power NP (W)": {"number": {"format": "number"}},
    "Power Max (W)": {"number": {"format": "number"}},
    "Cadence Avg (spm)": {"number": {"format": "number"}},
    "Step Length Avg (m)": {"number": {"format": "number"}},
    "Vertical Oscillation Avg (mm)": {"number": {"format": "number"}},
    "Vertical Ratio Avg (%)": {"number": {"format": "number"}},

    # GCT (derived if possible from session avg_stance_time + balance)
    "GCT Left Avg (ms)": {"number": {"format": "number"}},
    "GCT Right Avg (ms)": {"number": {"format": "number"}},
    "GCT Imbalance Avg (ms)": {"number": {"format": "number"}},

    # Load / calories
    "Calories Active (kcal)": {"number": {"format": "number"}},
    "Resting Calories (kcal)": {"number": {"format": "number"}},
    "Gross Calories (kcal)": {"number": {"format": "number"}},

    # Terrain totals
    "Total Ascent (m)": {"number": {"format": "number"}},
    "Total Descent (m)": {"number": {"format": "number"}},

    # Counts / misc
    "Pauses": {"number": {"format": "number"}},
    "Laps": {"number": {"format": "number"}},

    # Context / subjective
    "RPE": {"number": {"format": "number"}},
    "In-run Nutrition": {"rich_text": {}},
    "Post-run Notes": {"rich_text": {}},

    # Devices used (cleaned & deduped in your JSON)
    "Devices": {"multi_select": {}},
}

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
    # Notion select must exist or it will be created on first use in UI; here we just set it.
    return {"select": {"name": name}} if name else {"select": None}

def multi_select_values(names):
    opts = [{"name": n} for n in names if n]
    return {"multi_select": opts}

# ------------- Create database -------------
def create_database():
    db = notion.databases.create(
        parent={"type": "page_id", "page_id": PARENT_PAGE_ID},
        title=[{"type": "text", "text": {"content": DB_TITLE}}],
        properties=PROPERTIES,
    )
    print("Created DB:", db["id"])
    return db["id"]

# ------------- JSON -> Notion page mapping -------------
def build_page_props(run_json, shoes=None, run_category=None, rpe=None,
                     inrun_nutrition=None, post_notes=None):
    """
    Map your cleaned JSON to Notion properties. Manual fields can be passed in
    or left None (you can fill them later in Notion).
    """
    file_name = (run_json.get("file") or {}).get("name")
    summary = run_json.get("summary") or {}
    session_std = ((run_json.get("session") or {}).get("std") or {})
    # numeric metrics
    distance_km = (summary.get("distance_m") or 0) / 1000.0 if summary.get("distance_m") else None
    elapsed_min = (summary.get("elapsed_s") or 0) / 60.0 if summary.get("elapsed_s") else None
    moving_min  = (summary.get("timer_s") or 0) / 60.0 if summary.get("timer_s") else None
    pace_min_km = summary.get("avg_pace_min_per_km")
    pace_pretty = summary.get("avg_pace")   # "m:ss/km"
    speed_avg   = kmh_from_pace_min_per_km(pace_min_km)

    # cadence avg (spm) lives in session.derived as avg_cadence_total_spm in your JSON
    cadence_spm = ((run_json.get("session") or {}).get("derived") or {}).get("avg_cadence_total_spm")

    # mechanics
    step_len_m = ((run_json.get("session") or {}).get("derived") or {}).get("avg_step_length_m")
    vo_mm      = ((run_json.get("session") or {}).get("derived") or {}).get("avg_vertical_oscillation")
    vr_pct     = ((run_json.get("session") or {}).get("derived") or {}).get("avg_vertical_ratio")

    # GCT split at session level (derive if we have stance_time + balance)
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

    # devices (already filtered/deduped by your exporter)
    devices = []
    for d in (summary.get("devices_used") or []):
        man = d.get("manufacturer")
        prod = d.get("product")
        # compact label like "Garmin Forerunner 955" or just product if no mfg
        label = " ".join([x for x in [man, prod] if x])
        if label:
            devices.append(label)

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
        "RPE": to_number(rpe if rpe is not None else (summary.get("workout_rpe"))),
        "In-run Nutrition": to_rt(inrun_nutrition),
        "Post-run Notes": to_rt(post_notes),

        # Devices
        "Devices": multi_select_values(devices),
    }
    return props

# ------------- Create a page (row) -------------
def add_run_page(database_id: str, json_path: str, shoes=None, run_category=None,
                 rpe=None, inrun_nutrition=None, post_notes=None):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    props = build_page_props(
        data,
        shoes=shoes,
        run_category=run_category,
        rpe=rpe,
        inrun_nutrition=inrun_nutrition,
        post_notes=post_notes,
    )
    notion.pages.create(parent={"type": "database_id", "database_id": database_id},
                        properties=props)
    print(f"[OK] inserted: {os.path.basename(json_path)}")

# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(description="Create Notion DB and insert runs from JSON.")
    ap.add_argument("--create-db", action="store_true", help="Create the Notion database.")
    ap.add_argument("--db-id", help="Existing database ID (if you already created it).")
    ap.add_argument("--add", help="Path to a single JSON file to insert.")
    ap.add_argument("--add-batch", help="Folder with cleaned JSONs (globs **/*.json).")
    ap.add_argument("--shoes", help="Shoes select value.")
    ap.add_argument("--category", help="Run Category select value (easy/long/tempo/intervals/race/...).")
    ap.add_argument("--rpe", type=float, help="RPE value.")
    ap.add_argument("--inrun", help="In-run Nutrition text.")
    ap.add_argument("--notes", help="Post-run Notes text.")
    args = ap.parse_args()

    db_id = args.db_id
    if args.create_db:
        db_id = create_database()

    if not db_id and (args.add or args.add_batch):
        raise SystemExit("You must provide --db-id (or run with --create-db first).")

    if args.add:
        add_run_page(db_id, args.add, shoes=args.shoes, run_category=args.category,
                     rpe=args.rpe, inrun_nutrition=args.inrun, post_notes=args.notes)

    if args.add_batch:
        # insert all *.json in folder (both full and no-records are fine; full preferred)
        json_paths = sorted(glob.glob(os.path.join(args.add_batch, "**", "*.json"), recursive=True))
        if not json_paths:
            print("[WARN] No JSON files found in folder.")
        for p in json_paths:
            # prefer full over __no-records if both exist
            if p.endswith("__no-records.json"):
                base = p[:-len("__no-records.json")] + ".json"
                if os.path.exists(base):
                    continue
            add_run_page(db_id, p, shoes=args.shoes, run_category=args.category,
                         rpe=args.rpe, inrun_nutrition=args.inrun, post_notes=args.notes)

if __name__ == "__main__":
    main()
