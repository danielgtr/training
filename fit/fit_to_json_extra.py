#!/usr/bin/env python3
# fit_to_json_focus.py — Single-FIT → JSON (focused, running-first)
# Records (known-only):
#   timestamp, distance, enhanced_speed, enhanced_altitude, heart_rate, power, accumulated_power,
#   cadence_total_spm, stance_time_percent, stance_time, vertical_oscillation, vertical_ratio,
#   stance_time_balance, step_length_m, gct_left_ms/right_ms/imbalance_ms, geo (deg|sc|none),
#   pace_min_per_km (decimal, always).
# Laps:
#   filters laps <10s, keeps core + derived (average_pace_min_per_km, average_pace pretty, cadence_total_spm,
#   avg_step_length_m, avg_gct_* if available, total_strides from total_cycles).
# Summary (consolidated names):
#   local_time, timezone_offset_hours, location, sport, distance_m, timer_s, elapsed_s,
#   avg_pace_min_per_km (decimal) + avg_pace (pretty), avg_heart_rate, max_heart_rate,
#   avg_power, normalized_power, total_calories_active_kcal, resting_calories_kcal, gross_calories_kcal,
#   est_sweat_loss_ml, workout_rpe, workout_feel_0_100, stamina_*_pct, primary_benefit_code, vo2max_encoded_candidate,
#   avg_running_cadence_total_spm, avg_step_length_m, avg_vertical_oscillation, avg_vertical_ratio,
#   avg_stance_time, total_strides, total_ascent, total_descent, pause_count, lap_count, laps_removed, devices_used.
# Unknowns: only mapped session aliases (friendly names); no unknown_* leakage unless --keep-other-unknowns.
#
# CLI (unchanged except NO --with-pace):
#   python3 fit_to_json_focus.py input.fit [--out <path|->] [--no-records] [--geo deg|sc|none] [--keep-other-unknowns]
#
# Reqs: pip install fitdecode
# Optional (location label): pip install reverse_geocoder

import os, sys, json, argparse
from datetime import datetime, date, time, timedelta

try:
    import fitdecode
except Exception:
    sys.stderr.write("Instala fitdecode:  python -m pip install fitdecode\n")
    raise

# Optional reverse geocoding for location label in summary
try:
    import reverse_geocoder as rg
    _RG_OK = True
except Exception:
    _RG_OK = False

# ===== Aliases (FR955-oriented) mapped to friendly names =====
ALIAS_MAP_SESSION = {
    "unknown_196": ("resting_calories_kcal", "kcal", 1.0),
    "unknown_180": ("bmr_kcal_per_day", "kcal/day", 1.0),
    "unknown_178": ("est_sweat_loss_ml", "ml", 1.0),
    "unknown_192": ("workout_feel_0_100", None, 1.0),
    "unknown_193": ("workout_rpe", None, 1.0),
    "unknown_205": ("stamina_potential_start_pct", "%", 1.0),
    "unknown_206": ("stamina_potential_end_pct", "%", 1.0),
    "unknown_38":  ("end_position_lat_semicircles", "semicircles", 1.0),
    "unknown_39":  ("end_position_long_semicircles", "semicircles", 1.0),
    "unknown_188": ("primary_benefit_code", None, 1.0),
    "unknown_211": ("vo2max_encoded_candidate", None, 1.0),
    "unknown_207": ("body_battery_or_stamina_end_pct", "%", 1.0),
    "unknown_169": ("stride_counter_A_candidate", None, 1.0),
    "unknown_170": ("stride_counter_B_candidate", None, 1.0),
}

# ---------- utils ----------
def _json_default(o):
    if isinstance(o, (datetime, date, time)): return o.isoformat()
    if isinstance(o, (bytes, bytearray)):
        try: return o.decode("utf-8", "ignore")
        except Exception: return o.hex()
    try:
        import numpy as np
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
    except Exception:
        pass
    return str(o)

def write_json(path, payload):
    dirpath = os.path.dirname(path)
    if dirpath: os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

def to_text(x): return x.decode(errors="ignore") if isinstance(x, bytes) else x
def set_first_nonnull(d, k, v):
    if k not in d or d[k] is None: d[k] = v

def drop_nulls(d):  # quita claves con None
    return {k: v for k, v in d.items() if v is not None}

def semicircles_to_deg(v):
    return float(v) * (180.0 / (2**31)) if v is not None else None

def mm_to_m(v):
    return float(v) / 1000.0 if v is not None else None

def pace_min_per_km_from_mps(speed_mps):
    try:
        s = float(speed_mps)
        return (1000.0 / s) / 60.0 if s > 0 else None
    except Exception:
        return None

def pretty_pace_from_min(min_per_km):
    if min_per_km is None: return None
    minutes = int(min_per_km)
    seconds = int(round((min_per_km - minutes) * 60))
    if seconds == 60: minutes, seconds = minutes + 1, 0
    return f"{minutes}:{seconds:02d}/km"

def sanitize_list(dicts, keep_unknown=False):
    """Elimina unknown_* y None; opcionalmente conserva unknown_*."""
    out = []
    for d in dicts:
        if keep_unknown:
            clean = drop_nulls(dict(d))
        else:
            clean = {k: v for k, v in d.items()
                     if (v is not None and not (isinstance(k, str) and k.startswith("unknown_")))}
        out.append(clean)
    return out

# ---------- FIT reading ----------
def read_all_messages(path):
    out = {
        "file_id": {}, "file_creator": {},
        "developer_data_id": [], "field_description": [],
        "session": [], "lap": [], "record": [], "event": [],
        "device_info": [], "device_used": [],
        "gps_metadata_count": 0,
        "hr_zone": [], "power_zone": [], "time_in_zone": [],
        "activity": [],
        "other_messages": {}
    }
    with fitdecode.FitReader(path) as fr:
        for frame in fr:
            if not isinstance(frame, fitdecode.FitDataMessage): continue
            name = frame.name
            msg = {}
            for f in frame.fields:
                fname = to_text(getattr(f, "name", None))
                fval  = getattr(f, "value", None)
                set_first_nonnull(msg, fname, fval)

            if name == "file_id":
                out["file_id"] = msg
            elif name == "file_creator":
                out["file_creator"] = msg
            elif name == "gps_metadata":
                out["gps_metadata_count"] += 1
            elif name in out:
                if isinstance(out[name], list): out[name].append(msg)
                else: out["other_messages"].setdefault(name, []).append(msg)
            else:
                out["other_messages"].setdefault(name, []).append(msg)
    return out

# ---------- helpers / derived ----------
def extract_pauses(events, records):
    pauses, last_stop = [], None
    evs = sorted([e for e in events if "timestamp" in e], key=lambda x: x["timestamp"])
    for e in evs:
        if e.get("event") == "timer" and e.get("event_type") == "stop_all":
            last_stop = e.get("timestamp")
        elif e.get("event") == "timer" and e.get("event_type") == "start" and last_stop:
            start_ts = e.get("timestamp"); stop_ts = last_stop
            try: dur = (start_ts - stop_ts).total_seconds()
            except Exception: dur = None
            km_at = None
            try:
                dists = [r for r in records if r.get("timestamp") and r.get("distance") is not None]
                dists = [r for r in dists if r["timestamp"] <= stop_ts]
                if dists: km_at = float(dists[-1]["distance"]) / 1000.0
            except Exception: pass
            pauses.append({
                "start": _json_default(stop_ts),
                "end": _json_default(start_ts),
                "duration_s": dur,
                "km_at": km_at
            })
            last_stop = None
    return pauses

def apply_alias_session(sess):
    aliased, raw_unknowns = {}, {}
    for k, v in list(sess.items()):
        if str(k).startswith("unknown_"):
            raw_unknowns[k] = v
            if k in ALIAS_MAP_SESSION:
                aliased[ALIAS_MAP_SESSION[k][0]] = v

    std_keep = [
        "sport","sub_sport","trigger","message_index",
        "total_distance","total_timer_time","total_elapsed_time",
        "total_calories","total_ascent","total_descent","total_work",
        "avg_heart_rate","max_heart_rate","avg_power","max_power","normalized_power",
        "enhanced_avg_speed","enhanced_max_speed",
        "avg_running_cadence","avg_step_length","avg_vertical_oscillation","avg_vertical_ratio",
        "avg_stance_time","avg_stance_time_percent","avg_stance_time_balance",
        "total_cycles",
        "start_time","start_position_lat","start_position_long",
        "nec_lat","nec_long","swc_lat","swc_long",
        "max_temperature","avg_temperature",
        "intensity_factor","threshold_power","training_stress_score"
    ]
    std = {k: sess.get(k) for k in std_keep if k in sess}

    derived = {
        "start_lat_deg": semicircles_to_deg(sess.get("start_position_lat")),
        "start_lon_deg": semicircles_to_deg(sess.get("start_position_long")),
        "end_lat_deg": semicircles_to_deg(aliased.get("end_position_lat_semicircles")),
        "end_lon_deg": semicircles_to_deg(aliased.get("end_position_long_semicircles")),
        "bbox_deg": {
            "nec_lat_deg": semicircles_to_deg(sess.get("nec_lat")),
            "nec_lon_deg": semicircles_to_deg(sess.get("nec_long")),
            "swc_lat_deg": semicircles_to_deg(sess.get("swc_lat")),
            "swc_lon_deg": semicircles_to_deg(sess.get("swc_long")),
        },
        "avg_running_cadence_total_spm": (float(sess["avg_running_cadence"])*2.0) if sess.get("avg_running_cadence") is not None else None,
        "avg_step_length_m": mm_to_m(sess.get("avg_step_length")),
        "avg_vertical_oscillation": sess.get("avg_vertical_oscillation"),
        "avg_vertical_ratio": sess.get("avg_vertical_ratio"),
        "avg_stance_time": sess.get("avg_stance_time"),
        "total_strides": sess.get("total_cycles")
    }
    active = float(sess.get("total_calories") or 0.0)
    resting = aliased.get("resting_calories_kcal")
    derived["gross_calories_kcal"] = (active + float(resting)) if resting is not None else None

    return {"std": drop_nulls(std), "aliases": drop_nulls(aliased),
            "derived": drop_nulls(derived), "developer_raw_unknowns": raw_unknowns}

def enrich_lap(lp):
    row = dict(lp)
    if lp.get("avg_running_cadence") is not None:
        row["avg_running_cadence_total_spm"] = float(lp["avg_running_cadence"]) * 2.0
    if lp.get("max_running_cadence") is not None:
        row["max_running_cadence_total_spm"] = float(lp["max_running_cadence"]) * 2.0
    if lp.get("avg_step_length") is not None:
        row["avg_step_length_m"] = mm_to_m(lp["avg_step_length"])
    for key_sc, key_deg in [
        ("start_position_lat", "start_lat_deg"),
        ("start_position_long", "start_lon_deg"),
        ("end_position_lat", "end_lat_deg"),
        ("end_position_long", "end_lon_deg"),
    ]:
        row[key_deg] = semicircles_to_deg(lp.get(key_sc))
    if lp.get("total_cycles") is not None:
        row["total_strides"] = lp.get("total_cycles")
    # avg pace for lap (decimal min/km + pretty)
    td, tt = lp.get("total_distance"), lp.get("total_timer_time")
    if td and td > 0 and tt:
        pmk = (tt/60.0) / (td/1000.0)
        row["average_pace_min_per_km"] = pmk
        row["average_pace"] = pretty_pace_from_min(pmk)
    if lp.get("avg_stance_time") is not None and lp.get("avg_stance_time_balance") is not None:
        bal = float(lp["avg_stance_time_balance"]) / 100.0
        gct = float(lp["avg_stance_time"])
        row["avg_gct_left_ms"] = gct * bal
        row["avg_gct_right_ms"] = gct * (1.0 - bal)
        row["avg_gct_imbalance_ms"] = abs(row["avg_gct_left_ms"] - row["avg_gct_right_ms"])
    for k in ("start_position_lat","start_position_long","end_position_lat","end_position_long"):
        row.pop(k, None)
    return drop_nulls(row)

def build_record_focus(rec, geo="deg"):
    """
    Keep only agreed fields (+ derived):
      timestamp, distance,
      enhanced_speed,
      enhanced_altitude,
      heart_rate, power, accumulated_power,
      cadence_total_spm,
      stance_time_percent, stance_time, vertical_oscillation, vertical_ratio,
      stance_time_balance, step_length_m,
      gct_left_ms/right_ms/imbalance_ms,
      geo (deg|sc|none),
      pace_min_per_km (decimal).
    """
    out = {}
    out["timestamp"] = rec.get("timestamp")
    if rec.get("distance") is not None:
        out["distance"] = rec.get("distance")

    # speed/altitude: keep ONLY enhanced_* versions
    if rec.get("enhanced_speed") is not None:
        out["enhanced_speed"] = rec.get("enhanced_speed")
        pmk = pace_min_per_km_from_mps(rec.get("enhanced_speed"))
        if pmk is not None: out["pace_min_per_km"] = pmk

    if rec.get("enhanced_altitude") is not None:
        out["enhanced_altitude"] = rec.get("enhanced_altitude")

    # signals
    if rec.get("heart_rate") is not None: out["heart_rate"] = rec.get("heart_rate")
    if rec.get("power") is not None:      out["power"] = rec.get("power")
    if rec.get("accumulated_power") is not None: out["accumulated_power"] = rec.get("accumulated_power")

    # cadence total spm (no raw cadence, no fractional)
    cad = rec.get("cadence")
    if cad is not None:
        try: out["cadence_total_spm"] = round(float(cad) * 2.0, 1)
        except Exception: pass

    # running dynamics
    for k in ("stance_time_percent", "stance_time", "vertical_oscillation",
              "vertical_ratio", "stance_time_balance", "step_length"):
        if rec.get(k) is not None:
            out[k] = rec.get(k)

    if rec.get("step_length") is not None:
        out["step_length_m"] = mm_to_m(rec.get("step_length"))

    st, stb = rec.get("stance_time"), rec.get("stance_time_balance")
    if st is not None and stb is not None:
        try:
            bal = float(stb) / 100.0
            stf = float(st)
            out["gct_left_ms"] = stf * bal
            out["gct_right_ms"] = stf * (1.0 - bal)
            out["gct_imbalance_ms"] = abs(out["gct_left_ms"] - out["gct_right_ms"])
        except Exception:
            pass

    # geolocation
    if geo == "deg":
        lat_deg = semicircles_to_deg(rec.get("position_lat"))
        lon_deg = semicircles_to_deg(rec.get("position_long"))
        if lat_deg is not None: out["position_lat_deg"] = lat_deg
        if lon_deg is not None: out["position_long_deg"] = lon_deg
    elif geo == "sc":
        if rec.get("position_lat") is not None:  out["position_lat"]  = rec.get("position_lat")
        if rec.get("position_long") is not None: out["position_long"] = rec.get("position_long")

    return drop_nulls(out)

def first_valid_deg_coords(records):
    for r in records:
        lat = r.get("position_lat"); lon = r.get("position_long")
        if lat is not None and lon is not None and lat != 0 and lon != 0:
            return (semicircles_to_deg(lat), semicircles_to_deg(lon))
    return (None, None)

def detect_location_label(records):
    if not _RG_OK: return None
    lat, lon = first_valid_deg_coords(records)
    if lat is None or lon is None: return None
    try:
        res = rg.search([(lat, lon)])
        if res:
            r = res[0]
            parts = [p for p in [r.get("name",""), r.get("admin1",""), r.get("cc","")] if p]
            return ", ".join(parts) if parts else None
    except Exception:
        return None
    return None

def extract_local_time(activity_msgs, session_msg):
    """Prefer activity.local_timestamp; fallback session.local_timestamp."""
    def norm_pair(local_dt, utc_dt):
        if local_dt is None or utc_dt is None: return None, None
        if getattr(local_dt, "tzinfo", None) and not getattr(utc_dt, "tzinfo", None):
            utc_dt = utc_dt.replace(tzinfo=local_dt.tzinfo)
        elif getattr(utc_dt, "tzinfo", None) and not getattr(local_dt, "tzinfo", None):
            local_dt = local_dt.replace(tzinfo=utc_dt.tzinfo)
        try:
            off_h = int(round((local_dt - utc_dt).total_seconds() / 3600.0))
            return local_dt, off_h
        except Exception:
            return local_dt, None

    if activity_msgs:
        act = activity_msgs[0]
        lt, tz = norm_pair(act.get("local_timestamp"), act.get("timestamp"))
        if lt: return lt, tz

    if session_msg:
        utc = session_msg.get("start_time", session_msg.get("timestamp"))
        lt, tz = norm_pair(session_msg.get("local_timestamp"), utc)
        if lt: return lt, tz
    return None, None

# ---------- main pipeline ----------
def build_json_for_fit(fit_path, include_records=True, geo="deg",
                       keep_other_unknowns=False):
    data = read_all_messages(fit_path)

    session_src = data["session"][0] if data["session"] else {}
    session_block = apply_alias_session(session_src)

    # Local time + tz offset
    local_dt, tz_hours = extract_local_time(data["activity"], session_src)

    # Raw records for pauses/location
    records_raw = data["record"]

    # Laps: drop <10s
    removed_laps, kept_laps = [], []
    for idx, lp in enumerate(data["lap"], 1):
        dur = lp.get("total_timer_time")
        if dur is not None and dur < 10:
            removed_laps.append({"lap_index": idx, "duration_s": dur})
        else:
            kept_laps.append(enrich_lap(lp))
    laps_block = kept_laps

    # Records (focused)
    records_block = []
    if include_records:
        for r in records_raw:
            records_block.append(build_record_focus(r, geo=geo))

    # Pauses
    pauses = extract_pauses(data["event"], records_raw)

    # Clean messages (no unknown_*)
    activity_clean     = sanitize_list(data["activity"], keep_unknown=False)
    device_info_clean  = sanitize_list(data["device_info"], keep_unknown=False)
    device_used_clean  = sanitize_list(data["device_used"], keep_unknown=False)
    hr_zone_clean      = sanitize_list(data["hr_zone"], keep_unknown=False)
    power_zone_clean   = sanitize_list(data["power_zone"], keep_unknown=False)
    time_in_zone_clean = sanitize_list(data["time_in_zone"], keep_unknown=False)

    # Devices (compact list)
    devices_compact = []
    seen_serials = set()
    for d in device_info_clean:
        serial = d.get("serial_number")
        dtype  = d.get("antplus_device_type", d.get("device_type"))
        if serial and serial in seen_serials: continue
        if serial: seen_serials.add(serial)
        item = {
            "type": dtype,
            "manufacturer": d.get("manufacturer"),
            "product": d.get("garmin_product", d.get("product")),
            "serial": serial,
            "battery_status": d.get("battery_status"),
            "battery_voltage": d.get("battery_voltage")
        }
        devices_compact.append(drop_nulls(item))

    # Location label
    location_label = detect_location_label(records_raw)

    payload = {
        "file": {"name": os.path.basename(fit_path)},
        "metadata": {
            "file_id": data["file_id"],
            "file_creator": data["file_creator"],
            "developer_data_id": data["developer_data_id"],
            "field_descriptions": data["field_description"]
        },
        "session": {
            "std": session_block.get("std", {}),
            "derived": session_block.get("derived", {}),
            "aliases": session_block.get("aliases", {})
        },
        "laps": laps_block,
        "records": records_block if include_records else [],
        "events": {
            "raw": sanitize_list(data["event"], keep_unknown=False),
            "pause_count": len(pauses),
            "pauses": pauses
        },
        "messages": {
            "device_info": device_info_clean,
            "devices_used_compact": devices_compact,
            "device_used": device_used_clean,
            "gps_metadata_count": data["gps_metadata_count"],
            "hr_zone": hr_zone_clean,
            "power_zone": power_zone_clean,
            "time_in_zone": time_in_zone_clean,
            "activity": activity_clean
        },
        "provenance": {
            "alias_map": "FR955_v1_confirmed",
            "parser": "fitdecode",
            "notes": []
        }
    }

    if keep_other_unknowns:
        payload["messages"]["other"] = data["other_messages"]
        payload["session"]["developer_raw_unknowns"] = session_block.get("developer_raw_unknowns", {})

    # Consolidated summary with agreed names
    s_std = session_block.get("std", {})
    s_der = session_block.get("derived", {})
    s_alias = session_block.get("aliases", {})

    distance_m = s_std.get("total_distance")
    timer_s    = s_std.get("total_timer_time")
    avg_pace_min = (timer_s/60.0) / (distance_m/1000.0) if (timer_s and distance_m and distance_m > 0) else None

    payload["summary"] = drop_nulls({
        "local_time": _json_default(local_dt) if local_dt else None,
        "timezone_offset_hours": tz_hours,
        "location": location_label,
        "sport": s_std.get("sport"),
        "distance_m": distance_m,
        "timer_s": timer_s,
        "elapsed_s": s_std.get("total_elapsed_time"),
        "avg_pace_min_per_km": avg_pace_min,
        "avg_pace": pretty_pace_from_min(avg_pace_min) if avg_pace_min else None,
        "avg_heart_rate": s_std.get("avg_heart_rate"),
        "max_heart_rate": s_std.get("max_heart_rate"),
        "avg_power": s_std.get("avg_power"),
        "normalized_power": s_std.get("normalized_power"),
        "total_calories_active_kcal": s_std.get("total_calories"),
        "resting_calories_kcal": s_alias.get("resting_calories_kcal"),
        "gross_calories_kcal": s_der.get("gross_calories_kcal"),
        "est_sweat_loss_ml": s_alias.get("est_sweat_loss_ml"),
        "workout_rpe": s_alias.get("workout_rpe"),
        "workout_feel_0_100": s_alias.get("workout_feel_0_100"),
        "stamina_potential_start_pct": s_alias.get("stamina_potential_start_pct"),
        "stamina_potential_end_pct": s_alias.get("stamina_potential_end_pct"),
        "primary_benefit_code": s_alias.get("primary_benefit_code"),
        "vo2max_encoded_candidate": s_alias.get("vo2max_encoded_candidate"),
        "avg_running_cadence_total_spm": s_der.get("avg_running_cadence_total_spm"),
        "avg_step_length_m": s_der.get("avg_step_length_m"),
        "avg_vertical_oscillation": s_der.get("avg_vertical_oscillation"),
        "avg_vertical_ratio": s_der.get("avg_vertical_ratio"),
        "avg_stance_time": s_der.get("avg_stance_time"),
        "total_strides": s_der.get("total_strides"),
        "total_ascent": s_std.get("total_ascent"),
        "total_descent": s_std.get("total_descent"),
        "pause_count": len(pauses),
        "lap_count": len(laps_block),
        "laps_removed": removed_laps if removed_laps else None,
        "devices_used": devices_compact if devices_compact else None
    })

    return payload

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Single FIT → JSON (focused running metrics; no unknown_* leakage by default).")
    ap.add_argument("fit", help="Archivo .fit de entrada")
    ap.add_argument("--out", help="Ruta del JSON de salida (default: <fit>.json). Usa '-' para STDOUT.")
    ap.add_argument("--no-records", action="store_true", help="No incluir la serie temporal 'records'.")
    ap.add_argument("--geo", choices=["deg","sc","none"], default="deg",
                    help="Geolocalización por record: 'deg' (grados), 'sc' (semicírculos) o 'none'.")
    ap.add_argument("--keep-other-unknowns", action="store_true",
                    help="Conservar bloques unknown_* bajo messages.other y session.developer_raw_unknowns.")
    args = ap.parse_args()

    payload = build_json_for_fit(
        args.fit,
        include_records = (not args.no_records),
        geo = args.geo,
        keep_other_unknowns = args.keep_other_unknowns
    )

    if args.out in (None, ""):
        base = os.path.splitext(os.path.basename(args.fit))[0]
        out_path = f"{base}.json"
        write_json(out_path, payload)
        print(f"[OK] escrito {out_path}")
    elif args.out == "-":
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        write_json(args.out, payload)
        print(f"[OK] escrito {args.out}")

if __name__ == "__main__":
    main()
