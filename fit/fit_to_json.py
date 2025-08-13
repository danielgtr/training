#!/usr/bin/env python3
# fit_to_json_focus.py — Single-FIT → JSON “enfocado”
# Basado en tu versión, con:
# - Records solo con campos conocidos + derivados (pace_s_per_km ON por defecto, cadence_total_spm, geo en grados).
# - Laps limpias con average_pace_s_per_km.
# - Summary mucho más completo (sin unknown_*).
# - Sin route. Sin campos de bici.
# Reqs: pip install fitdecode

import os, sys, json, argparse, math
from datetime import datetime, date, time

try:
    import fitdecode
except Exception:
    sys.stderr.write("Instala fitdecode:  python -m pip install fitdecode\n")
    raise

# ================== utilidades ==================

def _json_default(o):
    if isinstance(o, (datetime, date, time)):
        return o.isoformat()
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
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

def to_text(x): return x.decode(errors="ignore") if isinstance(x, bytes) else x

def drop_nulls(d):  # quita claves con None
    return {k: v for k, v in d.items() if v is not None}

def semicircles_to_deg(v):
    return float(v) * (180.0 / (2**31)) if v is not None else None

def mm_to_m(v):
    return float(v) / 1000.0 if v is not None else None

def fmt_pace(p):  # p en s/km → "mm:ss"
    try:
        if p is None or not math.isfinite(p) or p <= 0: return None
        m = int(p // 60)
        s = int(round(p - m*60))
        if s == 60:
            m += 1; s = 0
        return f"{m:02d}:{s:02d}"
    except Exception:
        return None

def sanitize_list(dicts):
    """Elimina claves unknown_* y valores None en una lista de dicts."""
    out = []
    for d in dicts:
        clean = {k: v for k, v in d.items()
                 if (v is not None and not (isinstance(k, str) and k.startswith("unknown_")))}
        out.append(clean)
    return out

# =============== lectura FIT (SIN _fields_meta; gps_metadata solo conteo) ===============

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
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            name = frame.name
            msg = {}
            for f in frame.fields:
                fname = to_text(getattr(f, "name", None))
                fval  = getattr(f, "value", None)
                if fname is not None and fname not in msg:
                    msg[fname] = fval

            if name == "file_id":
                out["file_id"] = msg
            elif name == "file_creator":
                out["file_creator"] = msg
            elif name == "gps_metadata":
                out["gps_metadata_count"] += 1
            elif name in out:
                if isinstance(out[name], list):
                    out[name].append(msg)
                else:
                    out["other_messages"].setdefault(name, []).append(msg)
            else:
                out["other_messages"].setdefault(name, []).append(msg)
    return out

# ===================== helpers de eventos/pausas =====================

def extract_pauses(events):
    pauses, last_stop = [], None
    evs = sorted([e for e in events if "timestamp" in e], key=lambda x: x["timestamp"])
    for e in evs:
        if e.get("event") == "timer" and e.get("event_type") in ("stop_all","stop"):
            last_stop = e.get("timestamp")
        elif e.get("event") == "timer" and e.get("event_type") == "start" and last_stop:
            pauses.append({"start": str(last_stop), "end": str(e.get("timestamp"))})
            last_stop = None
    return pauses

# ===================== LAPS =====================

LAP_KEEP = (
    "start_time","timestamp","total_distance","total_timer_time","total_elapsed_time",
    "total_calories","avg_speed","max_speed","avg_heart_rate","max_heart_rate",
    "avg_running_cadence","max_running_cadence","avg_power","max_power",
    "total_ascent","total_descent","avg_step_length","avg_vertical_oscillation",
    "avg_vertical_ratio","avg_stance_time","avg_stance_time_percent","avg_stance_time_balance",
    "start_position_lat","start_position_long","end_position_lat","end_position_long"
)

def enrich_lap(lp):
    f = {k: lp.get(k) for k in LAP_KEEP if k in lp}
    out = {}

    if f.get("start_time") is not None:
        out["start_time"] = f["start_time"]
    if f.get("timestamp") is not None:
        out["end_time"] = f["timestamp"]

    # métricas base
    for k in ("total_distance","total_timer_time","total_elapsed_time","total_calories",
              "avg_speed","max_speed","avg_heart_rate","max_heart_rate",
              "avg_power","max_power","total_ascent","total_descent"):
        if f.get(k) is not None:
            out[k] = float(f[k]) if isinstance(f[k], (int,float)) else f[k]

    # running dynamics por lap
    if f.get("avg_running_cadence") is not None:
        out["avg_running_cadence"] = float(f["avg_running_cadence"])
        out["avg_running_cadence_total_spm"] = float(f["avg_running_cadence"]) * 2.0
    for k in ("avg_step_length","avg_vertical_oscillation","avg_vertical_ratio",
              "avg_stance_time","avg_stance_time_percent","avg_stance_time_balance"):
        if f.get(k) is not None:
            out[k] = float(f[k]) if isinstance(f[k], (int,float)) else f[k]

    # pace por lap
    avg_speed = out.get("avg_speed")
    if (avg_speed is None or avg_speed <= 0) and out.get("total_distance") and out.get("total_timer_time"):
        avg_speed = float(out["total_distance"]) / float(out["total_timer_time"])
    if avg_speed and avg_speed > 0:
        out["average_pace_s_per_km"] = 1000.0 / avg_speed
        out["average_pace"] = fmt_pace(out["average_pace_s_per_km"])

    # geo
    for k_sc, k_deg in [("start_position_lat","start_lat_deg"),
                        ("start_position_long","start_lon_deg"),
                        ("end_position_lat","end_lat_deg"),
                        ("end_position_long","end_lon_deg")]:
        if f.get(k_sc) is not None:
            out[k_deg] = semicircles_to_deg(f[k_sc])

    return drop_nulls(out)

# ===================== RECORDS =====================

RECORD_KEEP = (
    "timestamp","position_lat","position_long","distance",
    "enhanced_speed","speed",
    "enhanced_altitude","altitude",
    "heart_rate","power","accumulated_power",
    "cadence","fractional_cadence",
    "stance_time_percent","stance_time","vertical_oscillation",
    "vertical_ratio","stance_time_balance","step_length",
)

def build_record_focus(rec, geo="deg", with_pace=True):
    """
    Mantiene: timestamp, distance, speed/altitude (enhanced si hay),
              heart_rate, power, **accumulated_power**, cadence_total_spm,
              stance_time_percent, stance_time, vertical_oscillation, vertical_ratio,
              stance_time_balance, step_length,
              geo (deg|sc|none), y pace_s_per_km (por defecto).
    Quita unknown_* y cualquier basura.
    """
    r = {k: rec.get(k) for k in RECORD_KEEP if k in rec}
    out = {}

    out["timestamp"] = r.get("timestamp")

    if r.get("distance") is not None:
        out["distance"] = float(r["distance"])

    spd = r.get("enhanced_speed") if r.get("enhanced_speed") is not None else r.get("speed")
    if spd is not None:
        key = "enhanced_speed" if r.get("enhanced_speed") is not None else "speed"
        out[key] = float(spd)
        if with_pace and float(spd) > 0:
            out["pace_s_per_km"] = 1000.0 / float(spd)

    alt = r.get("enhanced_altitude") if r.get("enhanced_altitude") is not None else r.get("altitude")
    if alt is not None:
        key = "enhanced_altitude" if r.get("enhanced_altitude") is not None else "altitude"
        out[key] = float(alt)

    if r.get("heart_rate") is not None: out["heart_rate"] = float(r["heart_rate"])
    if r.get("power") is not None: out["power"] = float(r["power"])
    if r.get("accumulated_power") is not None: out["accumulated_power"] = float(r["accumulated_power"])

    # Cadencia: guardar SPM total, no la mitad.
    cad = r.get("cadence")
    if cad is not None:
        try:
            fc = float(r.get("fractional_cadence") or 0.0)
            out["cadence_total_spm"] = 2.0 * (float(cad) + fc)
        except Exception:
            out["cadence_total_spm"] = 2.0 * float(cad)

    for k in ("stance_time_percent","stance_time","vertical_oscillation",
              "vertical_ratio","stance_time_balance","step_length"):
        if r.get(k) is not None:
            out[k] = float(r[k]) if isinstance(r[k], (int,float)) else r[k]

    # geolocalización
    if geo == "deg":
        lat_deg = semicircles_to_deg(r.get("position_lat"))
        lon_deg = semicircles_to_deg(r.get("position_long"))
        if lat_deg is not None: out["position_lat_deg"] = lat_deg
        if lon_deg is not None: out["position_long_deg"] = lon_deg
    elif geo == "sc":
        if r.get("position_lat") is not None:  out["position_lat"]  = r["position_lat"]
        if r.get("position_long") is not None: out["position_long"] = r["position_long"]
    # geo == "none": no guardar posición

    return drop_nulls(out)

# ===================== PIPELINE PRINCIPAL =====================

def build_json_for_fit(fit_path, include_records=True, geo="deg", with_pace=True, keep_other_unknowns=False):
    data = read_all_messages(fit_path)

    # Laps enriquecidas (limpias)
    laps_block = [enrich_lap(lp) for lp in data["lap"]]

    # Records
    records_block = [build_record_focus(r, geo=geo, with_pace=with_pace)
                     for r in data["record"]] if include_records else []

    # Pausas
    pauses = extract_pauses(data["event"])

    # Bloques limpios
    activity_clean     = sanitize_list(data["activity"])
    device_info_clean  = sanitize_list(data["device_info"])
    device_used_clean  = sanitize_list(data["device_used"])
    hr_zone_clean      = sanitize_list(data["hr_zone"])
    power_zone_clean   = sanitize_list(data["power_zone"])
    time_in_zone_clean = sanitize_list(data["time_in_zone"])

    # --------- summary (potente, sin unknowns) ----------
    session_src = data["session"][0] if data["session"] else {}
    s = {k: v for k, v in session_src.items() if not str(k).startswith("unknown_")}
    # básicos
    sport      = s.get("sport")
    sub_sport  = s.get("sub_sport")
    start_time = s.get("start_time")
    distance_m = s.get("total_distance")
    timer_s    = s.get("total_timer_time")
    elapsed_s  = s.get("total_elapsed_time")
    pause_s    = (float(elapsed_s) - float(timer_s)) if (elapsed_s and timer_s) else None
    moving_ratio = (float(timer_s)/float(elapsed_s)) if (elapsed_s and timer_s and elapsed_s>0) else None

    # velocidades/paces
    avg_speed = (s.get("enhanced_avg_speed") or s.get("avg_speed"))
    if (not avg_speed) and distance_m and timer_s and timer_s > 0:
        avg_speed = float(distance_m)/float(timer_s)
    max_speed = (s.get("enhanced_max_speed") or s.get("max_speed"))
    avg_pace  = (1000.0/float(avg_speed)) if (avg_speed and avg_speed>0) else None
    best_lap_pace = None
    if laps_block:
        for lp in laps_block:
            p = lp.get("average_pace_s_per_km")
            if p and p>0:
                best_lap_pace = p if (best_lap_pace is None or p < best_lap_pace) else best_lap_pace

    # dinámica agregada
    avg_rcad   = s.get("avg_running_cadence")
    avg_spm    = float(avg_rcad)*2.0 if avg_rcad is not None else None
    step_len_m = mm_to_m(s.get("avg_step_length"))
    vo_mm      = s.get("avg_vertical_oscillation")
    vr_pct     = s.get("avg_vertical_ratio")
    st_ms      = s.get("avg_stance_time")
    st_pct     = s.get("avg_stance_time_percent")
    stb_pct    = s.get("avg_stance_time_balance")

    # potencia/te
    avg_hr  = s.get("avg_heart_rate"); max_hr = s.get("max_heart_rate")
    avg_pwr = s.get("avg_power"); max_pwr = s.get("max_power")
    np_w    = s.get("normalized_power")
    ifactor = s.get("intensity_factor")
    ftp     = s.get("threshold_power")
    tss     = s.get("training_stress_score")
    te_aer  = s.get("total_training_effect")
    te_ana  = s.get("total_anaerobic_training_effect")

    # elevación/temperatura
    ascent  = s.get("total_ascent"); descent = s.get("total_descent")
    t_avg   = s.get("avg_temperature"); t_max = s.get("max_temperature")

    # geo de inicio/fin (fin se estima del último record con grados)
    start_lat = semicircles_to_deg(s.get("start_position_lat"))
    start_lon = semicircles_to_deg(s.get("start_position_long"))
    end_lat = end_lon = None
    for rec in reversed(records_block):
        if rec.get("position_lat_deg") is not None and rec.get("position_long_deg") is not None:
            end_lat, end_lon = rec["position_lat_deg"], rec["position_long_deg"]
            break
    bbox = {
        "nec_lat_deg": semicircles_to_deg(s.get("nec_lat")),
        "nec_lon_deg": semicircles_to_deg(s.get("nec_long")),
        "swc_lat_deg": semicircles_to_deg(s.get("swc_lat")),
        "swc_lon_deg": semicircles_to_deg(s.get("swc_long")),
    }
    bbox = drop_nulls(bbox)

    summary = drop_nulls({
        "sport": sport,
        "sub_sport": sub_sport,
        "start_time": start_time,
        "distance_m": float(distance_m) if distance_m is not None else None,
        "distance_km": (float(distance_m)/1000.0) if distance_m is not None else None,
        "timer_s": float(timer_s) if timer_s is not None else None,
        "elapsed_s": float(elapsed_s) if elapsed_s is not None else None,
        "pause_s": pause_s,
        "moving_ratio": moving_ratio,

        "avg_speed_mps": float(avg_speed) if avg_speed is not None else None,
        "max_speed_mps": float(max_speed) if max_speed is not None else None,
        "avg_pace_s_per_km": avg_pace,
        "avg_pace": fmt_pace(avg_pace),
        "best_lap_pace_s_per_km": best_lap_pace,
        "best_lap_pace": fmt_pace(best_lap_pace),

        "avg_heart_rate": float(avg_hr) if avg_hr is not None else None,
        "max_heart_rate": float(max_hr) if max_hr is not None else None,

        "avg_power": float(avg_pwr) if avg_pwr is not None else None,
        "max_power": float(max_pwr) if max_pwr is not None else None,
        "normalized_power": float(np_w) if np_w is not None else None,
        "intensity_factor": ifactor,
        "threshold_power": ftp,
        "training_stress_score": tss,
        "training_effect_aerobic": te_aer,
        "training_effect_anaerobic": te_ana,

        "avg_running_cadence": float(avg_rcad) if avg_rcad is not None else None,
        "avg_running_cadence_total_spm": float(avg_spm) if avg_spm is not None else None,
        "avg_step_length_m": step_len_m,
        "avg_vertical_oscillation_mm": vo_mm,
        "avg_vertical_ratio_pct": vr_pct,
        "avg_stance_time_ms": st_ms,
        "avg_stance_time_percent": st_pct,
        "avg_stance_time_balance_pct": stb_pct,

        "total_ascent_m": float(ascent) if ascent is not None else None,
        "total_descent_m": float(descent) if descent is not None else None,
        "avg_temperature_c": t_avg,
        "max_temperature_c": t_max,

        "start_lat_deg": start_lat,
        "start_lon_deg": start_lon,
        "end_lat_deg": end_lat,
        "end_lon_deg": end_lon,
        "bbox_deg": bbox if bbox else None,

        "lap_count": len(laps_block),
        "record_count": len(records_block) if include_records else 0,
        "pause_count": len(pauses)
    })

    payload = {
        "file": {"name": os.path.basename(fit_path)},
        "metadata": {
            "file_id": data["file_id"],
            "file_creator": data["file_creator"],
            "developer_data_id": data["developer_data_id"],
            "field_descriptions": data["field_description"]
        },
        "laps": laps_block,
        "records": records_block,
        "events": {
            "raw": sanitize_list(data["event"]),
            "pause_count": len(pauses),
            "pauses": pauses
        },
        "messages": {
            "device_info": device_info_clean,
            "device_used": device_used_clean,
            "gps_metadata_count": data["gps_metadata_count"],
            "hr_zone": hr_zone_clean,
            "power_zone": power_zone_clean,
            "time_in_zone": time_in_zone_clean,
            "activity": activity_clean
        },
        "summary": summary
    }

    if keep_other_unknowns:
        # Solo si explícitamente quieres conservar lo demás (ojo, pesado)
        payload["messages"]["other"] = data["other_messages"]

    return payload

def main():
    ap = argparse.ArgumentParser(
        description="Single FIT → JSON enfocado (sin unknowns, con laps, records y summary robusto)."
    )
    ap.add_argument("fit", help="Archivo .fit de entrada")
    ap.add_argument("--out", help="Ruta del JSON de salida (default: <fit>.json). Usa '-' para STDOUT.")
    ap.add_argument("--no-records", action="store_true", help="No incluir la serie temporal 'records'")
    ap.add_argument("--geo", choices=["deg","sc","none"], default="deg",
                    help="Geolocalización por record: 'deg' (grados, default), 'sc' (semicírculos) o 'none'.")
    ap.add_argument("--no-pace", dest="with_pace", action="store_false",
                    help="No añadir 'pace_s_per_km' en records.")
    ap.add_argument("--keep-other-unknowns", action="store_true",
                    help="Conservar 'messages.other' (unknown_*). Por defecto se omite.")
    ap.set_defaults(with_pace=True)

    args = ap.parse_args()

    payload = build_json_for_fit(
        args.fit,
        include_records = (not args.no_records),
        geo = args.geo,
        with_pace = args.with_pace,
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
