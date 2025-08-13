#!/usr/bin/env python3
# fit_to_json_focus_plus.py — Extiende tu "fit_to_json_focus.py" SIN romper interfaz
# - Mantiene la misma estructura/CLI y bloques del payload.
# - Solo agrega campos seguros a records, laps y events (pausas) + summary.
#
# Reqs: pip install fitdecode

import os, sys, json, argparse
from datetime import datetime, date, time, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitdecode
except Exception:
    sys.stderr.write("Instala fitdecode:  python -m pip install fitdecode\n")
    raise

# === Aliases confirmados (FR955) a nivel sesión ===
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
    "unknown_188": ("primary_benefit_code", None, 0.9),
    "unknown_211": ("vo2max_encoded_candidate", None, 0.8),
    "unknown_207": ("body_battery_or_stamina_end_pct", "%", 0.6),
    "unknown_169": ("stride_counter_A_candidate", None, 0.5),
    "unknown_170": ("stride_counter_B_candidate", None, 0.5),
}

# ---------- utilidades ----------
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
def set_first_nonnull(d, k, v):
    if k not in d or d[k] is None: d[k] = v

def drop_nulls(d):  # quita claves con None
    return {k: v for k, v in d.items() if v is not None}

def semicircles_to_deg(v):
    return float(v) * (180.0 / (2**31)) if v is not None else None

def mm_to_m(v):
    return float(v) / 1000.0 if v is not None else None

def sanitize_list(dicts, keep_unknown=False):
    """Elimina claves unknown_* y valores None en una lista de dicts (opcionalmente conserva unknown)."""
    out = []
    for d in dicts:
        if keep_unknown:
            clean = drop_nulls(dict(d))
        else:
            clean = {k: v for k, v in d.items() if (v is not None and not (isinstance(k, str) and k.startswith("unknown_")))}
        out.append(clean)
    return out

def as_dt(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            # soporta '2025-08-01T00:42:52+00:00' y '2025-08-01T00:42:52'
            return datetime.fromisoformat(v.replace("Z","+00:00")) if "Z" in v or "+" in v else datetime.fromisoformat(v)
        except Exception:
            return None
    return None

# ---------- lectura FIT (SIN _fields_meta; gps_metadata solo conteo) ----------
def read_all_messages(path):
    out = {
        "file_id": {}, "file_creator": {},
        "developer_data_id": [], "field_description": [],
        "session": [], "lap": [], "record": [], "event": [],
        "device_info": [], "device_used": [],
        "gps_metadata_count": 0,
        "hr_zone": [], "power_zone": [], "time_in_zone": [],
        "activity": [],
        "other_messages": {}  # unknown_* grandes
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
                set_first_nonnull(msg, fname, fval)

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

# ---------- derivados / limpieza ----------
def extract_pauses(events: List[Dict[str, Any]], records: List[Dict[str, Any]]):
    """Parea stop_all → start. Devuelve lista con start, end, duration_s y distancia aproximada al inicio."""
    pauses, last_stop = [], None
    evs = sorted([e for e in events if "timestamp" in e], key=lambda x: as_dt(x["timestamp"]) or datetime.min)

    # índice simple timestamp→distancia (para estimar km al inicio de la pausa)
    rec_pairs = []
    for r in records:
        ts = as_dt(r.get("timestamp"))
        if ts is None: 
            continue
        dist = r.get("distance")
        if isinstance(dist, (int, float)):
            rec_pairs.append((ts, float(dist)))
    rec_pairs.sort(key=lambda x: x[0])

    def distance_at(ts: datetime) -> Optional[float]:
        if not rec_pairs or ts is None:
            return None
        # búsqueda lineal rápida hacia atrás (listas suelen ser densas)
        # si se quisiera, esto se puede cambiar a bisect
        last = None
        for t, d in rec_pairs:
            if t <= ts:
                last = d
            else:
                break
        return last

    for e in evs:
        if e.get("event") == "timer" and e.get("event_type") in ("stop_all", "stop"):
            last_stop = as_dt(e.get("timestamp"))
        elif e.get("event") == "timer" and e.get("event_type") == "start" and last_stop:
            end_ts = as_dt(e.get("timestamp"))
            if end_ts and last_stop and end_ts > last_stop:
                dur = (end_ts - last_stop).total_seconds()
                dist = distance_at(last_stop)
                km = (dist / 1000.0) if dist is not None else None
                typ = "micro" if dur < 15 else ("corta" if dur < 60 else "larga")
                pauses.append({
                    "start": last_stop,
                    "end": end_ts,
                    "duration_s": round(dur, 3),
                    "distance_m_at_start": dist,
                    "km_at_start": (round(km, 3) if km is not None else None),
                    "type": typ
                })
            last_stop = None
    total_pause_s = round(sum(p.get("duration_s", 0.0) for p in pauses), 3)
    return pauses, total_pause_s

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
        "start_time","start_position_lat","start_position_long",
        "nec_lat","nec_long","swc_lat","swc_long","max_temperature","avg_temperature",
        "intensity_factor","threshold_power","training_stress_score",
        "total_training_effect","total_anaerobic_training_effect"
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
        "avg_vertical_oscillation_mm": sess.get("avg_vertical_oscillation"),
        "avg_vertical_ratio_pct": sess.get("avg_vertical_ratio"),
        "avg_stance_time_ms": sess.get("avg_stance_time"),
    }
    active = float(sess.get("total_calories") or 0.0)
    resting = aliased.get("resting_calories_kcal")
    derived["gross_calories_kcal"] = (active + float(resting)) if resting is not None else None

    return {"std": std, "aliases": aliased, "developer_raw_unknowns": raw_unknowns, "derived": derived}

def enrich_lap(lp):
    row = dict(lp)
    if lp.get("avg_running_cadence") is not None:
        row["avg_running_cadence_total_spm"] = float(lp["avg_running_cadence"]) * 2.0
    if lp.get("avg_step_length") is not None:
        row["avg_step_length_m"] = mm_to_m(lp["avg_step_length"])
    # gct por pie a partir de avg_stance_time (ms) y avg_stance_time_balance (% izq)
    ast = lp.get("avg_stance_time")
    asb = lp.get("avg_stance_time_balance")  # percent (izq)
    try:
        if ast is not None and asb is not None:
            left = float(ast) * (float(asb)/100.0)
            right = float(ast) - left
            row["avg_gct_left_ms"] = round(left, 3)
            row["avg_gct_right_ms"] = round(right, 3)
            row["avg_gct_imbalance_ms"] = round(abs(left-right), 3)
    except Exception:
        pass

    # pace medio por lap si hay timer y distancia
    td = lp.get("total_distance")
    tt = lp.get("total_timer_time")
    if isinstance(td, (int,float)) and isinstance(tt, (int,float)) and td > 0:
        pace_s_km = (float(tt) / float(td)) * 1000.0
        row["average_pace_s_per_km"] = round(pace_s_km, 3)
        # mm:ss
        mm = int(pace_s_km//60)
        ss = int(round(pace_s_km - mm*60))
        row["average_pace"] = f"{mm}:{ss:02d}"

    for key_sc, key_deg in [
        ("start_position_lat", "start_lat_deg"),
        ("start_position_long", "start_lon_deg"),
        ("end_position_lat", "end_lat_deg"),
        ("end_position_long", "end_lon_deg"),
    ]:
        row[key_deg] = semicircles_to_deg(lp.get(key_sc))
    return drop_nulls(row)

# -------- records: campos pedidos (con geo en grados por defecto) --------
def _get_first(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return None

def build_record_focus(rec, geo="deg", with_pace=False, in_pause_intervals: List[Tuple[datetime, datetime]] = None):
    """
    Mantiene: timestamp, distance, enhanced_speed/speed, enhanced_altitude/altitude,
              heart_rate, power (+ accumulated_power), cadence_total_spm,
              stance_time_percent, stance_time, vertical_oscillation, vertical_ratio,
              stance_time_balance, step_length (+ step_length_m),
              respiration_rate, wrist_heart_rate, external_heart_rate,
              performance_condition, grade_adjusted_speed (+ gap pace),
              flags: in_pause, km_index; geo (deg|sc|none).
    """
    out = {}

    # tiempo / métricas base
    out["timestamp"] = rec.get("timestamp")
    dist = rec.get("distance")
    if dist is not None:
        out["distance"] = dist

    spd = rec.get("enhanced_speed") if rec.get("enhanced_speed") is not None else rec.get("speed")
    if spd is not None:
        out["enhanced_speed" if rec.get("enhanced_speed") is not None else "speed"] = spd
        if with_pace and float(spd) > 0:
            out["pace_s_per_km"] = 1000.0 / float(spd)

    alt = rec.get("enhanced_altitude") if rec.get("enhanced_altitude") is not None else rec.get("altitude")
    if alt is not None:
        out["enhanced_altitude" if rec.get("enhanced_altitude") is not None else "altitude"] = alt

    # señales
    for k in ("heart_rate","power","accumulated_power","performance_condition","grade_adjusted_speed"):
        if rec.get(k) is not None: out[k] = rec.get(k)

    # respiración y HR alternos (nombres posibles)
    rr = _get_first(rec, ["enhanced_respiration_rate","respiration_rate"])
    if rr is not None: out["respiration_rate"] = rr
    whr = _get_first(rec, ["wrist_heart_rate","wrist_hr"])
    if whr is not None: out["wrist_heart_rate"] = whr
    ehr = _get_first(rec, ["external_heart_rate","ext_heart_rate"])
    if ehr is not None: out["external_heart_rate"] = ehr

    # cadencia total (no guardar cadence crudo ni fractional)
    cad = rec.get("cadence")
    if cad is not None:
        try:
            out["cadence_total_spm"] = round(float(cad) * 2.0, 1)
        except Exception:
            pass

    # running dynamics
    for k in ("stance_time_percent", "stance_time", "vertical_oscillation",
              "vertical_ratio", "stance_time_balance", "step_length"):
        if rec.get(k) is not None:
            out[k] = rec.get(k)

    # derivado: step_length_m
    if rec.get("step_length") is not None:
        try: out["step_length_m"] = mm_to_m(rec.get("step_length"))
        except Exception: pass

    # GCT por pie a partir de stance_time y balance (% izq)
    try:
        st = float(rec.get("stance_time")) if rec.get("stance_time") is not None else None
        sb = float(rec.get("stance_time_balance")) if rec.get("stance_time_balance") is not None else None
        if st is not None and sb is not None:
            left = st * (sb/100.0)
            right = st - left
            out["gct_left_ms"] = round(left, 3)
            out["gct_right_ms"] = round(right, 3)
            out["gct_imbalance_ms"] = round(abs(left-right), 3)
    except Exception:
        pass

    # GAP a partir de grade_adjusted_speed si viene (pace equivalente)
    try:
        gas = out.get("grade_adjusted_speed")
        if with_pace and isinstance(gas, (int,float)) and gas > 0:
            gap_s_km = 1000.0 / float(gas)
            out["gap_s_per_km"] = round(gap_s_km, 3)
            mm = int(gap_s_km//60); ss = int(round(gap_s_km - mm*60))
            out["gap"] = f"{mm}:{ss:02d}"
    except Exception:
        pass

    # km_index entero
    try:
        if isinstance(dist, (int, float)):
            out["km_index"] = int(dist // 1000)
    except Exception:
        pass

    # geolocalización
    if geo == "deg":
        lat_deg = semicircles_to_deg(rec.get("position_lat"))
        lon_deg = semicircles_to_deg(rec.get("position_long"))
        if lat_deg is not None: out["position_lat_deg"] = lat_deg
        if lon_deg is not None: out["position_long_deg"] = lon_deg
    elif geo == "sc":
        if rec.get("position_lat") is not None:  out["position_lat"]  = rec.get("position_lat")
        if rec.get("position_long") is not None: out["position_long"] = rec.get("position_long")

    # flag de pausa
    if in_pause_intervals:
        ts = as_dt(rec.get("timestamp"))
        if ts:
            in_pause = any((ts>=a and ts<=b) for (a,b) in in_pause_intervals)
            if in_pause:
                out["in_pause"] = True

    # limpiar ruidos conocidos
    return drop_nulls(out)

# ---------- pipeline principal ----------
def build_json_for_fit(fit_path, include_records=True, geo="deg", with_pace=False, keep_other_unknowns=False):
    data = read_all_messages(fit_path)

    # Sesión con alias (conservamos developer_raw_unknowns aquí por trazabilidad)
    session_src = data["session"][0] if data["session"] else {}
    session_block = apply_alias_session(session_src)

    # Laps enriquecidas (+ pace + gct por pie)
    laps_src = data["lap"]
    laps_block = [enrich_lap(lp) for lp in laps_src]

    # Pausas (necesitan events + records originales para distancia)
    pauses_list, total_pause_time_s = extract_pauses(data["event"], data["record"])
    pause_intervals = [(as_dt(p["start"]), as_dt(p["end"])) for p in pauses_list if as_dt(p["start"]) and as_dt(p["end"])]

    # Records completos con campos seleccionados y geo
    records_block = [build_record_focus(r, geo=geo, with_pace=with_pace, in_pause_intervals=pause_intervals) for r in data["record"]] if include_records else []

    # acumulados por lap (cum_time_s / cum_dist_m)
    cum_t, cum_d = 0.0, 0.0
    for i, lp in enumerate(laps_block):
        t = float(lp.get("total_timer_time") or 0.0)
        d = float(lp.get("total_distance") or 0.0)
        cum_t += t; cum_d += d
        lp["cum_time_s"] = round(cum_t, 3)
        lp["cum_dist_m"] = round(cum_d, 3)

    # Mensajes limpios (sin unknown_* y sin nulls). other_messages se omite salvo que pidan conservarlo.
    activity_clean     = sanitize_list(data["activity"], keep_unknown=False)
    device_info_clean  = sanitize_list(data["device_info"], keep_unknown=False)
    device_used_clean  = sanitize_list(data["device_used"], keep_unknown=False)
    hr_zone_clean      = sanitize_list(data["hr_zone"], keep_unknown=False)
    power_zone_clean   = sanitize_list(data["power_zone"], keep_unknown=False)
    time_in_zone_clean = sanitize_list(data["time_in_zone"], keep_unknown=False)

    payload = {
        "file": {"name": os.path.basename(fit_path)},
        "metadata": {
            "file_id": data["file_id"],
            "file_creator": data["file_creator"],
            "developer_data_id": data["developer_data_id"],
            "field_descriptions": data["field_description"]
        },
        "session": session_block,
        "laps": laps_block,
        "records": records_block,
        "events": {
            "raw": sanitize_list(data["event"], keep_unknown=False),
            "pause_count": len(pauses_list),
            "total_pause_time_s": total_pause_time_s,
            "pauses": pauses_list
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
        "provenance": {
            "alias_map": "FR955_v1_confirmed",
            "parser": "fitdecode",
            "notes": []
        }
    }

    if keep_other_unknowns:
        # Solo si realmente quieres conservar los bloques unknown_* grandes
        payload["messages"]["other"] = data["other_messages"]

    # Resumen útil
    s_std, s_der, s_alias = session_block.get("std", {}), session_block.get("derived", {}), session_block.get("aliases", {})
    distance_m = s_std.get("total_distance")
    timer_s = s_std.get("total_timer_time")
    elapsed_s = s_std.get("total_elapsed_time")
    pace = (timer_s / distance_m * 1000.0) if (timer_s and distance_m and distance_m > 0) else None

    # TSS (si hay IF o si hay NP+FTP)
    tss = None
    IF = s_std.get("intensity_factor")
    NP = s_std.get("normalized_power")
    FTP = s_std.get("threshold_power")
    try:
        if IF is None and NP is not None and FTP is not None and FTP > 0:
            IF = float(NP)/float(FTP)
        if IF is not None and timer_s is not None:
            tss = (float(timer_s)/3600.0) * (float(IF)**2) * 100.0
    except Exception:
        tss = None

    payload["summary"] = {
        "start_time": s_std.get("start_time"),
        "sport": s_std.get("sport"),
        "distance_m": distance_m,
        "timer_s": timer_s,
        "elapsed_s": elapsed_s,
        "total_pause_time_s": total_pause_time_s if total_pause_time_s else ((elapsed_s - timer_s) if (elapsed_s and timer_s and elapsed_s>=timer_s) else None),
        "avg_pace_s_per_km": pace,
        "avg_hr_bpm": s_std.get("avg_heart_rate"),
        "max_hr_bpm": s_std.get("max_heart_rate"),
        "avg_power_w": s_std.get("avg_power"),
        "np_w": NP,
        "intensity_factor": IF,
        "tss_est": (round(tss,2) if tss is not None else None),
        "total_calories_active_kcal": s_std.get("total_calories"),
        "resting_calories_kcal": s_alias.get("resting_calories_kcal"),
        "gross_calories_kcal": s_der.get("gross_calories_kcal"),
        "sweat_loss_ml": s_alias.get("est_sweat_loss_ml"),
        "workout_rpe": s_alias.get("workout_rpe"),
        "workout_feel": s_alias.get("workout_feel_0_100"),
        "stamina_start_pct": s_alias.get("stamina_potential_start_pct"),
        "stamina_end_pct": s_alias.get("stamina_potential_end_pct"),
        "benefit_code": s_alias.get("primary_benefit_code"),
        "vo2max_code": s_alias.get("vo2max_encoded_candidate"),
        "pauses": len(pauses_list),
        "laps": len(laps_block),
        "records": len(records_block) if include_records else 0
    }
    return payload

def main():
    ap = argparse.ArgumentParser(description="Single FIT → JSON enfocado (con geo y running dynamics; sin unknown_* masivos).")
    ap.add_argument("fit", help="Archivo .fit de entrada")
    ap.add_argument("--out", help="Ruta del JSON de salida (default: <fit>.json). Usa '-' para STDOUT.")
    ap.add_argument("--no-records", action="store_true", help="No incluir la serie temporal 'record'")
    ap.add_argument("--geo", choices=["deg","sc","none"], default="deg",
                    help="Geolocalización por record: 'deg' (grados, default), 'sc' (semicírculos) o 'none'.")
    ap.add_argument("--with-pace", action="store_true",
                    help="Añadir 'pace_s_per_km' además de speed (y GAP si hay grade_adjusted_speed).")
    ap.add_argument("--keep-other-unknowns", action="store_true",
                    help="Conservar el bloque 'messages.other' (unknown_*). Por defecto se omite.")
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
