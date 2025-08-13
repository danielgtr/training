#!/usr/bin/env python3
# find_resting_calories_v4.py
# Detecta "resting calories" en mensajes 'session' evitando confundirlo con total_calories.
# Requiere:  pip install fitdecode

import argparse, json, sys

try:
    import fitdecode
except Exception:
    sys.stderr.write("Instala fitdecode:  python -m pip install fitdecode\n")
    raise

# ---------- utilidades ----------
def to_text(x):
    if isinstance(x, bytes): return x.decode(errors="ignore")
    return x

def norm(s):
    s = to_text(s)
    if not isinstance(s, str): return ""
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch in ("_", " ")).strip()

def get_field_safe(msg, key):
    try: return msg.get_value(key)
    except Exception: return None

# ---------- parsing ----------
def read_sessions_with_dev_fields(fit_path):
    sessions = []
    field_desc = {}  # (developer_data_index, field_definition_number) -> {"name":..., "units":...}

    with fitdecode.FitReader(fit_path) as fr:
        sidx = -1
        for frame in fr:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue

            if frame.name == "field_description":
                ddi = get_field_safe(frame, "developer_data_index")
                fdn = get_field_safe(frame, "field_definition_number")
                fname = to_text(get_field_safe(frame, "field_name"))
                funits = to_text(get_field_safe(frame, "units"))
                if ddi is not None and fdn is not None:
                    field_desc[(ddi, fdn)] = {"name": fname, "units": funits}
                continue

            if frame.name == "session":
                sidx += 1
                fields = []
                for f in frame.fields:
                    # compat: fitdecode expone is_dev / dev_data_index / def_num
                    is_dev = bool(getattr(f, "is_dev", False) or getattr(f, "is_developer_field", False))
                    name = to_text(getattr(f, "name", ""))
                    units = to_text(getattr(f, "units", ""))

                    if is_dev:
                        ddi = getattr(f, "dev_data_index", getattr(f, "developer_data_index", None))
                        fdn = getattr(f, "def_num", None)
                        meta = field_desc.get((ddi, fdn), {})
                        if name.startswith("unknown") or not name:
                            name = meta.get("name") or name
                        if not units:
                            units = meta.get("units") or units

                    fields.append({
                        "name": name,
                        "value": f.value,
                        "units": units,
                        "developer": is_dev,
                    })
                sessions.append({"index": sidx, "fields": fields})
    return sessions, field_desc

# ---------- matching ----------
EXCLUDE_NAMES = {
    "totalcalories", "total_calories",
    "totalfatcalories", "total_fat_calories",
    "avgcalories", "maxcalories", "calories"
}

def looks_like_resting_field(name, units):
    """Permite std o dev, pero excluye explícitos de total y requiere 'rest' y 'cal' en el nombre."""
    n = norm(name)
    if n in EXCLUDE_NAMES:
        return False
    if ("rest" not in n) or ("cal" not in n):
        return False
    # Unidades preferidas
    if units and str(units).lower().strip() == "kcal":
        return True
    # Si no trae units, igual aceptar; algunos developer fields no las incluyen
    return True

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_path")
    ap.add_argument("--dump-all", dest="dump_all", action="store_true", help="Lista todos los campos de session.")
    ap.add_argument("--show-dev", dest="show_dev", action="store_true", help="Lista solo los developer fields de session.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sessions, field_desc = read_sessions_with_dev_fields(args.fit_path)
    if not sessions:
        print("No hay mensajes 'session' en el FIT."); sys.exit(2)

    found = False
    for sess in sessions:
        # leer total_calories estándar
        total_cals = None
        for fld in sess["fields"]:
            if norm(fld["name"]) in ("totalcalories","total_calories"):
                total_cals = float(fld["value"]); break

        # localizar resting (std o dev) pero evitando confundir total_calories
        candidates = []
        for fld in sess["fields"]:
            if looks_like_resting_field(fld["name"], fld["units"]):
                v = fld["value"]
                if isinstance(v, (int, float)) and v >= 0:
                    # evita el caso donde el valor sea EXACTAMENTE igual a total_calories
                    if total_cals is not None and float(v) == total_cals:
                        continue
                    candidates.append(fld)

        # preferir los que tengan unidades kcal
        best = None
        for c in candidates:
            if (c["units"] or "").lower() == "kcal":
                best = c; break
        if best is None and candidates:
            best = candidates[0]

        if best:
            found = True
            rc = float(best["value"])
            units = (best.get("units") or "").strip()
            print(f"[OK] Session #{sess['index']}: resting calories = {rc} {units}".strip())
            if total_cals is not None:
                print(f"     active (total_calories) = {total_cals} kcal")
                print(f"     gross (active + resting)= {total_cals + rc} kcal")
        else:
            print(f"[--] Session #{sess['index']}: no encontré 'resting calories'")

        if args.show_dev or args.dump_all:
            print("── session fields ──")
            for f in sorted(sess["fields"], key=lambda x: norm(x["name"])):
                tag = "DEV" if f["developer"] else "STD"
                units = f["units"] or ""
                if args.show_dev and not f["developer"]:
                    continue
                print(f" [{tag}] {f['name']}: {f['value']} {units}".rstrip())
            print()

    if args.json:
        print(json.dumps({"fit_path": args.fit_path,
                          "sessions": sessions,
                          "developer_field_descriptions": field_desc},
                         ensure_ascii=False, indent=2))

    sys.exit(0 if found else 1)

if __name__ == "__main__":
    main()
