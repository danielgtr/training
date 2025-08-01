import os
from fitparse import FitFile
from datetime import datetime

def procesar_fit(filepath):
    try:
        fitfile = FitFile(filepath)

        distancia = None
        is_treadmill = False
        local_timestamp = None

        for msg in fitfile.get_messages():
            if msg.name == "session":
                distancia = msg.get_value("total_distance")
                if msg.get_value("sub_sport") == "treadmill":
                    is_treadmill = True
            elif msg.name == "activity":
                local_timestamp = msg.get_value("local_timestamp")

        if distancia is None or local_timestamp is None:
            print(f"❌ Incompleto: {os.path.basename(filepath)}")
            return

        km = round(distancia / 1000, 1)
        fecha_hora = local_timestamp.strftime("%d-%m-%y_%Hh%M")
        sufijo = "_treadmill" if is_treadmill else ""
        nuevo_nombre = f"{km}k_{fecha_hora}{sufijo}.fit"

        carpeta = os.path.dirname(filepath)
        nuevo_path = os.path.join(carpeta, nuevo_nombre)

        os.rename(filepath, nuevo_path)
        print(f"✅ Renombrado: {os.path.basename(filepath)} → {nuevo_nombre}")

    except Exception as e:
        print(f"⚠️ Error con {filepath}: {e}")

if __name__ == "__main__":
    for archivo in os.listdir("."):
        if archivo.lower().endswith(".fit"):
            procesar_fit(os.path.join(".", archivo))

