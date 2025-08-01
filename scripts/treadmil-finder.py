import os
from fitparse import FitFile

"""
Este script analiza los archivos fit de la carpeta
y si detecta que se hicieron en treadmill
les agrega "-treadmill" antes del .fit del nombre del archivo
"""

def is_treadmill(fitfile: FitFile):
    sub_sport = None
    unknown_110 = None
    gps_found = False

    for msg in fitfile.get_messages():
        if msg.name == "session":
            for field in msg:
                if field.name == "sub_sport":
                    sub_sport = field.value
                elif field.name == "unknown_110":
                    unknown_110 = field.value
        elif msg.name == "record":
            for field in msg:
                if field.name in ("position_lat", "position_long") and field.value is not None:
                    gps_found = True
                    break

    # Detección robusta
    if sub_sport == "treadmill":
        return True
    if unknown_110 and str(unknown_110).lower() == "treadmill":
        return True
    if not gps_found and sub_sport == "running":
        return True  # fallback
    return False

# Procesa todos los archivos .fit en la carpeta actual
for filename in os.listdir():
    if filename.lower().endswith(".fit") and "-treadmill.fit" not in filename.lower():
        try:
            fitfile = FitFile(filename)
            if is_treadmill(fitfile):
                name_part = filename[:-4]  # Remove .fit
                new_name = f"{name_part}-treadmill.fit"
                os.rename(filename, new_name)
                print(f"✅ Renombrado: {filename} → {new_name}")
            else:
                print(f"⏭ No es treadmill: {filename}")
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
