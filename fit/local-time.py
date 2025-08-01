from fitparse import FitFile
import sys
from datetime import datetime

def extraer_hora_local(path):
    fitfile = FitFile(path)

    for msg in fitfile.get_messages("activity"):
        local_timestamp = msg.get_value("local_timestamp")
        timestamp = msg.get_value("timestamp")

        print("🕒 Timestamp UTC:        ", timestamp)
        print("🕓 Local Timestamp:      ", local_timestamp)

        if local_timestamp and timestamp:
            offset = (local_timestamp - timestamp).total_seconds() / 3600
            print(f"🧭 Estimado de zona horaria: UTC{offset:+.0f}")
        return

    print("⚠️ No se encontró mensaje 'activity' con local_timestamp.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 extraer_hora_local.py archivo.fit")
    else:
        extraer_hora_local(sys.argv[1])
