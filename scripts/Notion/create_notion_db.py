#!/usr/bin/env python3
"""
Create a Notion database for your running log.

Requirements:
  pip install notion-client python-dotenv

Environment (.env file):
  NOTION_TOKEN          -> your internal integration secret
  NOTION_PARENT_PAGE    -> target parent page ID (give the integration access!)

Usage:
  python3 create_notion_db.py
"""

import os

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
PARENT_PAGE_ID = os.environ["NOTION_PARENT_PAGE"]

notion = Client(auth=NOTION_TOKEN)

DB_TITLE = "Running Log"

# --------- Database schema (properties) ----------
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
    "Pace Avg (min/km)": {"number": {"format": "number"}},
    "Pace Avg (pretty)": {"rich_text": {}},
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
    "RPE": {"rich_text": {}},
    "In-run Nutrition": {"rich_text": {}},
    "Post-run Notes": {"rich_text": {}},

    # Devices used
    "Devices": {"multi_select": {}},
}

def save_db_id_to_env(db_id):
    """Save the database ID to .env file"""
    env_file = ".env"
    
    # Read existing .env content
    env_lines = []
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            env_lines = f.readlines()
    
    # Remove existing NOTION_DATABASE_ID line if present
    env_lines = [line for line in env_lines if not line.startswith("NOTION_DATABASE_ID=")]
    
    # Add the new database ID
    env_lines.append(f"NOTION_DATABASE_ID={db_id}\n")
    
    # Write back to .env
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(env_lines)

def create_database():
    """Create the Notion database and save its ID to .env"""
    try:
        print("[INFO] Creating Notion database...")
        
        db = notion.databases.create(
            parent={"type": "page_id", "page_id": PARENT_PAGE_ID},
            title=[{"type": "text", "text": {"content": DB_TITLE}}],
            properties=PROPERTIES,
        )
        
        db_id = db["id"]
        print(f"[SUCCESS] Created database '{DB_TITLE}' with ID: {db_id}")
        
        # Save database ID to .env file
        save_db_id_to_env(db_id)
        print(f"[SUCCESS] Database ID saved to .env file")
        
        print(f"\n✅ Setup complete!")
        print(f"   You can now add runs using: python3 add_run_to_notion.py <json_file>")
        
        return db_id
        
    except Exception as e:
        print(f"[ERROR] Failed to create database: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check that NOTION_TOKEN is valid")
        print(f"  2. Check that NOTION_PARENT_PAGE is a valid page ID (not database ID)")
        print(f"  3. Make sure your integration has access to the parent page")
        raise

if __name__ == "__main__":
    create_database()
