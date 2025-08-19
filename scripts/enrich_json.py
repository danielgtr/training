#!/usr/bin/env python3
"""
Interactive script to enrich a JSON file with manual fields like shoes, category, RPE, etc.
Prompts for fields one by one, allowing empty responses to skip fields.
Saves the enriched JSON file for later use with add_run_to_notion.py.

Usage:
  python3 enrich_json.py <json_file>
"""

import os
import sys
import json
import argparse

def get_user_input(prompt, current_value=None):
    """Get user input with optional current value. Empty input returns None."""
    if current_value:
        full_prompt = f"{prompt} (current: {current_value}): "
    else:
        full_prompt = f"{prompt}: "
    
    response = input(full_prompt).strip()
    return response if response else None

def get_category_choice(current_value=None):
    """Get run category with numbered options."""
    categories = ["easy", "long", "tempo", "intervals", "race", "recovery", "progression", "fartlek"]
    
    print("\nRun Categories:")
    for i, cat in enumerate(categories, 1):
        marker = " (current)" if cat == current_value else ""
        print(f"  {i}. {cat}{marker}")
    
    while True:
        choice = input("Select category (1-8, or press Enter to skip): ").strip()
        if not choice:
            return current_value  # Keep existing value if user skips
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(categories):
                return categories[idx]
            else:
                print("Invalid choice. Please enter 1-8 or press Enter to skip.")
        except ValueError:
            print("Invalid input. Please enter a number 1-8 or press Enter to skip.")

def get_rpe_input(current_value=None):
    """Get RPE input with validation."""
    while True:
        prompt = f"RPE (1-10, current: {current_value}): " if current_value else "RPE (1-10, or press Enter to skip): "
        rpe_input = input(prompt).strip()
        if not rpe_input:
            return current_value  # Keep existing value if user skips
        try:
            rpe_value = float(rpe_input)
            if 1 <= rpe_value <= 10:
                return rpe_value
            else:
                print("RPE should be between 1 and 10. Please try again or press Enter to skip.")
        except ValueError:
            print("Invalid RPE. Please enter a number between 1-10 or press Enter to skip.")

def enrich_json_file(json_file):
    """
    Interactively collect manual fields and add them to the JSON file.
    """
    print(f"Enriching JSON file: {json_file}")
    
    # Load existing JSON
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        sys.exit(1)
    
    # Get current manual fields if they exist
    manual_fields = data.get("manual_fields", {})
    
    print("Fill in the following fields (press Enter to skip/keep current value):\n")
    
    # Collect manual fields
    shoes = get_user_input("Shoes (e.g., Nike Pegasus)", manual_fields.get("shoes"))
    run_category = get_category_choice(manual_fields.get("run_category"))
    rpe = get_rpe_input(manual_fields.get("rpe"))
    inrun_nutrition = get_user_input("In-run nutrition", manual_fields.get("inrun_nutrition"))
    post_notes = get_user_input("Post-run notes", manual_fields.get("post_notes"))
    
    # Update manual fields (only include non-None values)
    updated_fields = {}
    if shoes is not None:
        updated_fields["shoes"] = shoes
    if run_category is not None:
        updated_fields["run_category"] = run_category
    if rpe is not None:
        updated_fields["rpe"] = rpe
    if inrun_nutrition is not None:
        updated_fields["inrun_nutrition"] = inrun_nutrition
    if post_notes is not None:
        updated_fields["post_notes"] = post_notes
    
    # Add manual_fields section to JSON if there are any fields
    if updated_fields:
        data["manual_fields"] = updated_fields
    
    # Show summary
    print("\n" + "="*50)
    print("MANUAL FIELDS TO BE SAVED:")
    for key, value in updated_fields.items():
        print(f"  {key}: {value}")
    if not updated_fields:
        print("  (no manual fields provided)")
    print("="*50)
    
    # Confirm before saving
    confirm = input(f"\nSave enriched JSON to {json_file}? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Save enriched JSON
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Enriched JSON saved to {json_file}")
        print(f"You can now upload to Notion with: python3 Notion/add_run_to_notion.py {json_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Interactively enrich a JSON file with manual fields (shoes, category, RPE, etc.).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 enrich_json.py my_run.json
        """
    )
    
    parser.add_argument("json_file", help="Path to the JSON file to enrich")
    
    args = parser.parse_args()
    
    enrich_json_file(args.json_file)

if __name__ == "__main__":
    main()