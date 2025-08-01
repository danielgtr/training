#!/usr/bin/env python3
"""
Batch FIT Report Generator
Processes all FIT files and generates comprehensive reports organized by session
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import re

def extract_session_info(filename):
    """Extract session information from FIT filename for folder naming"""
    # Remove .fit extension
    base_name = filename.replace('.fit', '')
    
    # Extract components: distance, date, time, type
    parts = base_name.split('_')
    
    if len(parts) >= 3:
        distance = parts[0]  # e.g., "10.0k", "2.5k"
        date = parts[1]      # e.g., "18-07-25"
        time = parts[2]      # e.g., "21h49"
        
        # Check if it's treadmill
        is_treadmill = 'treadmill' in base_name.lower()
        activity_type = 'treadmill' if is_treadmill else 'outdoor'
        
        # Create descriptive folder name
        folder_name = f"{date}_{time}_{distance}_{activity_type}"
        
        return folder_name, {
            'distance': distance,
            'date': date,
            'time': time,
            'type': activity_type,
            'original_filename': filename
        }
    else:
        # Fallback for unexpected filename format
        folder_name = base_name
        return folder_name, {
            'original_filename': filename,
            'type': 'unknown'
        }

def process_fit_file(fit_file_path, output_base_dir, script_path):
    """Process a single FIT file and organize its outputs"""
    fit_filename = os.path.basename(fit_file_path)
    print(f"\n🔄 Processing: {fit_filename}")
    
    # Extract session info for folder naming
    folder_name, session_info = extract_session_info(fit_filename)
    
    # Create session-specific output directory
    session_output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(session_output_dir, exist_ok=True)
    
    # Create temporary output directory for this session
    temp_output_dir = os.path.join(session_output_dir, "temp_output")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        # Run the FIT report script with custom output directory
        cmd = [
            sys.executable, 
            script_path, 
            fit_file_path
        ]
        
        # Change to temp directory so output goes there
        original_cwd = os.getcwd()
        os.chdir(temp_output_dir)
        
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"✅ Successfully processed {fit_filename}")
            
            # Move generated files from temp_output/output to session directory
            temp_files_dir = os.path.join(temp_output_dir, "output")
            if os.path.exists(temp_files_dir):
                for file in os.listdir(temp_files_dir):
                    src = os.path.join(temp_files_dir, file)
                    dst = os.path.join(session_output_dir, file)
                    shutil.move(src, dst)
                    print(f"   📄 Moved: {file}")
            
            # Create session info file
            info_file = os.path.join(session_output_dir, "session_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"Session Information\n")
                f.write(f"==================\n")
                f.write(f"Original File: {session_info['original_filename']}\n")
                f.write(f"Distance: {session_info.get('distance', 'N/A')}\n")
                f.write(f"Date: {session_info.get('date', 'N/A')}\n")
                f.write(f"Time: {session_info.get('time', 'N/A')}\n")
                f.write(f"Activity Type: {session_info.get('type', 'N/A')}\n")
                f.write(f"Report Generated: {folder_name}\n")
            
            # Clean up temp directory
            shutil.rmtree(temp_output_dir)
            
        else:
            print(f"❌ Error processing {fit_filename}")
            print(f"   Error: {result.stderr}")
            # Clean up temp directory even on error
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout processing {fit_filename}")
        os.chdir(original_cwd)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        return False
    except Exception as e:
        print(f"💥 Exception processing {fit_filename}: {e}")
        os.chdir(original_cwd)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        return False
    
    return True

def main():
    # Paths
    base_dir = "/training"
    fit_dir = os.path.join(base_dir, "fit")
    reports_dir = os.path.join(base_dir, "reports")
    script_path = os.path.join(base_dir, "scripts", "fit-report.py")
    
    # Verify paths exist
    if not os.path.exists(fit_dir):
        print(f"❌ FIT directory not found: {fit_dir}")
        sys.exit(1)
    
    if not os.path.exists(script_path):
        print(f"❌ Report script not found: {script_path}")
        sys.exit(1)
    
    # Create reports directory
    os.makedirs(reports_dir, exist_ok=True)
    
    # Get all FIT files
    fit_files = [f for f in os.listdir(fit_dir) if f.endswith('.fit')]
    fit_files.sort()  # Process in order
    
    if not fit_files:
        print(f"❌ No FIT files found in {fit_dir}")
        sys.exit(1)
    
    print(f"🎯 Found {len(fit_files)} FIT files to process")
    print(f"📁 Output directory: {reports_dir}")
    print(f"🔧 Using script: {script_path}")
    
    # Process each FIT file
    successful = 0
    failed = 0
    
    for fit_file in fit_files:
        fit_file_path = os.path.join(fit_dir, fit_file)
        
        if process_fit_file(fit_file_path, reports_dir, script_path):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📊 BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Reports saved in: {reports_dir}")
    
    if failed > 0:
        print(f"\n⚠️  Some files failed to process. Check the error messages above.")
        sys.exit(1)
    else:
        print(f"\n🎉 All files processed successfully!")

if __name__ == "__main__":
    main()