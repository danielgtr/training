import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from scipy.ndimage import uniform_filter1d
import seaborn as sns

class RunningPaceAnalyzer:
    def __init__(self, json_file_path):
        """Initialize with path to JSON file from FIT data"""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.df = self._process_data()
        
    def _process_data(self):
        """Convert JSON records to pandas DataFrame"""
        records = self.data.get('records', [])
        
        # Extract data points
        data_points = []
        for record in records:
            if isinstance(record, dict):
                data_points.append(record)
        
        df = pd.DataFrame(data_points)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate elapsed time in seconds from start
            df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            df['elapsed_minutes'] = df['elapsed_seconds'] / 60
        
        # Convert pace from min/km string to seconds per km
        if 'pace_pretty' in df.columns:
            df['pace_sec_per_km'] = df['pace_pretty'].apply(self._parse_pace)
        
        # Convert distance from meters to km
        if 'distance_m' in df.columns:
            df['distance_km'] = df['distance_m'] / 1000
            
        return df
    
    def _parse_pace(self, pace_str):
        """Convert pace string like '5:31/km' to seconds per km"""
        if pd.isna(pace_str) or not isinstance(pace_str, str):
            return np.nan
        
        try:
            # Remove '/km' and split by ':'
            pace_clean = pace_str.replace('/km', '').replace('"', '')
            parts = pace_clean.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except:
            return np.nan
        
        return np.nan
    
    def _sec_to_pace_str(self, seconds):
        """Convert seconds per km back to MM:SS format"""
        if pd.isna(seconds):
            return "N/A"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def create_pace_graph(self, 
                         x_axis='distance_km',  # or 'elapsed_minutes' 
                         smooth_window=30,      # smoothing window
                         figsize=(12, 6),
                         title=None,
                         y_limit=None,          # tuple (min_pace_sec, max_pace_sec)
                         x_limit=None):         # tuple (min_x, max_x)
        """
        Create a pace graph with customizable axes
        
        Parameters:
        - x_axis: 'distance_km' or 'elapsed_minutes'
        - smooth_window: number of points for moving average (0 for no smoothing)
        - figsize: figure size tuple
        - title: custom title
        - y_limit: tuple of (min_pace, max_pace) in seconds per km
        - x_limit: tuple of (min_x, max_x) for x-axis
        """
        
        # Prepare data
        df_clean = self.df.dropna(subset=['pace_sec_per_km', x_axis])
        
        if len(df_clean) == 0:
            print("No valid pace data found")
            return None
            
        x_data = df_clean[x_axis]
        y_data = df_clean['pace_sec_per_km']
        
        # Apply smoothing if requested
        if smooth_window > 0:
            y_smoothed = uniform_filter1d(y_data.values, size=smooth_window)
        else:
            y_smoothed = y_data.values
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot raw data (light)
        plt.plot(x_data, y_data, alpha=0.3, color='lightblue', linewidth=0.5, label='Raw pace')
        
        # Plot smoothed data (main line)
        if smooth_window > 0:
            plt.plot(x_data, y_smoothed, color='darkblue', linewidth=2, label=f'Smoothed (window={smooth_window})')
        
        # Formatting
        xlabel = 'Distance (km)' if x_axis == 'distance_km' else 'Time (minutes)'
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Pace (min/km)', fontsize=12)
        
        # Set title
        if title is None:
            session_info = self.data.get('session', {})
            date_str = session_info.get('std', 'Unknown date')
            title = f'Running Pace Analysis - {date_str}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Format y-axis to show pace in MM:SS
        y_ticks = plt.yticks()[0]
        plt.yticks(y_ticks, [self._sec_to_pace_str(sec) for sec in y_ticks])
        
        # Set axis limits if provided
        if y_limit:
            plt.ylim(y_limit)
        if x_limit:
            plt.xlim(x_limit)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add summary statistics
        avg_pace = np.nanmean(y_data)
        stats_text = f'Avg Pace: {self._sec_to_pace_str(avg_pace)}'
        
        if x_axis == 'distance_km':
            total_distance = x_data.max()
            stats_text += f'\nTotal Distance: {total_distance:.2f} km'
        else:
            total_time = x_data.max()
            stats_text += f'\nTotal Time: {total_time:.1f} min'
            
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return plt.gcf()
    
    def create_elevation_pace_comparison(self, smooth_window=30, figsize=(12, 8)):
        """Create a comparison plot of pace vs elevation"""
        if 'altitude_m' not in self.df.columns:
            print("No elevation data available")
            return None
            
        df_clean = self.df.dropna(subset=['pace_sec_per_km', 'altitude_m', 'distance_km'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Pace plot
        x_data = df_clean['distance_km']
        pace_data = df_clean['pace_sec_per_km']
        
        if smooth_window > 0:
            pace_smoothed = uniform_filter1d(pace_data.values, size=smooth_window)
            ax1.plot(x_data, pace_smoothed, color='darkblue', linewidth=2)
        else:
            ax1.plot(x_data, pace_data, color='darkblue', linewidth=2)
            
        ax1.set_ylabel('Pace (min/km)')
        ax1.set_title('Pace vs Distance')
        ax1.grid(True, alpha=0.3)
        
        # Format pace y-axis
        y_ticks = ax1.get_yticks()
        ax1.set_yticklabels([self._sec_to_pace_str(sec) for sec in y_ticks])
        
        # Elevation plot
        elevation_data = df_clean['altitude_m']
        ax2.fill_between(x_data, elevation_data, alpha=0.6, color='green')
        ax2.plot(x_data, elevation_data, color='darkgreen', linewidth=1)
        ax2.set_ylabel('Elevation (m)')
        ax2.set_xlabel('Distance (km)')
        ax2.set_title('Elevation Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics of the run"""
        df_clean = self.df.dropna(subset=['pace_sec_per_km'])
        
        stats = {
            'total_distance_km': self.df['distance_km'].max() if 'distance_km' in self.df.columns else None,
            'total_time_minutes': self.df['elapsed_minutes'].max() if 'elapsed_minutes' in self.df.columns else None,
            'average_pace': self._sec_to_pace_str(np.nanmean(df_clean['pace_sec_per_km'])),
            'best_pace': self._sec_to_pace_str(np.nanmin(df_clean['pace_sec_per_km'])),
            'worst_pace': self._sec_to_pace_str(np.nanmax(df_clean['pace_sec_per_km'])),
            'elevation_gain': None,
            'avg_heart_rate': np.nanmean(self.df['heart_rate_bpm']) if 'heart_rate_bpm' in self.df.columns else None
        }
        
        # Calculate elevation gain if available
        if 'altitude_m' in self.df.columns:
            elevation_diff = np.diff(self.df['altitude_m'].dropna())
            stats['elevation_gain'] = np.sum(elevation_diff[elevation_diff > 0])
        
        return stats

def analyze_run(json_file_path, 
                x_axis='distance_km', 
                smooth_window=30, 
                show_elevation=True,
                save_plots=False,
                output_dir='.'):
    """
    Main function to analyze a running JSON file
    
    Parameters:
    - json_file_path: path to your JSON file
    - x_axis: 'distance_km' or 'elapsed_minutes'
    - smooth_window: smoothing factor (higher = smoother)
    - show_elevation: whether to create elevation comparison
    - save_plots: save plots instead of showing them
    - output_dir: directory to save plots
    """
    import os
    
    try:
        # Initialize analyzer
        analyzer = RunningPaceAnalyzer(json_file_path)
        
        # Get base filename for titles and saving
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        
        print(f"Analyzing run: {base_name}")
        
        # Create basic pace graph
        fig1 = analyzer.create_pace_graph(
            x_axis=x_axis,
            smooth_window=smooth_window,
            figsize=(14, 7),
            title=f"Pace Analysis - {base_name}"
        )
        
        if save_plots:
            fig1.savefig(os.path.join(output_dir, f"{base_name}_pace_graph.png"), dpi=300, bbox_inches='tight')
            print(f"Saved: {base_name}_pace_graph.png")
        else:
            plt.show()
        
        # Create elevation comparison if requested and data exists
        if show_elevation:
            fig2 = analyzer.create_elevation_pace_comparison(smooth_window=smooth_window)
            if fig2:
                if save_plots:
                    fig2.savefig(os.path.join(output_dir, f"{base_name}_elevation_pace.png"), dpi=300, bbox_inches='tight')
                    print(f"Saved: {base_name}_elevation_pace.png")
                else:
                    plt.show()
            else:
                print("No elevation data available for comparison")
        
        # Print summary statistics
        stats = analyzer.get_summary_stats()
        print(f"\n{'='*50}")
        print(f"RUN SUMMARY - {base_name}")
        print(f"{'='*50}")
        for key, value in stats.items():
            if value is not None:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        return analyzer
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file_path}'")
        print("Make sure the file path is correct!")
        return None
    except Exception as e:
        print(f"Error analyzing run: {e}")
        return None

# Example usage and command line interface
if __name__ == "__main__":
    import sys
    
    # Command line usage
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        
        # Optional arguments
        x_axis = sys.argv[2] if len(sys.argv) > 2 else 'distance_km'
        smooth_window = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        
        print(f"Command line analysis of: {json_file}")
        analyze_run(json_file, x_axis=x_axis, smooth_window=smooth_window)
    
    else:
        # Interactive usage example
        print("Usage examples:")
        print("1. Command line: python pace_analyzer.py your_file.json")
        print("2. Command line with options: python pace_analyzer.py your_file.json elapsed_minutes 50")
        print("3. In Python script:")
        print("   from pace_analyzer import analyze_run")
        print("   analyze_run('your_file.json')")
        print("\nAvailable functions:")
        print("- analyze_run(file_path) - Complete analysis")
        print("- RunningPaceAnalyzer(file_path) - For custom analysis")
