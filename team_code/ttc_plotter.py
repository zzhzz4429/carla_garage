import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json

class TTCPlotter:
    def __init__(self, save_dir="/home/ascc304/carla_garage/ttc_plots"):
        self.save_dir = save_dir
        self.ttc_data = []
        self.time_data = []
        self.risk_data = []
        self.object_ids = []
        self.distances = []
        self.gap_rates = []
        
        # Enhanced data for simplified risk approach
        self.object_types = []  # Track object class names
        self.emergency_flags = []  # Track emergency vehicle encounters
        self.unified_risks = []  # Track combined external risk (if available)
        
        # NEW: Signal compliance data
        self.signal_distances = []  # Distance to red lights/stop signs
        self.signal_types = []  # Type of signal (red light, stop sign)
        self.signal_risks = []  # Signal compliance risk values
        self.current_decelerations = []  # Current required deceleration
        self.ego_speeds = []  # Ego vehicle speed for deceleration calculation
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Generate unique session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_data_point(self, ttc_risk, ttc_object, frame_time, unified_risk=None, signal_data=None, ego_speed=0.0):
        """
        Add a data point for plotting (enhanced for simplified approach + signal compliance)
        
        Args:
            ttc_risk: TTC risk value
            ttc_object: Object information from TTC calculation
            frame_time: Current frame time
            unified_risk: Optional unified external risk value
            signal_data: Dictionary with signal compliance information
            ego_speed: Current ego vehicle speed (m/s)
        """
        self.ttc_data.append(ttc_risk)
        self.time_data.append(frame_time)
        self.risk_data.append(ttc_risk)
        self.ego_speeds.append(ego_speed)
        
        # Add unified risk if provided
        if unified_risk is not None:
            self.unified_risks.append(unified_risk)
        else:
            self.unified_risks.append(ttc_risk)  # Fallback to TTC risk
        
        # Handle TTC object data
        if ttc_object:
            self.object_ids.append(ttc_object.get('object_id', 'unknown'))
            
            # Handle distance: convert 0.0 to NaN for no object detected
            distance = ttc_object.get('distance', 0.0)
            if distance == 0.0:
                distance = np.nan
            self.distances.append(distance)
            
            # Handle gap_rate: convert 0.0 to NaN if distance is also 0.0/NaN
            gap_rate = ttc_object.get('gap_rate', 0.0)
            if np.isnan(distance) or gap_rate == 0.0:
                gap_rate = np.nan
            self.gap_rates.append(gap_rate)
            
            # Enhanced tracking for simplified approach
            self.object_types.append(ttc_object.get('class_name', 'unknown'))
            self.emergency_flags.append(ttc_object.get('is_emergency', False))
        else:
            # No object detected - use NaN instead of 0.0
            self.object_ids.append('none')
            self.distances.append(np.nan)
            self.gap_rates.append(np.nan)
            self.object_types.append('none')
            self.emergency_flags.append(False)
            
        # Handle signal compliance data
        if signal_data:
            self.signal_distances.append(signal_data.get('distance', np.nan))
            self.signal_types.append(signal_data.get('signal_type', 'none'))
            self.signal_risks.append(signal_data.get('signal_risk', 0.0))
            self.current_decelerations.append(signal_data.get('required_deceleration', 0.0))
        else:
            # No signal detected
            self.signal_distances.append(np.nan)
            self.signal_types.append('none')
            self.signal_risks.append(0.0)
            self.current_decelerations.append(0.0)
            
    def plot_ttc_timeline(self):
        """
        Create and save TTC timeline plot with signal compliance
        """
        if len(self.ttc_data) < 2:
            return  # Don't print anything if not enough data
            
        # Create enhanced subplots for risk-focused analysis (2x3 grid for signal data)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'External Risk Analysis (TTC + Signal Compliance) - Session {self.session_id}', fontsize=16)
        
        # Convert to numpy arrays for better handling
        time_array = np.array(self.time_data)
        ttc_array = np.array(self.ttc_data)
        distance_array = np.array(self.distances)
        gap_rate_array = np.array(self.gap_rates)
        unified_array = np.array(self.unified_risks)
        emergency_array = np.array(self.emergency_flags)
        
        # Signal compliance arrays
        signal_distance_array = np.array(self.signal_distances)
        signal_risk_array = np.array(self.signal_risks)
        deceleration_array = np.array(self.current_decelerations)
        ego_speed_array = np.array(self.ego_speeds)
        
        # Apply simple moving average smoothing to reduce zig-zag (window size of 3)
        def smooth_data(data, window_size=3):
            if len(data) < window_size:
                return data
            # Handle NaN values in smoothing
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data  # All NaN, return as is
            smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            # Pad the beginning to maintain array length
            padding = np.full(window_size-1, data[0])
            return np.concatenate([padding, smoothed])
        
        # Smooth the data (only non-NaN values)
        ttc_smoothed = smooth_data(ttc_array, window_size=3)
        signal_risk_smoothed = smooth_data(signal_risk_array, window_size=3)
        
        # For distance and gap rate, only smooth where we have valid data
        distance_smoothed = distance_array.copy()  # Keep NaN as NaN
        gap_rate_smoothed = gap_rate_array.copy()   # Keep NaN as NaN
        signal_distance_smoothed = signal_distance_array.copy()  # Keep NaN as NaN
        
        # Apply smoothing only to valid segments
        valid_distance_mask = ~np.isnan(distance_array)
        valid_gap_mask = ~np.isnan(gap_rate_array)
        valid_signal_mask = ~np.isnan(signal_distance_array)
        
        if np.any(valid_distance_mask):
            distance_smoothed[valid_distance_mask] = smooth_data(distance_array[valid_distance_mask], window_size=3)
        if np.any(valid_gap_mask):
            gap_rate_smoothed[valid_gap_mask] = smooth_data(gap_rate_array[valid_gap_mask], window_size=3)
        if np.any(valid_signal_mask):
            signal_distance_smoothed[valid_signal_mask] = smooth_data(signal_distance_array[valid_signal_mask], window_size=3)
        
        # Plot 1: Enhanced Risk Analysis (TTC + Unified + Signal)
        unified_smoothed = smooth_data(unified_array, window_size=3)
        
        # Plot TTC, unified, and signal risks
        ax1.plot(time_array, ttc_smoothed, 'b-', linewidth=2, label='TTC Risk', alpha=0.8)
        ax1.plot(time_array, unified_smoothed, 'purple', linewidth=2, label='Unified Risk', alpha=0.8)
        ax1.plot(time_array, signal_risk_smoothed, 'orange', linewidth=2, label='Signal Risk', alpha=0.8)
        
        # Highlight emergency vehicle encounters
        emergency_times = time_array[emergency_array]
        emergency_risks = unified_smoothed[emergency_array]
        if len(emergency_times) > 0:
            ax1.scatter(emergency_times, emergency_risks, color='red', s=50, marker='s', 
                       label='Emergency Vehicle', alpha=0.8, zorder=5)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Risk Level')
        ax1.set_title('External Risk Analysis (TTC + Signal Compliance)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add risk level thresholds
        ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Critical')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning')
        ax1.axhline(y=0.2, color='yellow', linestyle='--', alpha=0.7, label='Caution')
        
        # Plot 2: Distance to closest object (skip NaN values)
        valid_distance_indices = ~np.isnan(distance_array)
        if np.any(valid_distance_indices):
            ax2.plot(time_array[valid_distance_indices], distance_smoothed[valid_distance_indices], 
                    'g-', linewidth=2, label='Distance (smoothed)', alpha=0.8)
            ax2.plot(time_array[valid_distance_indices], distance_array[valid_distance_indices], 
                    'g-', linewidth=1, label='Distance (raw)', alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title('Distance to Closest Object')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add distance thresholds
        ax2.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Critical (5m)')
        ax2.axhline(y=15.0, color='orange', linestyle='--', alpha=0.7, label='Warning (15m)')
        ax2.axhline(y=30.0, color='yellow', linestyle='--', alpha=0.7, label='Safe (30m)')
        
        # Plot 3: Gap rate (closing speed) (skip NaN values)
        valid_gap_indices = ~np.isnan(gap_rate_array)
        if np.any(valid_gap_indices):
            ax3.plot(time_array[valid_gap_indices], gap_rate_smoothed[valid_gap_indices], 
                    'r-', linewidth=2, label='Gap Rate (smoothed)', alpha=0.8)
            ax3.plot(time_array[valid_gap_indices], gap_rate_array[valid_gap_indices], 
                    'r-', linewidth=1, label='Gap Rate (raw)', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Gap Rate (m/s)')
        ax3.set_title('Gap Rate (Closing Speed)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add zero line for reference
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Risk distribution histogram
        if len(self.ttc_data) > 10:
            ax4.hist(self.ttc_data, bins=20, alpha=0.7, color='blue', edgecolor='black', label='TTC Risk')
            if len([r for r in self.signal_risks if r > 0]) > 5:  # Only if we have signal data
                ax4.hist(self.signal_risks, bins=20, alpha=0.5, color='orange', edgecolor='black', label='Signal Risk')
            ax4.set_xlabel('Risk Level')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Risk Distribution')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Not enough data\nfor histogram', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Risk Distribution')
        
        # Plot 5: Signal Distance (NEW)
        valid_signal_indices = ~np.isnan(signal_distance_array)
        if np.any(valid_signal_indices):
            ax5.plot(time_array[valid_signal_indices], signal_distance_smoothed[valid_signal_indices], 
                    'orange', linewidth=2, label='Signal Distance (smoothed)', alpha=0.8)
            ax5.plot(time_array[valid_signal_indices], signal_distance_array[valid_signal_indices], 
                    'orange', linewidth=1, label='Signal Distance (raw)', alpha=0.3)
            
            # Add markers for different signal types
            for i, (t, d, signal_type) in enumerate(zip(time_array, signal_distance_array, self.signal_types)):
                if not np.isnan(d) and signal_type != 'none':
                    marker = 'o' if signal_type == 'Red Light' else 's'
                    color = 'red' if signal_type == 'Red Light' else 'brown'
                    ax5.scatter(t, d, marker=marker, color=color, s=30, alpha=0.7, zorder=5)
            
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Distance (m)')
        ax5.set_title('Distance to Traffic Signals')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Add stopping distance thresholds
        ax5.axhline(y=20.0, color='orange', linestyle='--', alpha=0.7, label='Warning (20m)')
        ax5.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Critical (10m)')
        
        # Plot 6: Required vs Safe Deceleration (NEW)
        safe_deceleration = 7.0  # m/s² - safe deceleration threshold
        
        # Only plot when we have deceleration data
        valid_decel_indices = deceleration_array > 0.1  # Only meaningful deceleration values
        if np.any(valid_decel_indices):
            ax6.plot(time_array[valid_decel_indices], deceleration_array[valid_decel_indices], 
                    'red', linewidth=2, label='Required Deceleration', alpha=0.8)
            
            # Add safe deceleration threshold
            ax6.axhline(y=safe_deceleration, color='green', linestyle='-', alpha=0.7, 
                       linewidth=2, label=f'Safe Deceleration ({safe_deceleration} m/s²)')
            
            # Fill area where deceleration exceeds safe threshold
            unsafe_mask = deceleration_array > safe_deceleration
            if np.any(unsafe_mask):
                ax6.fill_between(time_array, safe_deceleration, deceleration_array, 
                               where=unsafe_mask, color='red', alpha=0.3, 
                               label='Unsafe Deceleration Zone')
        
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Deceleration (m/s²)')
        ax6.set_title('Required vs Safe Deceleration for Signals')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"ttc_signal_analysis_{self.session_id}.png"
        plot_path = os.path.join(self.save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Only print if there was valid data
        valid_distances = [d for d in self.distances if not np.isnan(d)]
        valid_signals = [d for d in self.signal_distances if not np.isnan(d)]
        if valid_distances or valid_signals:
            print(f"TTC + Signal analysis plot saved to: {plot_path}")
        
    def save_data_json(self):
        """
        Save TTC data as JSON for further analysis
        """
        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'data_points': []
        }
        
        for i in range(len(self.ttc_data)):
            data_point = {
                'time': self.time_data[i],
                'ttc_risk': self.ttc_data[i],
                'object_id': self.object_ids[i],
                'distance': self.distances[i],
                'gap_rate': self.gap_rates[i]
            }
            data['data_points'].append(data_point)
            
        # Add enhanced summary statistics for simplified approach
        if self.ttc_data:
            # Emergency vehicle statistics
            emergency_encounters = sum(self.emergency_flags)
            emergency_frames = [i for i, flag in enumerate(self.emergency_flags) if flag]
            
            data['summary'] = {
                'total_frames': len(self.ttc_data),
                'max_ttc_risk': max(self.ttc_data),
                'min_ttc_risk': min(self.ttc_data),
                'avg_ttc_risk': np.mean(self.ttc_data),
                'std_ttc_risk': np.std(self.ttc_data),
                'critical_frames': sum(1 for risk in self.ttc_data if risk >= 0.8),
                'warning_frames': sum(1 for risk in self.ttc_data if 0.5 <= risk < 0.8),
                'caution_frames': sum(1 for risk in self.ttc_data if 0.2 <= risk < 0.5),
                'safe_frames': sum(1 for risk in self.ttc_data if risk < 0.2),
                # Enhanced statistics for simplified approach
                'emergency_encounters': emergency_encounters,
                'emergency_percentage': (emergency_encounters / len(self.ttc_data)) * 100,
                'max_unified_risk': max(self.unified_risks) if self.unified_risks else 0,
                'avg_unified_risk': np.mean(self.unified_risks) if self.unified_risks else 0,
                'object_type_distribution': {
                    obj_type: self.object_types.count(obj_type) 
                    for obj_type in set(self.object_types) if obj_type != 'none'
                }
            }
        
        # Save JSON file
        json_filename = f"ttc_data_{self.session_id}.json"
        json_path = os.path.join(self.save_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"TTC data saved to: {json_path}")
        
    def create_summary_report(self):
        """
        Create a text summary report
        """
        if not self.ttc_data:
            return
            
        # Filter out NaN values for statistics
        valid_distances = [d for d in self.distances if not np.isnan(d)]
        valid_gap_rates = [g for g in self.gap_rates if not np.isnan(g)]
        
        # Only create report if we have some valid data
        if not valid_distances and not valid_gap_rates:
            return  # Don't create report with no valid object data
            
        report_filename = f"ttc_summary_{self.session_id}.txt"
        report_path = os.path.join(self.save_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(f"TTC Risk Analysis Summary Report\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Frames Analyzed: {len(self.ttc_data)}\n")
            f.write(f"Frames with Valid Objects: {len(valid_distances)}\n")
            f.write(f"Maximum TTC Risk: {max(self.ttc_data):.3f}\n")
            f.write(f"Minimum TTC Risk: {min(self.ttc_data):.3f}\n")
            f.write(f"Average TTC Risk: {np.mean(self.ttc_data):.3f}\n")
            f.write(f"Standard Deviation: {np.std(self.ttc_data):.3f}\n\n")
            
            f.write("Risk Level Distribution:\n")
            f.write(f"  Critical (≥0.8): {sum(1 for risk in self.ttc_data if risk >= 0.8)} frames\n")
            f.write(f"  Warning (0.5-0.8): {sum(1 for risk in self.ttc_data if 0.5 <= risk < 0.8)} frames\n")
            f.write(f"  Caution (0.2-0.5): {sum(1 for risk in self.ttc_data if 0.2 <= risk < 0.5)} frames\n")
            f.write(f"  Safe (<0.2): {sum(1 for risk in self.ttc_data if risk < 0.2)} frames\n\n")
            
            if valid_distances:
                f.write(f"Distance Statistics (Valid Objects Only):\n")
                f.write(f"  Average Distance: {np.mean(valid_distances):.2f} m\n")
                f.write(f"  Minimum Distance: {min(valid_distances):.2f} m\n")
                f.write(f"  Maximum Distance: {max(valid_distances):.2f} m\n\n")
                
            if valid_gap_rates:
                f.write(f"Gap Rate Statistics (Valid Objects Only):\n")
                f.write(f"  Average Gap Rate: {np.mean(valid_gap_rates):.2f} m/s\n")
                f.write(f"  Minimum Gap Rate: {min(valid_gap_rates):.2f} m/s\n")
                f.write(f"  Maximum Gap Rate: {max(valid_gap_rates):.2f} m/s\n")
                f.write(f"  Frames with Closing Gap: {sum(1 for rate in valid_gap_rates if rate < 0)}\n")
                
        print(f"TTC summary report saved to: {report_path}")
        
    def finalize_session(self):
        """
        Create all plots and save all data for the current session
        """
        if self.ttc_data:
            self.plot_ttc_timeline()
            self.save_data_json()
            self.create_summary_report()
        else:
            print("No TTC data to plot") 