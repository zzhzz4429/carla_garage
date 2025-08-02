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
        
        # **SEPARATED RISK ARCHITECTURE** - Enhanced data tracking
        self.object_types = []  # Track object class names
        self.emergency_flags = []  # Track emergency vehicle encounters
        self.unified_risks = []  # Track combined external risk (if available)
        
        # **SEPARATED RISK COMPONENTS**
        self.ttc_risks = []  # Pure TTC risk (separated)
        self.signal_risks = []  # Pure signal compliance risk (separated)
        self.fused_ttc_risks = []  # TTC risk fused with driver state
        self.driver_states = []  # Driver state information
        
        # **SIGNAL COMPLIANCE DATA** - Enhanced for new system
        self.signal_distances = []  # Distance to red lights/stop signs
        self.signal_types = []  # Type of signal (red light, stop sign)
        self.signal_alert_levels = []  # Alert levels (CAUTIOUS, WARNING, CRITICAL, EMERGENCY, VIOLATION)
        self.current_decelerations = []  # Current required deceleration
        self.ego_speeds = []  # Ego vehicle speed for deceleration calculation
        
        # **VIOLATION TRACKING** - NEW
        self.violations = []  # Track detected violations
        self.violation_times = []  # Timestamps of violations
        self.max_deceleration_history = []  # Track maximum deceleration over time
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Generate unique session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_data_point(self, ttc_risk, ttc_object, frame_time, unified_risk=None, signal_data=None, ego_speed=0.0, risk_breakdown=None):
        """
        **UPDATED: Add data point for separated risk architecture with violation tracking**
        
        Args:
            ttc_risk: TTC risk value
            ttc_object: Object information from TTC calculation
            frame_time: Current frame time
            unified_risk: Optional unified external risk value
            signal_data: Dictionary with signal compliance information
            ego_speed: Current ego vehicle speed (m/s)
            risk_breakdown: Dictionary with separated risk components
        """
        self.ttc_data.append(ttc_risk)
        self.time_data.append(frame_time)
        self.risk_data.append(ttc_risk)
        self.ego_speeds.append(ego_speed)
        
        # **SEPARATED RISK COMPONENTS** - Extract from risk_breakdown if available
        if risk_breakdown:
            self.ttc_risks.append(risk_breakdown.get('ttc_risk', ttc_risk))
            self.signal_risks.append(risk_breakdown.get('signal_risk', 0.0))
            self.fused_ttc_risks.append(risk_breakdown.get('fused_ttc_risk', ttc_risk))
            
            # Driver state information
            driver_info = risk_breakdown.get('driver_state', {})
            self.driver_states.append(driver_info.get('current_state', 'unknown'))
        else:
            # Fallback to legacy data
            self.ttc_risks.append(ttc_risk)
            self.signal_risks.append(0.0)
            self.fused_ttc_risks.append(ttc_risk)
            self.driver_states.append('unknown')
        
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
            
        # **ENHANCED SIGNAL COMPLIANCE DATA**
        if signal_data:
            self.signal_distances.append(signal_data.get('distance', np.nan))
            self.signal_types.append(signal_data.get('signal_type', 'none'))
            self.signal_alert_levels.append(signal_data.get('alert_level', 'SAFE'))
            self.current_decelerations.append(signal_data.get('required_deceleration', 0.0))
            
            # **VIOLATION DETECTION** - Track violations when they occur
            if signal_data.get('alert_level') == 'VIOLATION':
                violation_entry = {
                    'time': frame_time,
                    'signal_type': signal_data.get('signal_type', 'Unknown'),
                    'distance': signal_data.get('distance', -1.0),
                    'ego_speed': ego_speed,
                    'ego_speed_kmh': ego_speed * 3.6
                }
                self.violations.append(violation_entry)
                self.violation_times.append(frame_time)
                print(f"📊 PLOTTER: Violation recorded at {frame_time:.1f}s - {signal_data.get('signal_type')}")
        else:
            # No signal detected
            self.signal_distances.append(np.nan)
            self.signal_types.append('none')
            self.signal_alert_levels.append('SAFE')
            self.current_decelerations.append(0.0)
        
        # **MAXIMUM DECELERATION TRACKING** - Track from risk breakdown if available
        if risk_breakdown and 'max_deceleration_so_far' in risk_breakdown:
            self.max_deceleration_history.append(risk_breakdown['max_deceleration_so_far'])
        else:
            # Use previous value or 0
            prev_max = self.max_deceleration_history[-1] if self.max_deceleration_history else 0.0
            self.max_deceleration_history.append(prev_max)
            
    def plot_ttc_timeline(self):
        """
        Create and save TTC timeline plot with signal compliance
        """
        if len(self.ttc_data) < 2:
            return  # Don't print anything if not enough data
            
        try:
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
            
            # Plot 1: **SEPARATED RISK ARCHITECTURE** - TTC vs Signal vs Unified
            # Plot separated risk components
            ax1.plot(time_array, np.array(self.ttc_risks), 'b-', linewidth=2, label='TTC Risk (Raw)', alpha=0.8)
            ax1.plot(time_array, np.array(self.fused_ttc_risks), 'navy', linewidth=2, label='TTC Risk (Fused)', alpha=0.8)
            ax1.plot(time_array, np.array(self.signal_risks), 'orange', linewidth=2, label='Signal Risk', alpha=0.8)
            ax1.plot(time_array, unified_smoothed, 'purple', linewidth=2, label='Unified Risk', alpha=0.8)
            
            # **VIOLATION MARKERS** - Highlight violations on risk plot
            if self.violation_times:
                for v_time in self.violation_times:
                    ax1.axvline(x=v_time, color='red', linestyle='--', linewidth=3, alpha=0.8)
                    ax1.text(v_time, 0.9, '🚨VIOLATION', rotation=90, fontsize=10, 
                            color='red', fontweight='bold', ha='center')
            
            # Highlight emergency vehicle encounters
            emergency_times = time_array[emergency_array]
            emergency_risks = unified_smoothed[emergency_array]
            if len(emergency_times) > 0:
                ax1.scatter(emergency_times, emergency_risks, color='red', s=50, marker='s', 
                           label='Emergency Vehicle', alpha=0.8, zorder=5)
            
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Risk Level')
            ax1.set_title('Separated Risk Architecture (TTC + Signal Compliance + Violations)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add risk level thresholds for new system
            ax1.axhline(y=0.95, color='darkred', linestyle='--', alpha=0.7, label='Emergency')
            ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Critical')
            ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax1.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Cautious')
            
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
            
        except (TypeError, AttributeError, Exception) as e:
            print(f"⚠️ Matplotlib error during plot generation: {e}")
            print("🔍 This is likely a logging configuration conflict - skipping plot generation")
            print("📊 TTC data is still saved, only plot generation is skipped")
            return
        
    def plot_signal_compliance_analysis(self):
        """
        **NEW: Create comprehensive signal compliance and violation analysis plot**
        """
        if len(self.ttc_data) < 2:
            return
            
        try:
            # Create 2x2 grid for signal compliance analysis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Signal Compliance Analysis with Violations - Session {self.session_id}', fontsize=16)
            
            time_array = np.array(self.time_data)
            
            # Plot 1: Signal Distance with Violations
            signal_distances = np.array(self.signal_distances)
            valid_signal_indices = ~np.isnan(signal_distances)
            
            if np.any(valid_signal_indices):
                ax1.plot(time_array[valid_signal_indices], signal_distances[valid_signal_indices], 
                        'orange', linewidth=2, label='Distance to Signal', alpha=0.8)
                
                # Mark different signal types
                for i, (t, d, signal_type, alert_level) in enumerate(zip(time_array, signal_distances, 
                                                                        self.signal_types, self.signal_alert_levels)):
                    if not np.isnan(d) and signal_type != 'none':
                        if alert_level == 'VIOLATION':
                            ax1.scatter(t, d, marker='X', color='red', s=100, alpha=1.0, zorder=10, 
                                       label='VIOLATION' if i == 0 else "")
                        elif signal_type == 'Red Light':
                            ax1.scatter(t, d, marker='o', color='red', s=30, alpha=0.7, zorder=5)
                        elif signal_type == 'Stop Sign':
                            ax1.scatter(t, d, marker='s', color='brown', s=30, alpha=0.7, zorder=5)
            
            # Add violation markers
            if self.violation_times:
                for v_time in self.violation_times:
                    ax1.axvline(x=v_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    ax1.text(v_time, ax1.get_ylim()[1] * 0.9, '🚨', fontsize=14, 
                            color='red', ha='center')
            
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Distance to Signal (m)')
            ax1.set_title('Signal Distance Tracking with Violations')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axhline(y=5.0, color='red', linestyle=':', alpha=0.7, label='Critical Distance')
            
            # Plot 2: Required Deceleration with Thresholds
            deceleration_array = np.array(self.current_decelerations)
            valid_decel_indices = deceleration_array > 0.1
            
            if np.any(valid_decel_indices):
                ax2.plot(time_array[valid_decel_indices], deceleration_array[valid_decel_indices], 
                        'red', linewidth=2, label='Required Deceleration', alpha=0.8)
                
                # Add straightforward deceleration thresholds
                ax2.axhline(y=4.0, color='yellow', linestyle='-', alpha=0.7, linewidth=2, label='Cautious (4.0 m/s²)')
                ax2.axhline(y=6.0, color='orange', linestyle='-', alpha=0.7, linewidth=2, label='Warning (6.0 m/s²)')
                ax2.axhline(y=8.0, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Critical (8.0 m/s²)')
                ax2.axhline(y=10.0, color='darkred', linestyle='-', alpha=0.7, linewidth=2, label='Emergency (10.0 m/s²)')
                
                # Fill zones
                ax2.fill_between(time_array, 0, 4.0, alpha=0.1, color='green', label='Safe Zone')
                ax2.fill_between(time_array, 4.0, 6.0, alpha=0.1, color='yellow', label='Cautious Zone')
                ax2.fill_between(time_array, 6.0, 8.0, alpha=0.1, color='orange', label='Warning Zone')
                ax2.fill_between(time_array, 8.0, 10.0, alpha=0.1, color='red', label='Critical Zone')
                ax2.fill_between(time_array, 10.0, 20.0, alpha=0.1, color='darkred', label='Emergency Zone')
            
            # Mark violations
            if self.violation_times:
                for v_time in self.violation_times:
                    ax2.axvline(x=v_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Required Deceleration (m/s²)')
            ax2.set_title('Signal Compliance Thresholds')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Separated Risk Components
            if hasattr(self, 'ttc_risks') and self.ttc_risks:
                ax3.plot(time_array, self.ttc_risks, 'blue', linewidth=2, label='TTC Risk', alpha=0.8)
                ax3.plot(time_array, self.signal_risks, 'orange', linewidth=2, label='Signal Risk', alpha=0.8)
                ax3.plot(time_array, self.fused_ttc_risks, 'navy', linewidth=2, label='TTC Risk (Fused)', alpha=0.8)
                
                # Mark violations
                if self.violation_times:
                    for v_time in self.violation_times:
                        ax3.axvline(x=v_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
                        ax3.scatter(v_time, 1.0, marker='X', color='red', s=100, alpha=1.0, zorder=10)
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Risk Level')
            ax3.set_title('Separated Risk Architecture')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(0, 1.1)
            
            # Plot 4: Violation Summary and Statistics
            ax4.axis('off')  # Turn off axis for text display
            
            # Create violation summary text
            summary_text = f"=== SIGNAL COMPLIANCE SUMMARY ===\n\n"
            summary_text += f"Total Violations: {len(self.violations)}\n"
            
            if self.violations:
                red_light_violations = sum(1 for v in self.violations if v['signal_type'] == 'Red Light')
                stop_sign_violations = sum(1 for v in self.violations if v['signal_type'] == 'Stop Sign')
                
                summary_text += f"Red Light Violations: {red_light_violations}\n"
                summary_text += f"Stop Sign Violations: {stop_sign_violations}\n\n"
                
                speeds = [v['ego_speed_kmh'] for v in self.violations]
                summary_text += f"Speed at Violations:\n"
                summary_text += f"  Average: {np.mean(speeds):.1f} km/h\n"
                summary_text += f"  Maximum: {max(speeds):.1f} km/h\n"
                summary_text += f"  Minimum: {min(speeds):.1f} km/h\n\n"
                
                summary_text += f"Violation Times:\n"
                for i, v in enumerate(self.violations):
                    summary_text += f"  {i+1}. {v['time']:.1f}s - {v['signal_type']} at {v['ego_speed_kmh']:.1f} km/h\n"
            else:
                summary_text += "No violations detected ✅\n"
            
            # Add signal encounter statistics
            total_signals = sum(1 for st in self.signal_types if st != 'none')
            summary_text += f"\nTotal Signal Encounters: {total_signals}\n"
            
            # Add maximum deceleration if available
            if hasattr(self, 'max_deceleration_history') and self.max_deceleration_history:
                max_decel = max(self.max_deceleration_history)
                summary_text += f"Maximum Deceleration Achieved: {max_decel:.1f} m/s²\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"signal_compliance_analysis_{self.session_id}.png"
            plot_path = os.path.join(self.save_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Signal compliance analysis plot saved to: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Error creating signal compliance plot: {e}")
            return
            
    def save_data_json(self):
        """
        **ENHANCED: Save comprehensive risk data including signal compliance**
        """
        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'data_points': []
        }
        
        for i in range(len(self.ttc_data)):
            data_point = {
                # **BASIC TTC DATA**
                'time': self.time_data[i],
                'ttc_risk': self.ttc_data[i],
                'object_id': self.object_ids[i],
                'distance': self.distances[i],
                'gap_rate': self.gap_rates[i],
                'ego_speed': self.ego_speeds[i] if i < len(self.ego_speeds) else 0.0,
                
                # **SEPARATED RISK ARCHITECTURE**
                'separated_risks': {
                    'ttc_risk': self.ttc_risks[i] if i < len(self.ttc_risks) else self.ttc_data[i],
                    'signal_risk': self.signal_risks[i] if i < len(self.signal_risks) else 0.0,
                    'fused_ttc_risk': self.fused_ttc_risks[i] if i < len(self.fused_ttc_risks) else self.ttc_data[i],
                    'unified_risk': self.unified_risks[i] if i < len(self.unified_risks) else self.ttc_data[i]
                },
                
                # **SIGNAL COMPLIANCE DATA** - This is the missing data!
                'signal_compliance': {
                    'distance_to_signal': self.signal_distances[i] if i < len(self.signal_distances) else None,
                    'signal_type': self.signal_types[i] if i < len(self.signal_types) else 'none',
                    'alert_level': self.signal_alert_levels[i] if i < len(self.signal_alert_levels) else 'SAFE',
                    'required_deceleration': self.current_decelerations[i] if i < len(self.current_decelerations) else 0.0
                },
                
                # **ENHANCED OBJECT DATA**
                'object_info': {
                    'type': self.object_types[i] if i < len(self.object_types) else 'none',
                    'is_emergency': self.emergency_flags[i] if i < len(self.emergency_flags) else False
                },
                
                # **DRIVER STATE** (if available)
                'driver_state': self.driver_states[i] if i < len(self.driver_states) else 'unknown',
                
                # **MAXIMUM DECELERATION TRACKING**
                'max_deceleration_so_far': self.max_deceleration_history[i] if i < len(self.max_deceleration_history) else 0.0
            }
            data['data_points'].append(data_point)
            
        # Add enhanced summary statistics for simplified approach
        if self.ttc_data:
            # Emergency vehicle statistics
            emergency_encounters = sum(self.emergency_flags)
            emergency_frames = [i for i, flag in enumerate(self.emergency_flags) if flag]
            
            data['summary'] = {
                # **BASIC TTC STATISTICS**
                'total_frames': len(self.ttc_data),
                'max_ttc_risk': max(self.ttc_data),
                'min_ttc_risk': min(self.ttc_data),
                'avg_ttc_risk': np.mean(self.ttc_data),
                'std_ttc_risk': np.std(self.ttc_data),
                'critical_frames': sum(1 for risk in self.ttc_data if risk >= 0.8),
                'warning_frames': sum(1 for risk in self.ttc_data if 0.5 <= risk < 0.8),
                'caution_frames': sum(1 for risk in self.ttc_data if 0.2 <= risk < 0.5),
                'safe_frames': sum(1 for risk in self.ttc_data if risk < 0.2),
                
                # **SEPARATED RISK STATISTICS**
                'separated_risks': {
                    'max_signal_risk': max(self.signal_risks) if self.signal_risks else 0.0,
                    'avg_signal_risk': np.mean(self.signal_risks) if self.signal_risks else 0.0,
                    'signal_events_detected': sum(1 for sr in self.signal_risks if sr > 0.01),
                    'max_unified_risk': max(self.unified_risks) if self.unified_risks else 0,
                    'avg_unified_risk': np.mean(self.unified_risks) if self.unified_risks else 0,
                },
                
                # **SIGNAL COMPLIANCE STATISTICS** - NEW comprehensive data
                'signal_compliance': {
                    'total_signal_encounters': sum(1 for st in self.signal_types if st != 'none'),
                    'red_light_encounters': sum(1 for st in self.signal_types if st == 'Red Light'),
                    'stop_sign_encounters': sum(1 for st in self.signal_types if st == 'Stop Sign'),
                    'violations_detected': len(self.violations),
                    'violation_details': self.violations,  # Complete violation data
                    'max_required_deceleration': max(self.current_decelerations) if self.current_decelerations else 0.0,
                    'avg_signal_distance': np.mean([d for d in self.signal_distances if not np.isnan(d)]) if self.signal_distances else 0.0,
                    'alert_level_distribution': {
                        level: self.signal_alert_levels.count(level)
                        for level in set(self.signal_alert_levels) if level != 'SAFE'
                    }
                },
                
                # **EMERGENCY VEHICLE STATISTICS** 
                'emergency_encounters': emergency_encounters,
                'emergency_percentage': (emergency_encounters / len(self.ttc_data)) * 100,
                
                # **OBJECT TYPE DISTRIBUTION**
                'object_type_distribution': {
                    obj_type: self.object_types.count(obj_type) 
                    for obj_type in set(self.object_types) if obj_type != 'none'
                },
                
                # **PERFORMANCE METRICS**
                'max_deceleration_achieved': max(self.max_deceleration_history) if self.max_deceleration_history else 0.0
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
        **UPDATED: Create all plots and save all data for the current session**
        """
        if self.ttc_data:
            self.plot_ttc_timeline()  # Original TTC plot
            self.plot_signal_compliance_analysis()  # NEW: Signal compliance plot
            self.save_data_json()
            self.create_summary_report()
        else:
            print("No TTC data to plot") 