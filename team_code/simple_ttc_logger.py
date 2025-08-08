import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional

class SimpleTTCLogger:
    """
    Simple logger focused on TTC data collection for experiments
    """
    
    def __init__(self, experiment_condition: str, participant_id: str, trial_number: str, scenario_name: str = None):
        """
        Initialize the simple TTC logger
        
        Args:
            experiment_condition: 'baseline', 'external_only', 'internal_only', 'fusion'
            participant_id: Participant identifier (e.g., 'P001')
            trial_number: Trial number (e.g., '1', '2', '3')
            scenario_name: Scenario identifier (e.g., 'SC-EV', 'SC-INT', 'SC-PED')
        """
        self.experiment_condition = experiment_condition
        self.participant_id = participant_id
        self.trial_number = trial_number
        
        # Get scenario from environment variable or parameter
        self.scenario_name = scenario_name or os.getenv('SCENARIO_NAME', 'SC-EV')
        
        self.start_time = time.time()
        
        # Data storage
        self.ttc_data = []
        self.alert_events = []  # Track alert events for timing analysis
        self.signal_violations = []  # Track signal compliance violations
        
        # **NEW: Signal compliance and deceleration tracking**
        self.signal_events = []           # All signal encounters
        self.speed_history = []           # Track speed for deceleration calculation
        self.max_deceleration = 0.0       # Maximum deceleration experienced
        self.deceleration_events = []     # All significant deceleration events
        
        # **NEW: Signal violation metrics**
        self.signal_metrics = {
            'total_signals_encountered': 0,
            'cautious_alerts': 0,         # 4.0+ m/s² required
            'warning_alerts': 0,          # 6.0+ m/s² required
            'critical_alerts': 0,         # 8.0+ m/s² required 
            'emergency_alerts': 0,        # 10.0+ m/s² required
            'max_required_deceleration': 0.0,
            'avg_signal_distance': 0.0,
            'red_light_encounters': 0,
            'stop_sign_encounters': 0
        }
        
        # **NEW: Post-violation tracking metrics**
        self.active_post_violations = {}     # Currently tracking violations {violation_id: tracking_data}
        self.completed_post_violations = []  # Completed post-violation analyses
        self._violation_counter = 0          # Counter for unique violation IDs
        
        # Experiment tracking
        self.hazard_first_detected = None  # NEW: When police car first detected
        self.first_alert_issued = None     # NEW: When first alert was issued
        
        # Signal compliance tracking
        self.violation_warnings = []  # Track violation warnings
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set absolute path for experiment logs
        self.log_directory = "/home/ascc304/carla_garage/experiment_logs/int_test"
        
        print(f"📊 Simple TTC Logger initialized")
        print(f"   Condition: {experiment_condition}")
        print(f"   Participant: {participant_id}")
        print(f"   Trial: {trial_number}")
        print(f"   Scenario: {self.scenario_name}")
        print(f"   Session: {self.session_id}")
        print(f"   Log directory: {self.log_directory}")
        
    def log_ttc_data(self, timestamp: float, ttc_object: Optional[Dict], ego_speed: float, 
                     internal_risk: float = 0.0, driver_state: str = "unknown"):
        """
        Log TTC data from the safety evaluator
        
        Args:
            timestamp: Current timestamp
            ttc_object: TTC object info from safety evaluator (None if external risk disabled)
            ego_speed: Current ego vehicle speed (m/s)
            internal_risk: Internal risk level (for internal-only experiments)
            driver_state: Current driver state (for internal-only experiments)
        """
        relative_time = timestamp - self.start_time
        
        # **NEW: Track speed and calculate deceleration**
        self._track_speed_and_deceleration(timestamp, ego_speed)
        
        # **NEW: Update post-violation tracking metrics**
        self.update_post_violation_tracking(timestamp, ego_speed)
        
        # Debug: Count all calls
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        
        # Debug output every 50 calls
        if self._call_count % 50 == 0:
            print(f"🔍 TTC Logger - Call {self._call_count}: ttc_object = {ttc_object is not None}")
        
        # Extract TTC information - RELAXED FILTERING
        if ttc_object:
            # Track hazard detection for timing analysis
            self.log_hazard_detection(timestamp, ttc_object)
            
            distance = ttc_object.get('distance', 0.0)
            ttc_value = ttc_object.get('ttc', float('inf'))
            object_type = ttc_object.get('class_name', 'unknown')
            is_emergency = ttc_object.get('is_emergency', False)
            ttc_risk = ttc_object.get('ttc_risk', 0.0)
            
            # Log ANY finite TTC value or any object with distance > 0
            # This captures cases where TTC might be large but still meaningful
            should_log = False
            log_reason = ""
            
            if ttc_value != float('inf') and ttc_value > 0:
                should_log = True
                log_reason = f"Valid TTC: {ttc_value:.2f}s"
            elif distance > 0 and ttc_risk > 0:
                should_log = True
                log_reason = f"Distance: {distance:.1f}m, Risk: {ttc_risk:.3f}"
            elif is_emergency and distance > 0:
                should_log = True
                log_reason = f"Emergency vehicle: {distance:.1f}m"
            elif distance > 0:
                should_log = True
                log_reason = f"Object detected: {distance:.1f}m"
            
            if should_log:
                data_entry = {
                    'timestamp': timestamp,
                    'relative_time': relative_time,
                    'distance': distance,
                    'ttc': ttc_value if ttc_value != float('inf') else None,
                    'ttc_risk': ttc_risk,
                    'object_type': object_type,
                    'is_emergency': is_emergency,
                    'ego_speed': ego_speed,
                    'ego_speed_kmh': ego_speed * 3.6,
                    'internal_risk': internal_risk,
                    'driver_state': driver_state,
                    'log_reason': log_reason,
                    'max_deceleration_so_far': self.max_deceleration  # **NEW: Current max decel**
                }
                
                self.ttc_data.append(data_entry)
                
                # Debug output for valid logs
                if self._call_count % 25 == 0:  # Less frequent debug
                    print(f"📝 Logged TTC data: {log_reason}")
        
        # **ALWAYS log basic frame data even without TTC object**
        else:
            # Still track basic info even when no TTC object present
            basic_entry = {
                'timestamp': timestamp,
                'relative_time': relative_time,
                'distance': None,
                'ttc': None,
                'ttc_risk': 0.0,
                'object_type': 'none',
                'is_emergency': False,
                'ego_speed': ego_speed,
                'ego_speed_kmh': ego_speed * 3.6,
                'internal_risk': internal_risk,
                'driver_state': driver_state,
                'log_reason': 'no_ttc_object',
                'max_deceleration_so_far': self.max_deceleration  # **NEW: Current max decel**
            }
            
            # Log every 5th frame when no objects present to reduce log size
            if len(self.ttc_data) == 0 or self._call_count % 5 == 0:
                self.ttc_data.append(basic_entry)
    
    def _track_speed_and_deceleration(self, timestamp: float, ego_speed: float):
        """
        **NEW: Track speed history and calculate maximum deceleration**
        
        Args:
            timestamp: Current timestamp
            ego_speed: Current ego vehicle speed (m/s)
        """
        # Add current speed to history
        speed_entry = {
            'timestamp': timestamp,
            'speed': ego_speed,
            'speed_kmh': ego_speed * 3.6
        }
        self.speed_history.append(speed_entry)
        
        # Calculate deceleration if we have at least 2 speed measurements
        if len(self.speed_history) >= 2:
            prev_entry = self.speed_history[-2]
            current_entry = self.speed_history[-1]
            
            dt = current_entry['timestamp'] - prev_entry['timestamp']
            if dt > 0.01:  # Avoid division by very small time differences
                dv = current_entry['speed'] - prev_entry['speed']
                acceleration = dv / dt  # m/s²
                
                # Track deceleration (negative acceleration)
                if acceleration < 0:
                    deceleration = abs(acceleration)
                    
                    # Update maximum deceleration
                    if deceleration > self.max_deceleration:
                        self.max_deceleration = deceleration
                        print(f"🛑 NEW MAX DECELERATION: {deceleration:.2f} m/s² at {ego_speed*3.6:.1f} km/h")
                    
                    # Log significant deceleration events (> 2 m/s²)
                    if deceleration > 2.0:
                        decel_event = {
                            'timestamp': timestamp,
                            'relative_time': timestamp - self.start_time,
                            'deceleration': deceleration,
                            'initial_speed': prev_entry['speed'],
                            'final_speed': current_entry['speed'],
                            'duration': dt,
                            'speed_change_kmh': dv * 3.6,
                            'severity': self._classify_deceleration(deceleration)
                        }
                        self.deceleration_events.append(decel_event)
                        
                        # Print significant deceleration events
                        if deceleration > 5.0:  # Hard braking
                            print(f"🚨 HARD BRAKING: {deceleration:.2f} m/s² at {ego_speed*3.6:.1f} km/h")
        
        # Keep only recent speed history to manage memory (last 100 seconds at 10Hz)
        if len(self.speed_history) > 1000:
            self.speed_history = self.speed_history[-500:]  # Keep last 50 seconds
    
    def _classify_deceleration(self, deceleration: float) -> str:
        """
        **NEW: Classify deceleration severity**
        
        Args:
            deceleration: Deceleration value in m/s²
            
        Returns:
            str: Severity classification matching signal compliance thresholds
        """
        if deceleration >= 10.0:
            return "EMERGENCY"       # Emergency braking (matches signal emergency threshold)
        elif deceleration >= 8.0:
            return "CRITICAL"        # Critical braking (matches signal critical threshold)  
        elif deceleration >= 6.0:
            return "WARNING"         # Warning level braking (matches signal warning threshold)
        elif deceleration >= 4.0:
            return "CAUTIOUS"        # Cautious braking (matches signal cautious threshold)
        elif deceleration >= 2.0:
            return "MODERATE"        # Moderate braking
        else:
            return "LIGHT"           # Light braking
    
    def log_alert_event(self, timestamp: float, alert_level: str, alert_type: str, 
                       current_ttc: Optional[float], current_distance: float, 
                       driver_state: str = "unknown"):
        """
        Log alert events for timing analysis
        
        Args:
            timestamp: When alert was issued
            alert_level: 'caution', 'warning', 'critical', 'emergency'
            alert_type: 'beep' or 'tts'
            current_ttc: TTC value when alert was issued
            current_distance: Distance when alert was issued
            driver_state: Driver state when alert was issued
        """
        alert_entry = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time,
            'alert_level': alert_level,
            'alert_type': alert_type,
            'ttc_at_alert': current_ttc,
            'distance_at_alert': current_distance,
            'driver_state': driver_state
        }
        
        self.alert_events.append(alert_entry)
        
        # Track first alert for lead time analysis
        if self.first_alert_issued is None:
            self.first_alert_issued = timestamp
            print(f"📍 First alert logged: {alert_level} at {current_distance:.1f}m")
        
        print(f"🚨 Alert logged: {alert_level} ({alert_type}) - Distance: {current_distance:.1f}m")
        
    def log_hazard_detection(self, timestamp: float, ttc_object: Dict):
        """
        Log when a hazard (emergency vehicle) is first detected
        
        Args:
            timestamp: When hazard was first detected
            ttc_object: The detected threat object
        """
        # Only log the first detection of an emergency vehicle
        if (self.hazard_first_detected is None and 
            ttc_object and 
            ttc_object.get('is_emergency', False)):
            
            self.hazard_first_detected = timestamp
            hazard_distance = ttc_object.get('distance', 0.0)
            
            print(f"🚔 Emergency vehicle first detected: {hazard_distance:.1f}m away")
            print(f"   Timestamp: {timestamp:.3f}")
            
    def log_signal_event(self, timestamp: float, signal_info: Dict, ego_speed: float):
        """
        **NEW: Log signal compliance events with comprehensive metrics tracking**
        
        Args:
            timestamp: Current timestamp
            signal_info: Signal information from safety evaluator
            ego_speed: Current ego vehicle speed (m/s)
        """
        if not signal_info:
            return
        
        # **DEBUG: Print all signal logging calls**
        signal_type = signal_info.get('signal_type', 'Unknown')
        alert_level = signal_info.get('alert_level', 'SAFE')
        distance = signal_info.get('distance_to_stop_line', 0.0)
        print(f"🔍 LOGGER DEBUG: Logging {signal_type} with alert_level='{alert_level}' at distance={distance:.2f}m")
        
        # Update signal encounter metrics
        self.signal_metrics['total_signals_encountered'] += 1
        
        signal_type = signal_info.get('signal_type', 'Unknown')
        if 'Red Light' in signal_type:
            self.signal_metrics['red_light_encounters'] += 1
        elif 'Stop Sign' in signal_type:
            self.signal_metrics['stop_sign_encounters'] += 1
        
        # Track alert levels based on deceleration thresholds
        required_decel = signal_info.get('required_deceleration', 0.0)
        alert_level = signal_info.get('alert_level', 'SAFE')
        
        if alert_level == 'EMERGENCY':
            self.signal_metrics['emergency_alerts'] += 1
        elif alert_level == 'CRITICAL':
            self.signal_metrics['critical_alerts'] += 1
        elif alert_level == 'WARNING':
            self.signal_metrics['warning_alerts'] += 1
        elif alert_level == 'CAUTIOUS':
            self.signal_metrics['cautious_alerts'] += 1
        
        # Update maximum required deceleration
        if required_decel > self.signal_metrics['max_required_deceleration']:
            self.signal_metrics['max_required_deceleration'] = required_decel
        
        # Update average signal distance
        distance = signal_info.get('distance_to_stop_line', 0.0)
        total_signals = self.signal_metrics['total_signals_encountered']
        current_avg = self.signal_metrics['avg_signal_distance']
        self.signal_metrics['avg_signal_distance'] = (
            (current_avg * (total_signals - 1) + distance) / total_signals
        )
        
        # Log detailed signal event
        signal_entry = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time,
            'signal_type': signal_type,
            'distance_to_stop_line': distance,
            'required_deceleration': required_decel,
            'alert_level': alert_level,
            'signal_risk': signal_info.get('signal_risk', 0.0),
            'ego_speed': ego_speed,
            'ego_speed_kmh': ego_speed * 3.6,
            'max_deceleration_at_time': self.max_deceleration,
            'deceleration_deficit': max(0, required_decel - self.max_deceleration),  # How much more decel needed
            'can_stop_safely': required_decel <= self.max_deceleration if self.max_deceleration > 0 else False
        }
        
        self.signal_events.append(signal_entry)
        
        # **SIMPLE VIOLATION DETECTION**
        # If safety evaluator detected a violation (alert_level = 'VIOLATION')
        if alert_level == 'VIOLATION':
            print(f"🚨 SIMPLE_TTC_LOGGER: Processing VIOLATION for {signal_type}")
            print(f"📊 Current violation count before: {len(self.signal_violations)}")
            
            # **FIXED: Use speed from signal_info for violations, not current ego_speed**
            violation_speed_ms = signal_info.get('ego_speed_at_violation', ego_speed)
            
            violation_entry = {
                'timestamp': timestamp,
                'relative_time': timestamp - self.start_time,
                'signal_type': signal_type,
                'distance_past_stop_line': abs(distance),  # How far past stop line
                'ego_speed_at_violation': violation_speed_ms,
                'ego_speed_kmh_at_violation': violation_speed_ms * 3.6,
                'violation_type': 'RAN_SIGNAL'
            }
            
            self.signal_violations.append(violation_entry)
            print(f"📊 VIOLATION LOGGED: {signal_type} - {abs(distance):.1f}m past stop line at {violation_speed_ms*3.6:.1f} km/h")
            print(f"📊 Current ego speed: {ego_speed*3.6:.1f} km/h, Violation speed: {violation_speed_ms*3.6:.1f} km/h")
            print(f"📊 Total violations after logging: {len(self.signal_violations)}")
            
            # **NEW: Start post-violation tracking for time-to-stop and distance metrics**
            self.start_post_violation_tracking(timestamp, signal_info, violation_speed_ms)
        
        # Print significant signal events
        if alert_level in ['CRITICAL', 'EMERGENCY', 'VIOLATION']:
            print(f"🚦 SIGNAL {alert_level}: {signal_type} at {distance:.1f}m")
            if alert_level != 'VIOLATION':
                print(f"   Required: {required_decel:.1f} m/s², Max achieved: {self.max_deceleration:.1f} m/s²")
                if signal_entry['deceleration_deficit'] > 0:
                    print(f"   ⚠️ Deficit: {signal_entry['deceleration_deficit']:.1f} m/s²")
        
    def start_post_violation_tracking(self, timestamp: float, violation_info: Dict, ego_speed: float):
        """
        **NEW: Start tracking post-violation metrics when a signal violation occurs**
        
        Args:
            timestamp: When the violation was detected
            violation_info: Information about the violation
            ego_speed: Current vehicle speed at violation detection (m/s) - often already slowed
        """
        self._violation_counter += 1
        violation_id = f"violation_{self._violation_counter:03d}"
        
        # **CRITICAL FIX**: Use the actual violation speed, not current speed
        # The violation speed is the max speed when approaching the signal
        actual_violation_speed = violation_info.get('ego_speed_at_violation', ego_speed)
        
        # **PHYSICS-BASED ESTIMATION**: Since violation detection happens late,
        # estimate how much distance was already traveled during deceleration
        speed_difference = actual_violation_speed - ego_speed
        if speed_difference > 0:
            # Estimate time taken to decelerate from violation speed to current speed
            # Using average deceleration of ~5 m/s² (reasonable braking)
            estimated_deceleration = 5.0  # m/s²
            estimated_decel_time = speed_difference / estimated_deceleration
            # Distance during deceleration: d = v₀t - ½at²
            estimated_distance_already_traveled = (actual_violation_speed * estimated_decel_time) - (0.5 * estimated_deceleration * estimated_decel_time ** 2)
        else:
            estimated_distance_already_traveled = 0.0
            
        # Initialize post-violation tracking data
        tracking_data = {
            'violation_id': violation_id,
            'start_timestamp': timestamp,
            'start_relative_time': timestamp - self.start_time,
            'signal_type': violation_info.get('signal_type', 'Unknown'),
            'violation_speed_ms': actual_violation_speed,  # **FIXED**: Use actual violation speed
            'violation_speed_kmh': actual_violation_speed * 3.6,
            'distance_past_stop_line': abs(violation_info.get('distance_to_stop_line', 0.0)),
            
            # Tracking state
            'is_active': True,
            'has_stopped': False,
            'stop_timestamp': None,
            'last_update_timestamp': timestamp,
            'last_speed': ego_speed,  # Current speed at detection time
            
            # **IMPROVED DISTANCE TRACKING**: Account for distance already traveled
            'distance_traveled_after_violation': estimated_distance_already_traveled,
            'estimated_distance_from_physics': estimated_distance_already_traveled > 0,
            
            # Time tracking - will be calculated when vehicle stops
            'time_to_stop': None,
        }
        
        self.active_post_violations[violation_id] = tracking_data
        
        return violation_id
    
    def update_post_violation_tracking(self, timestamp: float, ego_speed: float):
        """
        **NEW: Update all active post-violation tracking metrics**
        
        Args:
            timestamp: Current timestamp
            ego_speed: Current vehicle speed (m/s)
        """
        if not self.active_post_violations:
            return
        
        # Define when vehicle is considered "stopped"
        STOP_SPEED_THRESHOLD = 1.0  # m/s (3.6 km/h) - INCREASED from 0.5 for more realistic detection
        STOP_CONFIRMATION_TIME = 2.0  # seconds - require stopping for 2 seconds to confirm
        
        violations_to_complete = []
        
        for violation_id, tracking_data in self.active_post_violations.items():
            if not tracking_data['is_active']:
                continue
                
            # Calculate time elapsed since violation
            time_elapsed = timestamp - tracking_data['start_timestamp']
            
            # **IMPROVED: Initialize stop confirmation tracking**
            if 'stop_confirmation_start' not in tracking_data:
                tracking_data['stop_confirmation_start'] = None
                tracking_data['confirmed_stopped'] = False
            
            # Estimate distance traveled (simple integration using previous speed)
            if tracking_data['last_update_timestamp'] is not None:
                dt = timestamp - tracking_data['last_update_timestamp']
                # Use average speed over the interval for more accurate distance calculation
                avg_speed = (tracking_data['last_speed'] + ego_speed) / 2.0
                distance_increment = avg_speed * dt
                tracking_data['distance_traveled_after_violation'] += distance_increment
            
            # Update tracking state
            tracking_data['last_update_timestamp'] = timestamp
            tracking_data['last_speed'] = ego_speed
            
            # **IMPROVED: Check if vehicle has stopped with confirmation**
            if ego_speed <= STOP_SPEED_THRESHOLD:
                # Vehicle is currently below stop threshold
                if tracking_data['stop_confirmation_start'] is None:
                    # Start stop confirmation timer
                    tracking_data['stop_confirmation_start'] = timestamp
                else:
                    # Check if vehicle has been stopped long enough
                    stop_duration = timestamp - tracking_data['stop_confirmation_start']
                    if stop_duration >= STOP_CONFIRMATION_TIME and not tracking_data['confirmed_stopped']:
                        # Vehicle confirmed stopped
                        tracking_data['confirmed_stopped'] = True
                        tracking_data['has_stopped'] = True
                        tracking_data['stop_timestamp'] = tracking_data['stop_confirmation_start']
                        tracking_data['time_to_stop'] = tracking_data['stop_confirmation_start'] - tracking_data['start_timestamp']
                        
                        # Mark for completion
                        violations_to_complete.append(violation_id)
            else:
                # Vehicle speed above threshold - reset stop confirmation
                if tracking_data['stop_confirmation_start'] is not None:
                    tracking_data['stop_confirmation_start'] = None
            
            # Auto-complete tracking after reasonable time limit (60 seconds) or if speed very low
            if time_elapsed > 60.0 or (time_elapsed > 10.0 and ego_speed < 1.0):
                tracking_data['time_to_stop'] = time_elapsed if ego_speed <= STOP_SPEED_THRESHOLD else None
                violations_to_complete.append(violation_id)
        
        # Complete tracking for stopped violations
        for violation_id in violations_to_complete:
            self._complete_post_violation_tracking(violation_id)
    
    def _complete_post_violation_tracking(self, violation_id: str):
        """
        **NEW: Complete post-violation tracking and move to completed list**
        
        Args:
            violation_id: ID of the violation to complete
        """
        if violation_id not in self.active_post_violations:
            return
        
        tracking_data = self.active_post_violations[violation_id]
        tracking_data['is_active'] = False
        
        # Create completed violation analysis
        completed_analysis = {
            'violation_id': violation_id,
            'signal_type': tracking_data['signal_type'],
            'violation_timestamp': tracking_data['start_timestamp'],
            'violation_relative_time': tracking_data['start_relative_time'],
            'violation_speed_kmh': tracking_data['violation_speed_kmh'],
            'distance_past_stop_line': tracking_data['distance_past_stop_line'],
            
            # **PRIMARY METRICS REQUESTED**
            'time_to_stop_seconds': tracking_data.get('time_to_stop'),
            'distance_traveled_after_violation_meters': tracking_data['distance_traveled_after_violation'],
            
            # **ANALYSIS DETAILS**
            'vehicle_stopped': tracking_data['has_stopped'],
            'tracking_completed': True,
            'tracking_duration': tracking_data['last_update_timestamp'] - tracking_data['start_timestamp'],
            'used_physics_estimation': tracking_data.get('estimated_distance_from_physics', False),
            'detection_delay_compensated': tracking_data.get('estimated_distance_from_physics', False)
        }
        
        self.completed_post_violations.append(completed_analysis)
        
        # Remove from active tracking
        del self.active_post_violations[violation_id] 