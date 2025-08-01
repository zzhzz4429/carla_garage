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
        self.alert_events = []  # NEW: Track alert events for timing analysis
        self.signal_violations = []  # NEW: Track signal compliance violations
        
        # Experiment tracking
        self.hazard_first_detected = None  # NEW: When police car first detected
        self.first_alert_issued = None     # NEW: When first alert was issued
        
        # Signal compliance tracking
        self.signal_events = []  # Track all signal encounters
        self.violation_warnings = []  # Track violation warnings
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set absolute path for experiment logs
        self.log_directory = "/home/ascc304/carla_garage/experiment_logs/zhaohua/pedestrian"
        
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
                log_reason = f"Risk present: {ttc_risk:.3f}"
            elif distance > 0:
                should_log = True
                log_reason = f"Object detected: {distance:.1f}m"
                
            if should_log:
                ttc_entry = {
                    'timestamp': timestamp,
                    'relative_time': relative_time,
                    'ttc': ttc_value if ttc_value != float('inf') else None,
                    'distance': distance,
                    'ego_speed': ego_speed,
                    'object_type': object_type,
                    'is_emergency': is_emergency,
                    'ttc_risk': ttc_risk,
                    'internal_risk': internal_risk,
                    'driver_state': driver_state,
                    'log_reason': log_reason
                }
                
                self.ttc_data.append(ttc_entry)
                
                if len(self.ttc_data) % 10 == 1:  # Print every 10th entry
                    print(f"TTC logged #{len(self.ttc_data)}: {log_reason} - {object_type} at {distance:.1f}m")
            elif self._call_count % 50 == 0:
                print(f"🔍 TTC rejected: ttc={ttc_value}, distance={distance}, risk={ttc_risk}")
        elif self._call_count % 50 == 0:
            print(f"🔍 No TTC object provided")
            
        # Note: With new approach, TTC is always calculated for measurement
        # No need for special internal-only logging since ttc_object will always exist
            
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
        Log signal compliance events and violations
        
        Args:
            timestamp: Current timestamp
            signal_info: Signal information from safety evaluator
            ego_speed: Current ego vehicle speed (m/s)
        """
        if not signal_info:
            return
            
        signal_entry = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time,
            'signal_type': signal_info.get('signal_type', 'Unknown'),
            'distance': signal_info.get('distance', 0.0),
            'required_deceleration': signal_info.get('required_deceleration', 0.0),
            'signal_risk': signal_info.get('signal_risk', 0.0),
            'risk_category': signal_info.get('risk_category', 'UNKNOWN'),
            'violation_imminent': signal_info.get('violation_imminent', False),
            'violation_likely': signal_info.get('violation_likely', False),
            'ego_speed': ego_speed,
            'ego_speed_kmh': ego_speed * 3.6
        }
        
        self.signal_events.append(signal_entry)
        
        # Log violations separately for analysis
        if signal_info.get('violation_imminent', False):
            violation_entry = {
                'timestamp': timestamp,
                'relative_time': timestamp - self.start_time,
                'violation_type': 'IMMINENT',
                'signal_type': signal_info.get('signal_type', 'Unknown'),
                'distance': signal_info.get('distance', 0.0),
                'required_deceleration': signal_info.get('required_deceleration', 0.0),
                'ego_speed_kmh': ego_speed * 3.6,
                'severity': 'CRITICAL'
            }
            self.signal_violations.append(violation_entry)
            print(f"🚨 SIGNAL VIOLATION (IMMINENT): {signal_info.get('signal_type')} at {signal_info.get('distance', 0):.1f}m")
            print(f"   Required deceleration: {signal_info.get('required_deceleration', 0):.1f} m/s²")
            
        elif signal_info.get('violation_likely', False):
            violation_entry = {
                'timestamp': timestamp,
                'relative_time': timestamp - self.start_time,
                'violation_type': 'LIKELY',
                'signal_type': signal_info.get('signal_type', 'Unknown'),
                'distance': signal_info.get('distance', 0.0),
                'required_deceleration': signal_info.get('required_deceleration', 0.0),
                'ego_speed_kmh': ego_speed * 3.6,
                'severity': 'HIGH'
            }
            self.signal_violations.append(violation_entry)
            self.violation_warnings.append(violation_entry)
            print(f"⚠️  SIGNAL VIOLATION (LIKELY): {signal_info.get('signal_type')} at {signal_info.get('distance', 0):.1f}m")
            print(f"   Required deceleration: {signal_info.get('required_deceleration', 0):.1f} m/s²")
        
    def get_alert_lead_time(self) -> Optional[float]:
        """
        Calculate alert lead time (first_alert - hazard_detection)
        
        Returns:
            Lead time in seconds, or None if data incomplete
        """
        if self.hazard_first_detected and self.first_alert_issued:
            lead_time = self.first_alert_issued - self.hazard_first_detected
            return max(0.0, lead_time)  # Ensure non-negative
        return None
        
    def get_safety_outcome(self) -> str:
        """
        Classify safety outcome based on minimum distance achieved
        
        Returns:
            'collision', 'near_miss', 'safe', or 'no_data'
        """
        valid_distances = [entry['distance'] for entry in self.ttc_data if entry['distance'] is not None]
        
        if not valid_distances:
            return 'no_data'
            
        min_distance = min(valid_distances)
        
        if min_distance <= 0.5:
            return 'collision'
        elif min_distance <= 2.0:
            return 'near_miss'
        else:
            return 'safe'
            
    def get_enhanced_safety_outcome(self) -> Dict:
        """
        Enhanced safety analysis with margin details
        
        Returns:
            Dictionary with detailed safety assessment
        """
        valid_distances = [entry['distance'] for entry in self.ttc_data if entry['distance'] is not None]
        
        if not valid_distances:
            return {
                'basic_outcome': 'no_data',
                'minimum_distance': None,
                'safety_margin': None,
                'margin_quality': 'no_data',
                'safety_score': 0.0
            }
            
        min_distance = min(valid_distances)
        minimum_safe_distance = 2.0  # Baseline safe threshold
        safety_margin = min_distance - minimum_safe_distance
        
        # Determine outcome and quality
        if min_distance <= 0.5:
            basic_outcome = 'collision'
            margin_quality = 'collision'
            safety_score = 0.0
            risk_level = 'critical'
        elif min_distance <= 2.0:
            basic_outcome = 'near_miss'
            margin_quality = 'marginal'
            safety_score = min_distance / 2.0  # 0.25-1.0 range
            risk_level = 'high'
        elif min_distance <= 4.0:
            basic_outcome = 'safe'
            margin_quality = 'adequate'
            safety_score = min(1.0, min_distance / 4.0)
            risk_level = 'moderate'
        else:
            basic_outcome = 'safe'
            margin_quality = 'excellent'
            safety_score = 1.0
            risk_level = 'low'
        
        return {
            'basic_outcome': basic_outcome,
            'minimum_distance': min_distance,
            'safety_margin': safety_margin,
            'margin_quality': margin_quality,
            'safety_score': safety_score,
            'risk_level': risk_level
        }
        
    def classify_alert_timing(self) -> Dict:
        """
        Classify whether alerts were issued at appropriate times
        
        Returns:
            Dictionary with alert timing classification and analysis
        """
        if not self.alert_events:
            return {
                'timing_classification': 'no_alerts',
                'first_alert_distance': None,
                'timing_quality': 'missed' if self.get_safety_outcome() in ['collision', 'near_miss'] else 'no_alert_needed',
                'analysis': 'No alerts were issued during this scenario'
            }
        
        # Get first alert and corresponding distance
        first_alert = self.alert_events[0]
        first_alert_distance = first_alert.get('distance_at_alert', float('inf'))
        
        # Get driver state at first alert for context
        driver_state = first_alert.get('driver_state', 'unknown')
        is_attentive = driver_state == 'safe_driving'
        
        # Get safety outcome for context
        safety_outcome = self.get_safety_outcome()
        
        # Classify timing appropriateness
        if safety_outcome == 'collision' and first_alert_distance < 3.0:
            timing_classification = 'too_late'
            timing_quality = 'poor'
            analysis = f"Alert issued too late at {first_alert_distance:.1f}m - insufficient time to prevent collision"
            
        elif first_alert_distance > 25.0 and is_attentive:
            timing_classification = 'too_early'
            timing_quality = 'questionable'
            analysis = f"Alert may be unnecessary - issued at {first_alert_distance:.1f}m for attentive driver"
            
        elif 5.0 <= first_alert_distance <= 20.0:
            timing_classification = 'appropriate'
            timing_quality = 'good'
            analysis = f"Alert timing appropriate - issued at {first_alert_distance:.1f}m with adequate response time"
            
        elif first_alert_distance < 5.0:
            timing_classification = 'late'
            timing_quality = 'fair'
            analysis = f"Alert somewhat late - issued at {first_alert_distance:.1f}m, limited response time"
            
        else:  # first_alert_distance > 20.0 and not attentive
            timing_classification = 'early_appropriate'
            timing_quality = 'good'
            analysis = f"Early alert appropriate for {driver_state} driver - issued at {first_alert_distance:.1f}m"
        
        return {
            'timing_classification': timing_classification,
            'first_alert_distance': first_alert_distance,
            'driver_state_at_alert': driver_state,
            'timing_quality': timing_quality,
            'safety_outcome': safety_outcome,
            'analysis': analysis,
            'total_alerts': len(self.alert_events)
        }
        
    def measure_alert_effectiveness(self) -> Dict:
        """
        Measure how effective alerts were at improving safety outcomes
        
        Returns:
            Dictionary with alert effectiveness analysis
        """
        safety_outcome = self.get_safety_outcome()
        enhanced_safety = self.get_enhanced_safety_outcome()
        minimum_distance = enhanced_safety.get('minimum_distance')
        alerts_issued = len(self.alert_events) > 0
        
        # Determine driver context
        if self.alert_events:
            first_alert_driver_state = self.alert_events[0].get('driver_state', 'unknown')
        else:
            # Check TTC data for driver state if available
            driver_states = [entry.get('driver_state', 'unknown') for entry in self.ttc_data if entry.get('driver_state')]
            first_alert_driver_state = driver_states[0] if driver_states else 'unknown'
        
        is_attentive = first_alert_driver_state == 'safe_driving'
        
        # Classify alert effectiveness
        if alerts_issued and safety_outcome == 'safe' and minimum_distance is not None and minimum_distance < 8.0:
            # Alert issued and dangerous situation was safely navigated
            if minimum_distance < 2.0:
                effectiveness = 'highly_effective'
                impact_level = 'high'
                analysis = f"Alert prevented collision - achieved {minimum_distance:.1f}m safety margin"
            elif minimum_distance < 5.0:
                effectiveness = 'effective'
                impact_level = 'medium'
                analysis = f"Alert prevented near miss - achieved {minimum_distance:.1f}m safety margin"
            else:
                effectiveness = 'moderately_effective'
                impact_level = 'low'
                analysis = f"Alert provided safety benefit - achieved {minimum_distance:.1f}m margin"
                
        elif alerts_issued and safety_outcome == 'collision':
            effectiveness = 'insufficient'
            impact_level = 'none'
            analysis = "Alert was issued but collision still occurred - insufficient intervention"
            
        elif alerts_issued and safety_outcome == 'safe' and minimum_distance is not None and minimum_distance > 15.0:
            if is_attentive:
                effectiveness = 'false_positive'
                impact_level = 'negative'
                analysis = f"Unnecessary alert for attentive driver - no real danger at {minimum_distance:.1f}m"
            else:
                effectiveness = 'precautionary'
                impact_level = 'low'
                analysis = f"Precautionary alert for {first_alert_driver_state} driver at {minimum_distance:.1f}m"
                
        elif not alerts_issued and safety_outcome == 'collision':
            effectiveness = 'missed_opportunity'
            impact_level = 'critical_miss'
            analysis = "No alert issued and collision occurred - critical system failure"
            
        elif not alerts_issued and safety_outcome == 'near_miss':
            effectiveness = 'missed_intervention'
            impact_level = 'moderate_miss'
            analysis = f"No alert for near miss scenario - potential intervention opportunity missed"
            
        elif not alerts_issued and safety_outcome == 'safe':
            if is_attentive and minimum_distance is not None and minimum_distance > 10.0:
                effectiveness = 'correctly_suppressed'
                impact_level = 'appropriate'
                analysis = "Correctly no alert for safe scenario with attentive driver"
            else:
                effectiveness = 'lucky_safe'
                impact_level = 'questionable'
                analysis = f"No alert but safe outcome - potentially risky for {first_alert_driver_state} driver"
        else:
            effectiveness = 'indeterminate'
            impact_level = 'unknown'
            analysis = "Unable to determine alert effectiveness from available data"
        
        return {
            'effectiveness_classification': effectiveness,
            'impact_level': impact_level,
            'safety_outcome': safety_outcome,
            'minimum_distance': minimum_distance,
            'alerts_issued': alerts_issued,
            'total_alerts': len(self.alert_events),
            'driver_state': first_alert_driver_state,
            'analysis': analysis
        }
        
    def analyze_alert_precision(self) -> Dict:
        """
        Analyze alert precision - false positives and false negatives
        
        Returns:
            Dictionary with precision/recall analysis
        """
        safety_outcome = self.get_safety_outcome()
        enhanced_safety = self.get_enhanced_safety_outcome()
        minimum_distance = enhanced_safety.get('minimum_distance')
        alerts_issued = len(self.alert_events) > 0
        
        # Define what constitutes "real danger" (threshold where alert is justified)
        danger_threshold = 8.0  # meters - if minimum distance < 8m, alert was justified
        
        # Classify precision
        if alerts_issued and minimum_distance is not None and minimum_distance < danger_threshold:
            # Alert issued and real danger existed
            classification = 'true_positive'
            precision_quality = 'correct'
            if safety_outcome == 'safe':
                precision_analysis = f"Correct alert - prevented dangerous situation (min distance: {minimum_distance:.1f}m)"
            else:
                precision_analysis = f"Correct alert - real danger existed (min distance: {minimum_distance:.1f}m)"
                
        elif alerts_issued and minimum_distance is not None and minimum_distance >= danger_threshold:
            # Alert issued but no real danger
            classification = 'false_positive'
            precision_quality = 'incorrect'
            precision_analysis = f"False alarm - no real danger at {minimum_distance:.1f}m distance"
            
        elif not alerts_issued and safety_outcome in ['collision', 'near_miss']:
            # No alert but dangerous outcome
            classification = 'false_negative'
            precision_quality = 'missed'
            precision_analysis = f"Missed alert - {safety_outcome} occurred without warning"
            
        elif not alerts_issued and safety_outcome == 'safe' and minimum_distance is not None and minimum_distance >= danger_threshold:
            # No alert and no danger - correct suppression
            classification = 'true_negative'
            precision_quality = 'correct'
            precision_analysis = f"Correct suppression - no alert needed for safe scenario ({minimum_distance:.1f}m)"
            
        else:
            # Edge cases or insufficient data
            classification = 'indeterminate'
            precision_quality = 'unclear'
            precision_analysis = "Unable to determine alert precision from available data"
        
        # Calculate precision metrics
        if classification == 'true_positive':
            alert_necessity = 'high'
            false_positive_risk = 0.0
        elif classification == 'false_positive':
            alert_necessity = 'low'
            false_positive_risk = 1.0
        elif classification == 'false_negative':
            alert_necessity = 'critical'
            false_positive_risk = 0.0
        elif classification == 'true_negative':
            alert_necessity = 'none'
            false_positive_risk = 0.0
        else:
            alert_necessity = 'unknown'
            false_positive_risk = 0.5
        
        return {
            'precision_classification': classification,
            'precision_quality': precision_quality,
            'alert_necessity': alert_necessity,
            'false_positive_risk': false_positive_risk,
            'danger_threshold_used': danger_threshold,
            'minimum_distance': minimum_distance,
            'safety_outcome': safety_outcome,
            'alerts_issued': alerts_issued,
            'analysis': precision_analysis
        }
        
    def get_emergency_events(self) -> Dict:
        """
        Detect emergency situations based on risk levels
        
        Returns:
            Dictionary with emergency event analysis
        """
        emergency_events = []
        high_risk_events = []
        
        for entry in self.ttc_data:
            ttc_risk = entry.get('ttc_risk', 0.0)
            
            if ttc_risk >= 0.8:  # Emergency threshold
                emergency_events.append({
                    'timestamp': entry['timestamp'],
                    'distance': entry.get('distance'),
                    'ttc_risk': ttc_risk,
                    'ttc_value': entry.get('ttc'),
                    'situation': 'emergency_braking_needed'
                })
            elif ttc_risk >= 0.5:  # High risk threshold
                high_risk_events.append({
                    'timestamp': entry['timestamp'],
                    'distance': entry.get('distance'),
                    'ttc_risk': ttc_risk
                })
        
        # Calculate summary statistics
        all_risks = [entry.get('ttc_risk', 0.0) for entry in self.ttc_data]
        peak_risk = max(all_risks) if all_risks else 0.0
        
        return {
            'emergency_events_count': len(emergency_events),
            'high_risk_events_count': len(high_risk_events),
            'closest_emergency_distance': min([e['distance'] for e in emergency_events if e['distance'] is not None]) if emergency_events else None,
            'peak_risk_reached': peak_risk,
            'time_in_emergency_zone': len(emergency_events),  # Frames in emergency
            'time_in_danger_zone': len(emergency_events) + len(high_risk_events),  # Frames in danger
            'emergency_events': emergency_events[:5]  # Keep first 5 for details
        }
        
    def get_experiment_metrics(self) -> Dict:
        """
        Get key metrics for experiment analysis
        
        Returns:
            Dictionary with key metrics for ablation study
        """
        stats = self.get_summary_stats()
        enhanced_safety = self.get_enhanced_safety_outcome()
        emergency_analysis = self.get_emergency_events()
        
        # NEW: Alert effectiveness analysis
        alert_timing = self.classify_alert_timing()
        alert_effectiveness = self.measure_alert_effectiveness()
        alert_precision = self.analyze_alert_precision()
        
        return {
            # Basic safety metrics
            'safety_outcome': self.get_safety_outcome(),
            'minimum_distance': stats.get('min_distance'),
            
            # Enhanced safety analysis
            'enhanced_safety': enhanced_safety,
            'safety_margin': enhanced_safety.get('safety_margin'),
            'margin_quality': enhanced_safety.get('margin_quality'),
            'safety_score': enhanced_safety.get('safety_score'),
            
            # Emergency events analysis
            'emergency_events_count': emergency_analysis.get('emergency_events_count', 0),
            'peak_risk_reached': emergency_analysis.get('peak_risk_reached', 0.0),
            'time_in_danger_zone': emergency_analysis.get('time_in_danger_zone', 0),
            
            # Alert timing analysis
            'alert_timing_classification': alert_timing.get('timing_classification'),
            'first_alert_distance': alert_timing.get('first_alert_distance'),
            'timing_quality': alert_timing.get('timing_quality'),
            
            # Alert effectiveness analysis
            'alert_effectiveness': alert_effectiveness.get('effectiveness_classification'),
            'effectiveness_impact_level': alert_effectiveness.get('impact_level'),
            
            # Alert precision analysis
            'alert_precision': alert_precision.get('precision_classification'),
            'alert_necessity': alert_precision.get('alert_necessity'),
            'false_positive_risk': alert_precision.get('false_positive_risk'),
            
            # Alert metrics
            'alert_lead_time': self.get_alert_lead_time(),
            'total_alerts': stats.get('total_alerts_issued', 0),
            
            # Experiment context
            'emergency_vehicle_detected': stats.get('emergency_vehicle_encounters', 0) > 0,
            'experiment_condition': self.experiment_condition,
            'participant_id': self.participant_id,
            'trial_number': self.trial_number,
            'scenario_name': self.scenario_name
        }
        
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of logged TTC data"""
        if not self.ttc_data:
            return {'error': 'No TTC data logged'}
            
        # Filter out None TTC values for statistics
        valid_ttc_values = [entry['ttc'] for entry in self.ttc_data if entry['ttc'] is not None]
        
        # Filter out None distance values for statistics
        valid_distances = [entry['distance'] for entry in self.ttc_data if entry['distance'] is not None]
        
        ttc_risks = [entry.get('ttc_risk', 0.0) for entry in self.ttc_data]
        
        stats = {
            'total_logged_events': len(self.ttc_data),
            'events_with_valid_ttc': len(valid_ttc_values),
            'events_with_distance_only': len(self.ttc_data) - len(valid_ttc_values),
            'events_with_valid_distance': len(valid_distances),
            'min_distance': min(valid_distances) if valid_distances else None,
            'mean_distance': sum(valid_distances) / len(valid_distances) if valid_distances else None,
            'max_distance': max(valid_distances) if valid_distances else None,
            'emergency_vehicle_encounters': len([e for e in self.ttc_data if e.get('is_emergency', False)]),
            'max_ttc_risk': max(ttc_risks) if ttc_risks else 0,
            'mean_ttc_risk': sum(ttc_risks) / len(ttc_risks) if ttc_risks else 0,
            # Alert timing metrics
            'total_alerts_issued': len(self.alert_events),
            'alert_lead_time_seconds': self.get_alert_lead_time(),
            'hazard_detected': self.hazard_first_detected is not None,
            'alerts_by_level': {
                'caution': len([a for a in self.alert_events if a['alert_level'] == 'caution']),
                'warning': len([a for a in self.alert_events if a['alert_level'] == 'warning']),
                'critical': len([a for a in self.alert_events if a['alert_level'] == 'critical']),
                'emergency': len([a for a in self.alert_events if a['alert_level'] == 'emergency'])
            }
        }
        
        if valid_ttc_values:
            stats.update({
                'min_ttc': min(valid_ttc_values),
                'mean_ttc': sum(valid_ttc_values) / len(valid_ttc_values),
                'max_ttc': max(valid_ttc_values)
            })
        else:
            stats.update({
                'min_ttc': None,
                'mean_ttc': None,
                'max_ttc': None
            })
            
        return stats
        
    def save_data(self) -> str:
        """Save TTC data to JSON file in the specified absolute path"""
        # Create directory with absolute path
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Create filename
        filename = f"ttc_{self.scenario_name}_{self.experiment_condition}_{self.participant_id}_T{self.trial_number}_{self.session_id}.json"
        filepath = os.path.join(self.log_directory, filename)
        
        # Prepare data for saving
        save_data = {
            'metadata': {
                'experiment_condition': self.experiment_condition,
                'participant_id': self.participant_id,
                'trial_number': self.trial_number,
                'session_id': self.session_id,
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - self.start_time,
                'log_directory': self.log_directory,
                'hazard_first_detected': self.hazard_first_detected,
                'first_alert_issued': self.first_alert_issued,
                'alert_lead_time': self.get_alert_lead_time(),
                'scenario_name': self.scenario_name
            },
            'summary_stats': self.get_summary_stats(),
            'ttc_data': self.ttc_data,
            'alert_events': self.alert_events,
            'timing_analysis': {
                'total_alerts': len(self.alert_events),
                'alert_levels': [alert['alert_level'] for alert in self.alert_events],
                'alert_types': [alert['alert_type'] for alert in self.alert_events]
            },
            'experiment_metrics': self.get_experiment_metrics(),
            'enhanced_safety_analysis': {
                'safety_outcome': self.get_enhanced_safety_outcome(),
                'emergency_events': self.get_emergency_events()
            },
            'alert_effectiveness_analysis': {
                'timing_classification': self.classify_alert_timing(),
                'effectiveness_measurement': self.measure_alert_effectiveness(),
                'precision_analysis': self.analyze_alert_precision()
            },
            'signal_compliance_analysis': {
                'total_signal_events': len(self.signal_events),
                'total_violations': len(self.signal_violations),
                'violation_warnings': len(self.violation_warnings),
                'signal_events': self.signal_events,
                'signal_violations': self.signal_violations,
                'violation_summary': self.get_signal_violation_summary()
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"📁 TTC data saved: {filepath}")
        return filepath
        
    def generate_report(self) -> str:
        """Generate a simple text report"""
        stats = self.get_summary_stats()
        enhanced_safety = self.get_enhanced_safety_outcome()
        emergency_analysis = self.get_emergency_events()
        
        # Alert effectiveness analysis for Phase 1
        alert_timing = self.classify_alert_timing()
        alert_effectiveness = self.measure_alert_effectiveness()
        alert_precision = self.analyze_alert_precision()
        
        if 'error' in stats:
            return f"No TTC data available for {self.experiment_condition}"
            
        # Handle None values gracefully
        min_ttc = stats.get('min_ttc')
        mean_ttc = stats.get('mean_ttc') 
        max_ttc = stats.get('max_ttc')
        
        min_ttc_str = f"{min_ttc:.2f} seconds" if min_ttc is not None else "N/A"
        mean_ttc_str = f"{mean_ttc:.2f} seconds" if mean_ttc is not None else "N/A"
        max_ttc_str = f"{max_ttc:.2f} seconds" if max_ttc is not None else "N/A"
        
        report = f"""
=== SIMPLE TTC LOGGER REPORT ===
Experiment: {self.experiment_condition}
Participant: {self.participant_id}
Trial: {self.trial_number}
Session: {self.session_id}
Log Directory: {self.log_directory}
Scenario: {self.scenario_name}

=== LOGGING STATISTICS ===
Total Logged Events: {stats['total_logged_events']}
Events with Valid TTC: {stats['events_with_valid_ttc']}
Events with Distance Only: {stats['events_with_distance_only']}
Events with Valid Distance: {stats['events_with_valid_distance']}

=== TTC STATISTICS ===
Minimum TTC: {min_ttc_str}
Average TTC: {mean_ttc_str}
Maximum TTC: {max_ttc_str}

=== DISTANCE STATISTICS ===
Minimum Distance: {f"{stats['min_distance']:.2f} m" if stats['min_distance'] is not None else "N/A"}
Average Distance: {f"{stats['mean_distance']:.2f} m" if stats['mean_distance'] is not None else "N/A"}
Maximum Distance: {f"{stats['max_distance']:.2f} m" if stats['max_distance'] is not None else "N/A"}
Emergency Vehicle Encounters: {stats['emergency_vehicle_encounters']}

=== ALERT TIMING ANALYSIS ===
Total Alerts Issued: {stats['total_alerts_issued']}
Hazard Detected: {'Yes' if stats['hazard_detected'] else 'No'}
Alert Lead Time: {f"{stats['alert_lead_time_seconds']:.2f} seconds" if stats['alert_lead_time_seconds'] else "N/A"}

Alert Breakdown:
  - Caution: {stats['alerts_by_level']['caution']}
  - Warning: {stats['alerts_by_level']['warning']}
  - Critical: {stats['alerts_by_level']['critical']}
  - Emergency: {stats['alerts_by_level']['emergency']}

=== ENHANCED SAFETY ANALYSIS ===
Safety Outcome: {enhanced_safety['basic_outcome'].upper()}
Safety Margin: {f"{enhanced_safety['safety_margin']:.2f} m" if enhanced_safety['safety_margin'] is not None else "N/A"}
Margin Quality: {enhanced_safety['margin_quality'].title()}
Safety Score: {enhanced_safety['safety_score']:.2f}

Emergency Events: {emergency_analysis['emergency_events_count']}
Peak Risk Reached: {emergency_analysis['peak_risk_reached']:.3f}
Time in Danger Zone: {emergency_analysis['time_in_danger_zone']} frames

=== ALERT EFFECTIVENESS ANALYSIS ===
Alert Timing: {alert_timing['timing_classification'].replace('_', ' ').title()} ({alert_timing['timing_quality'].title()})
First Alert Distance: {f"{alert_timing['first_alert_distance']:.1f} m" if alert_timing['first_alert_distance'] is not None else "N/A"}
Driver State at Alert: {alert_timing.get('driver_state_at_alert', 'N/A').replace('_', ' ').title()}

Alert Effectiveness: {alert_effectiveness['effectiveness_classification'].replace('_', ' ').title()} ({alert_effectiveness['impact_level'].replace('_', ' ').title()})
Alert Precision: {alert_precision['precision_classification'].replace('_', ' ').title()} ({alert_precision['precision_quality'].title()})
Alert Necessity: {alert_precision['alert_necessity'].title()}
False Positive Risk: {alert_precision['false_positive_risk']:.1f}

Timing Analysis: {alert_timing.get('analysis', 'N/A')}
Effectiveness Analysis: {alert_effectiveness.get('analysis', 'N/A')}

=== SIGNAL COMPLIANCE ANALYSIS ===
Total Signal Events: {len(self.signal_events)}
Signal Violations: {len(self.signal_violations)}
Violation Warnings: {len(self.violation_warnings)}
"""
        
        # Add signal violation details if any exist
        if self.signal_violations:
            violation_summary = self.get_signal_violation_summary()
            report += f"""
Violation Rate: {violation_summary['violation_rate']:.1%}
Max Required Deceleration: {violation_summary['max_deceleration_required']:.1f} m/s²
Average Violation Distance: {violation_summary['average_violation_distance']:.1f} m

Violations by Signal Type:"""
            for signal_type, count in violation_summary['by_signal_type'].items():
                report += f"\n  {signal_type}: {count}"
                
            report += f"\n\nViolations by Severity:"
            for severity, count in violation_summary['by_severity'].items():
                report += f"\n  {severity}: {count}"
        else:
            report += "\nNo signal violations detected ✅"
            
        report += "\n"

        return report 

    def get_signal_violation_summary(self) -> Dict:
        """
        Generate summary statistics for signal compliance violations
        
        Returns:
            Dictionary with violation statistics
        """
        if not self.signal_events:
            return {
                'total_events': 0,
                'violations': 0,
                'violation_rate': 0.0,
                'by_signal_type': {},
                'by_severity': {},
                'max_deceleration_required': 0.0,
                'average_violation_distance': 0.0
            }
        
        # Count violations by type and severity
        violations_by_type = {}
        violations_by_severity = {}
        max_decel = 0.0
        violation_distances = []
        
        for violation in self.signal_violations:
            # Count by signal type
            signal_type = violation.get('signal_type', 'Unknown')
            violations_by_type[signal_type] = violations_by_type.get(signal_type, 0) + 1
            
            # Count by severity
            severity = violation.get('severity', 'Unknown')
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
            
            # Track max deceleration and distances
            decel = violation.get('required_deceleration', 0.0)
            max_decel = max(max_decel, decel)
            violation_distances.append(violation.get('distance', 0.0))
        
        # Calculate violation rate
        total_violations = len(self.signal_violations)
        violation_rate = total_violations / len(self.signal_events) if self.signal_events else 0.0
        
        # Average violation distance
        avg_violation_distance = sum(violation_distances) / len(violation_distances) if violation_distances else 0.0
        
        return {
            'total_events': len(self.signal_events),
            'violations': total_violations,
            'violation_rate': violation_rate,
            'by_signal_type': violations_by_type,
            'by_severity': violations_by_severity,
            'max_deceleration_required': max_decel,
            'average_violation_distance': avg_violation_distance
        } 