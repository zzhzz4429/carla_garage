import json
import time
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional

class FusionCoreMetricsLogger:
    """
    Simplified logger focusing on 5 core metrics using existing safety_evaluator TTC data
    """
    
    def __init__(self, experiment_condition: str, participant_id: str, trial_number: str):
        self.experiment_condition = experiment_condition
        self.participant_id = participant_id
        self.trial_number = trial_number
        self.start_time = time.time()
        
        # Create session ID and log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{experiment_condition}_{participant_id}_{trial_number}_{timestamp}"
        self.log_dir = "/home/ascc304/carla_garage/simple_fusion_metrics"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Simple data structure - just collect the data
        self.metrics_data = {
            'experiment_info': {
                'condition': experiment_condition,
                'participant_id': participant_id,
                'trial_number': trial_number,
                'session_id': self.session_id,
                'start_time': timestamp
            },
            
            # METRIC 1: Safety violations (simple list)
            'violations': [],
            
            # METRIC 2: TTC data (use safety_evaluator results directly)
            'ttc_data': [],
            
            # METRIC 3: Alert events with timing
            'alerts': [],
            
            # METRIC 4: False positive tracking
            'false_positives': [],
            
            # METRIC 5: Driver reactions
            'reactions': []
        }
        
        # Simple state tracking
        self.last_alert_time = 0.0
        self.last_alert_level = 'safe'
        self.min_ttc_this_session = float('inf')
        self.min_distance_this_session = float('inf')
        
        print(f"📊 Simple Fusion Metrics Logger: {self.session_id}")

    # ADD MISSING METHOD FOR SENSOR AGENT COMPATIBILITY
    def update_driver_state(self, timestamp: float, driver_state: str, attention_level: float):
        """Update current driver state for context-aware analysis"""
        # Simple implementation - just store the latest state
        self.current_driver_state = driver_state
        self.driver_attention_level = attention_level
        
    def log_frame_data(self, timestamp: float, ttc_risk: float, ttc_object: Dict, 
                      unified_assessment: Dict, ego_speed: float, control_data: Dict):
        """
        Log frame data using existing safety_evaluator TTC calculations
        """
        relative_time = timestamp - self.start_time
        
        # METRIC 2: TTC data (direct from safety_evaluator)
        # Fix: Check if ttc_object exists and has valid data instead of 'detected' field
        if ttc_object and ttc_object is not None:
            ttc_value = ttc_object.get('ttc', float('inf'))
            distance = ttc_object.get('distance', float('inf'))
            
            # Only log if we have meaningful values
            if distance != float('inf') and distance > 0:
                print(f"📊 TTC LOGGING: distance={distance:.2f}m, ttc={ttc_value:.2f}s, risk={ttc_risk:.3f}")
                
                # Track minimum values
                if ttc_value != float('inf') and ttc_value > 0 and ttc_value < self.min_ttc_this_session:
                    self.min_ttc_this_session = ttc_value
                    print(f"🎯 NEW MIN TTC: {ttc_value:.2f}s")
                    
                if distance < self.min_distance_this_session:
                    self.min_distance_this_session = distance
                    print(f"🎯 NEW MIN DISTANCE: {distance:.2f}m")
                    
                ttc_entry = {
                    'timestamp': timestamp,
                    'relative_time': relative_time,
                    'ttc': ttc_value,
                    'distance': distance,
                    'ttc_risk': ttc_risk,
                    'object_type': ttc_object.get('class_name', 'unknown'),
                    'is_emergency': ttc_object.get('is_emergency', False),
                    'ego_speed': ego_speed,
                    'driver_state': unified_assessment.get('driver_state', {}).get('current_state', 'safe_driving')
                }
                self.metrics_data['ttc_data'].append(ttc_entry)
                
                # METRIC 1: Simple collision detection
                if distance < 1.5:  # Simple threshold
                    self.log_violation(timestamp, 'collision', {
                        'distance': distance,
                        'ttc': ttc_value,
                        'object_type': ttc_object.get('class_name', 'unknown'),
                        'ego_speed': ego_speed
                    })
            else:
                print(f"🚫 SKIPPING TTC LOG: distance={distance}, ttc={ttc_value} (invalid values)")
        else:
            print(f"🚫 NO TTC OBJECT: ttc_object is None or empty")
                
        # METRIC 3 & 4: Alert tracking
        risk_level = unified_assessment.get('risk_level', 'safe')
        if risk_level != 'safe':
            self.log_alert(timestamp, risk_level, unified_assessment)
            
        # METRIC 5: Driver reaction detection
        self.detect_driver_reaction(timestamp, control_data, ego_speed)
        
    def log_violation(self, timestamp: float, violation_type: str, details: Dict):
        """METRIC 1: Log safety violations"""
        violation = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time,
            'type': violation_type,
            'details': details
        }
        self.metrics_data['violations'].append(violation)
        print(f"💥 VIOLATION: {violation_type}")
        
    def log_alert(self, timestamp: float, alert_level: str, unified_assessment: Dict):
        """METRIC 3 & 4: Log alerts"""
        # Check if this is a new alert (not repeated)
        if timestamp - self.last_alert_time > 2.0:  # 2 second gap between alerts
            
            # Get trigger distance
            external_breakdown = unified_assessment.get('external_breakdown', {})
            primary_threat = external_breakdown.get('primary_threat')
            trigger_distance = primary_threat.get('distance', float('inf')) if primary_threat else float('inf')
            
            alert = {
                'timestamp': timestamp,
                'relative_time': timestamp - self.start_time,
                'level': alert_level,
                'trigger_distance': trigger_distance,
                'unified_risk': unified_assessment.get('unified_risk', 0.0),
                'external_risk': unified_assessment.get('external_risk', 0.0),
                'internal_risk': unified_assessment.get('internal_risk', 0.0),
                'driver_state': unified_assessment.get('driver_state', {}).get('current_state', 'safe_driving')
            }
            
            # METRIC 4: Simple false positive detection
            if trigger_distance > 20.0 and unified_assessment.get('external_risk', 0.0) < 0.3:
                alert['is_false_positive'] = True
                self.metrics_data['false_positives'].append(alert)
            else:
                alert['is_false_positive'] = False
                
            self.metrics_data['alerts'].append(alert)
            
            # Setup for reaction tracking
            self.last_alert_time = timestamp
            self.last_alert_level = alert_level
            
            print(f"🔔 ALERT: {alert_level} at {trigger_distance:.1f}m")
            
    def detect_driver_reaction(self, timestamp: float, control_data: Dict, ego_speed: float):
        """METRIC 5: Simple driver reaction detection"""
        if self.last_alert_time == 0.0:
            return
            
        reaction_delay = timestamp - self.last_alert_time
        
        # Only check for reactions within 10 seconds of alert
        if reaction_delay > 10.0:
            return
            
        # Simple reaction detection
        significant_brake = control_data.get('brake', 0.0) > 0.3
        significant_steer = abs(control_data.get('steering', 0.0)) > 0.1
        
        if significant_brake or significant_steer:
            reaction = {
                'timestamp': timestamp,
                'relative_time': timestamp - self.start_time,
                'reaction_delay': reaction_delay,
                'alert_level': self.last_alert_level,
                'reaction_type': 'brake' if significant_brake else 'steer',
                'brake_magnitude': control_data.get('brake', 0.0),
                'steer_magnitude': abs(control_data.get('steering', 0.0))
            }
            
            self.metrics_data['reactions'].append(reaction)
            print(f"👤 REACTION: {reaction['reaction_type']} after {reaction_delay:.2f}s")
            
            # Reset tracking
            self.last_alert_time = 0.0
            
    def calculate_final_metrics(self) -> Dict:
        """Calculate the 5 core metrics simply"""
        
        trial_duration = time.time() - self.start_time
        
        # METRIC 1: Violation rate
        total_violations = len(self.metrics_data['violations'])
        violation_rate = total_violations / (trial_duration / 60.0) if trial_duration > 0 else 0.0
        
        # METRIC 2: TTC analysis (improved handling)
        ttc_values = [entry['ttc'] for entry in self.metrics_data['ttc_data'] 
                     if entry['ttc'] != float('inf') and entry['ttc'] > 0]
        
        print(f"🔍 TTC CALCULATION DEBUG:")
        print(f"   Total TTC entries: {len(self.metrics_data['ttc_data'])}")
        print(f"   Valid TTC values: {len(ttc_values)}")
        if ttc_values:
            print(f"   TTC values: {ttc_values[:5]}...")  # Show first 5
        print(f"   Min TTC this session: {self.min_ttc_this_session}")
        print(f"   Min distance this session: {self.min_distance_this_session}")
        
        # Use session minimums if available, otherwise calculate from data
        if self.min_ttc_this_session != float('inf'):
            min_ttc = self.min_ttc_this_session
        else:
            min_ttc = min(ttc_values) if ttc_values else float('inf')
            
        avg_ttc = np.mean(ttc_values) if ttc_values else float('inf')
        
        # Fix minimum distance
        if self.min_distance_this_session != float('inf'):
            min_distance = self.min_distance_this_session
        else:
            # Get from TTC data
            distances = [entry['distance'] for entry in self.metrics_data['ttc_data'] 
                        if entry['distance'] != float('inf') and entry['distance'] > 0]
            min_distance = min(distances) if distances else float('inf')
        
        # METRIC 3: Alert lead time (simple calculation)
        alerts = self.metrics_data['alerts']
        alert_count = len(alerts)
        
        # METRIC 4: False positive rate
        false_positives = len(self.metrics_data['false_positives'])
        fp_rate = false_positives / alert_count if alert_count > 0 else 0.0
        
        # METRIC 5: Reaction performance
        reactions = self.metrics_data['reactions']
        reaction_delays = [r['reaction_delay'] for r in reactions]
        avg_reaction_delay = np.mean(reaction_delays) if reaction_delays else float('inf')
        
        # Reaction by urgency
        urgent_reactions = [r for r in reactions if r['alert_level'] in ['critical', 'emergency']]
        urgent_avg_delay = np.mean([r['reaction_delay'] for r in urgent_reactions]) if urgent_reactions else float('inf')
        
        return {
            'experiment_condition': self.experiment_condition,
            'trial_duration_minutes': trial_duration / 60.0,
            
            # METRIC 1: Violations
            'metric_1_violations': {
                'total_count': total_violations,
                'rate_per_minute': violation_rate,
                'types': [v['type'] for v in self.metrics_data['violations']]
            },
            
            # METRIC 2: TTC Safety Margin (FIXED)
            'metric_2_ttc': {
                'minimum_ttc': min_ttc,
                'average_ttc': avg_ttc,
                'minimum_distance': min_distance,
                'valid_measurements': len(ttc_values),
                'total_entries': len(self.metrics_data['ttc_data'])
            },
            
            # METRIC 3: Alert Timing
            'metric_3_alerts': {
                'total_alerts': alert_count,
                'average_trigger_distance': np.mean([a['trigger_distance'] for a in alerts if a['trigger_distance'] != float('inf')]) if alerts else 0.0
            },
            
            # METRIC 4: False Positives
            'metric_4_false_positives': {
                'count': false_positives,
                'rate': fp_rate,
                'percentage': fp_rate * 100
            },
            
            # METRIC 5: Driver Reactions
            'metric_5_reactions': {
                'total_reactions': len(reactions),
                'average_delay': avg_reaction_delay,
                'urgent_average_delay': urgent_avg_delay,
                'fast_reactions': len([r for r in reactions if r['reaction_delay'] <= 2.0])
            }
        }
        
    def generate_simple_report(self) -> str:
        """Generate simple, focused report with better formatting"""
        metrics = self.calculate_final_metrics()
        
        # Format infinite values better
        def format_value(value, unit="", decimal_places=2):
            if value == float('inf'):
                return f"No data"
            else:
                return f"{value:.{decimal_places}f}{unit}"
        
        return f"""
🎯 SIMPLE FUSION METRICS REPORT
================================

Condition: {self.experiment_condition.upper()}
Participant: {self.participant_id} | Trial: {self.trial_number}
Duration: {metrics['trial_duration_minutes']:.1f} minutes

METRIC 1 - SAFETY VIOLATIONS: {metrics['metric_1_violations']['total_count']} 
  Rate: {metrics['metric_1_violations']['rate_per_minute']:.2f} per minute

METRIC 2 - MINIMUM TTC: {format_value(metrics['metric_2_ttc']['minimum_ttc'], 's')}
  Average TTC: {format_value(metrics['metric_2_ttc']['average_ttc'], 's')}
  Minimum Distance: {format_value(metrics['metric_2_ttc']['minimum_distance'], 'm')}
  Valid Measurements: {metrics['metric_2_ttc']['valid_measurements']} / {metrics['metric_2_ttc']['total_entries']}

METRIC 3 - ALERT PERFORMANCE: {metrics['metric_3_alerts']['total_alerts']} alerts
  Avg Trigger Distance: {metrics['metric_3_alerts']['average_trigger_distance']:.1f}m

METRIC 4 - FALSE POSITIVES: {metrics['metric_4_false_positives']['count']} 
  Rate: {metrics['metric_4_false_positives']['percentage']:.1f}%

METRIC 5 - DRIVER REACTIONS: {metrics['metric_5_reactions']['total_reactions']} reactions
  Average Delay: {format_value(metrics['metric_5_reactions']['average_delay'], 's')}
  Urgent Delay: {format_value(metrics['metric_5_reactions']['urgent_average_delay'], 's')}
  Fast Reactions: {metrics['metric_5_reactions']['fast_reactions']}

================================
"""
        
    def save_core_metrics(self) -> str:
        """Save metrics data"""
        self.metrics_data['final_metrics'] = self.calculate_final_metrics()
        
        filename = f"simple_fusion_metrics_{self.session_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics_data, f, indent=2, default=str)
            print(f"✅ Simple metrics saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            return "" 