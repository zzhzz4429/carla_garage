import json
import time
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import copy

class FusionExperimentLogger:
    """
    Comprehensive data logger for fusion system validation experiments
    Integrates with UnifiedRiskManager, SafetyEvaluator, and SensorAgent
    """
    
    def __init__(self, experiment_condition: str, participant_id: str, trial_number: str, scenario_name: str = ""):
        """
        Initialize the fusion experiment logger
        
        Args:
            experiment_condition: 'baseline', 'external_only', 'internal_only', 'unified'
            participant_id: Unique participant identifier
            trial_number: Trial number for this participant
            scenario_name: Name of the scenario being tested
        """
        self.experiment_condition = experiment_condition
        self.participant_id = participant_id
        self.trial_number = trial_number
        self.scenario_name = scenario_name
        self.start_time = time.time()
        
        # Create unique log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{experiment_condition}_{participant_id}_{trial_number}_{timestamp}"
        
        # Create log directory
        self.log_dir = "/home/ascc304/carla_garage/fusion_experiment_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Main data structure
        self.experiment_data = {
            'experiment_metadata': {
                'condition': experiment_condition,
                'participant_id': participant_id,
                'trial_number': trial_number,
                'scenario_name': scenario_name,
                'session_id': self.session_id,
                'start_time': timestamp,
                'start_timestamp': self.start_time
            },
            
            # CRITICAL FUSION VALIDATION DATA
            'fusion_validation': {
                'risk_predictions': [],        # Risk accuracy vs ground truth
                'weight_adaptations': [],      # When/why weights changed
                'synergy_events': [],          # When synergy factor > 1.0
                'alert_effectiveness': [],     # Alert appropriateness
                'temporal_performance': [],    # Early warning capability
                'context_awareness': []        # Different behavior per driver state
            },
            
            # FRAME-BY-FRAME DETAILED DATA
            'frame_data': [],               # All frame data
            
            # SAFETY OUTCOMES
            'safety_events': {
                'collisions': [],
                'near_misses': [],           # Distance < 2m
                'emergency_braking': [],     # Sudden deceleration > 6 m/s²
                'manual_interventions': [],
                'alert_events': []
            },
            
            # SYSTEM PERFORMANCE
            'system_metrics': {
                'processing_times': [],
                'alert_latencies': [],
                'false_positives': [],
                'false_negatives': [],
                'system_availability': []
            },
            
            # DRIVER BEHAVIOR
            'driver_behavior': {
                'steering_patterns': [],
                'speed_patterns': [],
                'reaction_times': [],
                'attention_patterns': [],
                'workload_indicators': []
            }
        }
        
        # State tracking for analysis
        self.previous_frame_data = None
        self.last_alert_time = 0.0
        self.last_weight_change_time = 0.0
        self.min_distance_this_trial = float('inf')
        self.collision_detected = False
        
        # Ground truth tracking
        self.ground_truth_danger_events = []
        self.current_danger_level = 0.0
        
        # Performance baselines for comparison
        self.baseline_risks = {'external_only': [], 'internal_only': [], 'simple_average': []}
        
        print(f"🔬 Fusion Experiment Logger initialized: {self.session_id}")
        print(f"📁 Logs will be saved to: {self.log_dir}")
        
    def log_frame_data(self, timestamp: float, ego_state: Dict, unified_assessment: Dict, 
                      control_data: Dict, detected_objects: List, sensor_data: Dict = None):
        """
        Log comprehensive frame data for fusion validation
        
        Args:
            timestamp: Current timestamp
            ego_state: Ego vehicle state (position, speed, etc.)
            unified_assessment: Full unified risk assessment result
            control_data: Control commands (steering, throttle, brake)
            detected_objects: List of detected bounding boxes
            sensor_data: Additional sensor information
        """
        relative_time = timestamp - self.start_time
        
        # Extract key components for analysis
        unified_risk = unified_assessment.get('unified_risk', 0.0)
        external_risk = unified_assessment.get('external_risk', 0.0)
        internal_risk = unified_assessment.get('internal_risk', 0.0)
        risk_level = unified_assessment.get('risk_level', 'safe')
        weights = unified_assessment.get('weights', {})
        driver_state_info = unified_assessment.get('driver_state', {})
        
        # Calculate comparison baselines
        external_only_risk = external_risk
        internal_only_risk = internal_risk
        simple_average_risk = (external_risk + internal_risk) / 2.0
        
        # Create comprehensive frame entry
        frame_entry = {
            'timestamp': timestamp,
            'relative_time': relative_time,
            'frame_id': len(self.experiment_data['frame_data']),
            
            # EGO VEHICLE STATE
            'ego_state': {
                'position': ego_state.get('position', [0, 0]),
                'speed': ego_state.get('speed', 0.0),
                'acceleration': self._calculate_acceleration(ego_state.get('speed', 0.0)),
                'heading': ego_state.get('heading', 0.0)
            },
            
            # CONTROL COMMANDS
            'control': {
                'steering': control_data.get('steering', 0.0),
                'throttle': control_data.get('throttle', 0.0),
                'brake': control_data.get('brake', 0.0),
                'manual_override': control_data.get('manual_override', False)
            },
            
            # FUSION SYSTEM OUTPUT
            'fusion_output': {
                'unified_risk': unified_risk,
                'risk_level': risk_level,
                'external_weight': weights.get('external_weight', 0.6),
                'internal_weight': weights.get('internal_weight', 0.4),
                'synergy_factor': unified_assessment.get('risk_factors', {}).get('synergy_factor', 1.0),
                'emergency_boost': unified_assessment.get('risk_factors', {}).get('emergency_boost', 0.0)
            },
            
            # COMPONENT RISKS (for comparison)
            'component_risks': {
                'external_risk': external_risk,
                'internal_risk': internal_risk,
                'external_only_prediction': external_only_risk,
                'internal_only_prediction': internal_only_risk,
                'simple_average_prediction': simple_average_risk
            },
            
            # DRIVER STATE
            'driver_state': {
                'current_state': driver_state_info.get('current_state', 'unknown'),
                'confidence': driver_state_info.get('confidence', 0.0),
                'is_stale': driver_state_info.get('is_stale', True),
                'last_update': driver_state_info.get('last_update', 0.0)
            },
            
            # DETECTED OBJECTS & THREATS
            'perception': {
                'total_objects': len(detected_objects),
                'emergency_vehicles': self._count_objects_by_class(detected_objects, 4),
                'vehicles': self._count_objects_by_class(detected_objects, 0),
                'pedestrians': self._count_objects_by_class(detected_objects, 1),
                'traffic_signals': self._count_objects_by_class(detected_objects, [2, 3]),
                'closest_object_distance': self._get_closest_object_distance(detected_objects),
                'primary_threat': self._extract_primary_threat_info(unified_assessment)
            },
            
            # TEMPORAL CONTEXT
            'temporal_context': unified_assessment.get('temporal_context', {}),
            
            # EXPERIMENTAL CONDITION
            'experiment_condition': self.experiment_condition
        }
        
        # Add to frame data
        self.experiment_data['frame_data'].append(frame_entry)
        
        # Analyze this frame for fusion validation
        self._analyze_frame_for_fusion_validation(frame_entry, unified_assessment)
        
        # Update tracking variables
        self.previous_frame_data = frame_entry
        self._update_safety_tracking(frame_entry)
        
    def _analyze_frame_for_fusion_validation(self, frame_entry: Dict, unified_assessment: Dict):
        """Analyze frame data specifically for fusion system validation"""
        
        # 1. RISK PREDICTION ACCURACY ANALYSIS
        self._analyze_risk_prediction_accuracy(frame_entry, unified_assessment)
        
        # 2. WEIGHT ADAPTATION ANALYSIS
        self._analyze_weight_adaptations(frame_entry)
        
        # 3. SYNERGY DETECTION
        self._analyze_synergy_events(frame_entry)
        
        # 4. TEMPORAL PERFORMANCE
        self._analyze_temporal_performance(frame_entry)
        
        # 5. CONTEXT AWARENESS
        self._analyze_context_awareness(frame_entry)
        
    def _analyze_risk_prediction_accuracy(self, frame_entry: Dict, unified_assessment: Dict):
        """Analyze how accurately different risk prediction methods work"""
        
        # Calculate ground truth danger level based on actual proximity and threat
        ground_truth_danger = self._calculate_ground_truth_danger(frame_entry)
        
        # Compare different risk prediction approaches
        fusion_risk = frame_entry['fusion_output']['unified_risk']
        external_only = frame_entry['component_risks']['external_risk']
        internal_only = frame_entry['component_risks']['internal_risk']
        simple_average = frame_entry['component_risks']['simple_average_prediction']
        
        # Calculate prediction errors
        prediction_analysis = {
            'timestamp': frame_entry['timestamp'],
            'relative_time': frame_entry['relative_time'],
            'ground_truth_danger': ground_truth_danger,
            
            'predictions': {
                'fusion': fusion_risk,
                'external_only': external_only,
                'internal_only': internal_only,
                'simple_average': simple_average
            },
            
            'errors': {
                'fusion_error': abs(fusion_risk - ground_truth_danger),
                'external_only_error': abs(external_only - ground_truth_danger),
                'internal_only_error': abs(internal_only - ground_truth_danger),
                'simple_average_error': abs(simple_average - ground_truth_danger)
            },
            
            'fusion_improvement': {
                'vs_external_only': abs(external_only - ground_truth_danger) - abs(fusion_risk - ground_truth_danger),
                'vs_internal_only': abs(internal_only - ground_truth_danger) - abs(fusion_risk - ground_truth_danger),
                'vs_simple_average': abs(simple_average - ground_truth_danger) - abs(fusion_risk - ground_truth_danger)
            },
            
            'context': {
                'driver_state': frame_entry['driver_state']['current_state'],
                'closest_object_distance': frame_entry['perception']['closest_object_distance'],
                'emergency_vehicle_present': frame_entry['perception']['emergency_vehicles'] > 0
            }
        }
        
        self.experiment_data['fusion_validation']['risk_predictions'].append(prediction_analysis)
        
    def _analyze_weight_adaptations(self, frame_entry: Dict):
        """Analyze when and why fusion weights change"""
        
        if self.previous_frame_data is None:
            return
            
        current_ext_weight = frame_entry['fusion_output']['external_weight']
        current_int_weight = frame_entry['fusion_output']['internal_weight']
        
        prev_ext_weight = self.previous_frame_data['fusion_output']['external_weight']
        prev_int_weight = self.previous_frame_data['fusion_output']['internal_weight']
        
        # Detect significant weight changes
        weight_change_threshold = 0.05
        ext_weight_change = abs(current_ext_weight - prev_ext_weight)
        int_weight_change = abs(current_int_weight - prev_int_weight)
        
        if ext_weight_change > weight_change_threshold or int_weight_change > weight_change_threshold:
            weight_adaptation = {
                'timestamp': frame_entry['timestamp'],
                'relative_time': frame_entry['relative_time'],
                
                'weight_changes': {
                    'external_weight': {'old': prev_ext_weight, 'new': current_ext_weight, 'change': current_ext_weight - prev_ext_weight},
                    'internal_weight': {'old': prev_int_weight, 'new': current_int_weight, 'change': current_int_weight - prev_int_weight}
                },
                
                'trigger_context': {
                    'driver_state_change': frame_entry['driver_state']['current_state'] != self.previous_frame_data['driver_state']['current_state'],
                    'external_risk_level': frame_entry['component_risks']['external_risk'],
                    'internal_risk_level': frame_entry['component_risks']['internal_risk'],
                    'driver_state': frame_entry['driver_state']['current_state'],
                    'data_staleness': frame_entry['driver_state']['is_stale']
                },
                
                'adaptation_magnitude': max(ext_weight_change, int_weight_change),
                'adaptation_direction': 'external_bias' if current_ext_weight > current_int_weight else 'internal_bias'
            }
            
            self.experiment_data['fusion_validation']['weight_adaptations'].append(weight_adaptation)
            self.last_weight_change_time = frame_entry['timestamp']
            
    def _analyze_synergy_events(self, frame_entry: Dict):
        """Analyze when synergy amplification occurs"""
        
        synergy_factor = frame_entry['fusion_output']['synergy_factor']
        
        if synergy_factor > 1.0:  # Synergy activation
            synergy_event = {
                'timestamp': frame_entry['timestamp'],
                'relative_time': frame_entry['relative_time'],
                'synergy_factor': synergy_factor,
                'amplification_amount': synergy_factor - 1.0,
                
                'contributing_factors': {
                    'external_risk': frame_entry['component_risks']['external_risk'],
                    'internal_risk': frame_entry['component_risks']['internal_risk'],
                    'unified_risk_before_synergy': frame_entry['component_risks']['external_risk'] * frame_entry['fusion_output']['external_weight'] + 
                                                 frame_entry['component_risks']['internal_risk'] * frame_entry['fusion_output']['internal_weight'],
                    'unified_risk_after_synergy': frame_entry['fusion_output']['unified_risk']
                },
                
                'context': {
                    'driver_state': frame_entry['driver_state']['current_state'],
                    'emergency_vehicle_present': frame_entry['perception']['emergency_vehicles'] > 0,
                    'closest_distance': frame_entry['perception']['closest_object_distance'],
                    'emergency_boost': frame_entry['fusion_output']['emergency_boost']
                }
            }
            
            self.experiment_data['fusion_validation']['synergy_events'].append(synergy_event)
            
    def _analyze_temporal_performance(self, frame_entry: Dict):
        """Analyze early warning and temporal response capabilities"""
        
        # Track when each system would have first triggered alerts
        current_distance = frame_entry['perception']['closest_object_distance']
        
        if current_distance < 20.0 and current_distance != float('inf'):  # Within alert range
            temporal_analysis = {
                'timestamp': frame_entry['timestamp'],
                'relative_time': frame_entry['relative_time'],
                'trigger_distance': current_distance,
                
                'system_responses': {
                    'fusion_would_alert': frame_entry['fusion_output']['unified_risk'] > 0.2,
                    'external_only_would_alert': frame_entry['component_risks']['external_risk'] > 0.2,
                    'internal_only_would_alert': frame_entry['component_risks']['internal_risk'] > 0.2,
                    'fusion_risk_level': frame_entry['fusion_output']['risk_level']
                },
                
                'early_warning_analysis': {
                    'fusion_detection_time': frame_entry['timestamp'] if frame_entry['fusion_output']['unified_risk'] > 0.2 else None,
                    'external_detection_time': frame_entry['timestamp'] if frame_entry['component_risks']['external_risk'] > 0.2 else None,
                    'internal_detection_time': frame_entry['timestamp'] if frame_entry['component_risks']['internal_risk'] > 0.2 else None
                }
            }
            
            self.experiment_data['fusion_validation']['temporal_performance'].append(temporal_analysis)
            
    def _analyze_context_awareness(self, frame_entry: Dict):
        """Analyze context-aware behavior across different driver states"""
        
        context_analysis = {
            'timestamp': frame_entry['timestamp'],
            'relative_time': frame_entry['relative_time'],
            
            'context': {
                'driver_state': frame_entry['driver_state']['current_state'],
                'driver_confidence': frame_entry['driver_state']['confidence'],
                'external_threat_level': frame_entry['component_risks']['external_risk'],
                'internal_risk_level': frame_entry['component_risks']['internal_risk']
            },
            
            'system_adaptation': {
                'weight_allocation': {
                    'external_weight': frame_entry['fusion_output']['external_weight'],
                    'internal_weight': frame_entry['fusion_output']['internal_weight']
                },
                'final_risk_assessment': frame_entry['fusion_output']['unified_risk'],
                'risk_level': frame_entry['fusion_output']['risk_level']
            },
            
            'expected_vs_actual': {
                'expected_external_bias': frame_entry['driver_state']['current_state'] in ['sleepy', 'using_phone', 'reaching_back'],
                'actual_external_bias': frame_entry['fusion_output']['external_weight'] > 0.6,
                'adaptation_appropriate': self._assess_adaptation_appropriateness(frame_entry)
            }
        }
        
        self.experiment_data['fusion_validation']['context_awareness'].append(context_analysis)
        
    def log_alert_event(self, timestamp: float, alert_data: Dict, unified_assessment: Dict):
        """Log when alerts are triggered and assess their appropriateness"""
        
        alert_event = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time,
            'alert_type': alert_data.get('type', 'unknown'),
            'alert_level': alert_data.get('level', 'unknown'),
            'trigger_risk': unified_assessment.get('unified_risk', 0.0),
            'trigger_distance': alert_data.get('trigger_distance'),
            
            'appropriateness_analysis': {
                'actual_danger_present': self._assess_actual_danger(unified_assessment),
                'alert_timing': self._assess_alert_timing(timestamp, alert_data),
                'risk_level_match': self._assess_risk_level_match(alert_data, unified_assessment)
            },
            
            'context': {
                'driver_state': unified_assessment.get('driver_state', {}).get('current_state', 'unknown'),
                'external_risk': unified_assessment.get('external_risk', 0.0),
                'internal_risk': unified_assessment.get('internal_risk', 0.0),
                'emergency_vehicle_present': self._check_emergency_vehicle_present(unified_assessment)
            }
        }
        
        self.experiment_data['fusion_validation']['alert_effectiveness'].append(alert_event)
        self.experiment_data['safety_events']['alert_events'].append(alert_event)
        self.last_alert_time = timestamp
        
    def log_manual_intervention(self, timestamp: float, intervention_data: Dict, context: Dict):
        """Log manual driver interventions"""
        
        intervention_event = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time,
            'intervention_type': intervention_data.get('type', 'unknown'),
            'trigger_reason': intervention_data.get('reason', 'unknown'),
            'reaction_time': intervention_data.get('reaction_time'),
            
            'system_state_at_intervention': {
                'unified_risk': context.get('unified_risk', 0.0),
                'risk_level': context.get('risk_level', 'unknown'),
                'last_alert_time': self.last_alert_time,
                'time_since_last_alert': timestamp - self.last_alert_time if self.last_alert_time > 0 else None
            },
            
            'effectiveness_indicators': {
                'prevented_collision': intervention_data.get('prevented_collision', False),
                'improved_safety_margin': intervention_data.get('improved_margin', False),
                'intervention_necessary': self._assess_intervention_necessity(context)
            }
        }
        
        self.experiment_data['safety_events']['manual_interventions'].append(intervention_event)
        
    def _calculate_ground_truth_danger(self, frame_entry: Dict) -> float:
        """Calculate ground truth danger level based on physical proximity and threat severity"""
        
        closest_distance = frame_entry['perception']['closest_object_distance']
        ego_speed = frame_entry['ego_state']['speed']
        
        if closest_distance == float('inf'):
            return 0.0
            
        # Basic danger calculation based on distance and speed
        # Closer objects and higher speeds = more danger
        distance_factor = max(0, 1.0 - (closest_distance / 20.0))  # Danger increases as distance < 20m
        speed_factor = min(1.0, ego_speed / 20.0)  # Speed contribution (cap at 20 m/s)
        
        base_danger = distance_factor * (0.7 + 0.3 * speed_factor)
        
        # Emergency vehicle bonus
        if frame_entry['perception']['emergency_vehicles'] > 0:
            base_danger *= 1.3
            
        # Driver state amplification (if driver is impaired, same situation is more dangerous)
        driver_state = frame_entry['driver_state']['current_state']
        if driver_state in ['sleepy', 'using_phone', 'reaching_back']:
            base_danger *= 1.2
            
        return min(1.0, base_danger)
        
    def _count_objects_by_class(self, detected_objects: List, class_ids) -> int:
        """Count detected objects by class ID(s)"""
        if isinstance(class_ids, int):
            class_ids = [class_ids]
            
        count = 0
        for obj in detected_objects:
            if len(obj) > 7 and obj[7] in class_ids:
                count += 1
        return count
        
    def _get_closest_object_distance(self, detected_objects: List) -> float:
        """Get distance to closest object"""
        if not detected_objects:
            return float('inf')
            
        min_distance = float('inf')
        for obj in detected_objects:
            if len(obj) > 0:
                distance = obj[0]
                if distance < min_distance:
                    min_distance = distance
                    
        return min_distance
        
    def _extract_primary_threat_info(self, unified_assessment: Dict) -> Dict:
        """Extract primary threat information"""
        external_breakdown = unified_assessment.get('external_breakdown', {})
        primary_threat = external_breakdown.get('primary_threat')
        
        if primary_threat:
            return {
                'present': True,
                'type': primary_threat.get('class_name', 'unknown'),
                'distance': primary_threat.get('distance', float('inf')),
                'is_emergency': primary_threat.get('is_emergency', False),
                'ttc': primary_threat.get('ttc', float('inf'))
            }
        else:
            return {'present': False}
            
    def _calculate_acceleration(self, current_speed: float) -> float:
        """Calculate acceleration from speed change"""
        if self.previous_frame_data is None:
            return 0.0
            
        prev_speed = self.previous_frame_data['ego_state']['speed']
        dt = 0.1  # Assuming 10Hz frame rate
        return (current_speed - prev_speed) / dt
        
    def _update_safety_tracking(self, frame_entry: Dict):
        """Update safety event tracking"""
        current_distance = frame_entry['perception']['closest_object_distance']
        
        # Update minimum distance
        if current_distance < self.min_distance_this_trial:
            self.min_distance_this_trial = current_distance
            
        # Detect near-miss events
        if current_distance < 2.0 and current_distance != float('inf'):
            near_miss = {
                'timestamp': frame_entry['timestamp'],
                'relative_time': frame_entry['relative_time'],
                'distance': current_distance,
                'ego_speed': frame_entry['ego_state']['speed'],
                'unified_risk': frame_entry['fusion_output']['unified_risk'],
                'driver_state': frame_entry['driver_state']['current_state']
            }
            self.experiment_data['safety_events']['near_misses'].append(near_miss)
            
        # Detect emergency braking
        acceleration = frame_entry['ego_state']['acceleration']
        if acceleration < -6.0:  # Emergency braking threshold
            emergency_brake = {
                'timestamp': frame_entry['timestamp'],
                'relative_time': frame_entry['relative_time'],
                'deceleration': abs(acceleration),
                'trigger_distance': current_distance,
                'unified_risk': frame_entry['fusion_output']['unified_risk']
            }
            self.experiment_data['safety_events']['emergency_braking'].append(emergency_brake)
            
    def _assess_adaptation_appropriateness(self, frame_entry: Dict) -> bool:
        """Assess whether weight adaptation was appropriate for the context"""
        driver_state = frame_entry['driver_state']['current_state']
        external_weight = frame_entry['fusion_output']['external_weight']
        internal_risk = frame_entry['component_risks']['internal_risk']
        
        # For impaired drivers, system should increase external weight
        if driver_state in ['sleepy', 'using_phone', 'reaching_back']:
            return external_weight > 0.6  # Should bias toward external risk
        elif driver_state == 'safe_driving':
            return 0.5 <= external_weight <= 0.7  # Should use balanced weights
        else:
            return True  # Unknown state, assume appropriate
            
    def _assess_actual_danger(self, unified_assessment: Dict) -> bool:
        """Assess whether actual danger was present when alert triggered"""
        external_breakdown = unified_assessment.get('external_breakdown', {})
        primary_threat = external_breakdown.get('primary_threat')
        
        if primary_threat:
            distance = primary_threat.get('distance', float('inf'))
            return distance < 15.0  # Danger if object < 15m
        return False
        
    def _assess_alert_timing(self, timestamp: float, alert_data: Dict) -> str:
        """Assess whether alert timing was appropriate"""
        trigger_distance = alert_data.get('trigger_distance')
        
        if trigger_distance is None:
            return 'unknown'
        elif trigger_distance > 25.0:
            return 'too_early'
        elif trigger_distance < 5.0:
            return 'too_late'
        else:
            return 'appropriate'
            
    def _assess_risk_level_match(self, alert_data: Dict, unified_assessment: Dict) -> bool:
        """Assess whether alert level matched actual risk level"""
        alert_level = alert_data.get('level', '')
        actual_risk_level = unified_assessment.get('risk_level', '')
        
        return alert_level.lower() == actual_risk_level.lower()
        
    def _check_emergency_vehicle_present(self, unified_assessment: Dict) -> bool:
        """Check if emergency vehicle is present in current assessment"""
        external_breakdown = unified_assessment.get('external_breakdown', {})
        primary_threat = external_breakdown.get('primary_threat')
        
        return primary_threat and primary_threat.get('is_emergency', False)
        
    def _assess_intervention_necessity(self, context: Dict) -> bool:
        """Assess whether manual intervention was necessary"""
        unified_risk = context.get('unified_risk', 0.0)
        return unified_risk > 0.6  # High risk situations may require intervention
        
    def calculate_fusion_effectiveness_metrics(self) -> Dict:
        """Calculate comprehensive metrics showing fusion system effectiveness"""
        
        if not self.experiment_data['fusion_validation']['risk_predictions']:
            return {'error': 'Insufficient data for analysis'}
            
        # PREDICTION ACCURACY ANALYSIS
        predictions = self.experiment_data['fusion_validation']['risk_predictions']
        
        fusion_errors = [p['errors']['fusion_error'] for p in predictions]
        external_errors = [p['errors']['external_only_error'] for p in predictions]
        internal_errors = [p['errors']['internal_only_error'] for p in predictions]
        simple_average_errors = [p['errors']['simple_average_error'] for p in predictions]
        
        # ADAPTATION EFFECTIVENESS
        adaptations = self.experiment_data['fusion_validation']['weight_adaptations']
        synergy_events = self.experiment_data['fusion_validation']['synergy_events']
        
        # ALERT EFFECTIVENESS
        alerts = self.experiment_data['fusion_validation']['alert_effectiveness']
        appropriate_alerts = [a for a in alerts if a['appropriateness_analysis']['risk_level_match']]
        
        # SAFETY OUTCOMES
        near_misses = len(self.experiment_data['safety_events']['near_misses'])
        emergency_brakes = len(self.experiment_data['safety_events']['emergency_braking'])
        interventions = len(self.experiment_data['safety_events']['manual_interventions'])
        
        metrics = {
            'prediction_accuracy': {
                'fusion_accuracy': 1.0 - np.mean(fusion_errors) if fusion_errors else 0.0,
                'external_only_accuracy': 1.0 - np.mean(external_errors) if external_errors else 0.0,
                'internal_only_accuracy': 1.0 - np.mean(internal_errors) if internal_errors else 0.0,
                'simple_average_accuracy': 1.0 - np.mean(simple_average_errors) if simple_average_errors else 0.0,
                
                'improvement_vs_external': (np.mean(external_errors) - np.mean(fusion_errors)) if external_errors and fusion_errors else 0.0,
                'improvement_vs_internal': (np.mean(internal_errors) - np.mean(fusion_errors)) if internal_errors and fusion_errors else 0.0,
                'improvement_vs_average': (np.mean(simple_average_errors) - np.mean(fusion_errors)) if simple_average_errors and fusion_errors else 0.0
            },
            
            'fusion_capabilities': {
                'adaptive_weight_changes': len(adaptations),
                'synergy_activations': len(synergy_events),
                'context_aware_responses': len(set([a['context']['driver_state'] for a in self.experiment_data['fusion_validation']['context_awareness']])),
                'appropriate_adaptations': len([a for a in adaptations if a.get('adaptation_magnitude', 0) > 0.1])
            },
            
            'alert_effectiveness': {
                'total_alerts': len(alerts),
                'appropriate_alerts': len(appropriate_alerts),
                'alert_accuracy_rate': len(appropriate_alerts) / len(alerts) if alerts else 0.0,
                'false_positive_rate': len([a for a in alerts if not a['appropriateness_analysis']['actual_danger_present']]) / len(alerts) if alerts else 0.0
            },
            
            'safety_outcomes': {
                'minimum_distance': self.min_distance_this_trial,
                'near_miss_events': near_misses,
                'emergency_braking_events': emergency_brakes,
                'manual_interventions': interventions,
                'collision_occurred': self.collision_detected
            },
            
            'overall_effectiveness_score': self._calculate_overall_effectiveness_score()
        }
        
        return metrics
        
    def _calculate_overall_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score (0-100)"""
        
        try:
            metrics = self.calculate_fusion_effectiveness_metrics()
            
            if 'error' in metrics:
                return 0.0
                
            # Weight different components
            accuracy_score = metrics['prediction_accuracy']['fusion_accuracy'] * 40  # 40% weight
            adaptation_score = min(metrics['fusion_capabilities']['adaptive_weight_changes'] / 10.0, 1.0) * 20  # 20% weight
            alert_score = metrics['alert_effectiveness']['alert_accuracy_rate'] * 20  # 20% weight
            safety_score = (1.0 - min(metrics['safety_outcomes']['near_miss_events'] / 5.0, 1.0)) * 20  # 20% weight
            
            return accuracy_score + adaptation_score + alert_score + safety_score
            
        except Exception as e:
            print(f"Error calculating effectiveness score: {e}")
            return 0.0
            
    def save_experiment_data(self) -> str:
        """Save all experiment data to file"""
        
        # Add final metrics and summary
        self.experiment_data['experiment_metadata']['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_data['experiment_metadata']['duration'] = time.time() - self.start_time
        self.experiment_data['experiment_metadata']['total_frames'] = len(self.experiment_data['frame_data'])
        
        # Calculate final effectiveness metrics
        self.experiment_data['final_analysis'] = self.calculate_fusion_effectiveness_metrics()
        
        # Save to file
        filename = f"fusion_experiment_{self.session_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.experiment_data, f, indent=2, default=str)
                
            print(f"✅ Experiment data saved: {filepath}")
            print(f"📊 Summary: {len(self.experiment_data['frame_data'])} frames, "
                  f"{len(self.experiment_data['fusion_validation']['risk_predictions'])} risk predictions, "
                  f"{len(self.experiment_data['fusion_validation']['weight_adaptations'])} adaptations")
            
            # Print key results
            if 'final_analysis' in self.experiment_data:
                analysis = self.experiment_data['final_analysis']
                if 'overall_effectiveness_score' in analysis:
                    print(f"🎯 Overall Effectiveness Score: {analysis['overall_effectiveness_score']:.1f}/100")
                    
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to save experiment data: {e}")
            return ""
            
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        
        metrics = self.calculate_fusion_effectiveness_metrics()
        
        if 'error' in metrics:
            return "Insufficient data for report generation."
            
        report = f"""
🔬 FUSION SYSTEM VALIDATION REPORT
{'='*50}

Experiment: {self.experiment_condition.upper()}
Participant: {self.participant_id}
Trial: {self.trial_number}
Duration: {self.experiment_data['experiment_metadata'].get('duration', 0):.1f} seconds

📊 PREDICTION ACCURACY RESULTS:
• Fusion System: {metrics['prediction_accuracy']['fusion_accuracy']:.3f}
• External Only: {metrics['prediction_accuracy']['external_only_accuracy']:.3f}
• Internal Only: {metrics['prediction_accuracy']['internal_only_accuracy']:.3f}
• Simple Average: {metrics['prediction_accuracy']['simple_average_accuracy']:.3f}

📈 FUSION IMPROVEMENTS:
• vs External Only: {metrics['prediction_accuracy']['improvement_vs_external']:.3f}
• vs Internal Only: {metrics['prediction_accuracy']['improvement_vs_internal']:.3f}
• vs Simple Average: {metrics['prediction_accuracy']['improvement_vs_average']:.3f}

🔄 FUSION CAPABILITIES:
• Weight Adaptations: {metrics['fusion_capabilities']['adaptive_weight_changes']}
• Synergy Activations: {metrics['fusion_capabilities']['synergy_activations']}
• Context-Aware Responses: {metrics['fusion_capabilities']['context_aware_responses']}

🚨 ALERT EFFECTIVENESS:
• Total Alerts: {metrics['alert_effectiveness']['total_alerts']}
• Accuracy Rate: {metrics['alert_effectiveness']['alert_accuracy_rate']:.2%}
• False Positive Rate: {metrics['alert_effectiveness']['false_positive_rate']:.2%}

🛡️ SAFETY OUTCOMES:
• Minimum Distance: {metrics['safety_outcomes']['minimum_distance']:.2f}m
• Near Misses: {metrics['safety_outcomes']['near_miss_events']}
• Emergency Braking: {metrics['safety_outcomes']['emergency_braking_events']}
• Manual Interventions: {metrics['safety_outcomes']['manual_interventions']}

🎯 OVERALL EFFECTIVENESS: {metrics.get('overall_effectiveness_score', 0):.1f}/100

{'='*50}
"""
        
        return report 