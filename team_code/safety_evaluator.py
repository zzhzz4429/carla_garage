import numpy as np
import carla
import transfuser_utils as t_u
from ttc_plotter import TTCPlotter

class SafetyEvaluator:
    def __init__(self, config):
        self.config = config
        # TTC thresholds
        self.critical_ttc = 2.5  # seconds - critical if TTC < 2s
        self.warning_ttc = 5.0   # seconds - warning if TTC < 8s
        
        # Risk assessment thresholds
        self.critical_distance = 3.0  # meters - critical if distance < 5m
        self.warning_distance = 10.0  # meters - warning if distance < 15m
        self.safe_distance = 20.0     # meters - safe if distance > 30m
        
        # Lateral risk thresholds
        self.lane_width = 3.5  # meters - typical lane width
        self.lateral_threshold = 2.0  # meters - lateral distance threshold
        
        # TTC parameters
        self.delta_t = 0.1  # seconds - time between frames at 10 Hz (fallback)
        self.max_ttc = 10.0  # seconds - maximum TTC to consider (increased from 5.0)
        self.alpha = 1.0    # TTC risk coefficient
        self.beta = 1.0     # Signal compliance risk coefficient
        self.a_safe = 7.0   # m/s² - safe deceleration threshold
        
        # Vehicle dimension compensation (center-to-center vs front-to-rear)
        self.ego_vehicle_length = 4.5   # meters - typical car length
        self.target_vehicle_length = 4.5  # meters - assume similar vehicle size
        self.safety_buffer = 2.0     # meters - minimum safe following distance
        
        # Total compensation: half ego + half target + safety buffer
        self.distance_compensation = (self.ego_vehicle_length / 2 + 
                                    self.target_vehicle_length / 2 + 
                                    self.safety_buffer)  # ~6.5 meters total
        
        # Object tracking for gap calculation (OLD METHOD - for reference)
        self.object_history = {}  # Track previous positions of objects
        self.gap_rate_history = {}  # Track gap rate history for smoothing
        self.frame_count = 0
        
        # Time tracking for accurate delta_t calculation
        self.last_timestamp = None
        self.current_timestamp = None
        
        # Simplified tracking variables (NEW METHOD)
        self.prev_closest_distance = None  # Previous frame's closest object distance
        self.gap_rate_smooth = 0.0  # Smoothed gap rate for noise reduction
        
        # Initialize TTC plotter
        self.ttc_plotter = TTCPlotter()
        self.current_time = 0.0
        
    def update_time(self, frame_time):
        """
        Update the current time for plotting
        
        Args:
            frame_time: Current time in seconds
        """
        self.current_time = frame_time
        
    def update_timestamp(self, timestamp):
        """
        Update timestamps and calculate actual delta_t
        
        Args:
            timestamp: Current timestamp from CARLA
        """
        self.last_timestamp = self.current_timestamp
        self.current_timestamp = timestamp
        
        # Calculate actual delta_t if we have previous timestamp
        if self.last_timestamp is not None:
            self.delta_t = timestamp - self.last_timestamp
            # Clamp delta_t to reasonable bounds (0.05 to 0.2 seconds)
            self.delta_t = max(0.05, min(0.2, self.delta_t))
        else:
            # First frame, use default delta_t
            self.delta_t = 0.1
            
        print(f"Timestamp: {timestamp:.3f}s, Delta_t: {self.delta_t:.3f}s")
        

        
    def calculate_signal_compliance_risk(self, ego_speed, bounding_boxes):
        """
        **STRAIGHTFORWARD SIGNAL COMPLIANCE RISK - Independent of driver state**
        
        Calculate risk based on required deceleration to stop at traffic signals.
        This is handled independently from TTC risk to allow different fusion strategies.
        
        Args:
            ego_speed: Current speed of the ego vehicle (m/s)
            bounding_boxes: List of detected bounding boxes
            
        Returns:
            tuple: (max_signal_risk, signal_info) where signal_info contains detailed data
        """
        if not bounding_boxes:
            return 0.0, None
            
        max_signal_risk = 0.0
        closest_signal_info = None
        
        # **STRAIGHTFORWARD DECELERATION THRESHOLDS**
        # Clear, intuitive thresholds based on required deceleration
        CAUTIOUS_DECEL_THRESHOLD = 4.0    # m/s² - Start cautious alerts (comfortable braking)
        WARNING_DECEL_THRESHOLD = 6.0     # m/s² - Warning level (firm braking)
        CRITICAL_DECEL_THRESHOLD = 8.0    # m/s² - Critical level (hard braking)
        EMERGENCY_DECEL_THRESHOLD = 10.0  # m/s² - Emergency level (maximum braking)
        
        for i, bb in enumerate(bounding_boxes):
            # Only consider traffic lights (2) and stop signs (3)
            if bb[7] not in [2, 3]:
                continue
                
            # Get distance to signal (distance to STOP LINE, not the traffic light itself)
            d_to_stop_line = bb[0]  # x-coordinate is distance to stop line
            signal_type = "Red Light" if bb[7] == 2 else "Stop Sign"
            
            # **DEBUG: Print all signal detections**
            print(f"🔍 SIGNAL DEBUG: {signal_type} at distance {d_to_stop_line:.2f}m")
            
            # **SIMPLE VIOLATION DETECTION**
            # If distance < 0, vehicle has passed through stop line → VIOLATION
            if d_to_stop_line < 0.0:
                print(f"🚨 VIOLATION DETECTED IN SAFETY_EVALUATOR: {signal_type} at {d_to_stop_line:.2f}m")
                
                # This is a violation - vehicle ran through the signal
                violation_info = {
                    'index': i,
                    'signal_type': signal_type,
                    'distance_to_stop_line': d_to_stop_line,  # Negative = past stop line
                    'required_deceleration': 0.0,  # Already past, no deceleration can help
                    'signal_risk': 1.0,  # Maximum risk - violation occurred
                    'alert_level': 'VIOLATION',
                    'position': (bb[0], bb[1]),
                    'violation_detected': True,  # Flag for logging
                    'ego_speed_at_violation': ego_speed,
                    
                    # **THRESHOLD FLAGS** - all false since violation already occurred
                    'is_cautious': False,
                    'is_warning': False,
                    'is_critical': False,
                    'is_emergency': False,
                    
                    # **THRESHOLDS** - for reference
                    'thresholds': {
                        'cautious': CAUTIOUS_DECEL_THRESHOLD,
                        'warning': WARNING_DECEL_THRESHOLD,
                        'critical': CRITICAL_DECEL_THRESHOLD,
                        'emergency': EMERGENCY_DECEL_THRESHOLD
                    }
                }
                
                print(f"🚨 SIGNAL VIOLATION: {signal_type} - Vehicle past stop line by {abs(d_to_stop_line):.1f}m at {ego_speed*3.6:.1f} km/h")
                
                # Return violation immediately with maximum risk
                return 1.0, violation_info
            
            # Skip if too close but not violation (between 0 and 1m)
            if d_to_stop_line < 1.0:
                print(f"🔍 SIGNAL DEBUG: Skipping {signal_type} too close but not violation ({d_to_stop_line:.2f}m)")
                continue
            
            # Calculate required deceleration to stop at the stop line
            required_deceleration = (ego_speed ** 2) / (2 * d_to_stop_line)
            
            # **STRAIGHTFORWARD RISK MAPPING**
            # Direct mapping from required deceleration to risk level
            signal_risk = 0.0
            alert_level = "SAFE"
            
            if required_deceleration >= EMERGENCY_DECEL_THRESHOLD:
                signal_risk = 0.95  # Emergency level - immediate action required
                alert_level = "EMERGENCY"
            elif required_deceleration >= CRITICAL_DECEL_THRESHOLD:
                signal_risk = 0.8   # Critical level - hard braking needed
                alert_level = "CRITICAL"
            elif required_deceleration >= WARNING_DECEL_THRESHOLD:
                signal_risk = 0.6   # Warning level - firm braking needed
                alert_level = "WARNING" 
            elif required_deceleration >= CAUTIOUS_DECEL_THRESHOLD:
                signal_risk = 0.3   # Cautious level - comfortable braking needed
                alert_level = "CAUTIOUS"
            # Below 4.0 m/s² = normal braking, no alert needed
            
            # Update maximum risk (closest/most critical signal)
            if signal_risk > max_signal_risk:
                max_signal_risk = signal_risk
                signal_type = "Red Light" if bb[7] == 2 else "Stop Sign"
                
                closest_signal_info = {
                    'index': i,
                    'signal_type': signal_type,
                    'distance_to_stop_line': d_to_stop_line,  # Clarified naming
                    'required_deceleration': required_deceleration,
                    'signal_risk': signal_risk,
                    'alert_level': alert_level,
                    'position': (bb[0], bb[1]),
                    
                    # **THRESHOLD FLAGS** - for easy checking in UnifiedRiskManager
                    'is_cautious': required_deceleration >= CAUTIOUS_DECEL_THRESHOLD,
                    'is_warning': required_deceleration >= WARNING_DECEL_THRESHOLD,
                    'is_critical': required_deceleration >= CRITICAL_DECEL_THRESHOLD,
                    'is_emergency': required_deceleration >= EMERGENCY_DECEL_THRESHOLD,
                    
                    # **THRESHOLDS** - for reference/logging
                    'thresholds': {
                        'cautious': CAUTIOUS_DECEL_THRESHOLD,
                        'warning': WARNING_DECEL_THRESHOLD,
                        'critical': CRITICAL_DECEL_THRESHOLD,
                        'emergency': EMERGENCY_DECEL_THRESHOLD
                    }
                }
                
        return max_signal_risk, closest_signal_info
        
 
        
    def get_risk_status(self, risk_score):
        """
        Convert risk score to a categorical status
        
        Args:
            risk_score: Float between 0 and 1
            
        Returns:
            str: Risk status category
        """
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.5:
            return "WARNING"
        elif risk_score >= 0.2:
            return "CAUTION"
        else:
            return "SAFE"
        
    def get_safe_distance(self, center_to_center_distance: float, object_class: int = 0) -> float:
        """
        Convert center-to-center distance to actual safe following distance
        
        Args:
            center_to_center_distance: Raw distance from BEV detection (meters)
            object_class: Object class (0=vehicle, 1=pedestrian, etc.)
            
        Returns:
            float: Actual distance between vehicle fronts minus safety requirements
        """
        if object_class == 1:  # Pedestrian
            # Pedestrians are smaller, less compensation needed
            pedestrian_compensation = 1.0 + self.safety_buffer  # ~3 meters
            return max(0.0, center_to_center_distance - pedestrian_compensation)
        else:  # Vehicle (0) or Emergency Vehicle (4)
            # Full vehicle dimension compensation
            return max(0.0, center_to_center_distance - self.distance_compensation)
    
    def is_collision_imminent(self, center_to_center_distance: float, object_class: int = 0) -> bool:
        """
        Check if vehicles are already in collision based on center-to-center distance
        
        Args:
            center_to_center_distance: Raw distance from BEV detection
            object_class: Object class type
            
        Returns:
            bool: True if vehicles are overlapping/colliding
        """
        if object_class == 1:  # Pedestrian
            collision_threshold = 2.0  # 2 meters center-to-center for pedestrian contact
        else:  # Vehicle
            collision_threshold = self.ego_vehicle_length / 2 + self.target_vehicle_length / 2
            
        return center_to_center_distance <= collision_threshold
        
    def finalize_plots(self):
        """Finalize and save TTC plots with error handling"""
        try:
            if hasattr(self, 'ttc_plotter'):
                self.ttc_plotter.finalize_session()
        except Exception as e:
            print(f"⚠️ Error during plot finalization: {e}")
            print("🔍 This is likely a matplotlib/logging configuration issue - continuing without plots")
            # Don't crash the shutdown process due to plotting errors
        

        
    def calculate_ttc_risk_simplified(self, ego_speed, bounding_boxes):
        """
        SIMPLIFIED TTC RISK CALCULATION - CURRENT METHOD
        Focus on immediate threats in ego lane for external risk assessment
        
        Args:
            ego_speed: Current speed of the ego vehicle (m/s)
            bounding_boxes: List of detected bounding boxes
            
        Returns:
            tuple: (ttc_risk, threat_object_info) where ttc_risk is the primary risk factor
        """
        if not bounding_boxes:
            # Add data point even when no objects detected
            self.ttc_plotter.add_data_point(0.0, None, self.current_time)
            return 0.0, None
        
        # Object class mapping for better identification
        class_names = {0: "vehicle", 1: "pedestrian", 2: "redlight", 3: "stopsign", 4: "emergency"}
        
        # Debug info
        print(f"Frame {self.frame_count}: Ego speed = {ego_speed:.2f} m/s, Delta_t = {self.delta_t:.3f}s")
        self.frame_count += 1
        
        # 1. Find immediate threats in ego lane
        lane_threats = []
        for i, bb in enumerate(bounding_boxes):
            # Include vehicles (0), pedestrians (1), and emergency vehicles (4) in front
            if bb[0] <= 0 or bb[7] not in [0, 1, 4]:  # Not in front or not relevant object type
                continue
                
            # Check if object is in ego lane (within ±half lane width)
            if abs(bb[1]) > self.lane_width / 2:
                continue
                
            obj_class = int(bb[7])
            class_name = class_names.get(obj_class, f"class{obj_class}")
            
            # Calculate actual safe distance (compensated for vehicle dimensions)
            raw_distance = bb[0]
            safe_distance = self.get_safe_distance(raw_distance, obj_class)
            
            # Check if collision is already happening
            if self.is_collision_imminent(raw_distance, obj_class):
                print(f"  ⚠️ COLLISION DETECTED: {class_name.title()} {i} at {raw_distance:.2f}m (center-to-center)")
                # Treat as immediate emergency
                safe_distance = 0.0
            
            lane_threats.append({
                'index': i,
                'distance': safe_distance,  # Use compensated distance
                'raw_distance': raw_distance,  # Keep original for reference
                'lateral_pos': bb[1],
                'class': obj_class,
                'class_name': class_name,
                'position': (bb[0], bb[1]),
                'is_emergency': obj_class == 4,
                'collision_imminent': self.is_collision_imminent(raw_distance, obj_class)
            })
            
            print(f"  {class_name.title()} {i}: Raw={raw_distance:.2f}m → Safe={safe_distance:.2f}m, Lateral={bb[1]:.2f}m")
        
        if not lane_threats:
            # No threats in lane
            self.prev_closest_distance = None
            self.gap_rate_smooth = 0.0
            self.ttc_plotter.add_data_point(0.0, None, self.current_time)
            print("  No threats detected in ego lane")
            return 0.0, None
        
        # 2. Sort by safe distance (closest first) - primary threat is closest
        lane_threats.sort(key=lambda x: x['distance'])
        closest_threat = lane_threats[0]
        current_safe_distance = closest_threat['distance']
        current_raw_distance = closest_threat['raw_distance']
        
        print(f"  Closest threat: {closest_threat['class_name']} at safe distance {current_safe_distance:.2f}m (raw: {current_raw_distance:.2f}m)")
        
        # Handle collision-imminent cases
        if closest_threat['collision_imminent']:
            print(f"  🚨 EMERGENCY: Collision imminent or already occurring!")
            # Force maximum risk for immediate collision
            ttc_risk = 1.0
            ttc_value = 0.1  # Nearly zero TTC
            
            threat_object = {
                'index': closest_threat['index'],
                'object_id': f"{closest_threat['class_name']}_COLLISION",
                'distance': current_safe_distance,
                'raw_distance': current_raw_distance,
                'lateral_pos': closest_threat['lateral_pos'],
                'gap_rate': -999.0,  # Indicate collision situation
                'ttc': ttc_value,
                'ttc_risk': ttc_risk,
                'object_class': closest_threat['class'],
                'class_name': closest_threat['class_name'],
                'position': closest_threat['position'],
                'detected': True,
                'is_emergency': closest_threat['is_emergency'],
                'collision_imminent': True
            }
            
            # Update state for next frame
            self.prev_closest_distance = current_safe_distance
            
            # Add data point to plotter
            self.ttc_plotter.add_data_point(ttc_risk, threat_object, self.current_time, ego_speed=ego_speed)
            
            print(f"  EMERGENCY TTC Risk: {ttc_risk:.3f}")
            return ttc_risk, threat_object
        
        # 3. Simple temporal tracking for gap rate calculation
        gap_rate = 0.0
        
        if hasattr(self, 'prev_closest_distance') and self.prev_closest_distance is not None:
            # Calculate raw gap rate using safe distance
            raw_gap_rate = (current_safe_distance - self.prev_closest_distance) / self.delta_t
            
            # Apply minimum time threshold to prevent noise amplification
            if self.delta_t < 0.05:
                gap_rate = 0.0
                print(f"    REJECTED: delta_t too small ({self.delta_t:.3f}s)")
            # Apply physics-based sanity check
            elif abs(raw_gap_rate) > 50.0:
                gap_rate = 0.0
                print(f"    REJECTED: gap_rate too large ({raw_gap_rate:.2f} m/s)")
            else:
                # Apply temporal smoothing to reduce noise
                if hasattr(self, 'gap_rate_smooth'):
                    alpha = 0.3  # Smoothing factor (0.3 = 30% new, 70% old)
                    gap_rate = alpha * raw_gap_rate + (1 - alpha) * self.gap_rate_smooth
                else:
                    gap_rate = raw_gap_rate
                
                self.gap_rate_smooth = gap_rate
                print(f"    Gap rate: raw={raw_gap_rate:.2f}, smoothed={gap_rate:.2f} m/s")
        else:
            # First frame or after reset
            gap_rate = 0.0
            self.gap_rate_smooth = 0.0
            print(f"    First detection or reset, gap_rate=0.0")
        
        # 4. Calculate TTC risk based on gap rate
        ttc_risk = 0.0
        ttc_value = float('inf')
        
        # Determine gap rate threshold based on object type
        if closest_threat['is_emergency']:
            # Emergency vehicles are often stationary, so use different logic
            gap_rate_threshold = -0.05  # More sensitive for emergency vehicles
            max_ttc_for_class = 45.0    # Longer planning horizon
            risk_multiplier = 1.5       # Higher risk for yielding behavior
        else:
            gap_rate_threshold = -0.1   # Normal threshold for vehicles/pedestrians
            max_ttc_for_class = self.max_ttc  # Standard TTC horizon
            risk_multiplier = 1.0       # Normal risk calculation
        
        if gap_rate < gap_rate_threshold:  # Approaching (distance decreasing)
            ttc_value = current_safe_distance / max(-gap_rate, 0.01)  # Prevent division by zero
            
            # Additional safety check: if distance is extremely small, treat as collision
            if current_safe_distance <= 0.01:  # Less than 1cm = collision
                print(f"    COLLISION: Distance too small ({current_safe_distance:.3f}m), forcing emergency risk")
                ttc_risk = 1.0
                ttc_value = 0.01  # Minimal TTC for collision
            elif ttc_value < max_ttc_for_class:
                # Prevent division by zero in risk calculation
                ttc_risk = (self.alpha * risk_multiplier) / max(ttc_value, 0.01)
                
                threat_type = "EMERGENCY" if closest_threat['is_emergency'] else "NORMAL"
                print(f"    TTC={ttc_value:.2f}s < max_ttc({max_ttc_for_class:.1f}s)")
                print(f"    Risk={ttc_risk:.3f} ({threat_type}, multiplier={risk_multiplier})")
            else:
                print(f"    TTC={ttc_value:.2f}s >= max_ttc({max_ttc_for_class:.1f}s), Risk=0.0 (too far)")
        else:
            reason = "not closing" if gap_rate >= 0 else "closing too slowly"
            print(f"    TTC=inf, Risk=0.0 (gap {reason}, gap_rate={gap_rate:.2f})")
        
        # Safety cap: Ensure TTC risk is within bounds
        ttc_risk = min(ttc_risk, 1.0)  # Never exceed maximum risk
        ttc_value = max(ttc_value, 0.01) if ttc_value != float('inf') else float('inf')  # Ensure positive TTC
        
        # 5. Prepare threat object info
        threat_object = {
            'index': closest_threat['index'],
            'object_id': f"{closest_threat['class_name']}_{current_safe_distance:.0f}",  # Simple ID
            'distance': current_safe_distance,  # Safe distance for TTC calculations
            'raw_distance': current_raw_distance,  # Original BEV distance for reference
            'lateral_pos': closest_threat['lateral_pos'],
            'gap_rate': gap_rate,
            'ttc': ttc_value,
            'ttc_risk': ttc_risk,
            'object_class': closest_threat['class'],
            'class_name': closest_threat['class_name'],
            'position': closest_threat['position'],
            'detected': True,
            'is_emergency': closest_threat['is_emergency'],
            'collision_imminent': False
        }
        
        # 6. Update state for next frame
        self.prev_closest_distance = current_safe_distance
        
        # Add data point to plotter (enhanced for simplified approach)
        self.ttc_plotter.add_data_point(ttc_risk, threat_object, self.current_time, ego_speed=ego_speed)
        
        print(f"  Final TTC Risk: {ttc_risk:.3f}")
        print(f"  Threat: {closest_threat['class_name']} at safe distance {current_safe_distance:.2f}m (raw: {current_raw_distance:.2f}m)")
        
        return ttc_risk, threat_object
        
    def calculate_external_risk_unified(self, ego_speed, bounding_boxes):
        """
        **SEPARATED EXTERNAL RISK ASSESSMENT - TTC and Signal Compliance handled independently**
        
        Returns both risks separately so UnifiedRiskManager can:
        - Apply driver state fusion to TTC risk (distracted drivers need earlier TTC warnings)
        - Handle signal compliance risk independently (red light violation is dangerous regardless)
        - Implement conditional alerting (TTC alerts can suppress signal alerts)
        """
        if not bounding_boxes:
            return 0.0, {
                'ttc_risk': 0.0,
                'signal_risk': 0.0,
                'combined_risk': 0.0,
                'primary_threat': None,
                'signal_info': None,
                'risk_level': 'SAFE',
                'separation_mode': True  # Indicates separated risk calculation
            }
        
        # **SEPARATED RISK CALCULATIONS**
        
        # 1. TTC Risk - Main collision threat assessment
        ttc_risk, ttc_object = self.calculate_ttc_risk_simplified(ego_speed, bounding_boxes)
        
        # 2. Signal Compliance Risk - Independent traffic law compliance  
        signal_risk, signal_object = self.calculate_signal_compliance_risk(ego_speed, bounding_boxes)
        
        # **ENHANCED TTC RISK with context**
        enhanced_ttc_risk = ttc_risk
        if ttc_object:
            # Emergency vehicle context boost for TTC
            if ttc_object.get('is_emergency', False):
                enhanced_ttc_risk *= 1.2  # 20% boost for emergency vehicles
                print(f"  🚨 Emergency vehicle TTC - risk boosted by 20%")
            
            # Close proximity emergency boost
            distance = ttc_object.get('distance', float('inf'))
            if distance < 5.0 and ttc_risk > 0.5:
                enhanced_ttc_risk = min(1.0, enhanced_ttc_risk * 1.3)
                print(f"  ⚠️ Very close object ({distance:.1f}m) - TTC risk boosted")
        
        # **TRADITIONAL COMBINED RISK for backward compatibility**
        # This is still calculated but UnifiedRiskManager can choose to use separated risks
        primary_weight = 0.7    # TTC risk weight
        secondary_weight = 0.3   # Signal compliance weight
        combined_risk = (primary_weight * enhanced_ttc_risk + secondary_weight * signal_risk)
        
        # **RETURN SEPARATED RISKS**
        return combined_risk, {
            # **SEPARATED COMPONENTS** - UnifiedRiskManager can use these independently
            'ttc_risk': enhanced_ttc_risk,
            'signal_risk': signal_risk,
            'raw_ttc_risk': ttc_risk,  # Before emergency vehicle boost
            
            # **THREAT OBJECTS**
            'primary_threat': ttc_object,
            'signal_info': signal_object,
            
            # **COMBINED RISK** - for backward compatibility
            'combined_risk': combined_risk,
            'risk_level': self.get_risk_status(combined_risk),
            
            # **SEPARATION FLAGS** - helps UnifiedRiskManager know it can handle them separately
            'separation_mode': True,
            'has_ttc_threat': ttc_object is not None,
            'has_signal_threat': signal_object is not None,
            
            # **ALERTING HINTS** - for conditional alerting logic
            'ttc_alert_level': self.get_risk_status(enhanced_ttc_risk) if ttc_object else 'NONE',
            'signal_alert_level': signal_object.get('alert_level', 'NONE') if signal_object else 'NONE'
        }
        
    def update_plotter_with_unified_risk(self, combined_risk, risk_breakdown):
        """
        Update the TTC plotter with unified risk data for enhanced visualization
        
        Args:
            combined_risk: The combined external risk value
            risk_breakdown: Dictionary containing risk breakdown details
        """
        if hasattr(self, 'ttc_plotter') and len(self.ttc_plotter.unified_risks) > 0:
            # Update the last entry with unified risk data
            self.ttc_plotter.unified_risks[-1] = combined_risk
            
            # If there's a primary threat, update emergency flag
            if risk_breakdown['primary_threat']:
                self.ttc_plotter.emergency_flags[-1] = risk_breakdown['primary_threat'].get('is_emergency', False)
                self.ttc_plotter.object_types[-1] = risk_breakdown['primary_threat'].get('class_name', 'unknown') 

