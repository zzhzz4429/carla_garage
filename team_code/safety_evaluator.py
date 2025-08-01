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
        Calculate risk based on signal compliance (red lights and stop signs)
        
        Args:
            ego_speed: Current speed of the ego vehicle (m/s)
            bounding_boxes: List of detected bounding boxes
            
        Returns:
            tuple: (max_signal_risk, closest_signal_info) where max_signal_risk is the maximum signal risk
        """
        if not bounding_boxes:
            return 0.0, None
            
        max_signal_risk = 0.0
        closest_signal_info = None
        
        for i, bb in enumerate(bounding_boxes):
            # Only consider traffic lights (2) and stop signs (3)
            if bb[7] not in [2, 3]:
                continue
                
            # Get distance to signal
            d_i = bb[0]  # x-coordinate is distance to signal
            
            # Calculate required deceleration to stop at the signal
            # Clamp distance to avoid division by zero
            d_i_clamped = max(d_i, 0.5)
            a_req = (ego_speed ** 2) / (2 * d_i_clamped)
            
            # Calculate signal compliance risk
            signal_risk = self.beta * max((a_req / self.a_safe) - 1, 0)
            
            # Update maximum risk
            if signal_risk > max_signal_risk:
                max_signal_risk = signal_risk
                signal_type = "Red Light" if bb[7] == 2 else "Stop Sign"
                closest_signal_info = {
                    'index': i,
                    'signal_type': signal_type,
                    'distance': d_i,
                    'required_deceleration': a_req,
                    'signal_risk': signal_risk,
                    'position': (bb[0], bb[1])
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
        """
        Create and save all TTC plots and data
        """
        self.ttc_plotter.finalize_session()
        

        
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
        UNIFIED EXTERNAL RISK ASSESSMENT - TTC + Signal Compliance
        """
        if not bounding_boxes:
            return 0.0, {
                'ttc_risk': 0.0,
                'signal_risk': 0.0,
                'combined_risk': 0.0,
                'primary_threat': None,
                'signal_threat': None,
                'risk_level': 'SAFE'
            }
        
        # Primary Risk Factor: TTC (main external risk)
        ttc_risk, ttc_object = self.calculate_ttc_risk_simplified(ego_speed, bounding_boxes)
        
        # Secondary Risk Factor: Signal Compliance (traffic lights, stop signs)
        signal_risk, signal_object = self.calculate_signal_compliance_risk(ego_speed, bounding_boxes)
        
        # Use TTC as the primary external risk (90% weight)
        # Signal compliance as secondary (10% weight for advisory)
        primary_weight = 0.7    # TTC risk weight
        secondary_weight = 0.3   # Signal compliance weight
        
        # Combined external risk
        combined_risk = (primary_weight * ttc_risk + secondary_weight * signal_risk)
        
        # Context-based risk adjustments (only for TTC objects)
        primary_threat = ttc_object
        
        if primary_threat:
            # Emergency vehicle context boost
            if primary_threat.get('is_emergency', False):
                combined_risk *= 1.2  # 20% boost for emergency vehicles
                print(f"  Emergency vehicle detected - risk boosted by 20%")
            
            # Close proximity emergency boost (only for very close objects)
            if primary_threat['distance'] < 8.0 and combined_risk > 0.1:
                proximity_boost = max(0.1, (8.0 - primary_threat['distance']) / 8.0 * 0.2)
                combined_risk += proximity_boost
                print(f"  Close proximity boost: +{proximity_boost:.3f}")
        
        # Cap the risk at 1.0
        combined_risk = min(combined_risk, 1.0)
        
        # Determine risk level based on combined risk
        if combined_risk >= 0.8:
            risk_level = "CRITICAL"
        elif combined_risk >= 0.5:
            risk_level = "WARNING" 
        elif combined_risk >= 0.2:
            risk_level = "CAUTION"
        else:
            risk_level = "SAFE"
        
        # Risk breakdown for analysis
        risk_breakdown = {
            'ttc_risk': ttc_risk,
            'signal_risk': signal_risk,
            'combined_risk': combined_risk,
            'primary_threat': primary_threat,
            'signal_threat': signal_object,
            'risk_level': risk_level,
            'weights': {
                'ttc_weight': primary_weight,
                'signal_weight': secondary_weight
            }
        }
        
        print(f"=== EXTERNAL RISK (TTC + Signal) ===")
        print(f"TTC Risk: {ttc_risk:.3f} (weight: {primary_weight})")
        print(f"Signal Risk: {signal_risk:.3f} (weight: {secondary_weight})")
        print(f"Combined Risk: {combined_risk:.3f}")
        print(f"Risk Level: {risk_level}")
        
        if primary_threat:
            threat_type = primary_threat.get('class_name', 'unknown')
            emergency_flag = " (EMERGENCY)" if primary_threat.get('is_emergency', False) else ""
            print(f"Primary Threat: {threat_type}{emergency_flag} at {primary_threat['distance']:.1f}m")
            
        if signal_object:
            print(f"Signal Alert: {signal_object['signal_type']} at {signal_object['distance']:.1f}m")
            print(f"Required Deceleration: {signal_object['required_deceleration']:.2f} m/s²")
            
        print("===================================")
        
        # Update plotter with both TTC and signal data
        signal_data = None
        if signal_object:
            signal_data = {
                'distance': signal_object['distance'],
                'signal_type': signal_object['signal_type'],
                'signal_risk': signal_object['signal_risk'],
                'required_deceleration': signal_object['required_deceleration']
            }
        
        # Update the TTC plotter with unified risk and signal data
        if hasattr(self, 'ttc_plotter'):
            # Update the last TTC data point with unified risk and signal data
            if len(self.ttc_plotter.ttc_data) > 0:
                # Remove the last entry added by calculate_ttc_risk_simplified
                self.ttc_plotter.ttc_data[-1] = ttc_risk
                self.ttc_plotter.unified_risks[-1] = combined_risk
                
                # Add signal data to the last entry
                if signal_data:
                    self.ttc_plotter.signal_distances[-1] = signal_data['distance']
                    self.ttc_plotter.signal_types[-1] = signal_data['signal_type']
                    self.ttc_plotter.signal_risks[-1] = signal_data['signal_risk']
                    self.ttc_plotter.current_decelerations[-1] = signal_data['required_deceleration']
        
        return combined_risk, risk_breakdown
        
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

