"""
Standalone Boundary-Based Risk Estimator

Implements the boundary-based risk field formulation from the research proposal.
This is a physics-grounded approach that replaces rule-based TTC calculations
with continuous 360° risk assessment.

Key Features:
- Polar boundary construction via ray casting
- Directional risk computation using physics-grounded formula
- Safe boundary extraction for trajectory constraints
- Multi-object and multi-directional threat assessment

Usage:
    estimator = BoundaryRiskEstimator()
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(ego_speed, bounding_boxes)
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import time


class BoundaryRiskEstimator:
    """
    Standalone implementation of boundary-based external risk estimation.
    
    This class implements the methodology from the research proposal:
    1. Polar boundary construction (360° ray casting)
    2. Directional risk computation (physics-grounded formula)
    3. Safe boundary extraction (risk threshold-based)
    """
    
    def __init__(self, 
                 angular_resolution: int = 10,
                 max_range: float = 50.0,
                 k_distance: float = -1.0,
                 alpha_coeff: float = 1.0,
                 beta_coeff: float = 0.5,
                 risk_threshold: float = 0.3):
        """
        Initialize the boundary risk estimator.
        
        Args:
            angular_resolution: Number of angular sectors (M in paper)
            max_range: Maximum detection range in meters
            k_distance: Distance penalty exponent (k < 0)
            alpha_coeff: Distance term coefficient (α)
            beta_coeff: Velocity term coefficient (β)
            risk_threshold: Risk threshold for safe boundary (τ)
        """
        self.M = angular_resolution  # Number of angular sectors
        self.max_range = max_range
        self.k = k_distance
        self.alpha = alpha_coeff
        self.beta = beta_coeff
        self.tau = risk_threshold
        
        # Pre-compute angular sectors for efficiency
        self.angles = np.linspace(0, 2*np.pi, self.M, endpoint=False)
        
        # Class-specific weights (from paper methodology)
        # m_O: Object presence weights   
        # m_D: Dynamic/velocity  
        self.class_weights = { 
            0: {'m_O': 1.0, 'm_D': 1.0},   # Vehicle
            1: {'m_O': 1.2, 'm_D': 0.8},   # Pedestrian (higher presence risk, lower velocity weight)
            2: {'m_O': 0.5, 'm_D': 0.0},   # Red Light (static, handled separately)
            3: {'m_O': 0.5, 'm_D': 0.0},   # Stop Sign (static, handled separately)
            4: {'m_O': 1.5, 'm_D': 1.3}    # Emergency Vehicle (highest weights)
        }
        
        # Vehicle dimensions for boundary detection
        self.ego_length = 4.5
        self.ego_width = 2.0
        
        # Debug/logging and temporal state
        self.debug = False
        self.frame_count = 0
        self.last_polar_boundary: Optional[List[Dict]] = None
        self.last_timestamp: Optional[float] = None
        self.last_delta_t: Optional[float] = None
        self.default_delta_t = 0.1  # seconds
        
    def set_debug(self, debug: bool):
        """Enable/disable debug output"""
        self.debug = debug
        
    def calculate_boundary_risk(self,
                                ego_speed: float,
                                bounding_boxes: List,
                                timestamp: Optional[float] = None) -> Tuple[np.ndarray, float, Dict]:
        """
        Main entry point: Calculate boundary-based risk field.
        
        Args:
            ego_speed: Current ego vehicle speed (m/s)
            bounding_boxes: List of detected objects [x, y, w, h, yaw, speed, brake, class]
            timestamp: Optional simulation timestamp (seconds) for delta-t estimation
            
        Returns:
            Tuple of:
            - risk_field: Array of risk values for each angular sector
            - max_risk: Maximum risk value across all directions
            - threat_info: Dictionary with detailed threat information
        """
        self.frame_count += 1
        start_time = time.time()
        
        if self.debug:
            print(f"\n=== BOUNDARY RISK FRAME {self.frame_count} ===")
            print(f"Ego speed: {ego_speed:.2f} m/s")
            print(f"Objects detected: {len(bounding_boxes)}")
        
        # Determine delta time
        if timestamp is not None and self.last_timestamp is not None:
            delta_t = max(0.01, min(0.5, timestamp - self.last_timestamp))
        else:
            delta_t = self.default_delta_t
        self.last_delta_t = delta_t

        # Step 1: Construct polar boundary
        polar_boundary = self.construct_polar_boundary(bounding_boxes, ego_speed, delta_t)
        
        # Step 2: Calculate directional risk for each sector
        risk_field = self.calculate_directional_risk(polar_boundary, ego_speed)
        
        # Step 3: Extract safe boundary and find maximum risk
        safe_boundary = self.extract_safe_boundary(risk_field, polar_boundary)
        max_risk = np.max(risk_field)
        max_risk_angle = self.angles[np.argmax(risk_field)]
        
        # Step 4: Generate threat information
        threat_info = self.generate_threat_info(polar_boundary, risk_field, max_risk_angle, ego_speed)

        # Persist temporal state for next frame
        self.last_polar_boundary = polar_boundary
        if timestamp is not None:
            self.last_timestamp = timestamp
        else:
            if self.last_timestamp is not None:
                self.last_timestamp += delta_t
            else:
                self.last_timestamp = delta_t
        
        computation_time = time.time() - start_time
        
        if self.debug:
            print(f"Max risk: {max_risk:.3f} at angle {np.degrees(max_risk_angle):.1f}°")
            print(f"Computation time: {computation_time*1000:.2f}ms")
            print("=" * 40)
        
        return risk_field, max_risk, threat_info
    
    def construct_polar_boundary(self,
                                 bounding_boxes: List,
                                 ego_speed: float,
                                 delta_t: float) -> List[Dict]:
        """
        Step 1: Construct polar boundary via ray casting.
        
        For each angular sector, find the nearest object surface.
        
        Args:
            bounding_boxes: List of detected objects
            ego_speed: Ego vehicle speed (m/s)
            delta_t: Time between frames (seconds)
            
        Returns:
            List of boundary points for each angular sector
        """
        polar_boundary = []
        
        for i, angle in enumerate(self.angles):
            # Ray direction (unit vector)
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Find nearest intersection with any object
            min_distance = self.max_range
            nearest_object = None
            fallback_velocity = 0.0
            
            for j, bb in enumerate(bounding_boxes):
                # Object position and properties
                obj_x, obj_y = bb[0], bb[1]
                obj_w, obj_h = bb[2], bb[3]
                obj_yaw = bb[4] if len(bb) > 4 else 0.0
                obj_speed = bb[5] if len(bb) > 5 else 0.0
                obj_class = int(bb[7]) if len(bb) > 7 else 0
                
                # Calculate distance to object boundary along ray
                distance = self.ray_object_intersection(ray_dir, obj_x, obj_y, obj_w, obj_h, obj_yaw)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = j
                    
                    # Calculate radial velocity (approaching speed)
                    # Project object velocity onto ray direction
                    if len(bb) > 5 and obj_speed > 0:
                        # Assume object moves in its heading direction
                        obj_vel_x = obj_speed * np.cos(obj_yaw)
                        obj_vel_y = obj_speed * np.sin(obj_yaw)
                        obj_velocity_vec = np.array([obj_vel_x, obj_vel_y])
                        
                        # Relative velocity (object - ego, projected onto ray)
                        ego_velocity_vec = np.array([ego_speed, 0.0])
                        relative_vel = obj_velocity_vec - ego_velocity_vec
                        
                        # Radial component (positive = approaching)
                        fallback_velocity = -np.dot(relative_vel, ray_dir)
            
            # Store boundary point information
            radial_velocity = 0.0
            if self.last_polar_boundary is not None and self.last_delta_t and nearest_object is not None:
                if i < len(self.last_polar_boundary):
                    prev_distance = self.last_polar_boundary[i]['distance']
                    if (prev_distance is not None and prev_distance < self.max_range
                            and min_distance < self.max_range):
                        radial_velocity = (prev_distance - min_distance) / max(delta_t, 1e-3)
                        # Limit excessively large magnitudes
                        radial_velocity = float(np.clip(radial_velocity, -100.0, 100.0))

            if radial_velocity <= 0.0 and fallback_velocity > 0.0:
                radial_velocity = fallback_velocity

            radial_velocity = max(0.0, radial_velocity)

            boundary_point = {
                'angle': angle,
                'distance': min_distance,
                'radial_velocity': radial_velocity,
                'object_index': nearest_object,
                'object_class': int(bounding_boxes[nearest_object][7]) if nearest_object is not None and len(bounding_boxes[nearest_object]) > 7 else 0
            }
            
            polar_boundary.append(boundary_point)
            
            if self.debug and i % 90 == 0:  # Debug every 90 degrees
                print(f"  Angle {np.degrees(angle):3.0f}°: dist={min_distance:.2f}m, v_r={object_velocity:.2f}m/s")
        
        return polar_boundary
    
    def ray_object_intersection(self,
                                ray_dir: np.ndarray,
                                obj_x: float,
                                obj_y: float,
                                obj_w: float,
                                obj_h: float,
                                obj_yaw: float) -> float:
        """
        Calculate distance from ego to object boundary along ray direction.

        Args:
            ray_dir: Ray direction (unit vector)
            obj_x, obj_y: Object center position
            obj_w, obj_h: Object dimensions
            obj_yaw: Object orientation (radians)

        Returns:
            Distance to object boundary, or max_range if no intersection
        """
        center = np.array([obj_x, obj_y])

        # Rotation matrix for object orientation
        cos_yaw = np.cos(obj_yaw)
        sin_yaw = np.sin(obj_yaw)
        rotation = np.array([[cos_yaw, -sin_yaw],
                             [sin_yaw,  cos_yaw]])

        # Transform ray origin and direction into the object's local frame
        origin_world = np.array([0.0, 0.0])
        origin_local = rotation.T @ (origin_world - center)
        dir_local = rotation.T @ ray_dir

        # Axis-aligned half extents in local frame
        half_extents = np.array([obj_w / 2.0, obj_h / 2.0])

        t_min = -np.inf
        t_max = np.inf

        for axis in range(2):
            if abs(dir_local[axis]) < 1e-6:
                # Ray is parallel to this slab; if origin outside slab, no intersection
                if abs(origin_local[axis]) > half_extents[axis]:
                    return self.max_range
            else:
                t1 = (-half_extents[axis] - origin_local[axis]) / dir_local[axis]
                t2 = (half_extents[axis] - origin_local[axis]) / dir_local[axis]
                t_near = min(t1, t2)
                t_far = max(t1, t2)

                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)

                if t_max < t_min:
                    return self.max_range

        if t_max < 0:
            # Intersection is behind the origin
            return self.max_range

        t_hit = t_min if t_min >= 0 else t_max
        if t_hit < 0:
            return self.max_range

        if t_hit > self.max_range:
            return self.max_range

        return float(t_hit)
    
    def calculate_directional_risk(self, polar_boundary: List[Dict], ego_speed: float) -> np.ndarray:
        """
        Step 2: Calculate directional risk using physics-grounded formula.
        
        R(θ) = α * m_O(c_θ) * r_θ^k + β * m_D(c_θ) * v_r^+(θ) * r_θ^(k+1)
        
        Args:
            polar_boundary: Boundary points from step 1
            ego_speed: Current ego speed
            
        Returns:
            Array of risk values for each angular sector
        """
        risk_field = np.zeros(self.M)
        
        for i, boundary_point in enumerate(polar_boundary):
            distance = boundary_point['distance']
            radial_velocity = boundary_point['radial_velocity']
            obj_class = boundary_point['object_class']
            
            # Skip if no object detected in this direction
            if boundary_point['object_index'] is None:
                continue
            
            # Get class-specific weights
            weights = self.class_weights.get(obj_class, {'m_O': 1.0, 'm_D': 1.0})
            m_O = weights['m_O']
            m_D = weights['m_D']
            
            # Avoid division by zero for very close objects
            safe_distance = max(distance, 0.1)
            
            # Physics-grounded risk formula from paper
            # Term 1: Distance-based risk (closer = higher risk)
            distance_term = self.alpha * m_O * (safe_distance ** self.k)
            
            # Term 2: Velocity-based risk (approaching faster = higher risk)
            velocity_term = self.beta * m_D * radial_velocity * (safe_distance ** (self.k + 1))
            
            # Combined risk
            risk_field[i] = distance_term + velocity_term
            
            if self.debug and risk_field[i] > 0.1:
                angle_deg = np.degrees(boundary_point['angle'])
                print(f"  Risk at {angle_deg:3.0f}°: {risk_field[i]:.3f} (dist={distance:.1f}m, v_r={radial_velocity:.1f}m/s, class={obj_class})")
        
        return risk_field
    
    def extract_safe_boundary(self, risk_field: np.ndarray, polar_boundary: List[Dict]) -> List[Dict]:
        """
        Step 3: Extract safe boundary by finding where R(θ) ≤ τ.
        
        Args:
            risk_field: Risk values for each direction
            polar_boundary: Original boundary points
            
        Returns:
            List of safe boundary points
        """
        safe_boundary = []
        
        for i, (risk, boundary_point) in enumerate(zip(risk_field, polar_boundary)):
            angle = boundary_point['angle']
            original_distance = boundary_point['distance']
            
            if risk <= self.tau:
                # Risk is already acceptable, use original distance
                safe_distance = original_distance
            else:
                # Need to shrink distance until risk ≤ τ
                # This is a simplified approach; full implementation would solve the equation
                safe_distance = original_distance * 0.7  # Conservative shrinking
                
                # Iterative refinement (simple approach)
                for _ in range(5):  # Max 5 iterations
                    # Recalculate risk at this distance
                    obj_class = boundary_point['object_class']
                    weights = self.class_weights.get(obj_class, {'m_O': 1.0, 'm_D': 1.0})
                    
                    test_distance = max(safe_distance, 0.1)
                    distance_term = self.alpha * weights['m_O'] * (test_distance ** self.k)
                    velocity_term = self.beta * weights['m_D'] * boundary_point['radial_velocity'] * (test_distance ** (self.k + 1))
                    test_risk = distance_term + velocity_term
                    
                    if test_risk <= self.tau:
                        break
                    else:
                        safe_distance *= 0.8  # Further shrinking
            
            safe_boundary.append({
                'angle': angle,
                'safe_distance': safe_distance,
                'original_distance': original_distance,
                'risk': risk
            })
        
        return safe_boundary
    
    def generate_threat_info(self, polar_boundary: List[Dict], risk_field: np.ndarray, 
                           max_risk_angle: float, ego_speed: float) -> Dict:
        """
        Step 4: Generate comprehensive threat information.
        
        Args:
            polar_boundary: Boundary points
            risk_field: Risk values
            max_risk_angle: Angle of maximum risk
            ego_speed: Current ego speed
            
        Returns:
            Dictionary with threat analysis
        """
        max_risk = np.max(risk_field)
        max_risk_idx = np.argmax(risk_field)
        
        # Find the boundary point with maximum risk
        max_risk_boundary = polar_boundary[max_risk_idx]
        
        # Risk level classification
        if max_risk >= 0.8:
            risk_level = "CRITICAL"
        elif max_risk >= 0.5:
            risk_level = "WARNING"
        elif max_risk >= 0.2:
            risk_level = "CAUTION"
        else:
            risk_level = "SAFE"
        
        # Primary threat information
        primary_threat = None
        if max_risk_boundary['object_index'] is not None:
            primary_threat = {
                'direction_deg': np.degrees(max_risk_angle),
                'distance': max_risk_boundary['distance'],
                'radial_velocity': max_risk_boundary['radial_velocity'],
                'object_class': max_risk_boundary['object_class'],
                'risk_contribution': max_risk
            }
        
        # Summary statistics
        total_risk = np.sum(risk_field)
        mean_risk = np.mean(risk_field)
        risk_directions = np.sum(risk_field > self.tau)
        
        threat_info = {
            'max_risk': max_risk,
            'risk_level': risk_level,
            'max_risk_direction_deg': np.degrees(max_risk_angle),
            'primary_threat': primary_threat,
            'total_risk': total_risk,
            'mean_risk': mean_risk,
            'risky_directions': risk_directions,
            'ego_speed': ego_speed,
            'frame_count': self.frame_count,
            
            # For compatibility with existing systems
            'boundary_based': True,
            'algorithm_version': '1.0'
        }
        
        return threat_info
    
    def get_risk_field_visualization_data(self, risk_field: np.ndarray) -> Dict:
        """
        Helper method to get data for visualization/plotting.
        
        Args:
            risk_field: Risk values for each direction
            
        Returns:
            Dictionary with visualization data
        """
        return {
            'angles_deg': np.degrees(self.angles),
            'risk_values': risk_field,
            'risk_threshold': self.tau,
            'max_risk': np.max(risk_field),
            'max_risk_angle_deg': np.degrees(self.angles[np.argmax(risk_field)])
        }


def test_boundary_risk_estimator():
    """
    Simple test function to validate the boundary risk estimator.
    """
    print("Testing Boundary Risk Estimator...")
    
    # Create estimator with debug enabled
    estimator = BoundaryRiskEstimator(angular_resolution=180, max_range=30.0)
    estimator.set_debug(True)
    
    # Test case 1: Single vehicle ahead
    print("\n--- Test Case 1: Single vehicle ahead ---")
    ego_speed = 15.0  # m/s
    bounding_boxes = [
        [10.0, 0.0, 4.0, 2.0, 0.0, 12.0, 0.0, 0]  # Vehicle 10m ahead, slightly slower
    ]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(ego_speed, bounding_boxes)
    print(f"Result: Max risk = {max_risk:.3f}, Level = {threat_info['risk_level']}")
    
    # Test case 2: Multiple objects
    print("\n--- Test Case 2: Multiple objects ---")
    bounding_boxes = [
        [15.0, 0.0, 4.0, 2.0, 0.0, 10.0, 0.0, 0],   # Vehicle ahead
        [8.0, -3.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1],    # Pedestrian to the left
        [20.0, 2.0, 5.0, 2.5, 0.0, 0.0, 0.0, 4]     # Emergency vehicle to the right
    ]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(ego_speed, bounding_boxes)
    print(f"Result: Max risk = {max_risk:.3f}, Level = {threat_info['risk_level']}")
    print(f"Primary threat direction: {threat_info['max_risk_direction_deg']:.1f}°")
    
    # Test case 3: No objects (baseline)
    print("\n--- Test Case 3: No objects ---")
    bounding_boxes = []
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(ego_speed, bounding_boxes)
    print(f"Result: Max risk = {max_risk:.3f}, Level = {threat_info['risk_level']}")
    
    print("\nBoundary Risk Estimator test completed!")
    
    return estimator, risk_field, threat_info


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_boundary_risk_estimator()
