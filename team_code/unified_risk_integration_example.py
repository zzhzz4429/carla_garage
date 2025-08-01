#!/usr/bin/env python3
"""
Unified Risk Integration Example
Shows how to integrate the unified risk manager into your existing CARLA sensor agent
"""

import time
import numpy as np
from unified_risk_manager import UnifiedRiskManager

class EnhancedSensorAgent:
    """
    Example integration of unified risk manager with your existing sensor agent
    """
    
    def __init__(self, config):
        """Initialize enhanced sensor agent with unified risk management"""
        self.config = config
        
        # Initialize unified risk manager
        self.risk_manager = UnifiedRiskManager(
            config=config,
            udp_port=9999,  # Listen for AGX Orin data on this port
            audio_enabled=True  # Enable audio alerts
        )
        
        # Your existing sensor agent initialization would go here
        # self.sensor_setup()
        # self.model_loading()
        # etc.
        
        print("Enhanced Sensor Agent with Unified Risk Management initialized")
        
    def sensor_tick(self, input_data, timestamp):
        """
        Main sensor processing loop with integrated risk assessment
        
        Args:
            input_data: Sensor data from CARLA
            timestamp: Current timestamp
            
        Returns:
            dict: Control commands and risk assessment
        """
        
        # Update risk manager timestamp
        self.risk_manager.safety_evaluator.update_timestamp(timestamp)
        self.risk_manager.safety_evaluator.update_time(timestamp)
        
        # Extract ego vehicle data (you would get this from your existing system)
        ego_speed = self.get_ego_speed(input_data)  # Implement based on your setup
        
        # Extract detected objects (you would get this from your existing perception)
        bounding_boxes = self.get_detected_objects(input_data)  # Implement based on your setup
        
        # =================================================================
        # UNIFIED RISK ASSESSMENT - This is the new addition
        # =================================================================
        
        risk_assessment = self.risk_manager.calculate_unified_risk(
            ego_speed=ego_speed,
            bounding_boxes=bounding_boxes
        )
        
        # Extract key risk information
        unified_risk = risk_assessment['unified_risk']
        risk_level = risk_assessment['risk_level']
        external_risk = risk_assessment['external_risk'] 
        internal_risk = risk_assessment['internal_risk']
        driver_state = risk_assessment['driver_state']['current_state']
        
        # =================================================================
        # RISK-AWARE CONTROL DECISIONS
        # =================================================================
        
        # Generate your normal control commands
        control_commands = self.generate_control_commands(input_data)
        
        # Apply risk-aware modifications to control commands
        control_commands = self.apply_risk_aware_control(
            control_commands, 
            risk_assessment
        )
        
        # =================================================================
        # ADDITIONAL RISK-BASED ACTIONS
        # =================================================================
        
        # Log critical situations for analysis
        if risk_level in ['critical', 'emergency']:
            self.log_critical_event(risk_assessment, timestamp)
            
        # Trigger emergency protocols if needed
        if risk_level == 'emergency':
            control_commands = self.emergency_intervention(control_commands, risk_assessment)
            
        # =================================================================
        # RETURN ENHANCED RESULTS
        # =================================================================
        
        result = {
            'control': control_commands,
            'risk_assessment': risk_assessment,
            'risk_metadata': {
                'unified_risk': unified_risk,
                'risk_level': risk_level,
                'external_risk': external_risk,
                'internal_risk': internal_risk,
                'driver_state': driver_state,
                'timestamp': timestamp
            }
        }
        
        return result
        
    def apply_risk_aware_control(self, control_commands, risk_assessment):
        """
        Modify control commands based on risk assessment
        
        Args:
            control_commands: Original control commands
            risk_assessment: Unified risk assessment results
            
        Returns:
            dict: Modified control commands
        """
        risk_level = risk_assessment['risk_level']
        unified_risk = risk_assessment['unified_risk']
        driver_state = risk_assessment['driver_state']['current_state']
        external_risk = risk_assessment['external_risk']
        
        # Create modified control commands
        modified_control = control_commands.copy()
        
        # Apply risk-based speed adjustments
        if risk_level == 'emergency':
            # Emergency: Aggressive braking
            modified_control['throttle'] = 0.0
            modified_control['brake'] = min(1.0, unified_risk)
            print(f"🚨 EMERGENCY BRAKING: brake={modified_control['brake']:.2f}")
            
        elif risk_level == 'critical':
            # Critical: Strong deceleration
            if driver_state in ['sleepy', 'using_phone']:
                # More aggressive if driver is inattentive
                modified_control['throttle'] *= 0.3
                modified_control['brake'] = max(modified_control.get('brake', 0), 0.4)
            else:
                modified_control['throttle'] *= 0.5
                modified_control['brake'] = max(modified_control.get('brake', 0), 0.3)
            print(f"⚠️ CRITICAL DECELERATION: throttle={modified_control['throttle']:.2f}")
            
        elif risk_level == 'warning':
            # Warning: Moderate speed reduction
            speed_factor = 1.0 - (unified_risk * 0.3)
            modified_control['throttle'] *= speed_factor
            print(f"⚠️ WARNING SPEED REDUCTION: factor={speed_factor:.2f}")
            
        elif risk_level == 'caution':
            # Caution: Slight speed reduction
            if driver_state != 'safe_driving':
                # Be more conservative if driver is not fully alert
                speed_factor = 1.0 - (unified_risk * 0.2)
                modified_control['throttle'] *= speed_factor
                print(f"⚠️ CAUTION (driver {driver_state}): factor={speed_factor:.2f}")
        
        # Apply emergency vehicle yielding behavior
        primary_threat = risk_assessment['external_breakdown'].get('primary_threat')
        if primary_threat and primary_threat.get('is_emergency', False):
            # Emergency vehicle detected - implement yielding behavior
            if external_risk > 0.3:
                modified_control['throttle'] *= 0.6  # Reduce speed for yielding
                print("🚑 EMERGENCY VEHICLE YIELDING")
                
        return modified_control
        
    def emergency_intervention(self, control_commands, risk_assessment):
        """
        Emergency intervention protocol for critical situations
        
        Args:
            control_commands: Current control commands
            risk_assessment: Risk assessment data
            
        Returns:
            dict: Emergency control commands
        """
        emergency_control = {
            'throttle': 0.0,
            'brake': 1.0,
            'steer': control_commands.get('steer', 0.0),  # Maintain steering
            'hand_brake': False,
            'reverse': False
        }
        
        # Log emergency intervention
        print("🆘 EMERGENCY INTERVENTION ACTIVATED")
        print(f"   Unified Risk: {risk_assessment['unified_risk']:.3f}")
        print(f"   Driver State: {risk_assessment['driver_state']['current_state']}")
        
        primary_threat = risk_assessment['external_breakdown'].get('primary_threat')
        if primary_threat:
            print(f"   Primary Threat: {primary_threat.get('class_name', 'unknown')} at {primary_threat.get('distance', 0):.1f}m")
            
        return emergency_control
        
    def log_critical_event(self, risk_assessment, timestamp):
        """
        Log critical events for later analysis
        
        Args:
            risk_assessment: Risk assessment data
            timestamp: Event timestamp
        """
        # This would typically write to a file or database
        critical_event = {
            'timestamp': timestamp,
            'unified_risk': risk_assessment['unified_risk'],
            'risk_level': risk_assessment['risk_level'],
            'external_risk': risk_assessment['external_risk'],
            'internal_risk': risk_assessment['internal_risk'],
            'driver_state': risk_assessment['driver_state']['current_state'],
            'driver_confidence': risk_assessment['driver_state']['confidence'],
            'primary_threat': risk_assessment['external_breakdown'].get('primary_threat'),
            'synergy_factor': risk_assessment['risk_factors']['synergy_factor']
        }
        
        # For this example, just print (in real implementation, save to file)
        print(f"💾 CRITICAL EVENT LOGGED: {risk_assessment['risk_level'].upper()} at {timestamp:.2f}s")
        
    def get_ego_speed(self, input_data):
        """
        Extract ego vehicle speed from input data
        
        Args:
            input_data: Sensor input data
            
        Returns:
            float: Ego speed in m/s
        """
        # TODO: Implement based on your existing sensor data structure
        # This is a placeholder - you would extract speed from your input_data
        
        # Example implementation (replace with your actual speed extraction):
        if hasattr(input_data, 'ego_vehicle'):
            velocity = input_data.ego_vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            return speed
        else:
            # Fallback for testing
            return 15.0  # 15 m/s (~54 km/h)
            
    def get_detected_objects(self, input_data):
        """
        Extract detected objects from perception system
        
        Args:
            input_data: Sensor input data
            
        Returns:
            list: Bounding boxes in the expected format
        """
        # TODO: Implement based on your existing perception pipeline
        # This is a placeholder - you would extract objects from your perception system
        
        # Expected format: [x, y, z, extent_x, extent_y, extent_z, rotation, class]
        # Where: x=forward distance, y=lateral offset, class: 0=vehicle, 1=pedestrian, 4=emergency
        
        # Example implementation (replace with your actual object detection):
        if hasattr(input_data, 'detected_objects'):
            return input_data.detected_objects
        else:
            # Fallback for testing - simulate some objects
            return [
                [20.0, 0.5, 0, 2.0, 1.0, 1.5, 0, 0],    # Vehicle 20m ahead, slightly right
                [35.0, -1.2, 0, 2.0, 1.0, 1.5, 0, 0],   # Vehicle 35m ahead, left lane
            ]
            
    def generate_control_commands(self, input_data):
        """
        Generate your normal control commands (placeholder)
        
        Args:
            input_data: Sensor input data
            
        Returns:
            dict: Control commands
        """
        # TODO: Replace with your existing control logic
        # This is a placeholder for your existing control system
        
        return {
            'throttle': 0.6,
            'brake': 0.0,
            'steer': 0.0,
            'hand_brake': False,
            'reverse': False
        }
        
    def shutdown(self):
        """Clean shutdown of the enhanced sensor agent"""
        print("Shutting down Enhanced Sensor Agent...")
        
        # Shutdown risk manager
        self.risk_manager.shutdown()
        
        # Your existing shutdown logic would go here
        print("Enhanced Sensor Agent shutdown complete")


# =================================================================
# EXAMPLE USAGE AND TESTING
# =================================================================

def main():
    """Example of how to use the enhanced sensor agent"""
    
    # Mock config (replace with your actual config)
    class MockConfig:
        def __init__(self):
            self.debug = True
            
    config = MockConfig()
    
    # Initialize enhanced sensor agent
    agent = EnhancedSensorAgent(config)
    
    print("\nEnhanced Sensor Agent started")
    print("Start the AGX Orin simulator in another terminal:")
    print("python team_code/test_agx_orin_simulator.py --mode scenario --scenario mixed_risks")
    print("\nPress Ctrl+C to stop")
    
    try:
        # Simulate sensor ticks
        for frame_id in range(100):  # Simulate 100 frames
            timestamp = time.time()
            
            # Mock input data (replace with your actual sensor data)
            class MockInputData:
                def __init__(self):
                    pass
                    
            input_data = MockInputData()
            
            # Process sensor tick with unified risk assessment
            result = agent.sensor_tick(input_data, timestamp)
            
            # Show results
            risk_meta = result['risk_metadata']
            print(f"\nFrame {frame_id}:")
            print(f"  Unified Risk: {risk_meta['unified_risk']:.3f}")
            print(f"  Risk Level: {risk_meta['risk_level']}")
            print(f"  Driver State: {risk_meta['driver_state']}")
            print(f"  Control: throttle={result['control']['throttle']:.2f}, brake={result['control']['brake']:.2f}")
            
            # Simulate 10Hz processing
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        agent.shutdown()

if __name__ == "__main__":
    main() 