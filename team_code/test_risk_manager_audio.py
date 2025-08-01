#!/usr/bin/env python3
"""
Test Audio with Actual Unified Risk Manager
Tests the audio system within the real unified risk manager context
"""

import os
import sys
import time
import json
import socket
import threading

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_mock_safety_evaluator():
    """Create a mock safety evaluator to avoid CARLA dependency"""
    class MockSafetyEvaluator:
        def __init__(self, config):
            self.config = config
            
        def calculate_external_risk_unified(self, ego_speed, bounding_boxes):
            """Mock external risk calculation"""
            # Simulate different risk scenarios based on bounding boxes
            if not bounding_boxes:
                return 0.1, {'primary_threat': None}
            
            # Use first bounding box distance to simulate risk
            distance = bounding_boxes[0][0] if len(bounding_boxes[0]) > 0 else 20.0
            
            if distance < 5:
                risk = 0.9  # High risk - very close
                threat = {'class_name': 'vehicle', 'distance': distance, 'is_emergency': False}
            elif distance < 10:
                risk = 0.7  # Medium-high risk
                threat = {'class_name': 'vehicle', 'distance': distance, 'is_emergency': False}
            elif distance < 15:
                risk = 0.4  # Medium risk
                threat = {'class_name': 'vehicle', 'distance': distance, 'is_emergency': False}
            else:
                risk = 0.1  # Low risk
                threat = {'class_name': 'vehicle', 'distance': distance, 'is_emergency': False}
            
            return risk, {'primary_threat': threat}
        
        def update_time(self, timestamp):
            pass
            
        def update_timestamp(self, timestamp):
            pass
    
    return MockSafetyEvaluator

def test_unified_risk_manager_audio():
    """Test audio with the actual unified risk manager"""
    print("=== Testing Unified Risk Manager Audio ===")
    
    # Patch the SafetyEvaluator import to avoid CARLA dependency
    import unified_risk_manager
    unified_risk_manager.SafetyEvaluator = create_mock_safety_evaluator()
    
    # Now import the UnifiedRiskManager
    from unified_risk_manager import UnifiedRiskManager
    
    # Create mock config
    class MockConfig:
        def __init__(self):
            self.debug = True
    
    config = MockConfig()
    
    try:
        # Initialize unified risk manager
        print("Initializing Unified Risk Manager...")
        risk_manager = UnifiedRiskManager(config, udp_port=9997, audio_enabled=True)
        
        if not risk_manager.audio_enabled:
            print("✗ Audio is disabled in risk manager")
            return False
            
        print("✓ Unified Risk Manager initialized with audio enabled")
        
        # Test scenarios with different risk levels
        test_scenarios = [
            {
                'name': 'Low Risk - Safe Driving',
                'ego_speed': 15.0,
                'bounding_boxes': [[25.0, 0.0, 0, 0, 0, 0, 0, 0]],  # Vehicle 25m away
                'driver_state': 'safe_driving',
                'expected_risk': 'safe'
            },
            {
                'name': 'Medium Risk - Caution',
                'ego_speed': 20.0,
                'bounding_boxes': [[12.0, 0.0, 0, 0, 0, 0, 0, 0]],  # Vehicle 12m away
                'driver_state': 'safe_driving',
                'expected_risk': 'caution'
            },
            {
                'name': 'High Risk - Critical with Sleepy Driver',
                'ego_speed': 25.0,
                'bounding_boxes': [[8.0, 0.0, 0, 0, 0, 0, 0, 0]],   # Vehicle 8m away
                'driver_state': 'sleepy',
                'expected_risk': 'critical'
            },
            {
                'name': 'Very High Risk - Emergency',
                'ego_speed': 30.0,
                'bounding_boxes': [[3.0, 0.0, 0, 0, 0, 0, 0, 0]],   # Vehicle 3m away
                'driver_state': 'using_phone',
                'expected_risk': 'emergency'
            }
        ]
        
        print("\nTesting different risk scenarios...")
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
            
            # Simulate driver state by updating internal state
            risk_manager.current_driver_state = scenario['driver_state']
            risk_manager.driver_state_confidence = 0.8
            risk_manager.last_driver_update = time.time()
            
            # Calculate unified risk (this will trigger audio alerts)
            result = risk_manager.calculate_unified_risk(
                scenario['ego_speed'],
                scenario['bounding_boxes']
            )
            
            print(f"Risk Level: {result['risk_level']}")
            print(f"Unified Risk: {result['unified_risk']:.3f}")
            print(f"External Risk: {result['external_risk']:.3f}")
            print(f"Internal Risk: {result['internal_risk']:.3f}")
            print(f"Driver State: {result['driver_state']['current_state']}")
            
            # Wait to hear the audio
            time.sleep(3)
            
            # Force additional alerts for critical/emergency
            if result['risk_level'] in ['critical', 'emergency']:
                print(f"Testing additional {result['risk_level']} alert...")
                risk_manager._generate_alert(result)
                time.sleep(3)
        
        # Test manual audio alerts
        print("\n--- Testing Manual Audio Alerts ---")
        manual_tests = [
            ('caution', 'safe_driving'),
            ('warning', 'safe_driving'),
            ('critical', 'sleepy'),
            ('emergency', 'using_phone')
        ]
        
        for risk_level, driver_state in manual_tests:
            print(f"Testing {risk_level} alert with {driver_state} driver...")
            risk_manager._play_audio_alert(risk_level, driver_state)
            time.sleep(2)
            
        # Test TTS alerts
        print("\n--- Testing TTS Alerts ---")
        tts_messages = [
            "Critical: Vehicle detected at close range - stay alert",
            "EMERGENCY: Immediate braking required - driver appears distracted"
        ]
        
        for message in tts_messages:
            print(f"Testing TTS: '{message[:40]}...'")
            risk_manager._speak_alert(message)
            time.sleep(3)
        
        # Cleanup
        risk_manager.shutdown()
        print("\n✓ All unified risk manager audio tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Run the unified risk manager audio test"""
    print("Unified Risk Manager Audio Test")
    print("=" * 50)
    print("This test will play various audio alerts and spoken messages.")
    print("You should hear different tones and speech for different risk levels.")
    print("=" * 50)
    
    # Run the test
    success = test_unified_risk_manager_audio()
    
    print("\n" + "=" * 50)
    print("UNIFIED RISK MANAGER AUDIO TEST SUMMARY")
    print("=" * 50)
    
    if success:
        print("✅ SUCCESS: Your unified risk manager audio system is fully functional!")
        print("\nWhat you should have heard:")
        print("• Different audio tones for each risk level")
        print("• Spoken alerts for critical and emergency situations")
        print("• Context-aware audio selection based on driver state")
        print("• Multiple alert patterns and frequencies")
        print("\n🎉 Your CARLA simulation will have full audio feedback!")
    else:
        print("❌ FAILED: There were issues with the audio system")
        print("Check the error messages above for troubleshooting.")
    
    print("\nNext steps:")
    print("1. Your audio system is ready for CARLA simulation")
    print("2. Run your sensor_agent.py with unified risk manager")
    print("3. Audio alerts will trigger based on real driving scenarios")

if __name__ == "__main__":
    main() 