#!/usr/bin/env python3
"""
AGX Orin Driver Detection Simulator
Simulates driver state detection results and sends them via UDP to the Unified Risk Manager
"""

import socket
import json
import time
import random
import argparse

class AGXOrinSimulator:
    def __init__(self, target_ip="127.0.0.1", target_port=9999):
        """
        Initialize the AGX Orin simulator
        
        Args:
            target_ip: IP address of the unified risk manager
            target_port: UDP port of the unified risk manager
        """
        self.target_ip = target_ip
        self.target_port = target_port
        
        # Driver states and their typical confidence ranges
        self.driver_states = {
            'safe_driving': {'min_conf': 0.7, 'max_conf': 0.95},
            'sleepy': {'min_conf': 0.6, 'max_conf': 0.9},
            'reaching_back': {'min_conf': 0.65, 'max_conf': 0.85},
            'using_phone': {'min_conf': 0.7, 'max_conf': 0.92}
        }
        
        # State transition probabilities (simulates realistic driver behavior)
        self.transition_matrix = {
            'safe_driving': {
                'safe_driving': 0.85,  # Likely to stay safe
                'sleepy': 0.08,        # May become sleepy
                'reaching_back': 0.05, # May reach back
                'using_phone': 0.02    # May use phone
            },
            'sleepy': {
                'safe_driving': 0.3,   # May become alert
                'sleepy': 0.65,        # Likely to stay sleepy
                'reaching_back': 0.03,
                'using_phone': 0.02
            },
            'reaching_back': {
                'safe_driving': 0.6,   # Usually temporary action
                'sleepy': 0.05,
                'reaching_back': 0.3,  # May continue reaching
                'using_phone': 0.05
            },
            'using_phone': {
                'safe_driving': 0.4,   # May put phone down
                'sleepy': 0.1,
                'reaching_back': 0.05,
                'using_phone': 0.45    # May continue using phone
            }
        }
        
        # Initialize current state
        self.current_state = 'safe_driving'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        print(f"AGX Orin Simulator initialized")
        print(f"Target: {self.target_ip}:{self.target_port}")
        print(f"Available states: {list(self.driver_states.keys())}")
        
    def get_next_state(self):
        """Generate next driver state based on transition probabilities"""
        transitions = self.transition_matrix[self.current_state]
        rand_val = random.random()
        
        cumulative_prob = 0.0
        for state, prob in transitions.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return state
                
        # Fallback (shouldn't happen with proper probabilities)
        return self.current_state
        
    def generate_confidence(self, state):
        """Generate realistic confidence score for a given state"""
        conf_range = self.driver_states[state]
        base_confidence = random.uniform(conf_range['min_conf'], conf_range['max_conf'])
        
        # Add some noise to make it more realistic
        noise = random.gauss(0, 0.05)  # Small gaussian noise
        confidence = max(0.0, min(1.0, base_confidence + noise))
        
        return confidence
        
    def send_driver_state(self, state=None, confidence=None):
        """
        Send driver state via UDP
        
        Args:
            state: Driver state to send (None for automatic generation)
            confidence: Confidence score (None for automatic generation)
        """
        if state is None:
            state = self.get_next_state()
            self.current_state = state
            
        if confidence is None:
            confidence = self.generate_confidence(state)
            
        # Create message
        message = {
            "driver_state": state,
            "confidence": round(confidence, 3),
            "timestamp": time.time()
        }
        
        try:
            # Send UDP message
            message_json = json.dumps(message)
            self.sock.sendto(message_json.encode('utf-8'), (self.target_ip, self.target_port))
            
            print(f"Sent: {state} (confidence: {confidence:.3f})")
            return True
            
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
            
    def run_simulation(self, duration=60, frequency=2.0):
        """
        Run continuous simulation
        
        Args:
            duration: Simulation duration in seconds (0 for infinite)
            frequency: Updates per second
        """
        print(f"\nStarting simulation...")
        print(f"Duration: {'infinite' if duration == 0 else f'{duration}s'}")
        print(f"Frequency: {frequency} Hz")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        update_interval = 1.0 / frequency
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration > 0 and (current_time - start_time) >= duration:
                    print("Simulation duration completed")
                    break
                    
                # Send driver state update
                self.send_driver_state()
                
                # Wait for next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            
        finally:
            self.sock.close()
            print("AGX Orin Simulator shut down")
            
    def run_scenario(self, scenario_name):
        """
        Run predefined test scenarios
        
        Args:
            scenario_name: Name of the scenario to run
        """
        scenarios = {
            'normal_driving': [
                ('safe_driving', 0.85, 5),
                ('safe_driving', 0.90, 5),
                ('safe_driving', 0.80, 5)
            ],
            'getting_sleepy': [
                ('safe_driving', 0.85, 3),
                ('sleepy', 0.65, 4),
                ('sleepy', 0.80, 4),
                ('sleepy', 0.75, 4)
            ],
            'phone_usage': [
                ('safe_driving', 0.90, 3),
                ('using_phone', 0.85, 5),
                ('using_phone', 0.78, 3),
                ('safe_driving', 0.85, 3)
            ],
            'distraction_event': [
                ('safe_driving', 0.88, 3),
                ('reaching_back', 0.75, 2),
                ('reaching_back', 0.70, 2),
                ('safe_driving', 0.85, 5)
            ],
            'mixed_risks': [
                ('safe_driving', 0.85, 2),
                ('sleepy', 0.70, 3),
                ('using_phone', 0.80, 3),
                ('reaching_back', 0.75, 2),
                ('sleepy', 0.85, 4),
                ('safe_driving', 0.90, 3)
            ]
        }
        
        if scenario_name not in scenarios:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return
            
        print(f"\nRunning scenario: {scenario_name}")
        scenario = scenarios[scenario_name]
        
        try:
            for state, confidence, duration in scenario:
                print(f"Phase: {state} (conf: {confidence:.2f}) for {duration}s")
                
                # Send updates at 2Hz during this phase
                updates = int(duration * 2)
                for _ in range(updates):
                    self.send_driver_state(state, confidence)
                    time.sleep(0.5)  # 2Hz
                    
        except KeyboardInterrupt:
            print("\nScenario stopped by user")
            
        finally:
            self.sock.close()
            print("Scenario completed")

def main():
    parser = argparse.ArgumentParser(description='AGX Orin Driver Detection Simulator')
    parser.add_argument('--ip', default='127.0.0.1', help='Target IP address')
    parser.add_argument('--port', type=int, default=9999, help='Target UDP port')
    parser.add_argument('--mode', choices=['simulation', 'scenario', 'manual'], 
                       default='simulation', help='Operation mode')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Simulation duration in seconds (0 for infinite)')
    parser.add_argument('--frequency', type=float, default=2.0, 
                       help='Update frequency in Hz')
    parser.add_argument('--scenario', choices=['normal_driving', 'getting_sleepy', 
                       'phone_usage', 'distraction_event', 'mixed_risks'],
                       default='mixed_risks', help='Scenario to run')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = AGXOrinSimulator(args.ip, args.port)
    
    if args.mode == 'simulation':
        # Run continuous simulation
        simulator.run_simulation(args.duration, args.frequency)
        
    elif args.mode == 'scenario':
        # Run predefined scenario
        simulator.run_scenario(args.scenario)
        
    elif args.mode == 'manual':
        # Manual mode for testing
        print("\nManual mode - enter driver states:")
        print("Valid states: safe_driving, sleepy, reaching_back, using_phone")
        print("Format: state [confidence] or 'quit' to exit")
        
        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break
                    
                parts = user_input.split()
                if len(parts) == 0:
                    continue
                    
                state = parts[0]
                confidence = float(parts[1]) if len(parts) > 1 else None
                
                if state not in simulator.driver_states:
                    print(f"Invalid state. Valid states: {list(simulator.driver_states.keys())}")
                    continue
                    
                simulator.send_driver_state(state, confidence)
                
        except KeyboardInterrupt:
            print("\nManual mode stopped")
        finally:
            simulator.sock.close()

if __name__ == "__main__":
    main() 