import os
import time
import carla
from copy import deepcopy
import sys
sys.path.append('/home/ascc304/transfuser')
sys.path.append('/home/ascc304/transfuser/team_code_autopilot') 

from leaderboard.autoagents import autonomous_agent
from submission_agent import HybridAgent
from autopilot_2 import EnhancedAutoPilot
import matplotlib.pyplot as plt
import numpy as np

def get_entry_point():
    return 'DualAgent'

class DualAgent(HybridAgent):
    def setup(self, path_to_conf_file, route_index=None):
        # Initialize TransFuser (parent class)
        super().setup(path_to_conf_file, route_index)
        
        # Initialize EnhancedAutoPilot with the config file
        self.enhanced_autopilot = EnhancedAutoPilot(path_to_conf_file)
        self.enhanced_autopilot.setup(path_to_conf_file, route_index)

        # Control switching variables
        self.use_transfuser = True
        self.last_switch_time = time.time()
        self.switch_interval = 10  # Switch every 10 seconds

        # Add new variables for hazard detection and timing
        self.last_hazard_time = None
        self.hazard_window = 5  # 5-second window to use EnhancedAutoPilot after hazard

        # Add data collection lists
        self.time_points = []
        self.transfuser_controls = {'steer': [], 'throttle': [], 'brake': []}
        self.enhanced_controls = {'steer': [], 'throttle': [], 'brake': []}
        self.hazard_points = []
        self.start_time = time.time()

    def _init(self):
        super()._init()
        # Initialize EnhancedAutoPilot with the same map data and route plan
        if hasattr(self, '_hd_map'):
            # Create a new list with tuples containing waypoints and road options
            self.enhanced_autopilot._global_plan = [(waypoint, road_option) 
                for waypoint, road_option in self._global_plan]
            
            # Create a new list with tuples containing locations and road options
            self.enhanced_autopilot._global_plan_world_coord = [(location, road_option) 
                for location, road_option in self._global_plan_world_coord]
            
            self.enhanced_autopilot._init(self._hd_map)

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        # Combine sensors from both agents
        sensors = super().sensors()
        enhanced_sensors = self.enhanced_autopilot.sensors()
        
        # Merge sensors, avoiding duplicates
        sensor_ids = set()
        merged_sensors = []
        
        for sensor in sensors + enhanced_sensors:
            if sensor['id'] not in sensor_ids:
                sensor_ids.add(sensor['id'])
                merged_sensors.append(sensor)
        
        return merged_sensors

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            if 'hd_map' in input_data:
                self._hd_map = input_data['hd_map']
            self._init()
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        # Get control commands from both agents
        control_transfuser = super().run_step(input_data, timestamp)
        control_enhanced = self.enhanced_autopilot.run_step(input_data, timestamp)

        # Record current time point
        current_time = time.time() - self.start_time
        self.time_points.append(current_time)

        # Record control values
        self.transfuser_controls['steer'].append(control_transfuser.steer)
        self.transfuser_controls['throttle'].append(control_transfuser.throttle)
        self.transfuser_controls['brake'].append(control_transfuser.brake)

        self.enhanced_controls['steer'].append(control_enhanced.steer)
        self.enhanced_controls['throttle'].append(control_enhanced.throttle)
        self.enhanced_controls['brake'].append(control_enhanced.brake)

        # Check vehicle hazard status
        vehicle_hazard = any(self.enhanced_autopilot.vehicle_hazard)
        if vehicle_hazard:
            self.hazard_points.append(current_time)
            self.last_hazard_time = current_time

        # Calculate the difference in brake values
        brake_difference = abs(control_transfuser.brake - control_enhanced.brake)

        # Define a threshold for the brake difference
        brake_threshold = 0.5  # Adjust this value as needed

        # Determine which control to use based on hazard, time window, and brake difference
        if (self.last_hazard_time is not None and current_time - self.last_hazard_time < self.hazard_window) or brake_difference > brake_threshold:
            print(f"Using EnhancedAutoPilot (Brake difference: {brake_difference:.2f})")
            return control_enhanced
        else:
            print("No recent hazard and brake difference within threshold. Using TransFuser")
            return control_transfuser

    def destroy(self):
        """
        Destroy (clean-up) the agent and plot the results
        """
        # Plot the control values
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot steer values
        ax1.plot(self.time_points, self.transfuser_controls['steer'], 'b-', label='TransFuser', alpha=0.7)
        ax1.plot(self.time_points, self.enhanced_controls['steer'], 'g-', label='Enhanced', alpha=0.7)
        ax1.set_ylabel('Steer')
        ax1.grid(True)
        ax1.legend()

        # Plot throttle values
        ax2.plot(self.time_points, self.transfuser_controls['throttle'], 'b-', label='TransFuser', alpha=0.7)
        ax2.plot(self.time_points, self.enhanced_controls['throttle'], 'g-', label='Enhanced', alpha=0.7)
        ax2.set_ylabel('Throttle')
        ax2.grid(True)
        ax2.legend()

        # Plot brake values
        ax3.plot(self.time_points, self.transfuser_controls['brake'], 'b-', label='TransFuser', alpha=0.7)
        ax3.plot(self.time_points, self.enhanced_controls['brake'], 'g-', label='Enhanced', alpha=0.7)
        ax3.set_ylabel('Brake')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        ax3.legend()

        # Add vertical lines for hazard points
        for hazard_time in self.hazard_points:
            ax1.axvline(x=hazard_time, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=hazard_time, color='r', linestyle='--', alpha=0.5)
            ax3.axvline(x=hazard_time, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('control_plots.png')
        plt.close()

        super().destroy()
        self.enhanced_autopilot.destroy()
