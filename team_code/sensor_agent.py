"""
Agent file that runs the evaluations for all models supported by this repo.
Run it by giving it as the agent option to the
leaderboard/leaderboard/leaderboard_evaluator.py file
"""

import os
from copy import deepcopy
from typing import Dict, List, Optional, Any

import cv2
import carla
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from model import LidarCenterNet
from config import GlobalConfig
from data import CARLA_Data
from nav_planner import RoutePlanner
from nav_planner import extrapolate_waypoint_route

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.optimize import fsolve
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from scenario_logger import ScenarioLogger
import transfuser_utils as t_u
from safety_evaluator import SafetyEvaluator
from unified_risk_manager import UnifiedRiskManager

import pathlib
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson  # Like json but faster
import gzip
import pygame
import time
import json
from datetime import datetime

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'SensorAgent'


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


class SensorAgent(autonomous_agent.AutonomousAgent):
  """
    Main class that runs the agents with the run_step function
    """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    # super().__init__(path_to_conf_file)
    
    # Initialize data recording variables
    self.data_recording = {
      'session_start_time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
      'control_commands': {
        'ai': [],
        'human': []
      },
      'interventions': [],
      'control_time': {
        'ai': 0,
        'human': 0
      },
      'steering_differences': [],
      'speed_differences': [],
      'last_mode_switch_time': time.time(),
      'current_mode': 'autonomous'
    }
    
    # Create a file to save the data with absolute path
    data_dir = os.path.join(os.getcwd(), "driving_data")
    os.makedirs(data_dir, exist_ok=True)
    self.data_file = os.path.join(data_dir, f"driving_data_{self.data_recording['session_start_time']}.json")
    print(f"Data will be saved to: {os.path.abspath(self.data_file)}")
    
    # Record data every N frames
    self.record_frequency = 10
    self.last_ai_control = None

    # Initialize pygame display with width for center camera only
    self.camera_width = 1920
    self.camera_height = 960
    self.width = self.camera_width  # Just center camera width
    self.height = self.camera_height
    self.display_current_speed = 0
    self.display_predicted_speed = 0
    
    # Initialize pygame with scaling support
    pygame.init()
    pygame.font.init()
    
    # Get the display info to calculate scaling
    display_info = pygame.display.Info()
    screen_w = display_info.current_w
    screen_h = display_info.current_h
    
    # Calculate scale factor to fit screen while maintaining aspect ratio
    scale_w = screen_w / self.width
    scale_h = screen_h / self.height
    scale = min(scale_w, scale_h) * 0.9  # Use 90% of available space
    
    # Calculate scaled dimensions
    scaled_width = int(self.width * scale)
    scaled_height = int(self.height * scale)
    
    # Center the window
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    # Create scaled display
    self._display = pygame.display.set_mode((scaled_width, scaled_height), 
                                          pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED)
    pygame.display.set_caption("Sensor Agent View")
    self._clock = pygame.time.Clock()
    
    # Initialize font for display
    font_name = 'courier' if os.name == 'nt' else 'mono'
    fonts = [x for x in pygame.font.get_fonts() if font_name in x]
    default_font = 'ubuntumono'
    mono = default_font if default_font in fonts else fonts[0]
    mono = pygame.font.match_font(mono)
    # Increase font size for better visibility
    self._font_mono = pygame.font.Font(mono, 24 if os.name == 'nt' else 28)
    
    # Initialize steering wheel control
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
      if joystick_count > 1:
        raise ValueError("Please Connect Just One Joystick")
      self._joystick = pygame.joystick.Joystick(0)
      self._joystick.init()
      self._steer_idx = 0
      self._throttle_idx = 2
      self._brake_idx = 3
      self._reverse_idx = 5
      self._handbrake_idx = 4
      self._square_idx = 1  # Square button index
      self.manual_control = False
      self.brake_pressed = False  # Track brake pedal state
      self.last_switch_time = 0  # Add debounce timer
      print("Joystick initialized - Starting in AUTONOMOUS mode")
    else:
      self._joystick = None
      self.manual_control = False  # Ensure this is set even without joystick
      print("No steering wheel detected - autonomous control only")

    # Set environment variables for better pygame performance
    os.environ['SDL_VIDEO_X11_VISUAL'] = '0'  # For Linux
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    torch.cuda.empty_cache()
    self.IS_BENCH2DRIVE = strtobool(os.environ.get('IS_BENCH2DRIVE', 'False'))
    print('IS_BENCH2DRIVE: ', self.IS_BENCH2DRIVE)
    self.track = autonomous_agent.Track.MAP if os.environ.get(
        'CHALLENGE_TRACK_CODENAME') == 'MAP' else autonomous_agent.Track.SENSORS
    if self.IS_BENCH2DRIVE:
      self.config_path = path_to_conf_file.split('+')[0]
    else:
      self.config_path = path_to_conf_file

    self.step = -1
    self.initialized = False
    self.device = torch.device('cuda:0')

    # Load the config saved during training
    with open(os.path.join(self.config_path, 'config.json'), 'rt', encoding='utf-8') as f:
      json_config = f.read()

    loaded_config = jsonpickle.decode(json_config)

    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the saved config.
    self.config.__dict__.update(loaded_config.__dict__)

    # For models supporting different output modalities we select which one to use here.
    # 0: Waypoints
    # 1: Path + Target Speed

    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    self.tuned_aim_distance = int(os.environ.get('TUNED_AIM_DISTANCE', 0))
    print('TUNED_AIM_DISTANCE for wp rep?: ', self.tuned_aim_distance)
    direct = os.environ.get('DIRECT', 1)
    self.config.inference_direct_controller = int(direct)
    print('Direct control prediction?: ', direct)
    self.stop_after_meter = int(os.environ.get('STOP_AFTER_METER', -1))
    print('STOP_AFTER_METER: ', self.stop_after_meter)

    # If set to true, will generate visualizations at SAVE_PATH
    self.config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0)) == 1

    self.compile = int(os.environ.get('COMPILE', 0)) == 1

    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
    print('Brake uncertainty threshold: ', self.config.brake_uncertainty_threshold)

    # Classification networks are known to be overconfident which leads to them braking a bit too late in our case.
    # Reducing the driving speed slightly counteracts that.
    if int(os.environ.get('SLOWER', 0)):
      print(f'Reduce target speeds during evaluation by factor {self.config.slower_factor}.')
      self.inference_target_speeds = [self.config.slower_factor * speed for speed in self.config.target_speeds]
    else:
      print('No speed reduction during inference.')
      self.inference_target_speeds = self.config.target_speeds

    if self.config.tp_attention:
      self.tp_attention_buffer = []

    # Stop signs can be occluded with our camera setup. This buffer remembers them until cleared.
    # Very useful on the LAV benchmark
    self.stop_sign_controller = int(os.environ.get('STOP_CONTROL', 1))
    print('Use stop sign controller:', self.stop_sign_controller)
    if self.stop_sign_controller:
      # There can be max 1 stop sign affecting the ego
      self.stop_sign_buffer = deque(maxlen=1)
      self.clear_stop_sign = 0  # Counter if we recently cleared a stop sign

    # Load model files
    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(self.config_path):
      if file.endswith('.pth') and file.startswith('model'):
        self.model_count += 1
        print(os.path.join(self.config_path, file))
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(self.config_path, file), map_location=self.device)
        net.load_state_dict(state_dict, strict=True)
        net.cuda(device=self.device)
        net.eval()

        if self.config.compile or self.compile:
          net = torch.compile(net, mode=self.config.compile_mode)

        self.nets.append(net)

    self.stuck_detector = 0
    self.force_move = 0

    self.bb_buffer = deque(maxlen=1)
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.target_point_prev = [1e5, 1e5, 1e5]

    # Filtering
    self.ego_model = EgoModel(dt=self.config.carla_frame_rate)
    self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
    # Still uses the leaderboard 1.0 bicycle model for the unscented kalman filter
    self.ukf = UKF(dim_x=4,
                   dim_z=4,
                   fx=bicycle_model_forward,
                   hx=measurement_function_hx,
                   dt=self.config.carla_frame_rate,
                   points=self.points,
                   x_mean_fn=state_mean,
                   z_mean_fn=measurement_mean,
                   residual_x=residual_state_x,
                   residual_z=residual_measurement_h)

    # State noise, same as measurement because we
    # initialize with the first measurement later
    self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
    # Measurement noise
    self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
    self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
    # Used to set the filter state equal the first measurement
    self.filter_initialized = False
    # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
    self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))

    #Temporal LiDAR
    self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)

    self.lidar_last = None

    # Forced stopping
    if self.stop_after_meter > 0:
      self.meters_travelled = 0

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    # Path to where visualizations and other debug output gets stored
    self.save_path = os.environ.get('SAVE_PATH', None)

    # Logger that generates logs used for infraction replay in the results_parser.
    if self.save_path is not None and route_index is not None:
      self.save_path = pathlib.Path(self.save_path) / route_index
      pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

      self.lon_logger = ScenarioLogger(
          save_path=self.save_path,
          route_index=route_index,
          logging_freq=self.config.logging_freq,
          log_only=True,
          route_only=False,  # with vehicles
          roi=self.config.logger_region_of_interest,
      )
    else:
      self.save_path = None

    self.metric_info = {}
    self._vehicle = CarlaDataProvider.get_hero_actor()
    self.world = self._vehicle.get_world()

    # Initialize safety evaluator (for backward compatibility)
    self.safety_evaluator = SafetyEvaluator(self.config)
    
    # Initialize unified risk manager (combines external TTC + internal driver state)
    self.unified_risk_manager = UnifiedRiskManager(
        config=self.config,
        udp_port=int(os.environ.get('DRIVER_STATE_UDP_PORT', 9999)),
        audio_enabled=strtobool(os.environ.get('AUDIO_ALERTS_ENABLED', 'True'))
    )

    # Add this to the setup() method, after the unified_risk_manager initialization:
    
    # Initialize simple TTC logger if enabled
    ttc_logging_env = os.environ.get('ENABLE_TTC_LOGGING', 'False')
    print(f"🔍 DEBUG: ENABLE_TTC_LOGGING = '{ttc_logging_env}'")
    
    if ttc_logging_env.lower() == 'true':
        print("🔍 DEBUG: Attempting to initialize TTC logger...")
        try:
            from simple_ttc_logger import SimpleTTCLogger
            
            experiment_condition = os.environ.get('EXPERIMENT_CONDITION', 'fusion')
            participant_id = os.environ.get('PARTICIPANT_ID', 'P001')
            trial_number = os.environ.get('TRIAL_NUMBER', '1')
            scenario_name = os.environ.get('SCENARIO_NAME', 'SC-EV')
            
            print(f"🔍 DEBUG: Creating logger with condition={experiment_condition}, participant={participant_id}, trial={trial_number}, scenario={scenario_name}")
            
            self.ttc_logger = SimpleTTCLogger(
                experiment_condition=experiment_condition,
                participant_id=participant_id,
                trial_number=trial_number,
                scenario_name=scenario_name
            )
            print(f"📊 TTC logging enabled successfully!")
            
            # Connect logger to unified risk manager for alert tracking
            self.unified_risk_manager.set_experiment_logger(self.ttc_logger)
            
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize TTC logger: {e}")
            self.ttc_logger = None
    else:
        print("🔍 DEBUG: TTC logging disabled")
        self.ttc_logger = None



  def _init(self):
    # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
    # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
    # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
    # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
    # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
    # of the GPS. It seems to work reasonably well, so we use this workaround for now.
    try:
      locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
      lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
      earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation

      def equations(variables):
        x, y = variables
        eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa) -
               math.cos(x * math.pi / 180.0) * y)
        eq2 = (math.log(math.tan(
            (lat + 90.0) * math.pi / 360.0)) * earth_radius_equa * math.cos(x * math.pi / 180.0) + locy -
               math.cos(x * math.pi / 180.0) * earth_radius_equa * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
        return [eq1, eq2]

      initial_guess = [0.0, 0.0]
      solution = fsolve(equations, initial_guess)
      self.lat_ref, self.lon_ref = solution[0], solution[1]
    except Exception as e:
      print(e, flush=True)
      self.lat_ref, self.lon_ref = 0.0, 0.0

    # During setup() not everything is available yet, so this _init is a second setup in run_step()
    if self.save_path is not None:
      # Privileged map access for logging and visualizations. Turned off during normal evaluation.
      from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=locally-disabled, import-outside-toplevel
      from nav_planner import interpolate_trajectory  # pylint: disable=locally-disabled, import-outside-toplevel
      self.world_map = CarlaDataProvider.get_map()
      trajectory = [item[0].location for item in self._global_plan_world_coord]
      self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)  # privileged

      self._waypoint_planner = RoutePlanner(self.config.log_route_planner_min_distance,
                                            self.config.route_planner_max_distance, self.lat_ref, self.lon_ref)
      self._waypoint_planner.set_route(self.dense_route, True)

      vehicle = CarlaDataProvider.get_hero_actor()
      self.lon_logger.ego_vehicle = vehicle
      self.lon_logger.world = vehicle.get_world()

      self.nets[0].init_visualization()

    self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance,
                                       self.lat_ref, self.lon_ref)
    self._route_planner.set_route(self._global_plan, True)
    self.initialized = True

  def sensors(self):
    """
    Define the sensor suite of the agent
    """
    sensors = [
      {
        'type': 'sensor.camera.rgb',
        'x': 1.4, 'y': 0.0, 'z': 1.2,
        'roll': 0.0, 'pitch': .0, 'yaw': 0.0,
        'width': self.camera_width, 'height': self.camera_height, 'fov': 110,
        'id': 'Center'
      },
			# Left mirror - reduced resolution
			{'type': 'sensor.camera.rgb',
			 'x': 0.7, 'y': -1.0, 'z': 1.0,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 210.0,
			 'width': int(self.camera_width * 0.3),
			 'height': int(self.camera_height * 0.3),
			 'fov': 100, 'id': 'Left'},

			# Right mirror - reduced resolution
			{'type': 'sensor.camera.rgb',
			 'x': 0.7, 'y': 1.0, 'z': 1.0,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 150.0,
			 'width': int(self.camera_width * 0.3),
			 'height': int(self.camera_height * 0.3),
			 'fov': 100, 'id': 'Right'},
    ]
    # Add existing sensors
    sensors.extend([
        {
            'type': 'sensor.camera.rgb',
            'x': self.config.camera_pos[0],
            'y': self.config.camera_pos[1],
            'z': self.config.camera_pos[2],
            'roll': self.config.camera_rot_0[0],
            'pitch': self.config.camera_rot_0[1],
            'yaw': self.config.camera_rot_0[2],
            'width': self.config.camera_width,
            'height': self.config.camera_height,
            'fov': self.config.camera_fov,
            'id': 'rgb_front'
        },
        {
            'type': 'sensor.other.imu',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'sensor_tick': self.config.carla_frame_rate,
            'id': 'imu'
        },
        {
            'type': 'sensor.other.gnss',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'sensor_tick': 0.01,
            'id': 'gps'
        },
        {
            'type': 'sensor.speedometer',
            'reading_frequency': self.config.carla_fps,
            'id': 'speed'
        }
    ])
    # Don't set up LiDAR for camera only approaches
    if self.config.backbone not in ('aim'):
      sensors.append({
          'type': 'sensor.lidar.ray_cast',
          'x': self.config.lidar_pos[0],
          'y': self.config.lidar_pos[1],
          'z': self.config.lidar_pos[2],
          'roll': self.config.lidar_rot[0],
          'pitch': self.config.lidar_rot[1],
          'yaw': self.config.lidar_rot[2],
          'id': 'lidar'
      })

    return sensors

  @torch.inference_mode()  # Turns off gradient computation
  def tick(self, input_data):
    """Pre-processes sensor data and runs the Unscented Kalman Filter"""
    rgb = []
    for camera_pos in ['front']:
      rgb_cam = 'rgb_' + camera_pos
      camera = input_data[rgb_cam][1][:, :, :3]

      # Also add jpg artifacts at test time, because the training data was saved as jpg.
      _, compressed_image_i = cv2.imencode('.jpg', camera)
      camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

      rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
      rgb_pos = t_u.crop_array(self.config, rgb_pos)

      # Switch to pytorch channel first order
      rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
      rgb.append(rgb_pos)
    rgb = np.concatenate(rgb, axis=1)
    rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

    gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
    speed = input_data['speed'][1]['speed']
    compass = t_u.preprocess_compass(input_data['imu'][1][-1])

    result = {
        'rgb': rgb,
        'compass': compass,
    }

    if self.config.backbone not in ('aim'):
      result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

    if not self.filter_initialized:
      # apply ukf only to x and y coordinates, append z coordinate afterwards
      self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
      self.filter_initialized = True

    # Safety checks for control values
    if not hasattr(self, 'control') or self.control is None:
      # If control is not initialized, use neutral values
      steer, throttle, brake = 0.0, 0.0, 0.0
    else:
      # Ensure control values are within valid ranges
      steer = np.clip(float(self.control.steer), -1.0, 1.0)
      throttle = np.clip(float(self.control.throttle), 0.0, 1.0)
      brake = float(self.control.brake > 0.0)  # Convert to binary value

    try:
      self.ukf.predict(steer=steer, throttle=throttle, brake=brake)
      self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
    except np.linalg.LinAlgError:
      print("UKF failed, reinitializing...")
      self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
      self.ukf.P = np.diag([1.0, 1.0, 0.1, 0.1])

    filtered_state = self.ukf.x
    self.state_log.append(filtered_state)
    result['gps'] = filtered_state[0:2]

    waypoint_route = self._route_planner.run_step(np.append(filtered_state[0:2], gps_pos[2]))

    if len(waypoint_route) > 2:
      target_point, far_command = waypoint_route[1]
      target_point_next, _ = waypoint_route[2]
    elif len(waypoint_route) > 1:
      target_point, far_command = waypoint_route[1]
      target_point_next = target_point
    else:
      target_point, far_command = waypoint_route[0]
      target_point_next = target_point

    if (target_point != self.target_point_prev).all():
      self.target_point_prev = target_point
      self.commands.append(far_command.value)

    one_hot_command = t_u.command_to_one_hot(self.commands[-2])
    result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

    ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])  # original

    ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

    result['target_point'] = ego_target_point

    if self.config.two_tp_input:
      ego_target_point_next = t_u.inverse_conversion_2d(target_point_next[:2], result['gps'], result['compass'])
      ego_target_point_next = torch.from_numpy(ego_target_point_next[np.newaxis]).to(self.device, dtype=torch.float32)
      result['target_point_next'] = ego_target_point_next

    result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

    if self.save_path is not None:
      pass
      waypoint_route = self._waypoint_planner.run_step(np.append(result['gps'], gps_pos[2]))
      waypoint_route = extrapolate_waypoint_route(waypoint_route, self.config.route_points)
      route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])[:self.config.route_points]
      self.lon_logger.log_step(route)

    return result



  def draw_unified_risk_info(self, surface, unified_assessment):
    """Draw unified risk assessment information display with speed and traffic info"""
    
    # Define colors for different risk levels
    status_colors = {
        "safe": (50, 255, 50),      # Green
        "caution": (255, 255, 50),  # Yellow
        "warning": (255, 165, 0),   # Orange
        "critical": (255, 50, 50),  # Red
        "emergency": (255, 0, 255)  # Magenta
    }
    
    # Emergency vehicle color (cyan/teal)
    emergency_color = (0, 255, 255)
    
    # Extract values from unified assessment
    unified_risk = unified_assessment['unified_risk']
    risk_level = unified_assessment['risk_level']
    external_risk = unified_assessment['external_risk']
    internal_risk = unified_assessment['internal_risk']
    driver_state = unified_assessment['driver_state']['current_state']
    driver_confidence = unified_assessment['driver_state']['confidence']
    is_stale = unified_assessment['driver_state']['is_stale']
    
    # Check for emergency vehicle
    emergency_indicator = ""
    primary_threat = unified_assessment['external_breakdown'].get('primary_threat')
    if primary_threat and primary_threat.get('is_emergency', False):
        emergency_indicator = " [EMERGENCY]"
    
    # Create text lines
    texts = []
    
    # Unified risk (main display)
    risk_color = status_colors.get(risk_level, (255, 255, 255))
    if emergency_indicator:
        risk_color = emergency_color
    
    texts.append((f"=== UNIFIED RISK ===", (255, 255, 255)))
    texts.append((f"Risk: {unified_risk:.3f}", risk_color))
    texts.append((f"Level: {risk_level.upper()}{emergency_indicator}", risk_color))
    texts.append(("", (255, 255, 255)))  # Spacer
    
    # External risk component
    texts.append((f"External: {external_risk:.3f}", status_colors.get("caution", (255, 255, 255))))
    if primary_threat:
        threat_distance = primary_threat.get('distance', 0)
        threat_type = primary_threat.get('class_name', 'object')
        texts.append((f"Threat: {threat_type} @ {threat_distance:.1f}m", (255, 255, 255)))
    
    texts.append(("", (255, 255, 255)))  # Spacer
    
    # Internal risk component
    driver_color = (255, 50, 50) if driver_state != 'safe_driving' else (50, 255, 50)
    if is_stale:
        driver_color = (128, 128, 128)  # Gray for stale data
        
    texts.append((f"Internal: {internal_risk:.3f}", driver_color))
    stale_indicator = " [STALE]" if is_stale else ""
    texts.append((f"Driver: {driver_state}{stale_indicator}", driver_color))
    texts.append((f"Confidence: {driver_confidence:.2f}", driver_color))
    
    texts.append(("", (255, 255, 255)))  # Spacer
    
    # === EGO SPEED (MOVED HERE) ===
    ego_speed_kmh = self.display_current_speed * 3.6
    speed_color = (255, 255, 255)
    if ego_speed_kmh > 60:
        speed_color = (255, 165, 0)  # Orange for high speed
    elif ego_speed_kmh > 80:
        speed_color = (255, 50, 50)  # Red for very high speed
    
    texts.append((f"=== VEHICLE STATUS ===", (255, 255, 255)))
    texts.append((f"Speed: {ego_speed_kmh:.1f} km/h", speed_color))
    
    # === TRAFFIC SIGNALS (MOVED HERE AND ENHANCED) ===
    # Get ALL detected traffic signals, not just dangerous ones
    all_signals = self.get_all_traffic_signals()
    
    if all_signals:
        texts.append(("", (255, 255, 255)))  # Spacer
        texts.append((f"=== TRAFFIC SIGNALS ===", (255, 255, 255)))
        
        for signal in all_signals[:3]:  # Show up to 3 closest signals
            signal_type = signal.get('signal_type', 'Unknown')
            distance = signal.get('distance', 0)
            risk = signal.get('signal_risk', 0)
            
            if signal_type == 'Red Light':
                signal_color = (255, 50, 50)
                icon = "🔴"
            elif signal_type == 'Stop Sign':
                signal_color = (255, 165, 0)
                icon = "🛑"
            else:
                signal_color = (255, 255, 255)
                icon = "⚠️"
                
            texts.append((f"{icon} {signal_type}: {distance:.1f}m", signal_color))
            
            # Show risk level if it's significant
            if risk > 0.1:
                risk_text = f"Risk: {risk:.2f}"
                texts.append((f"  {risk_text}", signal_color))
    else:
        texts.append(("", (255, 255, 255)))  # Spacer
        texts.append(("=== TRAFFIC SIGNALS ===", (255, 255, 255)))
        texts.append(("No Signals Detected", (50, 255, 50)))
    
    texts.append(("", (255, 255, 255)))  # Spacer
    
    # Risk weights
    weights = unified_assessment.get('weights', {})
    ext_weight = weights.get('external_weight', 0)
    int_weight = weights.get('internal_weight', 0)
    texts.append((f"Weights: Ext {ext_weight:.1f} Int {int_weight:.1f}", (200, 200, 200)))
    
    # Synergy info
    synergy_factor = unified_assessment.get('risk_factors', {}).get('synergy_factor', 1.0)
    if synergy_factor > 1.0:
        texts.append((f"Synergy: {synergy_factor:.2f}x", (255, 165, 0)))
    
    # Render text lines
    rendered_texts = []
    max_width = 0
    total_height = 0
    line_height = self._font_mono.get_height()
    
    for text, color in texts:
        if text:  # Skip empty spacer lines for width calculation
            rendered = self._font_mono.render(text, True, color)
            rendered_texts.append(rendered)
            max_width = max(max_width, rendered.get_width())
        else:
            rendered_texts.append(None)  # Spacer
        total_height += line_height + 2  # 2px spacing between lines
    
    # Position in bottom-left corner to avoid mirror overlap
    pos_x = 20  # 20px padding from left
    pos_y = surface.get_height() - total_height - 100  # 100px from bottom
    
    # Create semi-transparent background
    bg = pygame.Surface((max_width + 20, total_height + 20))
    bg.fill((0, 0, 0))
    bg.set_alpha(128)
    
    # Draw background
    surface.blit(bg, (pos_x - 10, pos_y - 10))
    
    # Draw text lines
    current_y = pos_y
    for rendered in rendered_texts:
        if rendered:  # Skip spacer lines
            surface.blit(rendered, (pos_x, current_y))
        current_y += line_height + 2

  def get_all_traffic_signals(self):
    """Get all detected traffic signals regardless of danger level"""
    all_signals = []
    
    # Check if we have bounding boxes
    if hasattr(self, 'bb_buffer') and len(self.bb_buffer) > 0:
        for bb in self.bb_buffer[-1]:
            if len(bb) > 7:
                class_id = int(bb[7])
                distance = bb[0]
                
                # Class IDs: 2 = Red Light, 3 = Stop Sign
                if class_id == 2:  # Red Light
                    all_signals.append({
                        'signal_type': 'Red Light',
                        'distance': distance,
                        'signal_risk': 0.0,  # Default, could be calculated
                        'class_id': class_id
                    })
                elif class_id == 3:  # Stop Sign
                    all_signals.append({
                        'signal_type': 'Stop Sign', 
                        'distance': distance,
                        'signal_risk': 0.0,  # Default, could be calculated
                        'class_id': class_id
                    })
    
    # Sort by distance (closest first)
    all_signals.sort(key=lambda x: x['distance'])
    
    return all_signals

  def draw_risk_info(self, surface, ttc_risk, ttc_status, signal_risk, signal_status, 
                     distance_risk, distance_status, braking_risk, braking_status):
    """Draw risk assessment information display (legacy version)"""
    
    # Define colors for different risk levels
    status_colors = {
        "SAFE": (50, 255, 50),      # Green
        "CAUTION": (255, 255, 50),  # Yellow
        "WARNING": (255, 165, 0),   # Orange
        "CRITICAL": (255, 50, 50)   # Red
    }
    
    # Emergency vehicle color (cyan/teal)
    emergency_color = (0, 255, 255)
    
    # Create text lines
    texts = []
    
    # Check if there's an emergency vehicle in TTC calculation
    emergency_indicator = ""
    ttc_color = status_colors.get(ttc_status, (255, 255, 255))
    
    # Check for emergency vehicle in the most recent TTC object
    if hasattr(self, 'current_ttc_object') and self.current_ttc_object:
        if self.current_ttc_object.get('is_emergency', False):
            emergency_indicator = " [EMERGENCY]"
            ttc_color = emergency_color
    
    texts.append((f"TTC Risk: {ttc_risk:.3f}{emergency_indicator}", ttc_color))
    texts.append((f"Status: {ttc_status}", ttc_color))
    texts.append(("", (255, 255, 255)))  # Spacer
    texts.append((f"Signal Risk: {signal_risk:.3f}", status_colors.get(signal_status, (255, 255, 255))))
    texts.append((f"Status: {signal_status}", status_colors.get(signal_status, (255, 255, 255))))
    texts.append(("", (255, 255, 255)))  # Spacer
    texts.append((f"Distance Risk: {distance_risk:.3f}", status_colors.get(distance_status, (255, 255, 255))))
    texts.append((f"Status: {distance_status}", status_colors.get(distance_status, (255, 255, 255))))
    texts.append(("", (255, 255, 255)))  # Spacer
    texts.append((f"Braking Risk: {braking_risk:.3f}", status_colors.get(braking_status, (255, 255, 255))))
    texts.append((f"Status: {braking_status}", status_colors.get(braking_status, (255, 255, 255))))
    
    # Render text lines
    rendered_texts = []
    max_width = 0
    total_height = 0
    line_height = self._font_mono.get_height()
    
    for text, color in texts:
        if text:  # Skip empty spacer lines for width calculation
            rendered = self._font_mono.render(text, True, color)
            rendered_texts.append(rendered)
            max_width = max(max_width, rendered.get_width())
        else:
            rendered_texts.append(None)  # Spacer
        total_height += line_height + 2  # 2px spacing between lines
    
    # Position in top-left corner with padding
    pos_x = 20  # 20px padding from left
    pos_y = 20  # 20px padding from top
    
    # Create semi-transparent background
    bg = pygame.Surface((max_width + 20, total_height + 20))
    bg.fill((0, 0, 0))
    bg.set_alpha(128)
    
    # Draw background
    surface.blit(bg, (pos_x - 10, pos_y - 10))
    
    # Draw text lines
    current_y = pos_y
    for rendered in rendered_texts:
        if rendered:  # Skip spacer lines
            surface.blit(rendered, (pos_x, current_y))
        current_y += line_height + 2

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp):
    self.step += 1
    current_time = time.time()
    
    # Define mirror dimensions
    mirror_width = int(self._display.get_width() * 0.2)  # 20% of display width
    mirror_height = int(self._display.get_height() * 0.2)  # 20% of display height

    if not self.initialized:
      self._init()
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.control = control
      tick_data = self.tick(input_data)
      if self.config.backbone not in ('aim'):
        self.lidar_last = deepcopy(tick_data['lidar'])
      return control

    # Need to run this every step for GPS filtering
    tick_data = self.tick(input_data)

    # Process all pygame events once
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        # Save data before quitting
        # self.save_recorded_data()
        return
      elif event.type == pygame.JOYBUTTONDOWN:
        if event.button == self._square_idx:  # Square button
          if current_time - self.last_switch_time > 0.2:
            # Record intervention if switching from autonomous to manual
            if not self.manual_control:
              self.data_recording['interventions'].append({
                'time': current_time,
                'step': self.step,
                'reason': 'button_press'
              })
            
            # Update control time before switching modes
            mode_duration = current_time - self.data_recording['last_mode_switch_time']
            if self.manual_control:
              self.data_recording['control_time']['human'] += mode_duration
            else:
              self.data_recording['control_time']['ai'] += mode_duration
            
            # Switch control mode
            self.manual_control = not self.manual_control
            self.data_recording['current_mode'] = 'manual' if self.manual_control else 'autonomous'
            self.data_recording['last_mode_switch_time'] = current_time
            self.last_switch_time = current_time
            


    lidar_indices = []
    for i in range(self.config.lidar_seq_len):
      lidar_indices.append(i * self.config.data_save_freq)

    #Current position of the car
    ego_x = self.state_log[-1][0]
    ego_y = self.state_log[-1][1]
    ego_theta = self.state_log[-1][2]

    ego_x_last = self.state_log[-2][0]
    ego_y_last = self.state_log[-2][1]
    ego_theta_last = self.state_log[-2][2]

    # We only get half a LiDAR at every time step. Aligns the last half into the current coordinate frame.
    if self.config.backbone not in ('aim'):
      lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    # Updates stop boxes by vehicle movement converting past predictions into the current frame.
    if self.stop_sign_controller:
      self.update_stop_box(self.stop_sign_buffer, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    if self.config.backbone not in ('aim'):
      lidar_current = deepcopy(tick_data['lidar'])
      lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

      self.lidar_buffer.append(lidar_full)

    if self.config.backbone not in ('aim'):
      # We wait until we have sufficient LiDARs
      if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
        self.lidar_last = deepcopy(tick_data['lidar'])
        tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
        self.control = tmp_control

        return tmp_control

    if self.config.backbone in ('aim'):  # Image only method
      # Dummy data
      lidar_bev = torch.zeros((1, 1 + int(self.config.use_ground_plane), self.config.lidar_resolution_height,
                               self.config.lidar_resolution_width)).to(self.device, dtype=torch.float32)
    else:
      # Voxelize LiDAR and stack temporal frames
      lidar_bev = []
      # prepare LiDAR input
      for i in lidar_indices:
        lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])

        # For single frame there is no point in realignment. The state_log index will also differ.
        if self.config.realign_lidar and self.config.lidar_seq_len > 1:
          # Position of the car when the LiDAR was collected
          curr_x = self.state_log[i][0]
          curr_y = self.state_log[i][1]
          curr_theta = self.state_log[i][2]

          # Voxelize to BEV for NN to process
          lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

        lidar_histogram = self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                                use_ground_plane=self.config.use_ground_plane)

        lidar_histogram = torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32)
        lidar_bev.append(lidar_histogram)

        lidar_bev = torch.cat(lidar_bev, dim=1)

    if self.config.backbone not in ('aim'):
      self.lidar_last = deepcopy(tick_data['lidar'])

    # prepare velocity input
    gt_velocity = tick_data['speed']
    velocity = gt_velocity.reshape(1, 1)  # used by transfuser

    compute_debug_output = self.config.debug and (self.save_path is not None)

    # new checkpoint lookahead: calculate which checkpoint to use for control
    speed = gt_velocity.item()

    if self.stop_after_meter > 0:
      dt = self.config.carla_frame_rate
      self.meters_travelled = self.meters_travelled + speed * dt

    # forward pass
    pred_wps = []
    pred_target_speeds = []
    pred_checkpoints = []
    bounding_boxes = []
    wp_selected = None
    for i in range(self.model_count):
      if self.config.backbone in ('transFuser', 'aim', 'bev_encoder'):
        pred_wp, \
        pred_target_speed, \
        pred_checkpoint, \
        pred_semantic, \
        pred_bev_semantic, \
        pred_depth, \
        pred_bb_features,\
        attention_weights,\
        pred_wp_1,\
        selected_path = self.nets[i].forward(
          rgb=tick_data['rgb'],
          lidar_bev=lidar_bev,
          target_point=tick_data['target_point'],
          target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
          ego_vel=velocity,
          command=tick_data['command'])
        # Only convert bounding boxes when they are used.
        if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                         self.stop_sign_controller):
          pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
        else:
          pred_bounding_box = None
      else:
        raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

      if self.config.use_wp_gru:
        if self.config.multi_wp_output:
          wp_selected = 0
          if F.sigmoid(selected_path)[0].item() > 0.5:
            wp_selected = 1
            pred_wps.append(pred_wp_1)
          else:
            pred_wps.append(pred_wp)
        else:
          pred_wps.append(pred_wp)
      if self.config.use_controller_input_prediction:
        pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
        pred_checkpoints.append(pred_checkpoint[0])

      bounding_boxes.append(pred_bounding_box)

    # Average the predictions from ensembles
    if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                     self.stop_sign_controller):
      # We average bounding boxes by using non-maximum suppression on the set of all detected boxes.
      bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)

      self.bb_buffer.append(bbs_vehicle_coordinate_system)
    else:
      bbs_vehicle_coordinate_system = None

    if self.stop_sign_controller:
      stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

    if self.config.tp_attention:
      self.tp_attention_buffer.append(attention_weights[2])

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    # calculate target speed scalar from model predictions
    if self.config.use_controller_input_prediction:
      pred_target_speed_ensemble = torch.stack(pred_target_speeds, dim=0).mean(dim=0)

      if self.uncertainty_weight:
        uncertainty = pred_target_speed_ensemble.detach().cpu().numpy()
        self.last_uncertainty = uncertainty  # Store for display
        if uncertainty[0] > self.config.brake_uncertainty_threshold:
          pred_target_speed_scalar = self.inference_target_speeds[0]
        else:
          pred_target_speed_scalar = sum(uncertainty * self.inference_target_speeds)
      else:
        pred_target_speed_index = torch.argmax(pred_target_speed_ensemble)
        pred_target_speed_scalar = self.inference_target_speeds[pred_target_speed_index]

    # Visualize the output of the last model
    if compute_debug_output:
      if self.config.use_controller_input_prediction:
        prob_target_speed = F.softmax(pred_target_speed, dim=1)
      else:
        prob_target_speed = pred_target_speed

      self.nets[0].visualize_model(
          self.save_path,
          self.step,
          tick_data['rgb'],
          lidar_bev,
          tick_data['target_point'],
          pred_wp,
          target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
          pred_semantic=pred_semantic,
          pred_bev_semantic=pred_bev_semantic,
          pred_depth=pred_depth,
          pred_checkpoint=pred_checkpoint,
          pred_speed=prob_target_speed,
          pred_target_speed_scalar=pred_target_speed_scalar,
          pred_bb=bbs_vehicle_coordinate_system,
          gt_speed=gt_velocity,
          gt_wp=pred_wp_1,
          wp_selected=wp_selected)

    if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
      pred_checkpoints = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_checkpoints, pred_target_speed_scalar, gt_velocity)
    elif self.config.use_wp_gru and not self.config.inference_direct_controller:
      steer, throttle, brake = self.nets[0].control_pid(self.pred_wp,
                                                        gt_velocity,
                                                        tuned_aim_distance=bool(self.tuned_aim_distance))
    else:
      raise ValueError('An output representation was chosen that was not trained.')
    # print(f"Predicted speed: {pred_target_speed_scalar}")
    # print(f"Current speed: {gt_velocity}")
    # Draw predicted waypoints trajectory
    if pred_checkpoints is not None:
        # Get current speed in km/h
        # current_speed = input_data['speed'][1]['speed'] * 3.6
        # Convert torch tensor to scalar value
        if torch.is_tensor(gt_velocity):
            current_speed = gt_velocity.item()
        else:
            current_speed = gt_velocity
        self.display_current_speed = current_speed
        
        
        # Get predicted target speed if available
        predicted_speed = pred_target_speed_scalar 
        self.display_predicted_speed = predicted_speed
        # print(f"Predicted speed: {predicted_speed}")
        # print(f"Current speed: {current_speed}")
        
        # Determine waypoint color based on speed difference
        speed_diff = abs(current_speed*3.6 - predicted_speed*3.6)
        waypoint_color = carla.Color(255, 0, 0) if speed_diff > 5.0 else carla.Color(0, 0, 255)  # Red if diff > 5km/h, blue otherwise
        
        # If pred_checkpoints is a torch tensor, convert to numpy
        if torch.is_tensor(pred_checkpoints):
            draw_wps = pred_checkpoints.detach().cpu().numpy()
        else:
            draw_wps = pred_checkpoints
            
        # Get ego vehicle transform
        ego_transform = self._vehicle.get_transform()
        
        # Draw all checkpoint sequences
        for checkpoint_idx, checkpoint_seq in enumerate(draw_wps):
            num_wp = len(checkpoint_seq)
            prev_wp_world = ego_transform.location
            
            for idx in range(num_wp//2):
                x_coord = checkpoint_seq[idx]
                y_coord = checkpoint_seq[idx + num_wp//2]
                
                # Convert from local to world coordinate system
                wp_local = carla.Location(
                    x=float(x_coord),
                    y=float(y_coord),
                    z=0.5
                )
                
                # Transform from local to world coordinates
                wp_world = ego_transform.transform(wp_local)
                
                # Draw point with color based on speed difference
                self.world.debug.draw_point(
                    wp_world,
                    size=0.3,
                    color=waypoint_color,
                    life_time=0.1
                )

    # 0.1 is just an arbitrary low number to threshold when the car is stopped
    if gt_velocity < 0.1:
      self.stuck_detector += 1
    else:
      self.stuck_detector = 0

    # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
    if self.stuck_detector > self.config.stuck_threshold:
      self.force_move = self.config.creep_duration

    if self.force_move > 0:
      emergency_stop = False
      if self.config.backbone not in ('aim'):
        # safety check
        safety_box = deepcopy(self.lidar_buffer[-1])

        # z-axis
        safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

        # y-axis
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

        # x-axis
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        emergency_stop = (len(safety_box) > 0)  # Checks if the List is empty

      if not emergency_stop:
        print('Detected agent being stuck. Step: ', self.step)
        throttle = max(self.config.creep_throttle, throttle)
        brake = False
        self.force_move -= 1
      else:
        print('Creeping stopped by safety box. Step: ', self.step)
        throttle = 0.0
        brake = True
        self.force_move = self.config.creep_duration

    if self.stop_sign_controller:
      if stop_for_stop_sign:
        throttle = 0.0
        brake = True

    if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
      print(f'Stopping after {self.stop_after_meter} meters.')
      throttle = 0.0
      brake = True

    # Create control command (throttle and brake may have been modified by unified risk manager)
    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
    ai_control = control

    # =================================================================
    # UNIFIED RISK ASSESSMENT - Combines external TTC + internal driver state
    # =================================================================
    if self.config.detect_boxes and hasattr(self, 'bb_buffer') and len(self.bb_buffer) > 0:
        # Update time for plotting using actual timestamp
        frame_time = timestamp  # Use actual timestamp instead of assuming 10 Hz
        self.unified_risk_manager.safety_evaluator.update_time(frame_time)
        self.unified_risk_manager.safety_evaluator.update_timestamp(timestamp)
        
        # Print all detected objects for debugging
        if strtobool(os.environ.get('DEBUG_OBJECTS', 'False')):
            print("=== All Detected Objects ===")
            object_types = {0: "Vehicle", 1: "Pedestrian", 2: "Red Light", 3: "Stop Sign", 4: "Emergency Vehicle"}
            for i, bb in enumerate(self.bb_buffer[-1]):
                if len(bb) > 7:
                    obj_type = object_types.get(bb[7], f"Unknown({bb[7]})")
                    speed = bb[5] if len(bb) > 5 and bb[5] is not None else "N/A"
                    distance = bb[0]
                    lateral_pos = bb[1]
                    print(f"  {obj_type} {i}: Speed={speed} m/s, Distance={distance:.1f}m, Lateral={lateral_pos:.1f}m")
            print("=============================")
        
        # UNIFIED RISK CALCULATION - Main risk assessment
        unified_risk_assessment = self.unified_risk_manager.calculate_unified_risk(
            ego_speed=gt_velocity.item(),
            bounding_boxes=self.bb_buffer[-1]
        )
        
        # Extract key risk information
        unified_risk = unified_risk_assessment['unified_risk']
        risk_level = unified_risk_assessment['risk_level']
        external_risk = unified_risk_assessment['external_risk']
        internal_risk = unified_risk_assessment['internal_risk']
        driver_state = unified_risk_assessment['driver_state']['current_state']
        
        # Store unified risk assessment for display
        self.current_unified_assessment = unified_risk_assessment
        
        # =================================================================
        # RISK-AWARE CONTROL MODIFICATIONS
        # =================================================================
        
        # Apply risk-based control adjustments if enabled
        if strtobool(os.environ.get('RISK_AWARE_CONTROL', 'True')):
            original_throttle, original_brake = throttle, brake
            
            # Apply risk-based speed adjustments
            if risk_level == 'emergency':
                # Emergency: Aggressive braking
                throttle = 0.0
                brake = min(1.0, unified_risk)
                print(f"🚨 EMERGENCY BRAKING: brake={brake:.2f}")
                
            elif risk_level == 'critical':
                # Critical: Strong deceleration
                if driver_state in ['sleepy', 'using_phone']:
                    # More aggressive if driver is inattentive
                    throttle *= 0.3
                    brake = max(brake, 0.4)
                else:
                    throttle *= 0.5
                    brake = max(brake, 0.3)
                print(f"⚠️ CRITICAL DECELERATION: throttle={throttle:.2f}")
                
            elif risk_level == 'warning':
                # Warning: Moderate speed reduction
                speed_factor = 1.0 - (unified_risk * 0.3)
                throttle *= speed_factor
                print(f"⚠️ WARNING SPEED REDUCTION: factor={speed_factor:.2f}")
                
            elif risk_level == 'caution':
                # Caution: Slight speed reduction
                if driver_state != 'safe_driving':
                    # Be more conservative if driver is not fully alert
                    speed_factor = 1.0 - (unified_risk * 0.2)
                    throttle *= speed_factor
                    print(f"⚠️ CAUTION (driver {driver_state}): factor={speed_factor:.2f}")
            
            # Apply emergency vehicle yielding behavior
            primary_threat = unified_risk_assessment['external_breakdown'].get('primary_threat')
            if primary_threat and primary_threat.get('is_emergency', False):
                # Emergency vehicle detected - implement yielding behavior
                if external_risk > 0.3:
                    throttle *= 0.6  # Reduce speed for yielding
                    print("🚑 EMERGENCY VEHICLE YIELDING")
        
        # =================================================================
        # LEGACY COMPATIBILITY - Keep old risk values for existing displays
        # =================================================================
        
        # Extract individual risk components for backward compatibility
        external_breakdown = unified_risk_assessment.get('external_breakdown', {})
        self.current_ttc_risk = external_breakdown.get('ttc_risk', 0.0)
        self.current_ttc_status = "SAFE" if self.current_ttc_risk < 0.2 else "WARNING" if self.current_ttc_risk < 0.5 else "CRITICAL"
        self.current_ttc_object = external_breakdown.get('primary_threat')
        
        # **UPDATED: Set signal risk from separated risk architecture**
        self.current_signal_risk = external_breakdown.get('signal_risk', 0.0)
        signal_info = external_breakdown.get('signal_info')
        if signal_info:
            # Use the actual signal alert level from the straightforward signal compliance
            signal_alert_level = signal_info.get('alert_level', 'SAFE')
            self.current_signal_status = signal_alert_level
            
            # **NEW: Handle VIOLATION alert level**
            if signal_alert_level == 'VIOLATION':
                print(f"🚨 SENSOR AGENT: Signal violation detected - {signal_info.get('signal_type', 'Unknown')}")
                # Set maximum signal risk for violations
                self.current_signal_risk = 1.0
        else:
            self.current_signal_status = "SAFE"
            
        # Set default values for other risk types (distance, braking) 
        self.current_distance_risk = external_breakdown.get('distance_risk', 0.0)
        self.current_distance_status = "SAFE" if self.current_distance_risk < 0.2 else "WARNING"
        self.current_braking_risk = 0.0
        self.current_braking_status = "SAFE"
        
        # =================================================================
        # TTC DATA LOGGING - Log risk assessment data for experiments
        # =================================================================
        
        # Log TTC data if logger is enabled
        if hasattr(self, 'ttc_logger') and self.ttc_logger:
            # Get TTC object from external breakdown
            ttc_object = external_breakdown.get('primary_threat')
            
            # Debug output (only every 100 frames to avoid spam)
            if self.step % 100 == 0:
                print(f"🔍 DEBUG Step {self.step}: TTC logger active, ttc_object={ttc_object is not None}")
                if ttc_object:
                    print(f"    TTC object: {ttc_object.get('class_name', 'unknown')} at {ttc_object.get('distance', 0):.1f}m")
            
            # Log the TTC data
            self.ttc_logger.log_ttc_data(
                timestamp=timestamp,
                ttc_object=ttc_object,
                ego_speed=gt_velocity.item(),
                internal_risk=unified_risk_assessment.get('internal_risk', 0.0),
                driver_state=unified_risk_assessment.get('driver_state', {}).get('current_state', 'unknown')
            )
            
            # **NEW: Log signal compliance events for signal compliance metrics**
            signal_info = external_breakdown.get('signal_info')
            if signal_info:
                # Log signal compliance event with comprehensive metrics
                self.ttc_logger.log_signal_event(
                    timestamp=timestamp,
                    signal_info=signal_info,
                    ego_speed=gt_velocity.item()
                )
                
                # Debug output for signal events (less frequent to avoid spam)
                if self.step % 50 == 0:
                    signal_type = signal_info.get('signal_type', 'Unknown')
                    distance = signal_info.get('distance_to_stop_line', 0.0)
                    decel = signal_info.get('required_deceleration', 0.0)
                    level = signal_info.get('alert_level', 'SAFE')
                    print(f"🚦 Signal: {signal_type} at {distance:.1f}m, {decel:.1f} m/s² ({level})")
        elif hasattr(self, 'ttc_logger') and self.step % 100 == 0:
            print(f"🔍 DEBUG Step {self.step}: TTC logger exists but no bounding boxes")
        
    else:
        # Initialize default values if no safety evaluation
        if not hasattr(self, 'current_unified_assessment'):
            self.current_unified_assessment = {
                'unified_risk': 0.0,
                'risk_level': 'safe',
                'external_risk': 0.0,
                'internal_risk': 0.0,
                'driver_state': {
                    'current_state': 'safe_driving',
                    'confidence': 0.0,
                    'is_stale': True
                },
                'weights': {'external_weight': 0.6, 'internal_weight': 0.4},
                'external_breakdown': {},
                'risk_factors': {'synergy_factor': 1.0}
            }
            
        # Legacy compatibility
        self.current_ttc_risk = 0.0
        self.current_ttc_status = "SAFE"
        self.current_ttc_object = None
        self.current_signal_risk = 0.0
        self.current_signal_status = "SAFE"
        self.current_distance_risk = 0.0
        self.current_distance_status = "SAFE"
        self.current_braking_risk = 0.0
        self.current_braking_status = "SAFE"

    if self.IS_BENCH2DRIVE:
      # TODO doesn't seem to work
      metric_info = self.get_metric_info()
      self.metric_info[self.step] = metric_info
      if self.save_path is not None and self.step % 1 == 0:
        with open(self.save_path / 'metric_info.json', 'w') as outfile:
          ujson.dump(self.metric_info, outfile, indent=4)

    # CARLA will not let the car drive in the initial frames.
    # We set the action to brake so that the filter does not get confused.
    if self.step < self.config.inital_frames_delay:
      self.control = carla.VehicleControl(0.0, 0.0, 1.0)
    else:
      self.control = control

    # Process and display camera views
    if 'Center' in input_data:
      # Clear the display
      self._display.fill((0, 0, 0))
      
      # Draw center camera view
      image = input_data['Center'][1][:, :, :3][:, :, ::-1]
      surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
      # Scale surface to match display size
      surface = pygame.transform.scale(surface, (self._display.get_width(), self._display.get_height()))
      self._display.blit(surface, (0, 0))
      
      # Draw mirrors if available
      if 'Right' in input_data:
        right_mirror = input_data['Right'][1][:, :, :3][:, :, ::-1]
        right_surface = pygame.surfarray.make_surface(right_mirror.swapaxes(0, 1))
        right_surface = pygame.transform.scale(right_surface, (mirror_width, mirror_height))
        mirror_x = self._display.get_width() - mirror_width - 10  # 10 pixels padding
        mirror_y = 10  # 10 pixels from top
        self._display.blit(right_surface, (mirror_x, mirror_y))
      
      if 'Left' in input_data:
        left_mirror = input_data['Left'][1][:, :, :3][:, :, ::-1]
        left_surface = pygame.surfarray.make_surface(left_mirror.swapaxes(0, 1))
        left_surface = pygame.transform.scale(left_surface, (mirror_width, mirror_height))
        mirror_x = 10  # 10 pixels padding
        mirror_y = 10  # 10 pixels from top
        self._display.blit(left_surface, (mirror_x, mirror_y))
      
      # Draw unified risk assessment information (now includes traffic signals and speed)
      if hasattr(self, 'current_unified_assessment'):
          self.draw_unified_risk_info(self._display, self.current_unified_assessment)
      
      # Draw FPS and control mode text
      fps = self._clock.get_fps()
      mode_text = "Manual Control" if self.manual_control else "Autonomous Control" 
      fps_text = self._font_mono.render(f'FPS: {fps:.0f} - {mode_text} (Press Square to switch)', True, (255, 255, 255))
      text_width = fps_text.get_width()
      text_height = fps_text.get_height()
      text_x = (self._display.get_width() - text_width) // 2  # Center horizontally
      text_y = self._display.get_height() - text_height - 10  # 10 pixels from bottom
      
      # Draw semi-transparent background for text
      text_background = pygame.Surface((text_width + 20, text_height + 10))
      text_background.fill((0, 0, 0))
      text_background.set_alpha(128)
      self._display.blit(text_background, (text_x - 10, text_y - 5))
      
      # Draw text
      self._display.blit(fps_text, (text_x, text_y))
      
      # Update display
      pygame.display.flip()

    # Handle steering wheel input if available
    if self._joystick is not None:
      # Get joystick inputs
      numAxes = self._joystick.get_numaxes()
      jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
      
      # Debug brake value
      if self.step < 10 or self.step % 100 == 0:
        print(f"Raw brake value: {jsInputs[self._brake_idx]}")
      
      # Process brake pedal for manual control
      brake_value = jsInputs[self._brake_idx]
      
      # Convert brake value to command before using it for control switching
      brake_cmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
      if brake_cmd <= 0:
        brake_cmd = 0
      elif brake_cmd > 0.3:
        brake_cmd = 1
      
      # Check brake pedal state when in autonomous mode
      if not self.manual_control and brake_cmd > 0.5:
        # Record intervention
        self.data_recording['interventions'].append({
          'time': current_time,
          'step': self.step,
          'reason': 'brake_press'
        })
        
        # Update control time before switching modes
        mode_duration = current_time - self.data_recording['last_mode_switch_time']
        self.data_recording['control_time']['ai'] += mode_duration
        self.data_recording['last_mode_switch_time'] = current_time
        
        self.manual_control = True
        self.data_recording['current_mode'] = 'manual'
        self.last_switch_time = current_time
        print(f"BRAKE PRESS: Switching to MANUAL Control - Brake value: {brake_value}")
      
      if self.manual_control:
        # Process inputs for manual control
        K1 = 1.0  # 0.55
        steer_cmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])
        
        K2 = 1.6  # 1.6
        throttle_cmd = K2 + (2.05 * math.log10(
          -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttle_cmd <= 0:
          throttle_cmd = 0
        elif throttle_cmd > 1:
          throttle_cmd = 1
        
        # Create manual control command
        human_control = carla.VehicleControl(
          steer=float(steer_cmd),
          throttle=float(throttle_cmd), 
          brake=float(brake_cmd))
        
        # Record human control command
        if self.step % self.record_frequency == 0:
          self.data_recording['control_commands']['human'].append({
            'step': self.step,
            'time': current_time,
            'steer': steer_cmd,
            'throttle': throttle_cmd,
            'brake': brake_cmd
          })
        
        # Update for UKF
        if self.step < self.config.inital_frames_delay:
          self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
          self.control = human_control
        
        # Get AI control for comparison (without applying it)
        # ai_control = self.get_control(tick_data)
        
        # Record AI control command and differences
        if self.step % self.record_frequency == 0:
          self.data_recording['control_commands']['ai'].append({
            'step': self.step,
            'time': current_time,
            'steer': ai_control.steer,
            'throttle': ai_control.throttle,
            'brake': ai_control.brake
          })
          
          # Calculate and record steering difference
          steering_diff = abs(ai_control.steer - steer_cmd)
          self.data_recording['steering_differences'].append({
            'step': self.step,
            'difference': steering_diff
          })
          
          # Calculate and record speed difference (using throttle-brake as proxy)
          ai_speed = ai_control.throttle - ai_control.brake
          human_speed = throttle_cmd - brake_cmd
          speed_diff = abs(ai_speed - human_speed)
          self.data_recording['speed_differences'].append({
            'step': self.step,
            'difference': speed_diff
          })
        
        return human_control
    
    # If we're in autonomous mode or no joystick is available
    # ai_control = self.get_control(tick_data)
    
    # Record AI control command
    if self.step % self.record_frequency == 0:
      self.data_recording['control_commands']['ai'].append({
        'step': self.step,
        'time': current_time,
        'steer': ai_control.steer,
        'throttle': ai_control.throttle,
        'brake': ai_control.brake
      })
      
      # Update control time
      if not self.manual_control:
        mode_duration = current_time - self.data_recording['last_mode_switch_time']
        self.data_recording['control_time']['ai'] += mode_duration
        self.data_recording['last_mode_switch_time'] = current_time
    
    # Save data periodically
    # if self.step % 1000 == 0:
    #   # self.save_recorded_data()
    
    self.control = ai_control



    
    return control

  def stop_sign_controller_step(self, ego_speed):
    """Checks whether the car is intersecting with one of the detected stop signs"""
    if self.clear_stop_sign > 0:
      self.clear_stop_sign -= 1

    if len(self.bb_buffer) < 1:
      return False
    stop_sign_stop_predicted = False
    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)
    origin = carla.Location(x=0.0, y=0.0, z=0.0)

    car_box = carla.BoundingBox(origin, extent)

    for bb in self.bb_buffer[-1]:
      if bb[7] == 3:  # Stop sign detected
        self.stop_sign_buffer.append(bb)

    if len(self.stop_sign_buffer) > 0:
      # Check if we need to stop
      stop_box = self.stop_sign_buffer[0]
      stop_origin = carla.Location(x=stop_box[0], y=stop_box[1], z=0.0)
      stop_extent = carla.Vector3D(stop_box[2], stop_box[3], 1.0)
      stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
      stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box[4]), 0.0)

      if t_u.check_obb_intersection(stop_carla_box, car_box) and self.clear_stop_sign <= 0:
        if ego_speed > 0.01:
          stop_sign_stop_predicted = True
        else:
          # We have cleared the stop sign
          stop_sign_stop_predicted = False
          self.stop_sign_buffer.pop()
          # Stop signs don't come in herds, so we know we don't need to clear one for a while.
          self.clear_stop_sign = 100

    if len(self.stop_sign_buffer) > 0:
      # Remove boxes that are too far away
      if np.linalg.norm(self.stop_sign_buffer[0][:2]) > abs(self.config.max_x):
        self.stop_sign_buffer.pop()

    return stop_sign_stop_predicted

  def bb_detected_in_front_of_vehicle(self, ego_speed):
    if len(self.bb_buffer) < 1:  # We only start after we have 4 time steps.
      return False

    collision_predicted = False

    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)

    # Safety box
    bremsweg = ((ego_speed.cpu().numpy().item() * 3.6) / 10.0)**2 / 2.0  # Bremsweg formula for emergency break
    safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # plus one meter is the car.

    center_safety_box = carla.Location(x=safety_x, y=0.0, z=1.0)

    safety_bounding_box = carla.BoundingBox(center_safety_box, extent)
    safety_bounding_box.rotation = carla.Rotation(0.0, 0.0, 0.0)

    for bb in self.bb_buffer[-1]:
      # We just give them some arbitrary height. Does not matter
      bb_extent_z = 1.0
      loc_local = carla.Location(bb[0], bb[1], 0.0)
      extent_det = carla.Vector3D(bb[2], bb[3], bb_extent_z)
      bb_local = carla.BoundingBox(loc_local, extent_det)
      bb_local.rotation = carla.Rotation(0.0, np.rad2deg(bb[4]).item(), 0.0)

      if t_u.check_obb_intersection(safety_bounding_box, bb_local):
        collision_predicted = True

    return collision_predicted

  def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                [np.sin(orientation_target),
                                 np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    return t_u.algin_lidar(lidar, pos_diff, rot_diff)

  def update_stop_box(self, boxes, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target]) - np.array([x, y])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target)],
                                [np.sin(orientation_target), np.cos(orientation_target)]])
    pos_diff = rotation_matrix.T @ pos_diff

    # Rotation matrix in local coordinate system
    local_rot_matrix = np.array([[np.cos(rot_diff), -np.sin(rot_diff)], [np.sin(rot_diff), np.cos(rot_diff)]])

    for _, box_pred in enumerate(boxes):
      box_pred[:2] = (local_rot_matrix.T @ (box_pred[:2] - pos_diff).T).T
      box_pred[4] = t_u.normalize_angle(box_pred[4] - rot_diff)

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
    Also writes logging files to disk.
    """
    # Finalize unified risk manager (includes TTC plots)
    if hasattr(self, 'unified_risk_manager'):
        self.unified_risk_manager.shutdown()
    
    # Fallback for legacy safety evaluator
    if hasattr(self, 'safety_evaluator'):
        self.safety_evaluator.finalize_plots()
        
    if self.save_path is not None:
      self.lon_logger.dump_to_json()
      if len(self.nets[0].speed_histogram) > 0:
        with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
          ujson.dump(self.nets[0].speed_histogram, f, indent=4)

      if self.config.tp_attention:
        if len(self.tp_attention_buffer) > 0:
          print('Average TP attention: ', sum(self.tp_attention_buffer) / len(self.tp_attention_buffer))
          with gzip.open(self.save_path / 'tp_attention.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(self.tp_attention_buffer, f, indent=4)

        del self.tp_attention_buffer

    # Save TTC data if logger exists
    if hasattr(self, 'ttc_logger') and self.ttc_logger:
        print("🔍 DEBUG: Saving TTC data...")
        filepath = self.ttc_logger.save_data()
        report = self.ttc_logger.generate_report()
        print(report)
        
        # Save report to file
        if filepath:
            report_file = filepath.replace('.json', '_report.txt')
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📋 TTC report saved: {report_file}")
    else:
        print("🔍 DEBUG: No TTC logger to save")

    del self.nets
    del self.config
    del self.metric_info



  def draw_traffic_status(self, surface, signal_info, ego_speed):
    """Draw traffic signal status and ego speed in top-right corner"""
    
    texts = []
    
    # === EGO SPEED ===
    ego_speed_kmh = ego_speed * 3.6
    speed_color = (255, 255, 255)
    if ego_speed_kmh > 60:
        speed_color = (255, 165, 0)  # Orange for high speed
    elif ego_speed_kmh > 80:
        speed_color = (255, 50, 50)  # Red for very high speed
    
    texts.append((f"Speed: {ego_speed_kmh:.1f} km/h", speed_color))
    
    # === TRAFFIC SIGNALS ===
    if signal_info:
        signal_type = signal_info.get('signal_type', 'Unknown')
        distance = signal_info.get('distance', 0)
        risk = signal_info.get('signal_risk', 0)
        decel = signal_info.get('required_deceleration', 0)
        
        if signal_type == 'Red Light':
            signal_color = (255, 50, 50)
            icon = "🔴"
        elif signal_type == 'Stop Sign':
            signal_color = (255, 165, 0)
            icon = "🛑"
        else:
            signal_color = (255, 255, 255)
            icon = "⚠️"
            
        texts.append((f"{icon} {signal_type}", signal_color))
        texts.append((f"{distance:.1f}m", signal_color))
        
        if decel > 7.0:
            texts.append(("EMERGENCY BRAKE", (255, 0, 0)))
        elif decel > 4.0:
            texts.append(("HARD BRAKE", (255, 165, 0)))
    else:
        texts.append(("No Signals", (50, 255, 50)))
    
    # Render and position
    rendered_texts = []
    max_width = 0
    total_height = 0
    line_height = self._font_mono.get_height()
    
    for text, color in texts:
        rendered = self._font_mono.render(text, True, color)
        rendered_texts.append((rendered, color))
        max_width = max(max_width, rendered.get_width())
        total_height += line_height + 2
    
    # Position in top-right corner
    pos_x = surface.get_width() - max_width - 30
    pos_y = 30
    
    # Background
    bg = pygame.Surface((max_width + 20, total_height + 20))
    bg.fill((0, 0, 0))
    bg.set_alpha(128)
    surface.blit(bg, (pos_x - 10, pos_y - 10))
    
    # Draw text
    current_y = pos_y
    for rendered, color in rendered_texts:
        surface.blit(rendered, (pos_x, current_y))
        current_y += line_height + 2





# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
  # Kinematic bicycle model.
  # Numbers are the tuned parameters from World on Rails
  front_wb = -0.090769015
  rear_wb = 1.4178275

  steer_gain = 0.36848336
  brake_accel = -4.952399
  throt_accel = 0.5633837

  locs_0 = x[0]
  locs_1 = x[1]
  yaw = x[2]
  speed = x[3]

  if brake:
    accel = brake_accel
  else:
    accel = throt_accel * throttle

  wheel = steer_gain * steer

  beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
  next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
  next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
  next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
  next_speed = speed + accel * dt
  next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

  next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

  return next_state_x


def measurement_function_hx(vehicle_state):
  '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
  return vehicle_state


def state_mean(state, wm):
  '''
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def measurement_mean(state, wm):
  '''
  We use the arctan of the average of sin and cos of the angle to
  calculate the average of orientations.
  :param state: array of states to be averaged. First index is the
  timestep.
  '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def residual_state_x(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


def residual_measurement_h(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


class EgoModel:
  """
      Kinematic bicycle model describing the motion of a car given it's state and
      action. Tuned parameters are taken from World on Rails.
      """

  def __init__(self, dt, ego_vehicle_model=True):
    self.dt = dt  # the following numbers are optimized for dt=1./20. = 20 FPS

    self.ego_vehicle_model = ego_vehicle_model

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275
    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

    # Numbers are tuned parameters for the polynomial equations below using
    # a dataset where the car drives on a straight highway, accelerates to
    # 80 km/h and brakes to 0 km/h
    self.throt_values = np.array([
        9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
        -1.05996152e-03, 6.71079346e-04
    ])
    self.brake_values = np.array([
        9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
        -4.91381935e-12
    ])

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt

    if self.ego_vehicle_model:
      if brake:
        spds = spds * 3.6
        features = np.array([spds, spds**2, spds**3, spds**4, spds**5, spds**6, spds**7]).T

        next_spds = (features @ self.brake_values).item() / 3.6
      else:
        throttle = np.clip(throt, 0., 1.0)
        # for a throttle value < 0.3 the car doesn't accelerate and the polynomial model below breaks
        if throttle < 0.3:
          next_spds = spds
        else:
          spds = spds * 3.6
          features = np.array([
              spds, spds**2, throttle, throttle**2, spds * throttle, spds * throttle**2, spds**2 * throttle,
              spds**2 * throttle**2
          ]).T

          next_spds = (features @ self.throt_values).item() / 3.6
    else:
      if brake:
        next_spds = spds + self.brake_accel * self.dt
      else:
        next_spds = spds + self.throt_accel * self.dt

    next_spds = max(0, next_spds)

    next_locs = np.array([next_locs_0, next_locs_1, locs[2]])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds


        

