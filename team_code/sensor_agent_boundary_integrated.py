"""
Agent file with integrated Boundary Risk Estimator

This is a modified version of the sensor agent that integrates the boundary-based
risk estimation system to replace traditional TTC-based risk assessment.

Key Changes:
- Imports BoundaryRiskEstimator
- Replaces TTC calculations with boundary risk assessment
- Maintains compatibility with existing control systems
"""

import os
from copy import deepcopy

import cv2
import carla
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import pygame

from leaderboard.autoagents import autonomous_agent
from model import LidarCenterNet
from config import GlobalConfig
from data import CARLA_Data
from nav_planner import RoutePlanner
from nav_planner import extrapolate_waypoint_route

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.optimize import fsolve

from scenario_logger import ScenarioLogger
import transfuser_utils as t_u

# NEW: Import boundary risk estimator
from boundary_risk_estimator import BoundaryRiskEstimator
from boundary_alert_system import BoundaryRiskAlertSystem
from boundary_driver_alert_manager import BoundaryDriverAlertManager

import pathlib
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson  # Like json but faster
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False
})

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
    Now includes boundary-based risk estimation
    """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    """Sets up the agent. route_index is for logging purposes"""
    torch.cuda.empty_cache()

    # Manual control / visualization configuration
    self.camera_width = 1920
    self.camera_height = 960
    self.display_current_speed = 0
    self.display_predicted_speed = 0

    self.enable_manual_control = strtobool(os.environ.get('ENABLE_MANUAL_CONTROL', 'False'))
    self.record_frequency = int(os.environ.get('MANUAL_RECORD_FREQUENCY', 10))
    self.data_recording = None
    self.data_file = None
    self.last_ai_control = None
    self.manual_control = False
    self.brake_pressed = False
    self.last_switch_time = 0
    self._display = None
    self._clock = None
    self._font_mono = None
    self._joystick = None
    self._steer_idx = 0
    self._throttle_idx = 1
    self._brake_idx = 2
    self._reverse_idx = 3
    self._handbrake_idx = 4
    self._square_idx = 0

    if self.enable_manual_control:
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

      data_dir = os.path.join(os.getcwd(), "driving_data")
      os.makedirs(data_dir, exist_ok=True)
      self.data_file = os.path.join(data_dir, f"driving_data_{self.data_recording['session_start_time']}.json")
      print(f"Data will be saved to: {os.path.abspath(self.data_file)}")

      # Initialize pygame display for manual monitoring
      pygame.init()
      pygame.font.init()
      display_info = pygame.display.Info()
      screen_w = display_info.current_w
      screen_h = display_info.current_h
      scale_w = screen_w / self.camera_width
      scale_h = screen_h / self.camera_height
      scale = min(scale_w, scale_h) * 0.9
      scaled_width = int(self.camera_width * scale)
      scaled_height = int(self.camera_height * scale)
      os.environ['SDL_VIDEO_CENTERED'] = '1'
      self._display = pygame.display.set_mode(
          (scaled_width, scaled_height),
          pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED)
      pygame.display.set_caption("Sensor Agent View")
      self._clock = pygame.time.Clock()
      font_name = 'courier' if os.name == 'nt' else 'mono'
      fonts = [x for x in pygame.font.get_fonts() if font_name in x]
      default_font = 'ubuntumono'
      mono = default_font if default_font in fonts else fonts[0]
      mono = pygame.font.match_font(mono)
      self._font_mono = pygame.font.Font(mono, 24 if os.name == 'nt' else 28)

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
        self._square_idx = 1
        self.manual_control = False
        self.brake_pressed = False
        self.last_switch_time = 0
        print("Joystick initialized - Starting in AUTONOMOUS mode")
      else:
        self._joystick = None
        self.manual_control = False
        print("No steering wheel detected - autonomous control only")
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

    # Access CARLA ego actor/world for debug visualizations
    try:
      self._vehicle = CarlaDataProvider.get_hero_actor()
      self.world = self._vehicle.get_world()
    except Exception as exc:  # pylint: disable=broad-except
      print(f"⚠️ Failed to access CARLA ego vehicle/world: {exc}")
      self._vehicle = None
      self.world = None

    # NEW: Initialize Boundary Risk Estimator
    self.use_boundary_risk = strtobool(os.environ.get('USE_BOUNDARY_RISK', 'True'))
    print('Use boundary risk estimation:', self.use_boundary_risk)
    
    self.show_boundary_overlay = strtobool(os.environ.get('SHOW_BOUNDARY_RISK_OVERLAY', 'False'))
    print('Show boundary risk overlay:', self.show_boundary_overlay)
    self.draw_boundary_debug = strtobool(os.environ.get('DRAW_BOUNDARY_RISK_DEBUG', 'False'))
    print('Draw boundary risk debug lines (CARLA server):', self.draw_boundary_debug)
    
    if self.use_boundary_risk:
      # Configure boundary risk estimator from environment variables
      angular_resolution = int(os.environ.get('BOUNDARY_ANGULAR_RESOLUTION', 10))
      max_range = float(os.environ.get('BOUNDARY_MAX_RANGE', 50.0))
      k_distance = float(os.environ.get('BOUNDARY_K_DISTANCE', -1.0))
      alpha_coeff = float(os.environ.get('BOUNDARY_ALPHA', 1.0))
      beta_coeff = float(os.environ.get('BOUNDARY_BETA', 0.5))
      risk_threshold = float(os.environ.get('BOUNDARY_RISK_THRESHOLD', 0.3))
      lateral_threshold = float(os.environ.get('BOUNDARY_LATERAL_THRESHOLD', 5.0))
      self.boundary_risk_estimator = BoundaryRiskEstimator(
          angular_resolution=angular_resolution,
          max_range=max_range,
          k_distance=k_distance,
          alpha_coeff=alpha_coeff,
          beta_coeff=beta_coeff,
          risk_threshold=risk_threshold,
          lateral_risk_threshold=lateral_threshold
      )
      
      # Enable debug if requested
      boundary_debug = strtobool(os.environ.get('BOUNDARY_DEBUG', 'False'))
      self.boundary_risk_estimator.set_debug(boundary_debug)
      
      print(f'Boundary Risk Estimator initialized:')
      print(f'  Angular resolution: {angular_resolution}°')
      print(f'  Max range: {max_range}m')
      print(f'  Risk threshold: {risk_threshold}')
      print(f'  Debug mode: {boundary_debug}')
    else:
      self.boundary_risk_estimator = None
      print('Boundary risk estimation disabled - using original TTC methods')

    self.boundary_driver_alert = BoundaryDriverAlertManager() if self.use_boundary_risk else None
    self.boundary_alert_system = BoundaryRiskAlertSystem() if self.use_boundary_risk else None
    self.boundary_latest_alert = None

    # NEW: Initialize boundary risk data logging
    self.boundary_risk_data = {
      'timestamps': [],
      'risk_scores': [],
      'risk_levels': [],
      'ego_speeds': [],
      'threat_directions': [],
      'threat_distances': [],
      'threat_lateral_offsets': [],
      'threat_classes': [],
      'risk_fields': [],  # Store full risk fields for detailed analysis
      'control_modifications': []  # Track when control was modified due to risk
    }
    # Run-level safety metrics (min distances, collision/critical timestamps)
    self.boundary_metrics = {
      'min_distance_per_class': {cls: float('inf') for cls in range(5)},  # 0-4 classes
      'critical_min_distance': float('inf'),
      'critical_min_time': None,
      'collision_time': None,
    }
    self.collision_distance_threshold = float(os.environ.get('BOUNDARY_COLLISION_DISTANCE', '0.5'))
    self.near_miss_distance_threshold = float(os.environ.get('BOUNDARY_NEAR_MISS_DISTANCE', '3.0'))
    self.log_boundary_data = strtobool(os.environ.get('LOG_BOUNDARY_DATA', 'True'))
    print(f'Boundary risk data logging: {self.log_boundary_data}')
    self.last_boundary_risk_field = None
    self.last_boundary_risk_info = None
    self.boundary_overlay_surface = None
    self.boundary_overlay_font = None
    self.boundary_overlay_window = False
    if self.show_boundary_overlay:
      if self.enable_manual_control:
        print("⚠️ Boundary risk overlay disabled when manual display is active.")
        self.show_boundary_overlay = False
      else:
        try:
          pygame.init()
          pygame.font.init()
          self.boundary_overlay_surface = pygame.display.set_mode((360, 360))
          pygame.display.set_caption('Boundary Risk Overview')
          self.boundary_overlay_font = pygame.font.SysFont('Arial', 16)
          self.boundary_overlay_window = True
        except pygame.error as e:
          print(f"⚠️ Failed to initialize boundary overlay window: {e}")
          self.show_boundary_overlay = False

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
    sensors = [
        {
            'type': 'sensor.camera.rgb',
            'x': 1.4,
            'y': 0.0,
            'z': 1.2,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'width': self.camera_width,
            'height': self.camera_height,
            'fov': 110,
            'id': 'Center'
        },
        {
            'type': 'sensor.camera.rgb',
            'x': 0.7,
            'y': -1.0,
            'z': 1.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 210.0,
            'width': int(self.camera_width * 0.3),
            'height': int(self.camera_height * 0.3),
            'fov': 100,
            'id': 'Left'
        },
        {
            'type': 'sensor.camera.rgb',
            'x': 0.7,
            'y': 1.0,
            'z': 1.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 150.0,
            'width': int(self.camera_width * 0.3),
            'height': int(self.camera_height * 0.3),
            'fov': 100,
            'id': 'Right'
        }
    ]

    sensors.extend([{
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
    }, {
        'type': 'sensor.other.imu',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': self.config.carla_frame_rate,
        'id': 'imu'
    }, {
        'type': 'sensor.other.gnss',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': 0.01,
        'id': 'gps'
    }, {
        'type': 'sensor.speedometer',
        'reading_frequency': self.config.carla_fps,
        'id': 'speed'
    }])
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

  def calculate_boundary_risk_assessment(self, ego_speed, bounding_boxes, timestamp):
    """
    NEW: Calculate boundary-based risk assessment
    
    Args:
        ego_speed: Current ego vehicle speed (m/s)
        bounding_boxes: List of detected bounding boxes
        
    Returns:
        Dictionary with risk assessment results
    """
    if not self.use_boundary_risk or self.boundary_risk_estimator is None:
      # Fallback to no risk assessment
      return {
          'max_risk': 0.0,
          'risk_level': 'SAFE',
          'primary_threat': None,
          'boundary_based': False,
          'fallback_mode': True
      }
    
    try:
      # Calculate boundary risk
      risk_field, max_risk, threat_info = self.boundary_risk_estimator.calculate_boundary_risk(
          ego_speed, bounding_boxes, timestamp
      )
      
      # Add boundary-specific information
      threat_info['risk_field'] = risk_field
      threat_info['boundary_based'] = True
      threat_info['fallback_mode'] = False
      self.last_boundary_risk_field = risk_field
      self.last_boundary_risk_info = threat_info
      
      return threat_info
      
    except Exception as e:
      print(f"⚠️ Boundary risk calculation failed: {e}")
      # Fallback to safe mode
      self.last_boundary_risk_field = None
      self.last_boundary_risk_info = None
      return {
          'max_risk': 0.0,
          'risk_level': 'SAFE',
          'primary_threat': None,
          'boundary_based': False,
          'fallback_mode': True,
          'error': str(e)
      }

  def log_boundary_risk_data(self, timestamp, ego_speed, boundary_risk_info):
    """
    NEW: Log boundary risk data for later plotting and analysis
    
    Args:
        timestamp: Current simulation timestamp
        ego_speed: Current ego vehicle speed
        boundary_risk_info: Risk assessment results from boundary estimator
    """
    if not self.log_boundary_data:
      return
    
    # Store basic risk data
    self.boundary_risk_data['timestamps'].append(timestamp)
    self.boundary_risk_data['risk_scores'].append(boundary_risk_info.get('max_risk', 0.0))
    self.boundary_risk_data['risk_levels'].append(boundary_risk_info.get('risk_level', 'SAFE'))
    self.boundary_risk_data['ego_speeds'].append(ego_speed)
    
    # Store threat information
    primary_threat = boundary_risk_info.get('primary_threat')
    if primary_threat:
      self.boundary_risk_data['threat_directions'].append(primary_threat.get('direction_deg', 0.0))
      self.boundary_risk_data['threat_distances'].append(primary_threat.get('distance', 0.0))
      self.boundary_risk_data['threat_lateral_offsets'].append(primary_threat.get('lateral_offset', 0.0))
      self.boundary_risk_data['threat_classes'].append(primary_threat.get('object_class', 0))

      # Update run-level minima
      obj_class = primary_threat.get('object_class', 0)
      obj_distance = primary_threat.get('distance')
      if obj_distance is not None:
        # Per-class near-miss/min distance
        prev_min = self.boundary_metrics['min_distance_per_class'].get(obj_class, float('inf'))
        self.boundary_metrics['min_distance_per_class'][obj_class] = min(prev_min, obj_distance)

        # Collision time (first time crossing collision threshold)
        if (self.boundary_metrics.get('collision_time') is None and
            obj_distance <= self.collision_distance_threshold):
          self.boundary_metrics['collision_time'] = timestamp

        # Critical min distance/time
        if str(boundary_risk_info.get('risk_level', 'SAFE')).upper() == 'CRITICAL':
          if obj_distance < self.boundary_metrics['critical_min_distance']:
            self.boundary_metrics['critical_min_distance'] = obj_distance
            self.boundary_metrics['critical_min_time'] = timestamp
    else:
      self.boundary_risk_data['threat_directions'].append(None)
      self.boundary_risk_data['threat_distances'].append(None)
      self.boundary_risk_data['threat_lateral_offsets'].append(None)
      self.boundary_risk_data['threat_classes'].append(None)
    
    # Store full risk field (optional, for detailed analysis)
    store_risk_fields = strtobool(os.environ.get('STORE_RISK_FIELDS', 'False'))
    if store_risk_fields and 'risk_field' in boundary_risk_info:
      self.boundary_risk_data['risk_fields'].append(boundary_risk_info['risk_field'])
    else:
      self.boundary_risk_data['risk_fields'].append(None)

  def update_boundary_risk_overlay(self):
    """Render a simple polar risk visualization in a dedicated pygame window."""
    if not self.show_boundary_overlay or self.boundary_overlay_surface is None:
      return

    pygame.event.pump()
    surface = self.boundary_overlay_surface
    surface.fill((12, 12, 18))

    risk_field = self.last_boundary_risk_field
    if risk_field is None or len(risk_field) == 0:
      if self.boundary_overlay_font:
        text = self.boundary_overlay_font.render('No boundary risk data', True, (200, 200, 200))
        surface.blit(text, (surface.get_width() // 2 - text.get_width() // 2,
                            surface.get_height() // 2 - text.get_height() // 2))
      pygame.display.update()
      return

    center = (surface.get_width() // 2, surface.get_height() // 2 + 40)
    base_radius = 50
    max_radius = 130
    segments = len(risk_field)

    max_display_risk = 1.0
    for value in risk_field:
      max_display_risk = max(max_display_risk, min(2.0, value))

    for idx, risk in enumerate(risk_field):
      normalized = max(0.0, min(1.0, risk / max_display_risk))
      color = self._risk_to_color(normalized)
      angle = 2 * math.pi * idx / segments - math.pi / 2  # start at top
      length = base_radius + normalized * (max_radius - base_radius)
      end_pos = (center[0] + length * math.cos(angle),
                 center[1] + length * math.sin(angle))
      pygame.draw.line(surface, color, center, end_pos, 4)

    pygame.draw.circle(surface, (80, 80, 120), center, base_radius, 1)
    pygame.draw.circle(surface, (150, 150, 200), center, max_radius, 1)

    if self.boundary_overlay_font:
      info = self.last_boundary_risk_info
      max_risk_text = f"Max Risk: {info['max_risk']:.2f}" if info else "Max Risk: --"
      level_text = f"Level: {info['risk_level']}" if info else "Level: --"
      texts = [
        max_risk_text,
        level_text,
        f"Sectors: {segments}",
        "Color scale: green→yellow→red"
      ]
      y = 10
      for line in texts:
        text_surface = self.boundary_overlay_font.render(line, True, (220, 220, 230))
        surface.blit(text_surface, (10, y))
        y += text_surface.get_height() + 4

    pygame.display.update()

  def draw_boundary_risk_debug(self):
    """Draw risk direction lines inside the CARLA world (server-side overlay)."""
    if (not self.draw_boundary_debug or self.world is None or self._vehicle is None or
        self.last_boundary_risk_field is None or len(self.last_boundary_risk_field) == 0):
      return

    try:
      transform = self._vehicle.get_transform()
      base_location = transform.location + carla.Location(z=0.5)
      max_range = getattr(self.boundary_risk_estimator, 'max_range', 30.0)
      angles = getattr(self.boundary_risk_estimator, 'angles', None)
      if angles is None:
        segments = len(self.last_boundary_risk_field)
        angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)

      max_risk = max(1e-3, float(np.max(self.last_boundary_risk_field)))

      for angle_local, risk in zip(angles, self.last_boundary_risk_field):
        normalized = max(0.0, min(1.0, risk / max_risk))
        color_tuple = self._risk_to_color(normalized)
        color = carla.Color(r=color_tuple[0], g=color_tuple[1], b=color_tuple[2])

        # Length transitions smoothly from short (low risk) to long (high risk)
        min_len = 3.0
        max_len = min(max_range, 15.0)
        length = min_len + normalized * (max_len - min_len)

        local_x = length * math.cos(angle_local)
        local_y = length * math.sin(angle_local)
        end_world = transform.transform(carla.Location(x=local_x, y=local_y, z=0.0))

        self.world.debug.draw_line(
            base_location,
            end_world + carla.Location(z=0.5),
            thickness=0.08,
            color=color,
            life_time=0.1)
    except Exception as exc:  # pylint: disable=broad-except
      print(f"⚠️ Failed to draw boundary risk debug lines: {exc}")

  @staticmethod
  def _risk_to_color(value: float):
    """Convert normalized risk value (0-1) to RGB color."""
    if value <= 0.3:
      t = value / 0.3 if value > 0 else 0
      r = int(30 + t * (180 - 30))
      g = int(180 + t * (220 - 180))
      b = int(60 + t * (80 - 60))
    elif value <= 0.7:
      t = (value - 0.3) / 0.4
      r = int(180 + t * (255 - 180))
      g = int(220 - t * (220 - 160))
      b = int(80 - t * 60)
    else:
      t = min(1.0, (value - 0.7) / 0.3)
      r = 255
      g = int(160 - t * 120)
      b = int(20 + t * 35)
    return (r, g, b)

  def draw_boundary_alert_banner(self, surface):
    """Draw the latest boundary-risk alert text on the pygame surface."""
    if self._font_mono is None or self.boundary_latest_alert is None:
      return

    alert = self.boundary_latest_alert
    if time.time() - alert['timestamp'] > 3.0:
      return

    color_map = {
        'caution': (240, 210, 10),
        'warning': (255, 165, 0),
        'critical': (255, 70, 70)
    }
    text = alert['text']
    color = color_map.get(alert['level'], (255, 255, 255))
    text_surface = self._font_mono.render(text, True, color)
    padding = 12
    bg = pygame.Surface((text_surface.get_width() + padding * 2, text_surface.get_height() + padding * 2))
    bg.fill((0, 0, 0))
    bg.set_alpha(150)
    pos_x = (surface.get_width() - bg.get_width()) // 2
    pos_y = 20
    surface.blit(bg, (pos_x, pos_y))
    surface.blit(text_surface, (pos_x + padding, pos_y + padding))

  def draw_boundary_driver_overlay(self, surface):
    """Compact status overlay for boundary risk + driver state."""
    if self._font_mono is None:
      return

    # Colors per level
    level_colors = {
        'safe': (120, 220, 120),
        'caution': (240, 210, 10),
        'warning': (255, 165, 0),
        'critical': (255, 70, 70)
    }

    lines = []

    # Boundary risk
    info = getattr(self, 'last_boundary_risk_info', None)
    if info:
      level = str(info.get('risk_level', 'safe')).lower()
      risk_val = float(info.get('max_risk', 0.0))
      color = level_colors.get(level, (255, 255, 255))
      lines.append((f"Boundary {risk_val:.2f} [{level.upper()}]", color))

      threat = info.get('primary_threat') or {}
      cls_name = threat.get('class_name') or self.boundary_driver_alert.CLASS_MAP.get(
          threat.get('object_class'), 'object') if self.boundary_driver_alert else threat.get('class_name', 'object')
      distance = threat.get('distance')
      direction = threat.get('direction_deg', info.get('max_risk_direction_deg'))
      dir_text = self._format_direction_short(direction)
      dist_text = f"{distance:.0f}m" if distance is not None else "near"
      lines.append((f"{cls_name} {dist_text} {dir_text}", (210, 210, 210)))
    else:
      lines.append(("Boundary: no data", (200, 200, 200)))

    # Driver state (if driver alert manager is available)
    if self.boundary_driver_alert:
      driver = self.boundary_driver_alert
      is_stale = (time.time() - driver.last_driver_update) > driver.driver_timeout
      stale_tag = " STALE" if is_stale else ""
      lines.append((f"Driver: {driver.driver_state} ({driver.driver_confidence:.2f}){stale_tag}",
                    (180, 210, 255) if not is_stale else (200, 160, 160)))
    else:
      lines.append(("Driver: n/a", (180, 180, 180)))

    # Latest alert text
    if self.boundary_latest_alert:
      alert = self.boundary_latest_alert
      lines.append((f"Alert: {alert.get('text', '')}", (255, 255, 255)))

    # Render block
    line_height = self._font_mono.get_height()
    rendered = []
    max_w = 0
    total_h = 0
    for text, color in lines:
      surf = self._font_mono.render(text, True, color)
      rendered.append(surf)
      max_w = max(max_w, surf.get_width())
      total_h += line_height + 2

    pos_x = 20
    pos_y = surface.get_height() - total_h - 140  # bottom-left, above FPS bar

    bg = pygame.Surface((max_w + 20, total_h + 20))
    bg.fill((0, 0, 0))
    bg.set_alpha(128)
    surface.blit(bg, (pos_x - 10, pos_y - 10))

    cur_y = pos_y
    for surf in rendered:
      surface.blit(surf, (pos_x, cur_y))
      cur_y += line_height + 2

  @staticmethod
  def _format_direction_short(deg):
    if deg is None:
      return "ahead"
    norm = ((deg + 180.0) % 360.0) - 180.0
    if abs(norm) < 10:
      return "ahead"
    side = "R" if norm > 0 else "L"
    return f"{abs(norm):.0f}{side}"

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1
    current_time = time.time()

    mirror_width = None
    mirror_height = None
    if self.enable_manual_control and self._display is not None:
      mirror_width = int(self._display.get_width() * 0.2)
      mirror_height = int(self._display.get_height() * 0.2)

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

    if self.enable_manual_control:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return
        elif event.type == pygame.JOYBUTTONDOWN and self._joystick is not None:
          if event.button == self._square_idx and self.data_recording is not None:
            if current_time - self.last_switch_time > 0.2:
              if not self.manual_control:
                self.data_recording['interventions'].append({
                    'time': current_time,
                    'step': self.step,
                    'reason': 'button_press'
                })
              mode_duration = current_time - self.data_recording['last_mode_switch_time']
              if self.manual_control:
                self.data_recording['control_time']['human'] += mode_duration
              else:
                self.data_recording['control_time']['ai'] += mode_duration
              self.manual_control = not self.manual_control
              if self.data_recording is not None:
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
                                         self.stop_sign_controller or self.use_boundary_risk):
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
                                     self.stop_sign_controller or self.use_boundary_risk):
      # We average bounding boxes by using non-maximum suppression on the set of all detected boxes.
      bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)

      self.bb_buffer.append(bbs_vehicle_coordinate_system)
    else:
      bbs_vehicle_coordinate_system = None

    manual_override = False
    manual_control_cmd = None

    # Optional visualization overlay for manual supervision
    if self.enable_manual_control and self._display is not None and 'Center' in input_data:
      self._display.fill((0, 0, 0))

      image = input_data['Center'][1][:, :, :3][:, :, ::-1]
      surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
      surface = pygame.transform.scale(surface, (self._display.get_width(), self._display.get_height()))
      self._display.blit(surface, (0, 0))

      if mirror_width and 'Right' in input_data:
        right_mirror = input_data['Right'][1][:, :, :3][:, :, ::-1]
        right_surface = pygame.surfarray.make_surface(right_mirror.swapaxes(0, 1))
        right_surface = pygame.transform.scale(right_surface, (mirror_width, mirror_height))
        mirror_x = self._display.get_width() - mirror_width - 10
        mirror_y = 10
        self._display.blit(right_surface, (mirror_x, mirror_y))

      if mirror_width and 'Left' in input_data:
        left_mirror = input_data['Left'][1][:, :, :3][:, :, ::-1]
        left_surface = pygame.surfarray.make_surface(left_mirror.swapaxes(0, 1))
        left_surface = pygame.transform.scale(left_surface, (mirror_width, mirror_height))
        mirror_x = 10
        mirror_y = 10
        self._display.blit(left_surface, (mirror_x, mirror_y))

      if hasattr(self, 'current_unified_assessment') and self._font_mono is not None:
        self.draw_unified_risk_info(self._display, self.current_unified_assessment)

      if self._font_mono is not None and self._clock is not None:
        self._clock.tick()
        fps = self._clock.get_fps()
        mode_text = "Manual Control" if self.manual_control else "Autonomous Control"
        fps_text = self._font_mono.render(
            f'FPS: {fps:.0f} - {mode_text} (Press Square to switch)', True, (255, 255, 255))
        text_width = fps_text.get_width()
        text_height = fps_text.get_height()
        text_x = (self._display.get_width() - text_width) // 2
        text_y = self._display.get_height() - text_height - 10
        text_background = pygame.Surface((text_width + 20, text_height + 10))
        text_background.fill((0, 0, 0))
        text_background.set_alpha(128)
        self._display.blit(text_background, (text_x - 10, text_y - 5))
        self._display.blit(fps_text, (text_x, text_y))

      self.draw_boundary_driver_overlay(self._display)
      self.draw_boundary_alert_banner(self._display)
      pygame.display.flip()

    # Handle manual control via joystick if enabled
    if self.enable_manual_control and self._joystick is not None and self.data_recording is not None:
      ai_reference = self.last_ai_control if self.last_ai_control is not None else carla.VehicleControl()
      num_axes = self._joystick.get_numaxes()
      js_inputs = [float(self._joystick.get_axis(i)) for i in range(num_axes)]

      if self.step < 10 or self.step % 100 == 0:
        print(f"Raw brake value: {js_inputs[self._brake_idx]}")

      brake_value = js_inputs[self._brake_idx]
      brake_cmd = 1.6 + (2.05 * math.log10(-0.7 * js_inputs[self._brake_idx] + 1.4) - 1.2) / 0.92
      if brake_cmd <= 0:
        brake_cmd = 0
      elif brake_cmd > 0:
        brake_cmd = 1

      if not self.manual_control and brake_cmd > 0.5:
        self.data_recording['interventions'].append({
            'time': current_time,
            'step': self.step,
            'reason': 'brake_press'
        })
        mode_duration = current_time - self.data_recording['last_mode_switch_time']
        self.data_recording['control_time']['ai'] += mode_duration
        self.data_recording['last_mode_switch_time'] = current_time
        self.manual_control = True
        self.data_recording['current_mode'] = 'manual'
        self.last_switch_time = current_time
        print(f"BRAKE PRESS: Switching to MANUAL Control - Brake value: {brake_value}")

      if self.manual_control:
        K1 = 1.0
        steer_cmd = K1 * math.tan(1.1 * js_inputs[self._steer_idx])

        K2 = 1.6
        throttle_cmd = K2 + (2.05 * math.log10(-0.7 * js_inputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttle_cmd <= 0:
          throttle_cmd = 0
        elif throttle_cmd > 1:
          throttle_cmd = 1

        human_control = carla.VehicleControl(
            steer=float(steer_cmd),
            throttle=float(throttle_cmd),
            brake=float(brake_cmd))

        if self.step % self.record_frequency == 0:
          self.data_recording['control_commands']['human'].append({
              'step': self.step,
              'time': current_time,
              'steer': steer_cmd,
              'throttle': throttle_cmd,
              'brake': brake_cmd
          })

        if self.step < self.config.inital_frames_delay:
          self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
          self.control = human_control

        # Mark manual override but do not return yet; allow risk to be computed
        manual_override = True
        manual_control_cmd = human_control

        if self.step % self.record_frequency == 0:
          self.data_recording['control_commands']['ai'].append({
              'step': self.step,
              'time': current_time,
              'steer': ai_reference.steer,
              'throttle': ai_reference.throttle,
              'brake': ai_reference.brake
          })
          steering_diff = abs(ai_reference.steer - steer_cmd)
          self.data_recording['steering_differences'].append({
              'step': self.step,
              'difference': steering_diff
          })
          ai_speed = ai_reference.throttle - ai_reference.brake
          human_speed = throttle_cmd - brake_cmd
          speed_diff = abs(ai_speed - human_speed)
          self.data_recording['speed_differences'].append({
              'step': self.step,
              'difference': speed_diff
          })

        # Do not return here; continue to risk computation

      if self.step % self.record_frequency == 0:
        self.data_recording['control_commands']['ai'].append({
            'step': self.step,
            'time': current_time,
            'steer': ai_reference.steer,
            'throttle': ai_reference.throttle,
            'brake': ai_reference.brake
        })
        if not self.manual_control:
          mode_duration = current_time - self.data_recording['last_mode_switch_time']
          self.data_recording['control_time']['ai'] += mode_duration
          self.data_recording['last_mode_switch_time'] = current_time


    # =================================================================
    # NEW: BOUNDARY RISK ASSESSMENT
    # =================================================================
    boundary_risk_info = None
    if self.use_boundary_risk:
      self.last_boundary_risk_field = None
      self.last_boundary_risk_info = None
    if self.use_boundary_risk and bbs_vehicle_coordinate_system is not None:
      boundary_risk_info = self.calculate_boundary_risk_assessment(speed, bbs_vehicle_coordinate_system, timestamp)
      self.last_boundary_risk_info = boundary_risk_info
      
      # NEW: Log boundary risk data for plotting
      if self.log_boundary_data and boundary_risk_info:
        self.log_boundary_risk_data(timestamp, speed, boundary_risk_info)
      
      # Log boundary risk information
      if self.step % 50 == 0:  # Log every 50 steps to avoid spam
        print(f"🎯 BOUNDARY RISK Step {self.step}: "
              f"Risk={boundary_risk_info['max_risk']:.3f}, "
              f"Level={boundary_risk_info['risk_level']}")
        
        if boundary_risk_info.get('primary_threat'):
          threat = boundary_risk_info['primary_threat']
          print(f"   Primary threat: {threat['direction_deg']:.1f}° "
                f"at {threat['distance']:.1f}m")

    if self.boundary_driver_alert:
      self.boundary_driver_alert.process(boundary_risk_info, speed, timestamp)
      self.boundary_latest_alert = self.boundary_driver_alert.get_latest_alert()
    elif self.boundary_alert_system:
      self.boundary_alert_system.process(boundary_risk_info, speed, timestamp)
      self.boundary_latest_alert = self.boundary_alert_system.get_latest_alert()

    # If manual override is active, return the human control after risk is computed
    if manual_override:
      return manual_control_cmd

    if self.stop_sign_controller:
      stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

    if self.config.tp_attention:
      self.tp_attention_buffer.append(attention_weights[2])

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    # calculate target speed scalar from model predictions
    if self.config.use_controller_input_prediction:
      pred_target_speed_ensemble = torch.stack(pred_target_speeds,
                                               dim=0).mean(dim=0)  # average across ensemble models' prediction

      if self.uncertainty_weight:
        uncertainty = pred_target_speed_ensemble.detach().cpu().numpy()
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

    # =================================================================
    # NEW: BOUNDARY RISK-BASED CONTROL MODIFICATIONS
    # =================================================================
    if (self.use_boundary_risk and boundary_risk_info and not boundary_risk_info.get('fallback_mode', False)
        and self.log_boundary_data):
      max_risk = boundary_risk_info['max_risk']
      risk_level = boundary_risk_info['risk_level']
      self.boundary_risk_data['control_modifications'].append({
        'timestamp': timestamp,
        'step': self.step,
        'risk_level': risk_level,
        'max_risk': max_risk,
        'original_throttle': throttle,
        'modified_throttle': throttle,
        'original_brake': brake,
        'modified_brake': brake
      })

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

    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
    ai_control = control
    self.last_ai_control = ai_control

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

    if self.draw_boundary_debug:
      self.draw_boundary_risk_debug()

    if self.show_boundary_overlay:
      self.update_boundary_risk_overlay()

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

  def save_boundary_risk_data(self):
    """
    NEW: Save boundary risk data to JSON file for later analysis
    """
    if not self.log_boundary_data or not self.boundary_risk_data['timestamps']:
      return
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "boundary_risk_logs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"boundary_risk_data_{timestamp_str}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for JSON serialization (convert numpy arrays if present)
    json_data = {}
    for key, value in self.boundary_risk_data.items():
      if key == 'risk_fields' and any(v is not None for v in value):
        # Convert numpy arrays to lists for JSON serialization
        json_data[key] = [v.tolist() if v is not None else None for v in value]
      else:
        json_data[key] = value
    
    # Add metadata
    json_data['metadata'] = {
      'total_steps': len(self.boundary_risk_data['timestamps']),
      'simulation_duration': self.boundary_risk_data['timestamps'][-1] - self.boundary_risk_data['timestamps'][0] if self.boundary_risk_data['timestamps'] else 0,
      'boundary_risk_estimator_config': {
        'angular_resolution': getattr(self.boundary_risk_estimator, 'M', None),
        'max_range': getattr(self.boundary_risk_estimator, 'max_range', None),
        'risk_threshold': getattr(self.boundary_risk_estimator, 'tau', None)
      } if self.boundary_risk_estimator else None
    }
    
    # Save to file
    try:
      with open(filepath, 'w') as f:
        ujson.dump(json_data, f, indent=2)
      print(f"📊 Boundary risk data saved: {filepath}")
    except Exception as e:
      print(f"❌ Error saving boundary risk data: {e}")

  def plot_boundary_risk_analysis(self):
    """
    NEW: Generate comprehensive boundary risk analysis plots
    """
    if not self.log_boundary_data or not self.boundary_risk_data['timestamps']:
      print("⚠️ No boundary risk data to plot")
      return
    
    print("📊 Generating boundary risk analysis plots...")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "boundary_risk_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
      # Convert timestamps to relative time (seconds from start)
      timestamps = np.array(self.boundary_risk_data['timestamps'])
      relative_time = timestamps - timestamps[0]
      
      # Plot 1: Risk Score Over Time
      self._plot_risk_timeline(relative_time, output_dir, timestamp_str)
      
      # Plot 2: Risk Level Distribution
      self._plot_risk_distribution(output_dir, timestamp_str)
      
      # Plot 3: Threat Analysis
      self._plot_threat_analysis(relative_time, output_dir, timestamp_str)
      
      # Plot 4: Control Modifications
      self._plot_control_modifications(relative_time, output_dir, timestamp_str)
      
      # Plot 5: Speed vs Risk Correlation
      self._plot_speed_risk_correlation(output_dir, timestamp_str)
      
      # Plot 6: Risk Field Heatmap (if available)
      if any(rf is not None for rf in self.boundary_risk_data['risk_fields']):
        self._plot_risk_field_heatmap(relative_time, output_dir, timestamp_str)
      
      print(f"📊 Boundary risk plots saved to: {output_dir}")
      
    except Exception as e:
      print(f"❌ Error generating plots: {e}")
      import traceback
      traceback.print_exc()

  def _plot_risk_timeline(self, relative_time, output_dir, timestamp_str):
    """Plot risk score and ego speed over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot risk score
    risk_scores = np.array(self.boundary_risk_data['risk_scores'])
    ego_speeds = np.array(self.boundary_risk_data['ego_speeds'])
    
    ax1.plot(relative_time, risk_scores, 'r-', linewidth=2, label='Boundary Risk Score')
    ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Caution Threshold')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
    ax1.set_ylabel('Risk Score')
    ax1.set_title('Boundary Risk Score Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Color background by risk level
    risk_levels = self.boundary_risk_data['risk_levels']
    for i in range(len(relative_time)-1):
      color = {'SAFE': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange', 'CRITICAL': 'red'}.get(risk_levels[i], 'gray')
      ax1.axvspan(relative_time[i], relative_time[i+1], alpha=0.1, color=color)
    
    # Plot ego speed
    ax2.plot(relative_time, ego_speeds * 3.6, 'b-', linewidth=2, label='Ego Speed (km/h)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('Ego Vehicle Speed Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'risk_timeline_{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

  def _plot_risk_distribution(self, output_dir, timestamp_str):
    """Plot risk level distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Risk level pie chart
    risk_levels = self.boundary_risk_data['risk_levels']
    level_counts = {}
    for level in risk_levels:
      level_counts[level] = level_counts.get(level, 0) + 1
    
    colors = {'SAFE': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange', 'CRITICAL': 'red'}
    pie_colors = [colors.get(level, 'gray') for level in level_counts.keys()]
    
    ax1.pie(level_counts.values(), labels=level_counts.keys(), colors=pie_colors, autopct='%1.1f%%')
    ax1.set_title('Risk Level Distribution')
    
    # Risk score histogram
    risk_scores = self.boundary_risk_data['risk_scores']
    ax2.hist(risk_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=0.3, color='orange', linestyle='--', label='Caution Threshold')
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Warning Threshold')
    ax2.set_xlabel('Risk Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Risk Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'risk_distribution_{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

  def _plot_threat_analysis(self, relative_time, output_dir, timestamp_str):
    """Plot threat direction and distance analysis"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Filter out None values
    valid_indices = [i for i, d in enumerate(self.boundary_risk_data['threat_directions']) if d is not None]
    
    if valid_indices:
      valid_times = relative_time[valid_indices]
      threat_directions = [self.boundary_risk_data['threat_directions'][i] for i in valid_indices]
      threat_distances = [self.boundary_risk_data['threat_distances'][i] for i in valid_indices]
      threat_classes = [self.boundary_risk_data['threat_classes'][i] for i in valid_indices]
      
      # Plot threat directions
      class_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'orange', 4: 'purple'}
      class_names = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Red Light', 3: 'Stop Sign', 4: 'Emergency'}
      
      for class_id in set(threat_classes):
        class_indices = [i for i, c in enumerate(threat_classes) if c == class_id]
        if class_indices:
          class_times = [valid_times[i] for i in class_indices]
          class_directions = [threat_directions[i] for i in class_indices]
          ax1.scatter(class_times, class_directions, c=class_colors.get(class_id, 'gray'), 
                     label=class_names.get(class_id, f'Class {class_id}'), alpha=0.7)
      
      ax1.set_ylabel('Threat Direction (degrees)')
      ax1.set_title('Primary Threat Direction Over Time')
      ax1.legend()
      ax1.grid(True, alpha=0.3)
      
      # Plot threat distances
      for class_id in set(threat_classes):
        class_indices = [i for i, c in enumerate(threat_classes) if c == class_id]
        if class_indices:
          class_times = [valid_times[i] for i in class_indices]
          class_distances = [threat_distances[i] for i in class_indices]
          ax2.scatter(class_times, class_distances, c=class_colors.get(class_id, 'gray'), 
                     label=class_names.get(class_id, f'Class {class_id}'), alpha=0.7)
      
      ax2.set_xlabel('Time (seconds)')
      ax2.set_ylabel('Threat Distance (meters)')
      ax2.set_title('Primary Threat Distance Over Time')
      ax2.legend()
      ax2.grid(True, alpha=0.3)
    else:
      ax1.text(0.5, 0.5, 'No threat data available', ha='center', va='center', transform=ax1.transAxes)
      ax2.text(0.5, 0.5, 'No threat data available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'threat_analysis_{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

  def _plot_control_modifications(self, relative_time, output_dir, timestamp_str):
    """Plot control modifications due to boundary risk"""
    if not self.boundary_risk_data['control_modifications']:
      print("⚠️ No control modifications to plot")
      return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Extract control modification data
    mod_times = []
    mod_throttle_orig = []
    mod_throttle_new = []
    mod_brake_orig = []
    mod_brake_new = []
    mod_risk_levels = []
    
    for mod in self.boundary_risk_data['control_modifications']:
      mod_time = mod['timestamp'] - relative_time[0] if relative_time[0] > 0 else mod['timestamp']
      mod_times.append(mod_time)
      mod_throttle_orig.append(mod['original_throttle'])
      mod_throttle_new.append(mod['modified_throttle'])
      mod_brake_orig.append(mod['original_brake'])
      mod_brake_new.append(mod['modified_brake'])
      mod_risk_levels.append(mod['risk_level'])
    
    # Plot throttle modifications
    ax1.plot(mod_times, mod_throttle_orig, 'b-', alpha=0.7, label='Original Throttle')
    ax1.plot(mod_times, mod_throttle_new, 'r-', alpha=0.7, label='Modified Throttle')
    ax1.scatter(mod_times, mod_throttle_new, c='red', s=30, alpha=0.8)
    ax1.set_ylabel('Throttle')
    ax1.set_title('Throttle Control Modifications')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot brake modifications
    ax2.plot(mod_times, mod_brake_orig, 'b-', alpha=0.7, label='Original Brake')
    ax2.plot(mod_times, mod_brake_new, 'r-', alpha=0.7, label='Modified Brake')
    ax2.scatter(mod_times, mod_brake_new, c='red', s=30, alpha=0.8)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Brake')
    ax2.set_title('Brake Control Modifications')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'control_modifications_{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

  def _plot_speed_risk_correlation(self, output_dir, timestamp_str):
    """Plot correlation between ego speed and risk score"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ego_speeds = np.array(self.boundary_risk_data['ego_speeds']) * 3.6  # Convert to km/h
    risk_scores = np.array(self.boundary_risk_data['risk_scores'])
    
    # Create scatter plot with color coding by risk level
    risk_levels = self.boundary_risk_data['risk_levels']
    colors = {'SAFE': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange', 'CRITICAL': 'red'}
    
    for level in set(risk_levels):
      indices = [i for i, l in enumerate(risk_levels) if l == level]
      if indices:
        ax.scatter([ego_speeds[i] for i in indices], [risk_scores[i] for i in indices], 
                  c=colors.get(level, 'gray'), label=level, alpha=0.6)
    
    ax.set_xlabel('Ego Speed (km/h)')
    ax.set_ylabel('Risk Score')
    ax.set_title('Risk Score vs Ego Speed Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(ego_speeds, risk_scores)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speed_risk_correlation_{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

  def _plot_risk_field_heatmap(self, relative_time, output_dir, timestamp_str):
    """Plot risk field heatmap over time (if risk fields are stored)"""
    # Get valid risk fields
    valid_fields = [rf for rf in self.boundary_risk_data['risk_fields'] if rf is not None]
    if not valid_fields:
      return
    
    # Sample every N frames to avoid too dense visualization
    sample_rate = max(1, len(valid_fields) // 100)  # Max 100 time samples
    sampled_fields = valid_fields[::sample_rate]
    sampled_times = relative_time[::sample_rate][:len(sampled_fields)]
    
    # Create heatmap
    risk_matrix = np.array(sampled_fields).T  # Transpose for proper orientation
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    im = ax.imshow(risk_matrix, aspect='auto', cmap='hot', interpolation='bilinear')
    
    # Set labels
    ax.set_xlabel('Time Sample')
    ax.set_ylabel('Angular Direction (degrees)')
    ax.set_title('Boundary Risk Field Heatmap Over Time')
    
    # Set y-axis ticks to show angles
    num_angles = risk_matrix.shape[0]
    angle_ticks = np.linspace(0, num_angles-1, 8)
    angle_labels = [f'{int(360 * tick / num_angles)}°' for tick in angle_ticks]
    ax.set_yticks(angle_ticks)
    ax.set_yticklabels(angle_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Risk Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'risk_field_heatmap_{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

  def _save_boundary_run_metrics(self):
    """Persist run-level boundary safety metrics (min distances, collision time)."""
    # Prepare output directory
    if self.save_path is not None:
      out_dir = self.save_path
    else:
      out_dir = pathlib.Path(os.getcwd()) / "boundary_metrics"
      out_dir.mkdir(parents=True, exist_ok=True)

    # Convert Path to string for os.path operations
    out_dir_str = str(out_dir)

    # Normalize metrics (replace inf with None)
    min_distances = {}
    class_names = {
      0: "vehicle",
      1: "pedestrian",
      2: "traffic_light",
      3: "stop_sign",
      4: "emergency"
    }
    for cls_id, dist in self.boundary_metrics['min_distance_per_class'].items():
      label = class_names.get(cls_id, f"class_{cls_id}")
      min_distances[label] = None if dist == float('inf') else dist

    metrics = {
      "min_distance_per_class": min_distances,
      "critical_min_distance": None if self.boundary_metrics['critical_min_distance'] == float('inf') else self.boundary_metrics['critical_min_distance'],
      "critical_min_time": self.boundary_metrics['critical_min_time'],
      "collision_time": self.boundary_metrics['collision_time'],
      "collision_distance_threshold": self.collision_distance_threshold,
      "near_miss_distance_threshold": self.near_miss_distance_threshold,
      "frames_logged": len(self.boundary_risk_data.get('timestamps', []))
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir_str, f"boundary_run_metrics_{timestamp_str}.json")
    with open(filename, "w", encoding="utf-8") as f:
      json.dump(metrics, f, indent=2)
    print(f"📄 Boundary run metrics saved: {filename}")

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
    Also writes logging files to disk.
    """
    # Save run-level boundary safety metrics (min distances, collision time)
    if self.use_boundary_risk and self.log_boundary_data:
      try:
        self._save_boundary_run_metrics()
      except Exception as e:
        print(f"⚠️ Error saving boundary run metrics: {e}")

    # NEW: Generate boundary risk plots before cleanup
    if self.use_boundary_risk and self.log_boundary_data:
      try:
        self.plot_boundary_risk_analysis()
        self.save_boundary_risk_data()
      except Exception as e:
        print(f"⚠️ Error generating boundary risk plots: {e}")
    
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

    del self.nets
    del self.config
    del self.metric_info


# Filter Functions (unchanged from original)
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
