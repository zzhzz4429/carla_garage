"""
Agent file that runs the evaluations for all models supported by this repo.
Run it by giving it as the agent option to the
leaderboard/leaderboard/leaderboard_evaluator.py file
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
from safety_evaluator import SafetyEvaluator

import pathlib
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson  # Like json but faster
import gzip
import json
from pathlib import Path

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
    """Sets up the agent. route_index is for logging purposes"""
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

    self.safety_evaluator = SafetyEvaluator(self.config)

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
    sensors = [{
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
    }]
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

    self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
    self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
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

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1

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

    if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                    self.config.backbone in ('transFuser') or self.config.backbone in ('geometric_fusion')):
        # Collect environment data
        env_data = self.collect_environment_data(bbs_vehicle_coordinate_system)
        
        # Add predicted target speed (convert from m/s to km/h)
        if 'pred_target_speed_scalar' in locals():
            # Store both the original m/s value and the converted km/h value
            env_data["predicted_target_speed_ms"] = float(pred_target_speed_scalar)
            env_data["predicted_target_speed_kmh"] = float(pred_target_speed_scalar) * 3.6  # Convert m/s to km/h
        
        # Add current speed (convert from m/s to km/h)
        env_data["current_speed_ms"] = float(gt_velocity.item())
        env_data["current_speed_kmh"] = float(gt_velocity.item()) * 3.6  # Convert m/s to km/h
        
        # Add command information
        env_data["command"] = int(self.commands[-1])
        
        # Store the data
        if not hasattr(self, 'environment_data_buffer'):
            self.environment_data_buffer = []
        
        self.environment_data_buffer.append(env_data)
        
        # Add to metric info if using bench2drive
        if self.IS_BENCH2DRIVE:
            self.metric_info[self.step]['environment_data'] = env_data

    # After processing bounding boxes and before returning control
    if hasattr(self, 'bb_buffer') and len(self.bb_buffer) > 0:
        # Update time for plotting using actual timestamp
        frame_time = timestamp  # Use actual timestamp instead of assuming 10 Hz
        self.safety_evaluator.update_time(frame_time)
        
        # Print all detected objects for debugging
        print("=== All Detected Objects ===")
        object_types = {0: "Vehicle", 1: "Pedestrian", 2: "Red Light", 3: "Stop Sign", 4: "Emergency Vehicle"}
        for i, bb in enumerate(self.bb_buffer[-1]):
            if len(bb) > 7:
                obj_type = object_types.get(bb[7], f"Unknown({bb[7]})")
                speed = bb[5] if len(bb) > 5 and bb[5] is not None else "N/A"
                distance = bb[0]
                lateral_pos = bb[1]
                print(f"  {obj_type} {i}: Speed={speed} m/s, Distance={distance:.1f}m, Lateral={lateral_pos:.1f}m")
                # Debug: Print raw bounding box data
                print(f"    Raw bb data: {bb}")
        print("=============================")
        
        # Update timestamp for accurate delta_t calculation
        self.safety_evaluator.update_timestamp(timestamp)
        
        # Calculate TTC with gap tracking (new method)
        ttc_risk, ttc_object = self.safety_evaluator.calculate_ttc_with_gap_tracking(
            ego_speed=gt_velocity.item(),
            bounding_boxes=self.bb_buffer[-1]
        )
        
        # Calculate signal compliance risk
        signal_risk, signal_object = self.safety_evaluator.calculate_signal_compliance_risk(
            ego_speed=gt_velocity.item(),
            bounding_boxes=self.bb_buffer[-1]
        )
        
        # Calculate alternative risk assessments
        distance_risk, distance_object = self.safety_evaluator.calculate_distance_based_risk(
            ego_speed=gt_velocity.item(),
            bounding_boxes=self.bb_buffer[-1]
        )
        
        braking_risk, braking_object = self.safety_evaluator.calculate_braking_distance_risk(
            ego_speed=gt_velocity.item(),
            bounding_boxes=self.bb_buffer[-1]
        )
        
        # Get status for each method
        ttc_risk_status = self.safety_evaluator.get_risk_status(ttc_risk)
        signal_status = self.safety_evaluator.get_risk_status(signal_risk)
        distance_status = self.safety_evaluator.get_risk_status(distance_risk)
        braking_status = self.safety_evaluator.get_risk_status(braking_risk)
        
        # Print risk information for debugging
        print(f"=== Risk Assessment ===")
        print(f"TTC (gap tracking): {ttc_risk:.3f}, Status: {ttc_risk_status}")
        print(f"Signal Compliance: {signal_risk:.3f}, Status: {signal_status}")
        print(f"Distance Risk: {distance_risk:.3f}, Status: {distance_status}")
        print(f"Braking Risk: {braking_risk:.3f}, Status: {braking_status}")
        
        if ttc_object:
            object_type = object_types.get(ttc_object['object_class'], f"Unknown({ttc_object['object_class']})")
            print(f"  TTC Risk Object: {object_type} at {ttc_object['distance']:.1f}m")
            print(f"  Gap Rate: {ttc_object['gap_rate']:.2f} m/s, TTC: {ttc_object['ttc']:.2f}s")
            print(f"  TTC Risk: {ttc_object['ttc_risk']:.3f}")
            
        if signal_object:
            print(f"  Signal Risk: {signal_object['signal_type']} at {signal_object['distance']:.1f}m")
            print(f"  Required Deceleration: {signal_object['required_deceleration']:.2f} m/s²")
            print(f"  Signal Risk: {signal_object['signal_risk']:.3f}")
            
        if distance_object:
            object_type = object_types.get(distance_object['object_class'], f"Unknown({distance_object['object_class']})")
            print(f"  Highest distance risk: {object_type} at {distance_object['distance']:.1f}m")
            print(f"  Longitudinal risk: {distance_object['longitudinal_risk']:.3f}")
            print(f"  Lateral risk: {distance_object['lateral_risk']:.3f}")
            print(f"  Speed risk: {distance_object['speed_risk']:.3f}")
            
        if braking_object:
            object_type = object_types.get(braking_object['object_class'], f"Unknown({braking_object['object_class']})")
            print(f"  Highest braking risk: {object_type} at {braking_object['distance']:.1f}m")
            print(f"  Stopping distance needed: {braking_object['stopping_distance']:.1f}m")
            print(f"  Risk: {braking_object['risk']:.3f}")
        print("======================")
        
        # # Adjust control based on safety score if needed
        # if safety_status == "CRITICAL":           
        #     control.throttle = 0.0
        #     control.brake = 1.0
        # elif safety_status == "WARNING":
        #     control.throttle = max(0.0, control.throttle - 0.3)
        #     control.brake = min(1.0, control.brake + 0.3)
        # elif safety_status == "CAUTION":
        #     control.throttle = max(0.0, control.throttle - 0.1)
        #     control.brake = min(1.0, control.brake + 0.1)
    
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

  def collect_environment_data(self, pred_bb):
    """
    Collects data about the environment including categorized distance ranges
    
    Args:
        pred_bb: List of predicted bounding boxes
        
    Returns:
        dict: Dictionary containing the environment data
    """
    # Initialize the data structure
    env_data = {
        # Presence flags (binary)
        "vehicle_in_front": False,
        "pedestrian_crossing": False,
        "red_light_ahead": False,
        "stop_sign_ahead": False,
        "emergency_vehicle_ahead": False,
        
        # Exact distances (meters)
        "distance_to_closest_front_vehicle": float('inf'),
        "distance_to_closest_pedestrian": float('inf'),            
        "distance_to_nearest_red_light": float('inf'),
        "distance_to_nearest_stop_sign": float('inf'),
        "distance_to_nearest_emergency_vehicle": float('inf'),
        
        # Categorized distances
        "vehicles_by_distance": {
            "very_close": 0,  # < 10m
            "close": 0,       # 10-30m
            "medium": 0,      # 30-50m
            "far": 0          # > 50m
        },
        "pedestrians_by_distance": {
            "very_close": 0,  # < 10m
            "close": 0,       # 10-30m
            "medium": 0,      # 30-50m
            "far": 0          # > 50m
        },
        "red_lights_by_distance": {
            "very_close": 0,  # < 10m
            "close": 0,       # 10-30m
            "medium": 0,      # 30-50m
            "far": 0          # > 50m
        },
        "stop_signs_by_distance": {
            "very_close": 0,  # < 10m
            "close": 0,       # 10-30m
            "medium": 0,      # 30-50m
            "far": 0          # > 50m
        },
        "emergency_vehicles_by_distance": {
            "very_close": 0,  # < 10m
            "close": 0,       # 10-30m
            "medium": 0,      # 30-50m
            "far": 0          # > 50m
        }
    }
    
    if pred_bb is None or len(pred_bb) == 0:
        return env_data
    
    # Define the object classes
    VEHICLE = 0           # Car (orange)
    PEDESTRIAN = 1        # Pedestrian (green)
    RED_LIGHT = 2         # Red light (red)
    STOP_SIGN = 3         # Stop sign (pink)
    EMERGENCY_VEHICLE = 4 # Emergency vehicle (teal)
    
    # Helper function to categorize distance
    def categorize_distance(distance):
        if distance < 5.0:
            return "very_close"
        elif distance < 10.0:
            return "close"
        elif distance < 20.0:
            return "medium"
        else:
            return "far"
    
    # Initialize pedestrian tracking if not already done
    if not hasattr(self, 'pedestrian_tracking'):
        self.pedestrian_tracking = {}
    
    # Process each bounding box
    for box in pred_bb:
        # Verify the class index is within expected range
        if len(box) <= 7 or box[7] < 0 or box[7] > 4:
            print(f"Warning: Invalid class index in bounding box: {box}")
            continue
            
        object_type = int(box[7])
        
        # Get position (x, y) in vehicle coordinate system
        x, y = box[0], box[1]
        
        # Only consider objects in front of the ego vehicle
        if x > 0:
            # Calculate distance
            if object_type == VEHICLE:
                # For vehicles, consider lateral distance
                lateral_distance = abs(y)
                lane_threshold = 2.0  # meters from center
                
                if lateral_distance <= lane_threshold:
                    # This is a vehicle in our lane
                    longitudinal_distance = x
                    
                    # Update closest distance
                    if longitudinal_distance < env_data["distance_to_closest_front_vehicle"]:
                        env_data["vehicle_in_front"] = True
                        env_data["distance_to_closest_front_vehicle"] = longitudinal_distance
                    
                    # Increment the appropriate distance category
                    category = categorize_distance(longitudinal_distance)
                    env_data["vehicles_by_distance"][category] += 1
            
            elif object_type == PEDESTRIAN:
                # For pedestrians, we need to track their lateral movement
                # Generate a unique ID for this pedestrian based on its position
                # This is a simple approach - in a real system you'd want more robust tracking
                ped_id = f"ped_{int(x*10)}_{int(y*10)}"
                distance = np.sqrt(x**2 + y**2)
                
                if ped_id not in self.pedestrian_tracking:
                    # First time seeing this pedestrian
                    self.pedestrian_tracking[ped_id] = {
                        "initial_y": y,
                        "current_y": y,
                        "has_crossed": False,
                        "last_seen": self.step
                    }
                    
                    # New pedestrian in front is considered crossing
                    env_data["pedestrian_crossing"] = True
                    env_data["distance_to_closest_pedestrian"] = min(
                        env_data["distance_to_closest_pedestrian"], distance)
                    
                    # Increment the appropriate distance category
                    category = categorize_distance(distance)
                    env_data["pedestrians_by_distance"][category] += 1
                else:
                    # Update the tracking info
                    self.pedestrian_tracking[ped_id]["current_y"] = y
                    self.pedestrian_tracking[ped_id]["last_seen"] = self.step
                    
                    # Check if pedestrian has crossed based on lateral position change
                    initial_y = self.pedestrian_tracking[ped_id]["initial_y"]
                    current_y = y
                    has_crossed = self.pedestrian_tracking[ped_id]["has_crossed"]
                    
                    # Determine if pedestrian has crossed based on your criteria
                    if not has_crossed:
                        if (initial_y < 0 and current_y > 1.0) or (initial_y > 0 and current_y < -2.0):
                            # Pedestrian has crossed from one side to the other
                            self.pedestrian_tracking[ped_id]["has_crossed"] = True
                            # Don't count this pedestrian anymore
                            continue
                    
                    # If pedestrian hasn't crossed yet, include in environment data
                    if not self.pedestrian_tracking[ped_id]["has_crossed"]:
                        env_data["pedestrian_crossing"] = True
                        env_data["distance_to_closest_pedestrian"] = min(
                            env_data["distance_to_closest_pedestrian"], distance)
                        
                        # Increment the appropriate distance category
                        category = categorize_distance(distance)
                        env_data["pedestrians_by_distance"][category] += 1
            
            else:
                # For other objects, use Euclidean distance
                distance = np.sqrt(x**2 + y**2)
                
                # Update based on object type
                if object_type == RED_LIGHT:
                    env_data["red_light_ahead"] = True
                    env_data["distance_to_nearest_red_light"] = min(
                        env_data["distance_to_nearest_red_light"], distance)
                    
                    # Increment the appropriate distance category
                    category = categorize_distance(distance)
                    env_data["red_lights_by_distance"][category] += 1
                    
                elif object_type == STOP_SIGN:
                    env_data["stop_sign_ahead"] = True
                    env_data["distance_to_nearest_stop_sign"] = min(
                        env_data["distance_to_nearest_stop_sign"], distance)
                    
                    # Increment the appropriate distance category
                    category = categorize_distance(distance)
                    env_data["stop_signs_by_distance"][category] += 1
                    
                elif object_type == EMERGENCY_VEHICLE:
                    env_data["emergency_vehicle_ahead"] = True
                    env_data["distance_to_nearest_emergency_vehicle"] = min(
                        env_data["distance_to_nearest_emergency_vehicle"], distance)
                    
                    # Increment the appropriate distance category
                    category = categorize_distance(distance)
                    env_data["emergency_vehicles_by_distance"][category] += 1
    
    # Clean up pedestrian tracking - remove pedestrians not seen recently
    pedestrians_to_remove = []
    for ped_id, tracking_info in self.pedestrian_tracking.items():
        if self.step - tracking_info["last_seen"] > 10:  # Not seen for 10 frames
            pedestrians_to_remove.append(ped_id)
    
    for ped_id in pedestrians_to_remove:
        del self.pedestrian_tracking[ped_id]
    
    return env_data

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
    # Finalize TTC plots before destroying
    if hasattr(self, 'safety_evaluator'):
        self.safety_evaluator.finalize_plots()
    
    # Original save path for other data
    if self.save_path is not None:
      self.lon_logger.dump_to_json()
      if len(self.nets[0].speed_histogram) > 0:
        with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
          ujson.dump(self.nets[0].speed_histogram, f, indent=4)

    # Custom save path for environment data
    env_data_path = Path("/home/ascc304/carla_garage/env_data")
    
    # Create the directory if it doesn't exist
    os.makedirs(env_data_path, exist_ok=True)
    
    # Generate a unique filename using timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_data_filename = f"environment_data_{timestamp}"
    
    # Save environment data if available
    if hasattr(self, 'environment_data_buffer') and self.environment_data_buffer:
      # Save as regular JSON for easier access
      json_path = env_data_path / f"{env_data_filename}.json"
      with open(json_path, 'w') as f:
        json.dump(self.environment_data_buffer, f, indent=4)
      
      # Save a simplified version to a text file
      txt_path = env_data_path / f"{env_data_filename}_summary.txt"
      with open(txt_path, 'w') as f:
        f.write(f"Environment Data Summary - {len(self.environment_data_buffer)} timesteps\n")
        f.write("=" * 50 + "\n\n")
        
        # Calculate averages
        vehicle_in_front_count = sum(1 for d in self.environment_data_buffer if d["vehicle_in_front"])
        pedestrian_count = sum(1 for d in self.environment_data_buffer if d["pedestrian_crossing"])
        red_light_count = sum(1 for d in self.environment_data_buffer if d["red_light_ahead"])
        stop_sign_count = sum(1 for d in self.environment_data_buffer if d["stop_sign_ahead"])
        emergency_count = sum(1 for d in self.environment_data_buffer if d["emergency_vehicle_ahead"])
        
        # Get valid distances (non-infinite)
        vehicle_distances = [d["distance_to_closest_front_vehicle"] for d in self.environment_data_buffer 
                            if d["distance_to_closest_front_vehicle"] != float('inf')]
        pedestrian_distances = [d["distance_to_closest_pedestrian"] for d in self.environment_data_buffer 
                               if d["distance_to_closest_pedestrian"] != float('inf')]
        red_light_distances = [d["distance_to_nearest_red_light"] for d in self.environment_data_buffer 
                              if d["distance_to_nearest_red_light"] != float('inf')]
        stop_sign_distances = [d["distance_to_nearest_stop_sign"] for d in self.environment_data_buffer 
                              if d["distance_to_nearest_stop_sign"] != float('inf')]
        emergency_distances = [d["distance_to_nearest_emergency_vehicle"] for d in self.environment_data_buffer 
                              if d["distance_to_nearest_emergency_vehicle"] != float('inf')]
        
        # Write summary statistics
        f.write(f"Vehicle in front detected: {vehicle_in_front_count} times ({vehicle_in_front_count/len(self.environment_data_buffer)*100:.1f}%)\n")
        if vehicle_distances:
            f.write(f"  Average distance: {sum(vehicle_distances)/len(vehicle_distances):.2f} meters\n")
            f.write(f"  Minimum distance: {min(vehicle_distances):.2f} meters\n\n")
        else:
            f.write("  No valid distance measurements\n\n")
            
        f.write(f"Pedestrian detected: {pedestrian_count} times ({pedestrian_count/len(self.environment_data_buffer)*100:.1f}%)\n")
        if pedestrian_distances:
            f.write(f"  Average distance: {sum(pedestrian_distances)/len(pedestrian_distances):.2f} meters\n")
            f.write(f"  Minimum distance: {min(pedestrian_distances):.2f} meters\n\n")
        else:
            f.write("  No valid distance measurements\n\n")
            
        f.write(f"Red light detected: {red_light_count} times ({red_light_count/len(self.environment_data_buffer)*100:.1f}%)\n")
        if red_light_distances:
            f.write(f"  Average distance: {sum(red_light_distances)/len(red_light_distances):.2f} meters\n")
            f.write(f"  Minimum distance: {min(red_light_distances):.2f} meters\n\n")
        else:
            f.write("  No valid distance measurements\n\n")
            
        f.write(f"Stop sign detected: {stop_sign_count} times ({stop_sign_count/len(self.environment_data_buffer)*100:.1f}%)\n")
        if stop_sign_distances:
            f.write(f"  Average distance: {sum(stop_sign_distances)/len(stop_sign_distances):.2f} meters\n")
            f.write(f"  Minimum distance: {min(stop_sign_distances):.2f} meters\n\n")
        else:
            f.write("  No valid distance measurements\n\n")
            
        f.write(f"Emergency vehicle detected: {emergency_count} times ({emergency_count/len(self.environment_data_buffer)*100:.1f}%)\n")
        if emergency_distances:
            f.write(f"  Average distance: {sum(emergency_distances)/len(emergency_distances):.2f} meters\n")
            f.write(f"  Minimum distance: {min(emergency_distances):.2f} meters\n\n")
        else:
            f.write("  No valid distance measurements\n\n")
      
        # In the destroy method, add this to the summary text file
        f.write("\nObjects by Distance Category:\n")
        f.write("-" * 30 + "\n")

        # Calculate average counts for each category
        if self.environment_data_buffer:
            # Initialize counters
            total_counts = {
                "vehicles": {"very_close": 0, "close": 0, "medium": 0, "far": 0},
                "pedestrians": {"very_close": 0, "close": 0, "medium": 0, "far": 0},
                "red_lights": {"very_close": 0, "close": 0, "medium": 0, "far": 0},
                "stop_signs": {"very_close": 0, "close": 0, "medium": 0, "far": 0},
                "emergency_vehicles": {"very_close": 0, "close": 0, "medium": 0, "far": 0}
            }
            
            # Sum up all counts
            for data in self.environment_data_buffer:
                if "vehicles_by_distance" in data:
                    for category in ["very_close", "close", "medium", "far"]:
                        total_counts["vehicles"][category] += data["vehicles_by_distance"][category]
                        total_counts["pedestrians"][category] += data["pedestrians_by_distance"][category]
                        total_counts["red_lights"][category] += data["red_lights_by_distance"][category]
                        total_counts["stop_signs"][category] += data["stop_signs_by_distance"][category]
                        total_counts["emergency_vehicles"][category] += data["emergency_vehicles_by_distance"][category]
            
            # Write the summary
            for obj_type in ["vehicles", "pedestrians", "red_lights", "stop_signs", "emergency_vehicles"]:
                f.write(f"\n{obj_type.replace('_', ' ').title()}:\n")
                for category in ["very_close", "close", "medium", "far"]:
                    count = total_counts[obj_type][category]
                    avg = count / len(self.environment_data_buffer)
                    f.write(f"  {category.replace('_', ' ').title()}: {count} total ({avg:.2f} avg per frame)\n")
      
        f.write("\nSpeed Information:\n")
        f.write("-" * 30 + "\n")

        # Calculate average speeds in km/h
        if any("current_speed_kmh" in d for d in self.environment_data_buffer):
            current_speeds_kmh = [d["current_speed_kmh"] for d in self.environment_data_buffer if "current_speed_kmh" in d]
            f.write(f"Average actual speed: {sum(current_speeds_kmh)/len(current_speeds_kmh):.2f} km/h\n")
            f.write(f"Maximum actual speed: {max(current_speeds_kmh):.2f} km/h\n")

        if any("predicted_target_speed_kmh" in d for d in self.environment_data_buffer):
            predicted_speeds_kmh = [d["predicted_target_speed_kmh"] for d in self.environment_data_buffer 
                                   if "predicted_target_speed_kmh" in d]
            f.write(f"Average predicted target speed: {sum(predicted_speeds_kmh)/len(predicted_speeds_kmh):.2f} km/h\n")
            f.write(f"Maximum predicted target speed: {max(predicted_speeds_kmh):.2f} km/h\n")
      
      print(f"Saved environment data for {len(self.environment_data_buffer)} timesteps")
      print(f"Data saved to: {json_path}")
      print(f"Summary saved to: {txt_path}")
    else:
      print("No environment data collected during this run")

    # Continue with the rest of the destroy method
    if self.save_path is not None:
      if self.config.tp_attention:
        if len(self.tp_attention_buffer) > 0:
          print('Average TP attention: ', sum(self.tp_attention_buffer) / len(self.tp_attention_buffer))
          with gzip.open(self.save_path / 'tp_attention.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(self.tp_attention_buffer, f, indent=4)

        del self.tp_attention_buffer
    
    # Clean up environment data buffer
    if hasattr(self, 'environment_data_buffer'):
      del self.environment_data_buffer

    del self.nets
    del self.config
    del self.metric_info


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
