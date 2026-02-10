"""
Child of the autopilot that additionally runs data collection and storage.
"""

import cv2
import carla
import random
import torch
import numpy as np
import json
import os
import gzip
import laspy
import math
import time
import pygame
from shapely.geometry import Polygon
from pathlib import Path

from autopilot import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from PIL import Image

from agents.tools.misc import (is_within_distance, get_trafficlight_trigger_location, compute_distance)

from agents.navigation.local_planner import LocalPlanner

# Boundary risk estimator (GT-based)
from boundary_risk_estimator import BoundaryRiskEstimator


def get_entry_point():
  return 'DataAgent'


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


class DataAgent(AutoPilot):
  """
        Child of the autopilot that additionally runs data collection and storage.
        """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager=None)
    self.weather_tmp = None
    self.step_tmp = 0

    # Disable waypoint/debug visualization for DataAgent
    self.visualize = 0
    # Disable data collection
    self.datagen = False
    self.save_path = None

    # Manual control / visualization configuration
    self.camera_width = 1920
    self.camera_height = 960

    self.enable_manual_control = strtobool(os.environ.get('ENABLE_MANUAL_CONTROL', 'False'))
    self.record_frequency = int(os.environ.get('MANUAL_RECORD_FREQUENCY', 10))
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
      pygame.display.set_caption("Data Agent View")
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

    self.tm = traffic_manager

    self.scenario_name = Path(path_to_conf_file).parent.name
    self.cutin_vehicle_starting_position = None

    if self.save_path is not None and self.datagen:
      (self.save_path / 'lidar').mkdir()
      (self.save_path / 'rgb').mkdir()
      (self.save_path / 'semantics').mkdir()
      (self.save_path / 'semantics_augmented').mkdir()
      (self.save_path / 'depth').mkdir()
      (self.save_path / 'depth_augmented').mkdir()
      (self.save_path / 'rgb_augmented').mkdir()
      (self.save_path / 'bev_semantics').mkdir()
      (self.save_path / 'bev_semantics_augmented').mkdir()
      (self.save_path / 'boxes').mkdir()

    self.tmp_visu = int(os.environ.get('TMP_VISU', 0))

    self._active_traffic_light = None
    self.last_lidar = None
    self.last_ego_transform = None
    self._last_tick_timestamp = None

    # Boundary risk estimation (GT-based from CARLA actors)
    self.use_boundary_risk = strtobool(os.environ.get('USE_BOUNDARY_RISK', 'True'))
    print('Use boundary risk estimation (DataAgent GT):', self.use_boundary_risk)

    self.boundary_risk_estimator = None
    self.boundary_risk_data = None
    self.last_boundary_risk_field = None
    self.last_boundary_risk_info = None
    self.last_ego_speed = 0.0
    self.collision_distance_threshold = float(os.environ.get('BOUNDARY_COLLISION_DISTANCE', '0.5'))
    self.near_miss_distance_threshold = float(os.environ.get('BOUNDARY_NEAR_MISS_DISTANCE', '3.0'))

    # Visualization/debug overlays
    self.draw_boundary_debug = strtobool(os.environ.get('DRAW_BOUNDARY_RISK_DEBUG', 'True'))
    self.enable_relative_pos_debug = strtobool(os.environ.get('DRAW_RELATIVE_POS_DEBUG', 'False'))
    self.debug_tick_rate = strtobool(os.environ.get('DEBUG_TICK_RATE', 'False'))
    self.debug_tick_rate_freq = int(os.environ.get('DEBUG_TICK_RATE_FREQ', 30))
    self.collision_warn_log_path = os.path.join(os.getcwd(), "collision_warning_debug.txt")

    if self.use_boundary_risk:
      angular_resolution = int(os.environ.get('BOUNDARY_ANGULAR_RESOLUTION', 10))
      max_range = float(os.environ.get('BOUNDARY_MAX_RANGE', 50.0))
      k_distance = float(os.environ.get('BOUNDARY_K_DISTANCE', -1.0))
      alpha_coeff = float(os.environ.get('BOUNDARY_ALPHA', 1.0))
      beta_coeff = float(os.environ.get('BOUNDARY_BETA', 0.5))
      risk_threshold = float(os.environ.get('BOUNDARY_RISK_THRESHOLD', 0.3))
      lateral_threshold = float(os.environ.get('BOUNDARY_LATERAL_THRESHOLD', 2.5))

      self.boundary_risk_estimator = BoundaryRiskEstimator(
          angular_resolution=angular_resolution,
          max_range=max_range,
          k_distance=k_distance,
          alpha_coeff=alpha_coeff,
          beta_coeff=beta_coeff,
          risk_threshold=risk_threshold,
          lateral_risk_threshold=lateral_threshold
      )
      boundary_debug = strtobool(os.environ.get('BOUNDARY_DEBUG', 'False'))
      self.boundary_risk_estimator.set_debug(boundary_debug)

      print('Boundary Risk Estimator initialized (DataAgent):')
      print(f'  Angular resolution: {angular_resolution}°')
      print(f'  Max range: {max_range}m')
      print(f'  Risk threshold: {risk_threshold}')
      print(f'  Debug mode: {boundary_debug}')



  def _init(self, hd_map):
    super()._init(hd_map)
    if self.datagen:
      self.shuffle_weather()

    obs_config = {
        'width_in_pixels': self.config.lidar_resolution_width,
        'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
        'pixels_per_meter': self.config.pixels_per_meter_collection,
        'history_idx': [-1],
        'scale_bbox': True,
        'scale_mask_col': 1.0,
        'map_folder': 'maps_2ppm_cv'
    }

    self.stop_sign_criteria = RunStopSign(self._world)
    self.ss_bev_manager = ObsManager(obs_config, self.config)
    self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

    self.ss_bev_manager_augmented = ObsManager(obs_config, self.config)

    bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
    transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
    # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
    self.augmented_vehicle_dummy = t_u.CarlaActorDummy(self._vehicle.get_world(), bb_copy, transform_copy,
                                                       self._vehicle.id)
    self.ss_bev_manager_augmented.attach_ego_vehicle(self.augmented_vehicle_dummy,
                                                     criteria_stop=self.stop_sign_criteria)

    self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)

  def sensors(self):
    # workaraound that only does data augmentation at the beginning of the route
    if self.config.augment:
      self.augmentation_translation = np.random.uniform(low=self.config.camera_translation_augmentation_min,
                                                        high=self.config.camera_translation_augmentation_max)
      self.augmentation_rotation = np.random.uniform(low=self.config.camera_rotation_augmentation_min,
                                                     high=self.config.camera_rotation_augmentation_max)

    result = super().sensors()

    if self.enable_manual_control:
      result += [
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

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      result += [{
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
          'id': 'rgb'
      }, {
          'type': 'sensor.camera.rgb',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1] + self.augmentation_translation,
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'rgb_augmented'
      }, {
          'type': 'sensor.camera.semantic_segmentation',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'semantics'
      }, {
          'type': 'sensor.camera.semantic_segmentation',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1] + self.augmentation_translation,
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'semantics_augmented'
      }, {
          'type': 'sensor.camera.depth',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'depth'
      }, {
          'type': 'sensor.camera.depth',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1] + self.augmentation_translation,
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'depth_augmented'
      }]

    result.append({
        'type': 'sensor.lidar.ray_cast',
        'x': self.config.lidar_pos[0],
        'y': self.config.lidar_pos[1],
        'z': self.config.lidar_pos[2],
        'roll': self.config.lidar_rot[0],
        'pitch': self.config.lidar_rot[1],
        'yaw': self.config.lidar_rot[2],
        'rotation_frequency': self.config.lidar_rotation_frequency,
        'points_per_second': self.config.lidar_points_per_second,
        'id': 'lidar'
    })

    return result

  def tick(self, input_data):
    result = {}

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      rgb = input_data['rgb'][1][:, :, :3]
      rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]

      # We store depth at 8 bit to reduce the filesize. 16 bit would be ideal, but we can't afford the extra storage.
      depth = input_data['depth'][1][:, :, :3]
      depth = (t_u.convert_depth(depth) * 255.0 + 0.5).astype(np.uint8)

      depth_augmented = input_data['depth_augmented'][1][:, :, :3]
      depth_augmented = (t_u.convert_depth(depth_augmented) * 255.0 + 0.5).astype(np.uint8)

      semantics = input_data['semantics'][1][:, :, 2]
      semantics_augmented = input_data['semantics_augmented'][1][:, :, 2]

    else:
      rgb = None
      rgb_augmented = None
      semantics = None
      semantics_augmented = None
      depth = None
      depth_augmented = None

    # The 10 Hz LiDAR only delivers half a sweep each time step at 20 Hz.
    # Here we combine the 2 sweeps into the same coordinate system
    if self.last_lidar is not None:
      ego_transform = self._vehicle.get_transform()
      ego_location = ego_transform.location
      last_ego_location = self.last_ego_transform.location
      relative_translation = np.array([
          ego_location.x - last_ego_location.x, ego_location.y - last_ego_location.y,
          ego_location.z - last_ego_location.z
      ])

      ego_yaw = ego_transform.rotation.yaw
      last_ego_yaw = self.last_ego_transform.rotation.yaw
      relative_rotation = np.deg2rad(t_u.normalize_angle_degree(ego_yaw - last_ego_yaw))

      orientation_target = np.deg2rad(ego_yaw)
      # Rotate difference vector from global to local coordinate system.
      rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                  [np.sin(orientation_target),
                                   np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
      relative_translation = rotation_matrix.T @ relative_translation

      lidar_last = t_u.algin_lidar(self.last_lidar, relative_translation, relative_rotation)
      # Combine back and front half of LiDAR
      lidar_360 = np.concatenate((input_data['lidar'], lidar_last), axis=0)
    else:
      lidar_360 = input_data['lidar']  # The first frame only has 1 half

    bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)

    self.stop_sign_criteria.tick(self._vehicle)
    bev_semantics = self.ss_bev_manager.get_observation(self.close_traffic_lights)
    bev_semantics_augmented = self.ss_bev_manager_augmented.get_observation(self.close_traffic_lights)

    if self.tmp_visu:
      self.visualuize(bev_semantics['rendered'], rgb)

    result.update({
        'lidar': lidar_360,
        'rgb': rgb,
        'rgb_augmented': rgb_augmented,
        'semantics': semantics,
        'semantics_augmented': semantics_augmented,
        'depth': depth,
        'depth_augmented': depth_augmented,
        'bev_semantics': bev_semantics['bev_semantic_classes'],
        'bev_semantics_augmented': bev_semantics_augmented['bev_semantic_classes'],
        'bounding_boxes': bounding_boxes,
    })

    return result

  def _map_gt_class(self, box):
    label = box.get('class')
    if label == 'car':
      type_id = str(box.get('type_id', '')).lower()
      role_name = str(box.get('role_name', '')).lower()
      if any(tag in type_id for tag in ('ambulance', 'police', 'firetruck')) or 'emergency' in role_name:
        return 4
      return 0
    if label == 'walker':
      return 1
    if label == 'traffic_light':
      return 2
    if label == 'stop_sign':
      return 3
    return None

  def _gt_boxes_to_boundary_format(self, boxes):
    formatted = []
    for box in boxes:
      if box.get('class') == 'ego_car':
        continue
      obj_class = self._map_gt_class(box)
      if obj_class is None:
        continue
      pos = box.get('position', [0.0, 0.0, 0.0])
      extent = box.get('extent', [0.0, 0.0, 0.0])
      yaw = float(box.get('yaw', 0.0))
      speed = float(box.get('speed', 0.0))
      brake = float(box.get('brake', 0.0)) if box.get('brake') is not None else 0.0

      formatted.append([
          float(pos[0]),
          float(pos[1]),
          float(extent[0]),
          float(extent[1]),
          yaw,
          speed,
          brake,
          int(obj_class),
          int(box.get('id')) if box.get('id') is not None else None
      ])
    return formatted

  def calculate_boundary_risk_assessment(self, ego_speed, bounding_boxes, timestamp):
    if not self.use_boundary_risk or self.boundary_risk_estimator is None:
      return {
          'max_risk': 0.0,
          'risk_level': 'SAFE',
          'primary_threat': None,
          'boundary_based': False,
          'fallback_mode': True
      }

    try:
      # Disable frame-to-frame boundary velocity; use GT speed instead
      self.boundary_risk_estimator.last_polar_boundary = None

      risk_field, _max_risk, threat_info = self.boundary_risk_estimator.calculate_boundary_risk(
          ego_speed, bounding_boxes, timestamp
      )
      threat_info['risk_field'] = risk_field
      threat_info['boundary_based'] = True
      threat_info['fallback_mode'] = False
      self.last_boundary_risk_field = risk_field
      self.last_boundary_risk_info = threat_info
      return threat_info
    except Exception as e:
      print(f"⚠️ Boundary risk calculation failed (DataAgent): {e}")
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


  # def draw_boundary_risk_debug(self):
  #   if (not self.draw_boundary_debug or self._world is None or self._vehicle is None or
  #       self.last_boundary_risk_field is None or len(self.last_boundary_risk_field) == 0):
  #     return

  #   try:
  #     transform = self._vehicle.get_transform()
  #     base_location = transform.location + carla.Location(z=0.5)
  #     max_range = getattr(self.boundary_risk_estimator, 'max_range', 30.0)
  #     angles = getattr(self.boundary_risk_estimator, 'angles', None)
  #     if angles is None:
  #       segments = len(self.last_boundary_risk_field)
  #       angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)

  #     max_risk = max(1e-3, float(np.max(self.last_boundary_risk_field)))

  #     polar_boundary = getattr(self.boundary_risk_estimator, 'last_polar_boundary', None)
  #     has_class_info = polar_boundary is not None and len(polar_boundary) == len(self.last_boundary_risk_field)

  #     for idx, (angle_local, risk) in enumerate(zip(angles, self.last_boundary_risk_field)):
  #       if risk <= 0.0:
  #         continue
  #       normalized = max(0.0, min(1.0, risk / max_risk))
  #       risk_color = self._risk_to_color(normalized)

  #       color_tuple = risk_color
  #       if has_class_info:
  #         obj_class = polar_boundary[idx].get('object_class')
  #         if obj_class == 4:  # emergency vehicle -> blue
  #           base_tuple = (60, 140, 255)
  #         elif obj_class == 1:  # pedestrian -> purple
  #           base_tuple = (180, 70, 220)
  #         else:
  #           base_tuple = None

  #         if base_tuple:
  #           brightness = 0.35 + 0.65 * normalized
  #           color_tuple = (
  #               int(base_tuple[0] * brightness),
  #               int(base_tuple[1] * brightness),
  #               int(base_tuple[2] * brightness),
  #           )

  #       color = carla.Color(r=color_tuple[0], g=color_tuple[1], b=color_tuple[2])

  #       min_len = 3.0
  #       max_len = min(max_range, 15.0)
  #       length = min_len + normalized * (max_len - min_len)

  #       local_x = length * math.cos(angle_local)
  #       local_y = length * math.sin(angle_local)
  #       end_world = transform.transform(carla.Location(x=local_x, y=local_y, z=0.0))

  #       self._world.debug.draw_line(
  #           base_location,
  #           end_world + carla.Location(z=0.5),
  #           thickness=0.08,
  #           color=color,
  #           life_time=0.1)
  #   except Exception as exc:  # pylint: disable=broad-except
  #     print(f"⚠️ Failed to draw boundary risk debug lines (DataAgent): {exc}")

  @staticmethod
  def _risk_to_color(value: float):
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
  def draw_boundary_risk_debug(self):
    if (not self.draw_boundary_debug or self._world is None or self._vehicle is None or
        self.last_boundary_risk_field is None or len(self.last_boundary_risk_field) == 0):
      return

    try:
      transform = self._vehicle.get_transform()
      base_location = transform.location + carla.Location(z=1.0) # 稍微调高一点，避免穿模
      
      # 1. 改变归一化逻辑：使用固定阈值，而不是全局最大值
      # 建议使用 self.boundary_risk_estimator.risk_threshold 或固定值 (如 1.0)
      # 这样风险 0.1 就是绿色，1.0 以上才是深红，视觉反馈更客观
      reference_risk = 1.0 

      polar_boundary = getattr(self.boundary_risk_estimator, 'last_polar_boundary', None)
      if polar_boundary is None: return

      # 为了性能，如果分辨率很高(如360)，可以每隔2-3度画一根线
      step = 1 if len(self.last_boundary_risk_field) <= 72 else 3

      for idx in range(0, len(self.last_boundary_risk_field), step):
        risk = float(self.last_boundary_risk_field[idx])
        if risk <= 0.01: # 忽略极小风险，保持画面整洁
          continue
          
        angle_local = self.boundary_risk_estimator.angles[idx]
        
        # 统一归一化：超过 reference_risk 的都算 1.0
        normalized = max(0.0, min(1.0, risk / reference_risk))
        
        # 2. 颜色逻辑增强
        obj_class = polar_boundary[idx].get('object_class', 0)
        
        # 默认使用风险色
        risk_color = self._risk_to_color(normalized)
        
        # 如果是行人，强制使用醒目的紫色调
        if obj_class == 1:
            color_tuple = (255, 0, 255) # 纯霓虹紫，确保实验者一眼看到行人
        elif obj_class == 4:
            color_tuple = (0, 191, 255) # 深天蓝，用于警车
        else:
            color_tuple = risk_color

        color = carla.Color(r=color_tuple[0], g=color_tuple[1], b=color_tuple[2])

        # 3. 长度逻辑：不要让线太短，行人即使风险小，也要画得够长，才能起到预警作用
        min_len = 2.0
        if obj_class == 1: min_len = 5.0 # 行人的线保底长一些
        
        length = min_len + normalized * 10.0 # 长度上限设为 12-15米 即可

        local_x = length * math.cos(angle_local)
        local_y = length * math.sin(angle_local)
        
        # 坐标转换
        end_world = transform.transform(carla.Location(x=local_x, y=local_y, z=0.0))

        # 绘制
        self._world.debug.draw_line(
            base_location,
            end_world + carla.Location(z=1.0),
            thickness=0.1, # 稍微加粗
            color=color,
            life_time=0.1)
            
    except Exception as exc:
      print(f"⚠️ Debug Drawing Error: {exc}")

  def draw_relative_pos_debug(self):
    if self._world is None or self._vehicle is None:
      return
    if getattr(self, "_relpos_last_draw_step", None) == self.step:
      return
    try:
      self._relpos_last_draw_step = self.step
      ego_transform = self._vehicle.get_transform()
      ego_matrix = np.array(ego_transform.get_matrix())
      vehicles = self._world.get_actors().filter('*vehicle*')
      for vehicle in vehicles:
        if vehicle.id == self._vehicle.id:
          continue
        vehicle_transform = vehicle.get_transform()
        vehicle_matrix = np.array(vehicle_transform.get_matrix())
        relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
        text = f"rel x={relative_pos[0]:.1f}, y={relative_pos[1]:.1f}"
        location = vehicle_transform.location + carla.Location(z=2.2)
        self._world.debug.draw_string(
            location,
            text,
            draw_shadow=False,
            color=carla.Color(255, 255, 0),
            life_time=0.05,
            persistent_lines=False)
    except Exception as exc:  # pylint: disable=broad-except
      print(f"⚠️ Failed to draw relative positions: {exc}")
  # def _render_risk_radar(self, size=250):
  #   """Render a polar risk radar HUD surface."""
  #   if self.last_boundary_risk_field is None or len(self.last_boundary_risk_field) == 0:
  #     return None
  #   if self.boundary_risk_estimator is None:
  #     return None

  #   surface = pygame.Surface((size, size), pygame.SRCALPHA)
  #   surface = surface.convert_alpha()
  #   center = (size // 2, size // 2)
  #   radius = size // 2 - 8

  #   # Background
  #   surface.fill((0, 0, 0, 0))
  #   pygame.draw.circle(surface, (40, 40, 40, 160), center, radius)
  #   pygame.draw.circle(surface, (120, 120, 120, 160), center, radius, 1)

  #   # Crosshairs
  #   pygame.draw.line(surface, (90, 90, 90, 140), (center[0], center[1] - radius),
  #                    (center[0], center[1] + radius), 1)
  #   pygame.draw.line(surface, (90, 90, 90, 140), (center[0] - radius, center[1]),
  #                    (center[0] + radius, center[1]), 1)

  #   risk_field = self.last_boundary_risk_field
  #   angles = getattr(self.boundary_risk_estimator, 'angles', None)
  #   if angles is None:
  #     angles = np.linspace(0.0, 2.0 * math.pi, len(risk_field), endpoint=False)

  #   max_display_risk = 1.0
  #   for value in risk_field:
  #     max_display_risk = max(max_display_risk, float(value))

  #   for angle, risk in zip(angles, risk_field):
  #     if risk <= 0.0:
  #       continue
  #     normalized = max(0.0, min(1.0, float(risk) / max_display_risk))
  #     color = self._risk_to_color(normalized)
  #     display_angle = angle - math.pi / 2.0  # 0° forward -> up
  #     length = int(normalized * radius)
  #     end_pos = (
  #         int(center[0] + length * math.cos(display_angle)),
  #         int(center[1] + length * math.sin(display_angle)),
  #     )
  #     pygame.draw.line(surface, color, center, end_pos, 3)

  #   # Ego triangle (pointing up)
  #   ego_size = 8
  #   ego_points = [
  #       (center[0], center[1] - ego_size),
  #       (center[0] - ego_size // 2, center[1] + ego_size // 2),
  #       (center[0] + ego_size // 2, center[1] + ego_size // 2),
  #   ]
  #   pygame.draw.polygon(surface, (240, 240, 240, 220), ego_points)

  #   return surface
  def _render_risk_radar(self, size=250):
    """
    渲染极坐标风险雷达 HUD。
    优化点：固定比例尺、行人视觉增强、坐标对齐。
    """
    if self.last_boundary_risk_field is None or len(self.last_boundary_risk_field) == 0:
      return None
    if self.boundary_risk_estimator is None:
      return None

    # 1. 初始化 Surface (支持透明度)
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface = surface.convert_alpha()
    center = (size // 2, size // 2)
    radius = size // 2 - 10

    # 2. 绘制背景圆盘与装饰线 (深色半透明，提升对比度)
    surface.fill((0, 0, 0, 0))
    pygame.draw.circle(surface, (20, 20, 20, 180), center, radius) 
    pygame.draw.circle(surface, (180, 180, 180, 255), center, radius, 2) # 外边框

    # 绘制十字参考线 (前方、后方、左右)
    line_color = (100, 100, 100, 150)
    pygame.draw.line(surface, line_color, (center[0], center[1] - radius), (center[0], center[1] + radius), 1)
    pygame.draw.line(surface, line_color, (center[0] - radius, center[1]), (center[0] + radius, center[1]), 1)

    # 3. 获取风险数据与类别信息
    risk_field = self.last_boundary_risk_field
    angles = getattr(self.boundary_risk_estimator, 'angles', None)
    polar_boundary = getattr(self.boundary_risk_estimator, 'last_polar_boundary', None)
    
    # 【核心优化】固定参考量纲：
    # 不再使用 max(risk_field)，因为那会导致远处的弱威胁被强行放大。
    # 我们设定 1.0 为“满额风险”长度。
    reference_risk = 1.0 

    for i, (angle, risk) in enumerate(zip(angles, risk_field)):
      if risk <= 0.01: # 忽略背景杂讯
        continue

      # 归一化 (0.0 到 1.0)
      normalized = max(0.0, min(1.0, float(risk) / reference_risk))
      
      # 获取物体类别
      obj_class = 0
      if polar_boundary is not None and i < len(polar_boundary):
        obj_class = polar_boundary[i].get('object_class', 0)

      # 4. 颜色分配逻辑
      if obj_class == 1: # 行人：绝对醒目的紫色
        color = (255, 0, 255, 255)
      elif obj_class == 4: # 紧急车辆：蓝色
        color = (0, 191, 255, 255)
      else:
        # 使用类名调用静态方法，修复之前的属性错误
        color_rgb = DataAgent._risk_to_color(normalized)
        color = (color_rgb[0], color_rgb[1], color_rgb[2], 255)

      # 5. 坐标转换 (CARLA 0°是右 -> Pygame 0°是上)
      display_angle = angle - math.pi / 2.0 
      
      # 6. 长度与线宽优化
      # 行人保底长度为半径的 35%，确保即使风险低也能被看见
      min_len_ratio = 0.35 if obj_class == 1 else 0.1
      length = int(max(min_len_ratio, normalized) * radius)
      
      # 行人线条加粗到 5 像素，普通车辆为 3 像素
      line_width = 5 if obj_class == 1 else 3
      
      end_pos = (
          int(center[0] + length * math.cos(display_angle)),
          int(center[1] + length * math.sin(display_angle)),
      )
      
      pygame.draw.line(surface, color, center, end_pos, line_width)

    # 7. 绘制自车图标 (白色小三角形)
    ego_size = 10
    ego_points = [
        (center[0], center[1] - ego_size), # 前端
        (center[0] - ego_size//1.5, center[1] + ego_size//1.5), # 左后
        (center[0] + ego_size//1.5, center[1] + ego_size//1.5)  # 右后
    ]
    pygame.draw.polygon(surface, (255, 255, 255, 255), ego_points)

    # 8. 显示自车速度（m/s）
    speed_value = float(self.last_ego_speed) if self.last_ego_speed is not None else 0.0
    speed_text = f"Ego: {speed_value:.1f} m/s"
    if pygame.font.get_init():
      font = self._font_mono or pygame.font.SysFont('monospace', 18)
      text_surf = font.render(speed_text, True, (255, 255, 255))
      text_x = (size - text_surf.get_width()) // 2
      text_y = 6
      surface.blit(text_surf, (text_x, text_y))

    return surface

  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    self.step_tmp += 1
    current_time = time.time()
    if self.debug_tick_rate:
      if self._last_tick_timestamp is not None:
        delta_t = float(timestamp - self._last_tick_timestamp)
        if delta_t > 0 and (self.step_tmp % max(1, self.debug_tick_rate_freq) == 0):
          tick_rate = 1.0 / delta_t
          print(f"[TickRate] game_dt={delta_t:.4f}s rate={tick_rate:.2f} Hz step={self.step_tmp}")
      self._last_tick_timestamp = float(timestamp)

    mirror_width = None
    mirror_height = None
    if self.enable_manual_control and self._display is not None:
      mirror_width = int(self._display.get_width() * 0.2)
      mirror_height = int(self._display.get_height() * 0.2)

    # Convert LiDAR into the coordinate frame of the ego vehicle
    input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

    # Must be called before run_step, so that the correct augmentation shift is saved
    if self.datagen:
      self.augment_camera(sensors)

    control = super().run_step(input_data, timestamp, plant=plant)

    tick_data = self.tick(input_data)

    manual_override = False
    manual_control_cmd = None

    if self.enable_manual_control:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return
        elif event.type == pygame.JOYBUTTONDOWN and self._joystick is not None:
          if event.button == self._square_idx:
            if current_time - self.last_switch_time > 0.2:
              self.manual_control = not self.manual_control
              self.last_switch_time = current_time

      # Optional visualization overlay for manual supervision
      if self._display is not None and 'Center' in input_data:
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

        # Risk radar HUD
        radar_surface = self._render_risk_radar(size=240)
        if radar_surface is not None:
          margin = 20
          radar_x = self._display.get_width() - radar_surface.get_width() - margin
          radar_y = self._display.get_height() - radar_surface.get_height() - margin
          self._display.blit(radar_surface, (radar_x, radar_y))

          max_risk = float(np.max(self.last_boundary_risk_field)) if self.last_boundary_risk_field is not None else 0.0
          if max_risk >= 1.0 and self._font_mono is not None:
            warn_text = self._font_mono.render("WARNING: COLLISION RISK", True, (255, 80, 80))
            self._display.blit(warn_text, (radar_x - warn_text.get_width() - 10, radar_y + 10))
            # Log class-0 vehicle data only when: no pedestrian, no emergency, ego speed < 1 m/s
            try:
              boundary_info = self.last_boundary_risk_info or {}
              risk_field = boundary_info.get('risk_field')
              polar_boundary = getattr(self.boundary_risk_estimator, 'last_polar_boundary', None)

              has_ped_or_emergency = False
              if polar_boundary is not None:
                for boundary_point in polar_boundary:
                  if boundary_point.get('object_index') is None:
                    continue
                  obj_class = boundary_point.get('object_class')
                  if obj_class in (1, 4):
                    has_ped_or_emergency = True
                    break

              if (not has_ped_or_emergency) and float(self.last_ego_speed) < 1.0:
                vehicle_entries = []
                if risk_field is not None and polar_boundary is not None:
                  for risk_val, boundary_point in zip(risk_field, polar_boundary):
                    if risk_val is None or float(risk_val) <= 1.0:
                      continue
                    if boundary_point.get('object_class') != 0:
                      continue
                    vehicle_entries.append({
                        "radial_speed": boundary_point.get("radial_velocity_signed"),
                        "radial_speed_unsigned": boundary_point.get("radial_velocity"),
                        "risk": float(risk_val) if risk_val is not None else None,
                        "distance": boundary_point.get("distance"),
                        "lateral_offset": boundary_point.get("lateral_offset"),
                        "object_id": boundary_point.get("object_id"),
                        "object_x": boundary_point.get("object_x"),
                        "object_y": boundary_point.get("object_y"),
                    })

                log_entry = {
                    "timestamp": timestamp,
                    "vehicles": vehicle_entries,
                }
                with open(self.collision_warn_log_path, "a", encoding="utf-8") as f:
                  f.write(json.dumps(log_entry) + "\n")
            except Exception as exc:
              print(f"⚠️ Failed to log collision warning data: {exc}")

        pygame.display.flip()

      # Handle manual control via joystick if enabled
      if self._joystick is not None:
        num_axes = self._joystick.get_numaxes()
        js_inputs = [float(self._joystick.get_axis(i)) for i in range(num_axes)]

        brake_cmd = 1.6 + (2.05 * math.log10(-0.7 * js_inputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brake_cmd <= 0:
          brake_cmd = 0
        elif brake_cmd > 0:
          brake_cmd = 1

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

          manual_override = True
          manual_control_cmd = human_control

    # =================================================================
    # Boundary risk assessment using GT bounding boxes
    # =================================================================
    if self.use_boundary_risk and self.boundary_risk_estimator is not None:
      ego_speed = self._get_forward_speed(transform=self._vehicle.get_transform(),
                                          velocity=self._vehicle.get_velocity())
      self.last_ego_speed = ego_speed
      gt_boxes = tick_data.get('bounding_boxes', [])
      bbs_vehicle_coordinate_system = self._gt_boxes_to_boundary_format(gt_boxes)

      if bbs_vehicle_coordinate_system:
        boundary_risk_info = self.calculate_boundary_risk_assessment(
            ego_speed, bbs_vehicle_coordinate_system, timestamp
        )
        if self.draw_boundary_debug:
          self.draw_boundary_risk_debug()
        if self.enable_relative_pos_debug:
          self.draw_relative_pos_debug()

    if self.step % self.config.data_save_freq == 0:
      if self.save_path is not None and self.datagen:
        self.save_sensors(tick_data)

    self.last_lidar = input_data['lidar']
    self.last_ego_transform = self._vehicle.get_transform()

    if plant:
      # Control contains data when run with plant
      return {**tick_data, **control}
    else:
      if manual_override:
        return manual_control_cmd
      return control

  def augment_camera(self, sensors):
    # Update dummy vehicle
    if self.initialized:
      # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
      last_translation = self.augmentation_translation
      last_rotation = self.augmentation_rotation
      bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
      transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
      augmented_loc = transform_copy.transform(carla.Location(0.0, last_translation, 0.0))
      transform_copy.location = augmented_loc
      transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
      self.augmented_vehicle_dummy.bounding_box = bb_copy
      self.augmented_vehicle_dummy.transform = transform_copy

  def _get_night_mode(self, weather):
    """Check wheather or not the street lights need to be turned on"""
    SUN_ALTITUDE_THRESHOLD_1 = 15
    SUN_ALTITUDE_THRESHOLD_2 = 165

    # For higher fog and cloudness values, the amount of light in scene starts to rapidly decrease
    CLOUDINESS_THRESHOLD = 80
    FOG_THRESHOLD = 40

    # In cases where more than one weather conditition is active, decrease the thresholds
    COMBINED_THRESHOLD = 10

    altitude_dist = weather.sun_altitude_angle - SUN_ALTITUDE_THRESHOLD_1
    altitude_dist = min(altitude_dist, SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle)
    cloudiness_dist = CLOUDINESS_THRESHOLD - weather.cloudiness
    fog_density_dist = FOG_THRESHOLD - weather.fog_density

    # Check each parameter independetly
    if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
      return True

    # Check if two or more values are close to their threshold
    joined_threshold = int(altitude_dist < COMBINED_THRESHOLD)
    joined_threshold += int(cloudiness_dist < COMBINED_THRESHOLD)
    joined_threshold += int(fog_density_dist < COMBINED_THRESHOLD)

    if joined_threshold >= 2:
      return True

    return False

  def shuffle_weather(self):
    # change weather for visual diversity
    if self.weather_tmp is None:
      t = carla.WeatherParameters
      options = dir(t)[:22]
      chosen_preset = random.choice(options)
      self.chosen_preset = chosen_preset
      weather = t.__getattribute__(t, chosen_preset)
      self.weather_tmp = weather

    self._world.set_weather(self.weather_tmp)

    # night mode
    vehicles = self._world.get_actors().filter('*vehicle*')
    if self._get_night_mode(weather):
      for vehicle in vehicles:
        vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
    else:
      for vehicle in vehicles:
        vehicle.set_light_state(carla.VehicleLightState.NONE)

  def save_sensors(self, tick_data):
    frame = self.step // self.config.data_save_freq

    # CARLA images are already in opencv's BGR format.
    cv2.imwrite(str(self.save_path / 'rgb' / (f'{frame:04}.jpg')), tick_data['rgb'])
    cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.jpg')), tick_data['rgb_augmented'])

    cv2.imwrite(str(self.save_path / 'semantics' / (f'{frame:04}.png')), tick_data['semantics'])
    cv2.imwrite(str(self.save_path / 'semantics_augmented' / (f'{frame:04}.png')), tick_data['semantics_augmented'])

    cv2.imwrite(str(self.save_path / 'depth' / (f'{frame:04}.png')), tick_data['depth'])
    cv2.imwrite(str(self.save_path / 'depth_augmented' / (f'{frame:04}.png')), tick_data['depth_augmented'])

    cv2.imwrite(str(self.save_path / 'bev_semantics' / (f'{frame:04}.png')), tick_data['bev_semantics'])
    cv2.imwrite(str(self.save_path / 'bev_semantics_augmented' / (f'{frame:04}.png')),
                tick_data['bev_semantics_augmented'])

    # Specialized LiDAR compression format
    header = laspy.LasHeader(point_format=self.config.point_format)
    header.offsets = np.min(tick_data['lidar'], axis=0)
    header.scales = np.array([self.config.point_precision, self.config.point_precision, self.config.point_precision])

    with laspy.open(self.save_path / 'lidar' / (f'{frame:04}.laz'), mode='w', header=header) as writer:
      point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
      point_record.x = tick_data['lidar'][:, 0]
      point_record.y = tick_data['lidar'][:, 1]
      point_record.z = tick_data['lidar'][:, 2]

      writer.write_points(point_record)

    with gzip.open(self.save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
      json.dump(tick_data['bounding_boxes'], f, indent=4)

  def destroy(self, results=None):
    torch.cuda.empty_cache()


    if results is not None and self.save_path is not None:
      with gzip.open(os.path.join(self.save_path, 'results.json.gz'), 'wt', encoding='utf-8') as f:
        json.dump(results.__dict__, f, indent=2)

    super().destroy(results)

  def get_bounding_boxes(self, lidar=None):
    results = []

    ego_transform = self._vehicle.get_transform()
    ego_control = self._vehicle.get_control()
    ego_velocity = self._vehicle.get_velocity()
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_rotation = ego_transform.rotation
    ego_extent = self._vehicle.bounding_box.extent
    ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
    ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
    ego_yaw = np.deg2rad(ego_rotation.yaw)
    ego_brake = ego_control.brake

    relative_yaw = 0.0
    relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

    # Check for possible vehicle obstacles
    # Retrieve all relevant actors
    self._actors = self._world.get_actors()
    vehicle_list = self._actors.filter('*vehicle*')

    result = {
        'class': 'ego_car',
        'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
        'yaw': relative_yaw,
        'num_points': -1,
        'distance': -1,
        'speed': ego_speed,
        'brake': ego_brake,
        'id': int(self._vehicle.id),
        'matrix': ego_transform.get_matrix()
    }
    results.append(result)

    for vehicle in vehicle_list:
      if vehicle.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        if vehicle.id != self._vehicle.id:
          vehicle_transform = vehicle.get_transform()
          vehicle_rotation = vehicle_transform.rotation
          vehicle_matrix = np.array(vehicle_transform.get_matrix())
          vehicle_control = vehicle.get_control()
          vehicle_velocity = vehicle.get_velocity()
          vehicle_extent = vehicle.bounding_box.extent
          vehicle_id = vehicle.id

          vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
          yaw = np.deg2rad(vehicle_rotation.yaw)

          relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
          relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
          vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
          vehicle_brake = vehicle_control.brake
          vehicle_steer = vehicle_control.steer
          vehicle_throttle = vehicle_control.throttle

          # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
          if not lidar is None:
            num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list, lidar)
          else:
            num_in_bbox_points = -1

          distance = np.linalg.norm(relative_pos)

          result = {
              'class': 'car',
              'extent': vehicle_extent_list,
              'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
              'yaw': relative_yaw,
              'num_points': int(num_in_bbox_points),
              'distance': distance,
              'speed': vehicle_speed,
              'brake': vehicle_brake,
              'steer': vehicle_steer,
              'throttle': vehicle_throttle,
              'id': int(vehicle_id),
              'role_name': vehicle.attributes['role_name'],
              'type_id': vehicle.type_id,
              'matrix': vehicle_transform.get_matrix()
          }
          results.append(result)

    walkers = self._actors.filter('*walker*')
    for walker in walkers:
      if walker.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        walker_transform = walker.get_transform()
        walker_velocity = walker.get_velocity()
        walker_rotation = walker.get_transform().rotation
        walker_matrix = np.array(walker_transform.get_matrix())
        walker_id = walker.id
        walker_extent = walker.bounding_box.extent
        walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
        yaw = np.deg2rad(walker_rotation.yaw)

        relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
        relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

        walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
        if not lidar is None:
          num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, walker_extent, lidar)
        else:
          num_in_bbox_points = -1

        distance = np.linalg.norm(relative_pos)

        result = {
            'class': 'walker',
            'extent': walker_extent,
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': int(num_in_bbox_points),
            'distance': distance,
            'speed': walker_speed,
            'id': int(walker_id),
            'matrix': walker_transform.get_matrix()
        }
        results.append(result)

    # Note this only saves static actors, which does not include static background objects
    static_list = self._actors.filter('*static*')
    for static in static_list:
      if static.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        static_transform = static.get_transform()
        static_velocity = static.get_velocity()
        static_rotation = static.get_transform().rotation
        static_matrix = np.array(static_transform.get_matrix())
        static_id = static.id
        static_extent = static.bounding_box.extent
        static_extent = [static_extent.x, static_extent.y, static_extent.z]
        yaw = np.deg2rad(static_rotation.yaw)

        relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
        relative_pos = t_u.get_relative_transform(ego_matrix, static_matrix)

        static_speed = self._get_forward_speed(transform=static_transform, velocity=static_velocity)

        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
        if not lidar is None:
          num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, static_extent, lidar)
        else:
          num_in_bbox_points = -1

        distance = np.linalg.norm(relative_pos)

        result = {
            'class': 'static',
            'extent': static_extent,
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': int(num_in_bbox_points),
            'distance': distance,
            'speed': static_speed,
            'id': int(static_id),
            'matrix': static_transform.get_matrix(),
            'type_id': static.type_id,
            'mesh_path': static.attributes['mesh_path'] if 'mesh_path' in static.attributes else None
        }
        results.append(result)

    for traffic_light in self.close_traffic_lights:
      traffic_light_extent = [traffic_light[0].extent.x, traffic_light[0].extent.y, traffic_light[0].extent.z]

      traffic_light_transform = carla.Transform(traffic_light[0].location, traffic_light[0].rotation)
      traffic_light_rotation = traffic_light_transform.rotation
      traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
      yaw = np.deg2rad(traffic_light_rotation.yaw)

      relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
      relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)

      distance = np.linalg.norm(relative_pos)

      result = {
          'class': 'traffic_light',
          'extent': traffic_light_extent,
          'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
          'yaw': relative_yaw,
          'distance': distance,
          'state': str(traffic_light[1]),
          'id': int(traffic_light[2]),
          'affects_ego': traffic_light[3],
          'matrix': traffic_light_transform.get_matrix()
      }
      results.append(result)

    for stop_sign in self.close_stop_signs:
      stop_sign_extent = [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z]

      stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
      stop_sign_rotation = stop_sign_transform.rotation
      stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
      yaw = np.deg2rad(stop_sign_rotation.yaw)

      relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
      relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

      distance = np.linalg.norm(relative_pos)

      result = {
          'class': 'stop_sign',
          'extent': stop_sign_extent,
          'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
          'yaw': relative_yaw,
          'distance': distance,
          'id': int(stop_sign[1]),
          'affects_ego': stop_sign[2],
          'matrix': stop_sign_transform.get_matrix()
      }
      results.append(result)

    return results

  def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
    """
        Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
        :param extent: List, Extent of the bounding box
        :param lidar: LiDAR point cloud
        :return: Returns the number of LiDAR hits within the bounding box of the
        vehicle
        """

    rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

    # LiDAR in the with the vehicle as origin
    vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

    # check points in bbox
    x, y, z = extent[0], extent[1], extent[2]
    num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                  (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
    return num_points

  def visualuize(self, rendered, visu_img):
    rendered = cv2.resize(rendered, dsize=(visu_img.shape[1], visu_img.shape[1]), interpolation=cv2.INTER_LINEAR)
    visu_img = cv2.cvtColor(visu_img, cv2.COLOR_BGR2RGB)

    final = np.concatenate((visu_img, rendered), axis=0)

    Image.fromarray(final).save(self.save_path / (f'{self.step:04}.jpg'))

  def _vehicle_obstacle_detected(self,
                                 vehicle_list=None,
                                 max_distance=None,
                                 up_angle_th=90,
                                 low_angle_th=0,
                                 lane_offset=0):
    """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
    self._use_bbs_detection = False
    self._offset = 0

    def get_route_polygon():
      route_bb = []
      extent_y = self._vehicle.bounding_box.extent.y
      r_ext = extent_y + self._offset
      l_ext = -extent_y + self._offset
      r_vec = ego_transform.get_right_vector()
      p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
      p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
      route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

      for wp, _ in self._local_planner.get_plan():
        if ego_location.distance(wp.transform.location) > max_distance:
          break

        r_vec = wp.transform.get_right_vector()
        p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
        p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
        route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

      # Two points don't create a polygon, nothing to check
      if len(route_bb) < 3:
        return None

      return Polygon(route_bb)

    if not vehicle_list:
      vehicle_list = self._world.get_actors().filter("*vehicle*")

    ego_transform = self._vehicle.get_transform()
    ego_location = ego_transform.location
    ego_wpt = self.world_map.get_waypoint(ego_location, lane_type=carla.libcarla.LaneType.Any)

    # Get the right offset
    if ego_wpt.lane_id < 0 and lane_offset != 0:
      lane_offset *= -1

    # Get the transform of the front of the ego
    ego_front_transform = ego_transform
    ego_front_transform.location += carla.Location(self._vehicle.bounding_box.extent.x *
                                                   ego_transform.get_forward_vector())

    opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
    use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

    # Get the route bounding box
    route_polygon = get_route_polygon()

    for target_vehicle in vehicle_list:
      if target_vehicle.id == self._vehicle.id:
        continue

      target_transform = target_vehicle.get_transform()
      if target_transform.location.distance(ego_location) > max_distance:
        continue

      target_wpt = self.world_map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

      # General approach for junctions and vehicles invading other lanes due to the offset
      if (use_bbs or target_wpt.is_junction) and route_polygon:

        target_bb = target_vehicle.bounding_box
        target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
        target_list = [[v.x, v.y, v.z] for v in target_vertices]
        target_polygon = Polygon(target_list)

        if route_polygon.intersects(target_polygon):
          return (True, target_vehicle.id, compute_distance(target_vehicle.get_location(), ego_location))

      # Simplified approach, using only the plan waypoints (similar to TM)
      else:

        if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
          next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
          if not next_wpt:
            continue
          if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
            continue

        target_forward_vector = target_transform.get_forward_vector()
        target_extent = target_vehicle.bounding_box.extent.x
        target_rear_transform = target_transform
        target_rear_transform.location -= carla.Location(
            x=target_extent * target_forward_vector.x,
            y=target_extent * target_forward_vector.y,
        )

        if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
          return (True, target_vehicle.id, compute_distance(target_transform.location, ego_transform.location))

    return (False, None, -1)

  def _get_forward_speed(self, transform=None, velocity=None):
    """
        Calculate the forward speed of the vehicle based on its transform and velocity.

        Args:
            transform (carla.Transform, optional): The transform of the vehicle. If not provided, it will be obtained from the vehicle.
            velocity (carla.Vector3D, optional): The velocity of the vehicle. If not provided, it will be obtained from the vehicle.

        Returns:
            float: The forward speed of the vehicle in m/s.
        """
    if not velocity:
      velocity = self._vehicle.get_velocity()

    if not transform:
      transform = self._vehicle.get_transform()

    # Convert the velocity vector to a NumPy array
    velocity_np = np.array([velocity.x, velocity.y, velocity.z])

    # Convert rotation angles from degrees to radians
    pitch_rad = np.deg2rad(transform.rotation.pitch)
    yaw_rad = np.deg2rad(transform.rotation.yaw)

    # Calculate the orientation vector based on pitch and yaw angles
    orientation_vector = np.array(
        [np.cos(pitch_rad) * np.cos(yaw_rad),
         np.cos(pitch_rad) * np.sin(yaw_rad),
         np.sin(pitch_rad)])

    # Calculate the forward speed by taking the dot product of velocity and orientation vectors
    forward_speed = np.dot(velocity_np, orientation_vector)

    return forward_speed

