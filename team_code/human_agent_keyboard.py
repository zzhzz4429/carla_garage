#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import numpy as np
import json

try:
	import pygame
	from pygame.locals import K_DOWN
	from pygame.locals import K_LEFT
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SPACE
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_d
	from pygame.locals import K_s
	from pygame.locals import K_w
	from pygame.locals import K_q
	from pygame.locals import K_r
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla
import math
import os
from datetime import datetime, timedelta
from collections import Counter, deque

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from boundary_risk_estimator import BoundaryRiskEstimator
from sotif_risk_estimator import SOTIFRiskEstimator
from risk_curve_plotter import RiskCurvePlotter


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
	def __init__(self, font, dim, pos):
		self.font = font
		self.dim = dim
		self.pos = pos
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)

	def set_text(self, text, color=(255, 255, 255), seconds=2.0):
		text_texture = self.font.render(text, True, color)
		self.surface = pygame.Surface(self.dim)
		self.seconds_left = seconds
		self.surface.fill((0, 0, 0, 0))
		self.surface.blit(text_texture, (10, 11))

	def tick(self, delta_seconds):
		self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
		self.surface.set_alpha(500.0 * self.seconds_left)

	def render(self, display):
		display.blit(self.surface, self.pos)


def get_entry_point():
	return 'HumanAgentSteeringWheel'


def strtobool(v):
	return str(v).lower() in ('yes', 'y', 'true', 't', '1')


def normalize_angle(x):
	x = x % (2 * np.pi)
	if x > np.pi:
		x -= 2 * np.pi
	return x


def get_relative_transform(ego_matrix, actor_matrix):
	relative_pos = actor_matrix[:3, 3] - ego_matrix[:3, 3]
	rot = ego_matrix[:3, :3].T
	return rot @ relative_pos


class HumanInterface(object):
	"""
	Class to control a vehicle manually for debugging purposes
	"""

	def __init__(self, width, height, side_scale, left_mirror=False, right_mirror=False):
		self._width = width
		self._height = height
		self.dim = (width, height)
		self._scale = side_scale
		self._surface = None
		self.camera_idx = 0
		
		# Add camera dimensions
		self.camera_width = width
		self.camera_height = height

		self._left_mirror = left_mirror
		self._right_mirror = right_mirror
		self.vehicle = "Unknown"
		self.map = "Unknown"

		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		
		display_info = pygame.display.Info()
		screen_w = display_info.current_w
		screen_h = display_info.current_h
		scale_w = screen_w / max(1, self.camera_width)
		scale_h = screen_h / max(1, self.camera_height)
		scale = min(scale_w, scale_h) * 0.9
		display_width = int(self.camera_width * scale)
		display_height = int(self.camera_height * scale)
		self._width = display_width
		self._height = display_height
		self.dim = (display_width, display_height)
		self._display = pygame.display.set_mode((display_width, display_height),
		                                      pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED)
		pygame.display.set_caption("Human Agent")

		font = pygame.font.Font(pygame.font.get_default_font(), 20)
		self._notifications = FadingText(font, (width, 40), (0, height - 40))

		font_name = 'courier' if os.name == 'nt' else 'mono'
		fonts = [x for x in pygame.font.get_fonts() if font_name in x]
		default_font = 'ubuntumono'
		mono = default_font if default_font in fonts else fonts[0]
		mono = pygame.font.match_font(mono)
		self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
		self._font_speed = pygame.font.Font(mono, 50)

		self._info_text = []

		self._show_info = True

		self.camera_surfaces = [None, None, None]  # For left, center, right views
		self._mirror_width = int(self._display.get_width() * 0.2)
		self._mirror_height = int(self._display.get_height() * 0.2)
		self._center_raw_surface = pygame.Surface((self.camera_width, self.camera_height))
		self._center_surface = pygame.Surface((self._display.get_width(), self._display.get_height()))
		self._left_raw_surface = pygame.Surface((int(self.camera_width * self._scale), int(self.camera_height * self._scale)))
		self._right_raw_surface = pygame.Surface((int(self.camera_width * self._scale), int(self.camera_height * self._scale)))
		self._mirror_surface_l = pygame.Surface((self._mirror_width, self._mirror_height))
		self._mirror_surface_r = pygame.Surface((self._mirror_width, self._mirror_height))
		
		# Optional: Set SDL video driver hints
		os.environ['SDL_VIDEO_X11_VISUAL'] = '0'  # For Linux
		os.environ['SDL_VIDEO_CENTERED'] = '1'

	def tick(self, control, fps, timestamp):
		self._info_text = [
			'Framerate:  % 14.0f FPS' % fps,
			'',
			'Vehicle: % 20s' % self.vehicle,
			'Map:     % 20s' % self.map,
			'Time:            % 12s' % timedelta(seconds=int(timestamp)),
			'',
			'Speed:%2.0f km/h' % 1,
			'',
			'']

		self._info_text += [
			('Throttle:', control.throttle, 0.0, 1.0),
			('Steer:', control.steer, -1.0, 1.0),
			('Reverse:', control.reverse),
		]



	def run_interface(self, input_data):
		"""
		DataAgent-like interface:
		- Center camera fills display
		- Left/Right mirrors shown as small overlays at top corners
		"""
		self._display.fill((0, 0, 0))

		if 'Center' in input_data:
			image = input_data['Center'][1][:, :, :3][:, :, ::-1]
			pygame.surfarray.blit_array(self._center_raw_surface, image.swapaxes(0, 1))
			if self._center_raw_surface.get_size() == self._center_surface.get_size():
				self._display.blit(self._center_raw_surface, (0, 0))
			else:
				pygame.transform.scale(self._center_raw_surface, self._center_surface.get_size(), self._center_surface)
				self._display.blit(self._center_surface, (0, 0))

		if 'Right' in input_data:
			right_mirror = input_data['Right'][1][:, :, :3][:, :, ::-1]
			pygame.surfarray.blit_array(self._right_raw_surface, right_mirror.swapaxes(0, 1))
			pygame.transform.scale(self._right_raw_surface, self._mirror_surface_r.get_size(), self._mirror_surface_r)
			mirror_x = self._display.get_width() - self._mirror_width - 10
			mirror_y = 10
			self._display.blit(self._mirror_surface_r, (mirror_x, mirror_y))

		if 'Left' in input_data:
			left_mirror = input_data['Left'][1][:, :, :3][:, :, ::-1]
			pygame.surfarray.blit_array(self._left_raw_surface, left_mirror.swapaxes(0, 1))
			pygame.transform.scale(self._left_raw_surface, self._mirror_surface_l.get_size(), self._mirror_surface_l)
			mirror_x = 10
			mirror_y = 10
			self._display.blit(self._mirror_surface_l, (mirror_x, mirror_y))

		# Display flip is performed by agent after optional HUD overlays are drawn.

	def set_black_screen(self):
		"""Set the surface to black"""
		self._display.fill((0, 0, 0))
		pygame.display.flip()

	def _quit(self):
		pygame.quit()


class HumanAgentSteeringWheel(AutonomousAgent):
	"""
	Human agent to control the ego vehicle via keyboard
	"""

	current_control = None
	agent_engaged = False

	def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
		"""
		Setup the agent parameters
		"""
		self.track = Track.MAP

		self.agent_engaged = False
		# Full HD resolution (1080p)
		self.camera_width = 1920
		self.camera_height = 1080
		self._side_scale = 0.3
		self._left_mirror = False
		self._right_mirror = False

		self._hic = HumanInterface(
			self.camera_width,
			self.camera_height,
			self._side_scale,
			self._left_mirror,
			self._right_mirror
		)

		self._controller = SteeringWheelControl(path_to_conf_file, self._hic)
		self._prev_timestamp = 0

		self._clock = pygame.time.Clock()
		self._world = None
		self._vehicle = None
		self._actors = None
		self.last_ego_speed = 0.0
		self.last_boundary_risk_field = None
		self.last_boundary_risk_info = None
		self.enable_post_risk_plotting = strtobool(os.environ.get('ENABLE_POST_RISK_PLOTTING', 'True'))
		self.risk_plot_output_dir = os.environ.get('RISK_PLOT_OUTPUT_DIR', 'risk_post_plots')
		self.risk_history_maxlen = max(1000, int(os.environ.get('RISK_HISTORY_MAXLEN', '36000')))
		self.enable_emergency_bbox_recording = strtobool(os.environ.get('ENABLE_EMERGENCY_BBOX_RECORDING', 'True'))
		self._emergency_bbox_records = []
		self._emergency_presence_frames = 0
		self._risk_history = {
			'timestamp': deque(maxlen=self.risk_history_maxlen),
			'max_risk': deque(maxlen=self.risk_history_maxlen),
			'risk_level': deque(maxlen=self.risk_history_maxlen),
			'risk_direction_deg': deque(maxlen=self.risk_history_maxlen),
			'risk_type': deque(maxlen=self.risk_history_maxlen),
			'primary_object_id': deque(maxlen=self.risk_history_maxlen),
			'primary_object_class': deque(maxlen=self.risk_history_maxlen),
			'primary_object_class_id': deque(maxlen=self.risk_history_maxlen),
			'primary_object_distance': deque(maxlen=self.risk_history_maxlen),
			'primary_object_x': deque(maxlen=self.risk_history_maxlen),
			'primary_object_y': deque(maxlen=self.risk_history_maxlen),
			'related_object_count': deque(maxlen=self.risk_history_maxlen),
			'related_object_ids': deque(maxlen=self.risk_history_maxlen),
			'related_object_classes': deque(maxlen=self.risk_history_maxlen),
		}

		# Boundary/SOTIF risk estimation config (GT actor based).
		self.use_boundary_risk = strtobool(os.environ.get('USE_BOUNDARY_RISK', 'True'))
		self.enable_sotif_heatmap_hud = strtobool(os.environ.get('DRAW_SOTIF_HEATMAP_HUD', 'True'))
		self.enable_risk_curve_plot = strtobool(os.environ.get('ENABLE_RISK_CURVE_PLOT', 'False'))
		# Performance knobs for SOTIF-heavy runs.
		self.risk_compute_every_n_frames = max(1, int(os.environ.get('RISK_COMPUTE_EVERY_N_FRAMES', '1')))
		self.sotif_hud_update_every_n_frames = max(1, int(os.environ.get('SOTIF_HUD_UPDATE_EVERY_N_FRAMES', '2')))
		self.sotif_hud_downsample = max(1, int(os.environ.get('SOTIF_HUD_DOWNSAMPLE', '2')))
		self.sotif_hud_norm_percentile = float(os.environ.get('SOTIF_HUD_NORM_PERCENTILE', '99.0'))
		self._frame_index = 0
		self._cached_sotif_hud_surface = None
		self._last_sotif_hud_frame = -1
		self._sotif_hud_panel = None
		self.boundary_risk_estimator = None
		risk_estimator_type = str(os.environ.get('RISK_ESTIMATOR_TYPE', 'sotif')).strip().lower()
		
		# Real-time risk curve plotter
		self.risk_curve_plotter = None
		if self.enable_risk_curve_plot:
			plot_type = str(os.environ.get('RISK_CURVE_PLOT_TYPE', 'both')).strip().lower()
			update_rate = float(os.environ.get('RISK_CURVE_UPDATE_RATE', '10.0'))
			try:
				self.risk_curve_plotter = RiskCurvePlotter(
					enable=True,
					plot_type=plot_type,
					update_rate=update_rate
				)
				print(f'Risk curve plotter enabled (type={plot_type}, rate={update_rate}Hz)')
			except Exception as e:
				print(f'⚠️ Failed to initialize risk curve plotter: {e}')
				self.risk_curve_plotter = None

		if self.use_boundary_risk:
			angular_resolution = int(os.environ.get('BOUNDARY_ANGULAR_RESOLUTION', 72))
			max_range = float(os.environ.get('BOUNDARY_MAX_RANGE', 50.0))
			k_distance = float(os.environ.get('BOUNDARY_K_DISTANCE', -1.0))
			alpha_coeff = float(os.environ.get('BOUNDARY_ALPHA', 1.0))
			beta_coeff = float(os.environ.get('BOUNDARY_BETA', 0.5))
			risk_threshold = float(os.environ.get('BOUNDARY_RISK_THRESHOLD', 0.3))
			lateral_threshold = float(os.environ.get('BOUNDARY_LATERAL_THRESHOLD', 15))

			if risk_estimator_type == 'sotif':
				self.boundary_risk_estimator = SOTIFRiskEstimator(
					angular_resolution=angular_resolution,
					max_range=max_range,
					k_distance=k_distance,
					alpha_coeff=alpha_coeff,
					beta_coeff=beta_coeff,
					risk_threshold=risk_threshold,
					lateral_risk_threshold=lateral_threshold
				)
			else:
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
			print(f"[HumanAgent] Risk estimator: {risk_estimator_type} (debug={boundary_debug})")

		# Prime world/map info for HUD labels.
		self._ensure_world_and_ego()

	def sensors(self):
		"""
		Define the sensor suite required by the agent

		:return: a list containing the required sensors in the following format:

		[
			{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					  'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

			{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					  'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

			{'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
			 'id': 'LIDAR'}
		]
		"""

		sensors = [
			# Center camera
			{'type': 'sensor.camera.rgb',
			 'x': 1.4, 'y': 0.0, 'z': 1.1,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
			 'width': self.camera_width, 'height': self.camera_height,
			 'fov': 60, 'id': 'Center'},

			# Left mirror - reduced resolution
			{'type': 'sensor.camera.rgb',
			 'x': 0.7, 'y': -1.0, 'z': 1.0,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 210.0,
			 'width': int(self.camera_width * self._side_scale),
			 'height': int(self.camera_height * self._side_scale),
			 'fov': 100, 'id': 'Left'},

			# Right mirror - reduced resolution
			{'type': 'sensor.camera.rgb',
			 'x': 0.7, 'y': 1.0, 'z': 1.0,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 150.0,
			 'width': int(self.camera_width * self._side_scale),
			 'height': int(self.camera_height * self._side_scale),
			 'fov': 100, 'id': 'Right'},

			{'type': 'sensor.speedometer', 'id': 'speedometer'}
		]

		return sensors

	def run_step(self, input_data, timestamp):
		"""
		Execute one step of navigation.
		"""

		passed_milliseconds = self._clock.tick()
		self._frame_index += 1
		self.agent_engaged = True
		self._controller.tick(passed_milliseconds, self._clock.get_fps(), timestamp)
		self._hic.run_interface(input_data)
		self._update_boundary_risk(timestamp)
		self._render_risk_hud()
		pygame.display.flip()

		control = self._controller.parse_events(timestamp - self._prev_timestamp)
		self._prev_timestamp = timestamp

		return control

	def destroy(self, results=None):
		"""
		Cleanup
		"""
		if self.enable_post_risk_plotting:
			self._generate_post_simulation_risk_plots()
		self._save_emergency_bbox_records()

		# Close risk curve plotter
		if self.risk_curve_plotter is not None:
			self.risk_curve_plotter.close()
			self.risk_curve_plotter = None
		
		self._hic.set_black_screen()
		self._hic._quit = True

	def _record_emergency_vehicle_bboxes(self, timestamp, gt_boxes):
		"""Record all emergency-vehicle GT boxes for this frame when present."""
		if not self.enable_emergency_bbox_recording:
			return
		frame_entries = []
		for box in gt_boxes:
			if box.get('class') == 'ego_car':
				continue
			obj_class = self._map_gt_class(box)
			if obj_class != 4:
				continue
			pos = box.get('position', [0.0, 0.0, 0.0])
			extent = box.get('extent', [0.0, 0.0, 0.0])
			yaw_rad = float(box.get('yaw', 0.0))
			speed = float(box.get('speed', 0.0))
			brake = float(box.get('brake', 0.0)) if box.get('brake') is not None else 0.0
			frame_entries.append({
				'object_id': int(box.get('id')) if box.get('id') is not None else None,
				'class_id': 4,
				'class_name': 'emergency',
				'role_name': str(box.get('role_name', '')),
				'type_id': str(box.get('type_id', '')),
				'position': [float(pos[0]), float(pos[1]), float(pos[2])],
				'extent': [float(extent[0]), float(extent[1]), float(extent[2])],
				'yaw_rad': yaw_rad,
				'yaw_deg': float(np.degrees(yaw_rad)),
				'speed': speed,
				'brake': brake,
			})
		if frame_entries:
			self._emergency_presence_frames += 1
			self._emergency_bbox_records.append({
				'timestamp': float(timestamp),
				'count': int(len(frame_entries)),
				'vehicles': frame_entries,
			})

	def _save_emergency_bbox_records(self):
		"""Persist emergency-vehicle GT box timeline to JSON for offline reference."""
		if not self.enable_emergency_bbox_recording:
			return
		if not self._emergency_bbox_records:
			print("[HumanAgent] No emergency-vehicle presence observed; no emergency bbox log saved.")
			return
		try:
			os.makedirs(self.risk_plot_output_dir, exist_ok=True)
			run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			payload = {
				'frames_with_emergency_presence': int(self._emergency_presence_frames),
				'total_records': int(len(self._emergency_bbox_records)),
				'records': self._emergency_bbox_records,
			}
			output_path = os.path.join(self.risk_plot_output_dir, f'emergency_vehicle_bboxes_{run_stamp}.json')
			with open(output_path, 'w', encoding='utf-8') as f:
				json.dump(payload, f, indent=2)
			print(f"[HumanAgent] Emergency bbox reference saved: {output_path}")
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[HumanAgent] Failed to save emergency bbox reference: {exc}")

	@staticmethod
	def _risk_to_color(value: float):
		if value <= 0.3:
			t = value / 0.3 if value > 0 else 0.0
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

	def _ensure_world_and_ego(self):
		try:
			if self._world is None:
				client = carla.Client('localhost', 2000)
				client.set_timeout(2.0)
				self._world = client.get_world()
				self._hic.map = self._world.get_map().name.split('/')[-1]
			if self._world is None:
				return
			if self._vehicle is None or not self._vehicle.is_alive:
				actors = self._world.get_actors()
				self._actors = actors
				vehicles = actors.filter('*vehicle*')
				hero = None
				for vehicle in vehicles:
					role_name = vehicle.attributes.get('role_name', '')
					if role_name in ('hero', 'ego_vehicle'):
						hero = vehicle
						break
				self._vehicle = hero
				if hero is not None:
					self._hic.vehicle = hero.type_id.split('.')[-1]
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[HumanAgent] Failed to bind world/ego: {exc}")

	def _get_forward_speed(self, transform=None, velocity=None):
		if self._vehicle is None:
			return 0.0
		if velocity is None:
			velocity = self._vehicle.get_velocity()
		if transform is None:
			transform = self._vehicle.get_transform()
		pitch = math.radians(transform.rotation.pitch)
		yaw = math.radians(transform.rotation.yaw)
		cos_p = math.cos(pitch)
		ox = cos_p * math.cos(yaw)
		oy = cos_p * math.sin(yaw)
		oz = math.sin(pitch)
		return float(velocity.x * ox + velocity.y * oy + velocity.z * oz)

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
			# Internally, GT relative yaw is stored in radians; estimator expects degrees.
			yaw_deg = float(np.degrees(float(box.get('yaw', 0.0))))
			speed = float(box.get('speed', 0.0))
			brake = float(box.get('brake', 0.0)) if box.get('brake') is not None else 0.0
			formatted.append([
				float(pos[0]), float(pos[1]),
				float(extent[0]), float(extent[1]),
				yaw_deg, speed, brake, int(obj_class),
				int(box.get('id')) if box.get('id') is not None else None
			])
		return formatted

	def _collect_gt_bounding_boxes(self, radius=60.0, ego_transform=None, ego_velocity=None, actors=None):
		if self._world is None or self._vehicle is None:
			return []
		results = []
		if ego_transform is None:
			ego_transform = self._vehicle.get_transform()
		if ego_velocity is None:
			ego_velocity = self._vehicle.get_velocity()
		ego_location = ego_transform.location
		ego_matrix = np.array(ego_transform.get_matrix())
		ego_rotation = ego_transform.rotation
		ego_yaw = np.deg2rad(ego_rotation.yaw)
		ego_extent = self._vehicle.bounding_box.extent
		ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
		ego_control = self._vehicle.get_control()

		results.append({
			'class': 'ego_car',
			'extent': [ego_extent.x, ego_extent.y, ego_extent.z],
			'position': [0.0, 0.0, 0.0],
			'yaw': 0.0,
			'speed': ego_speed,
			'brake': ego_control.brake,
			'id': int(self._vehicle.id)
		})

		if actors is None:
			actors = self._world.get_actors()
		self._actors = actors
		for actor in actors:
			tid = actor.type_id
			is_vehicle = 'vehicle' in tid
			is_walker = 'walker' in tid
			is_traffic_light = 'traffic_light' in tid
			is_stop_sign = 'traffic.stop' in tid
			if not (is_vehicle or is_walker or is_traffic_light or is_stop_sign):
				continue
			if is_vehicle and actor.id == self._vehicle.id:
				continue
			actor_location = actor.get_location()
			if actor_location.distance(ego_location) > radius:
				continue

			transform = actor.get_transform()
			matrix = np.array(transform.get_matrix())
			yaw = np.deg2rad(transform.rotation.yaw)
			relative_yaw = normalize_angle(yaw - ego_yaw)
			relative_pos = get_relative_transform(ego_matrix, matrix)
			extent = actor.bounding_box.extent

			if is_vehicle:
				control = actor.get_control()
				results.append({
					'class': 'car',
					'extent': [extent.x, extent.y, extent.z],
					'position': [float(relative_pos[0]), float(relative_pos[1]), float(relative_pos[2])],
					'yaw': float(relative_yaw),
					'speed': self._get_forward_speed(transform=transform, velocity=actor.get_velocity()),
					'brake': float(control.brake),
					'id': int(actor.id),
					'role_name': actor.attributes.get('role_name', ''),
					'type_id': tid
				})
			elif is_walker:
				results.append({
					'class': 'walker',
					'extent': [extent.x, extent.y, extent.z],
					'position': [float(relative_pos[0]), float(relative_pos[1]), float(relative_pos[2])],
					'yaw': float(relative_yaw),
					'speed': self._get_forward_speed(transform=transform, velocity=actor.get_velocity()),
					'id': int(actor.id)
				})
			elif is_traffic_light:
				results.append({
					'class': 'traffic_light',
					'extent': [extent.x, extent.y, extent.z],
					'position': [float(relative_pos[0]), float(relative_pos[1]), float(relative_pos[2])],
					'yaw': float(relative_yaw),
					'id': int(actor.id)
				})
			elif is_stop_sign:
				results.append({
					'class': 'stop_sign',
					'extent': [extent.x, extent.y, extent.z],
					'position': [float(relative_pos[0]), float(relative_pos[1]), float(relative_pos[2])],
					'yaw': float(relative_yaw),
					'id': int(actor.id)
				})

		return results

	def _update_boundary_risk(self, timestamp):
		if not self.use_boundary_risk or self.boundary_risk_estimator is None:
			self._record_risk_sample(timestamp, 0.0, "DISABLED", None)
			return
		rebound_ego = False
		if self._vehicle is None or not self._vehicle.is_alive:
			self._ensure_world_and_ego()
			rebound_ego = True
		if self._vehicle is None or not self._vehicle.is_alive:
			self._record_risk_sample(timestamp, 0.0, "NO_EGO", None)
			return
		try:
			self.boundary_risk_estimator.last_polar_boundary = None
		except Exception:
			pass
		ego_transform = self._vehicle.get_transform()
		ego_velocity = self._vehicle.get_velocity()
		ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
		self.last_ego_speed = ego_speed
		actors_override = self._actors if rebound_ego else None
		gt_boxes = self._collect_gt_bounding_boxes(
			ego_transform=ego_transform,
			ego_velocity=ego_velocity,
			actors=actors_override
		)
		self._record_emergency_vehicle_bboxes(timestamp, gt_boxes)
		should_compute_risk = (
			self.last_boundary_risk_field is None
			or self.last_boundary_risk_info is None
			or (self._frame_index % self.risk_compute_every_n_frames == 0)
		)
		if not should_compute_risk:
			max_risk_cached = float(np.max(self.last_boundary_risk_field)) if self.last_boundary_risk_field is not None else 0.0
			threat_info_cached = self.last_boundary_risk_info if isinstance(self.last_boundary_risk_info, dict) else {}
			self._record_risk_sample(
				timestamp=timestamp,
				max_risk=max_risk_cached,
				risk_level=str(threat_info_cached.get('risk_level', 'UNKNOWN')),
				risk_direction_deg=threat_info_cached.get('max_risk_direction_deg'),
				threat_info=threat_info_cached
			)
			return
		bbs = self._gt_boxes_to_boundary_format(gt_boxes)
		if not bbs:
			self._record_risk_sample(timestamp, 0.0, "SAFE", None)
			return
		risk_field, _max_risk, threat_info = self.boundary_risk_estimator.calculate_boundary_risk(
			ego_speed, bbs, timestamp
		)
		threat_info['risk_field'] = risk_field
		self.last_boundary_risk_field = risk_field
		self.last_boundary_risk_info = threat_info
		self._record_risk_sample(
			timestamp=timestamp,
			max_risk=float(_max_risk),
			risk_level=str(threat_info.get('risk_level', 'UNKNOWN')),
			risk_direction_deg=threat_info.get('max_risk_direction_deg'),
			threat_info=threat_info
		)
		
		# Update real-time risk curve plot
		if self.risk_curve_plotter is not None and self.risk_curve_plotter.is_enabled():
			angles = getattr(self.boundary_risk_estimator, 'angles', None)
			if angles is not None and len(angles) == len(risk_field):
				max_risk = float(np.max(risk_field)) if risk_field.size > 0 else 0.0
				self.risk_curve_plotter.update(risk_field, angles, max_risk, threat_info)

	def _render_risk_radar(self, size=220):
		if self.last_boundary_risk_field is None or len(self.last_boundary_risk_field) == 0:
			return None
		if self.boundary_risk_estimator is None:
			return None
		surface = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
		center = (size // 2, size // 2)
		radius = size // 2 - 10
		surface.fill((0, 0, 0, 0))
		pygame.draw.circle(surface, (20, 20, 20, 180), center, radius)
		pygame.draw.circle(surface, (180, 180, 180, 255), center, radius, 2)
		pygame.draw.line(surface, (100, 100, 100, 150), (center[0], center[1] - radius), (center[0], center[1] + radius), 1)
		pygame.draw.line(surface, (100, 100, 100, 150), (center[0] - radius, center[1]), (center[0] + radius, center[1]), 1)

		risk_field = self.last_boundary_risk_field
		angles = getattr(self.boundary_risk_estimator, 'angles', None)
		polar_boundary = getattr(self.boundary_risk_estimator, 'last_polar_boundary', None)
		if angles is None:
			angles = np.linspace(0.0, 2.0 * np.pi, len(risk_field), endpoint=False)
		for i, (angle, risk) in enumerate(zip(angles, risk_field)):
			if risk <= 0.01:
				continue
			normalized = max(0.0, min(1.0, float(risk)))
			obj_class = 0
			if polar_boundary is not None and i < len(polar_boundary):
				obj_class = polar_boundary[i].get('object_class', 0)
			if obj_class == 1:
				color = (255, 0, 255, 255)
			elif obj_class == 4:
				color = (0, 191, 255, 255)
			else:
				color_rgb = HumanAgentSteeringWheel._risk_to_color(normalized)
				color = (color_rgb[0], color_rgb[1], color_rgb[2], 255)
			display_angle = angle - math.pi / 2.0
			length = int(max(0.1, normalized) * radius)
			end_pos = (
				int(center[0] + length * math.cos(display_angle)),
				int(center[1] + length * math.sin(display_angle)),
			)
			pygame.draw.line(surface, color, center, end_pos, 3)
		ego_size = 9
		pygame.draw.polygon(surface, (255, 255, 255, 255), [
			(center[0], center[1] - ego_size),
			(center[0] - ego_size // 2, center[1] + ego_size // 2),
			(center[0] + ego_size // 2, center[1] + ego_size // 2),
		])
		return surface

	def _render_sotif_heatmap_hud(self, size=220):
		if self.boundary_risk_estimator is None:
			return None
		if not isinstance(self.boundary_risk_estimator, SOTIFRiskEstimator):
			return None
		if self.last_boundary_risk_info is None:
			return None
		if (
			self._cached_sotif_hud_surface is not None
			and self.sotif_hud_update_every_n_frames > 1
			and self._last_sotif_hud_frame >= 0
			and (self._frame_index - self._last_sotif_hud_frame) < self.sotif_hud_update_every_n_frames
		):
			return self._cached_sotif_hud_surface
		heatmap_r = self.last_boundary_risk_info.get('heatmap')
		heatmap_c = self.last_boundary_risk_info.get('heatmap_C')

		def _normalize_map(arr):
			if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.size == 0:
				return None
			max_val = float(np.max(arr))
			if max_val <= 1e-8:
				return np.zeros_like(arr, dtype=np.float32)
			if self.sotif_hud_norm_percentile > 0.0:
				ref = max(1e-8, float(np.percentile(arr, self.sotif_hud_norm_percentile)))
			else:
				ref = max_val
			return np.clip(arr / ref, 0.0, 1.0).astype(np.float32)

		norm_r = _normalize_map(heatmap_r)
		norm_c = _normalize_map(heatmap_c)
		if norm_r is None and norm_c is None:
			return None
		if norm_r is not None and norm_c is not None:
			normalized = 0.4 * norm_r + 0.6 * norm_c
		elif norm_c is not None:
			normalized = norm_c
		else:
			normalized = norm_r

		if self.sotif_hud_downsample > 1:
			normalized = normalized[::self.sotif_hud_downsample, ::self.sotif_hud_downsample]
		display_map = np.flipud(normalized.T)
		r = (255.0 * display_map).astype(np.uint8)
		g = (200.0 * (1.0 - display_map)).astype(np.uint8)
		b = (45.0 * (1.0 - display_map)).astype(np.uint8)
		rgb = np.stack((r, g, b), axis=2)
		if self._sotif_hud_panel is None or self._sotif_hud_panel.get_size() != (size, size):
			self._sotif_hud_panel = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
		panel = self._sotif_hud_panel
		panel.fill((12, 12, 12, 185))
		map_size = size - 24
		heat_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
		heat_surface = pygame.transform.scale(heat_surface, (map_size, map_size))
		panel.blit(heat_surface, (12, 12))
		pygame.draw.rect(panel, (220, 220, 220, 220), pygame.Rect(11, 11, map_size + 2, map_size + 2), 1)
		center = (12 + map_size // 2, 12 + map_size // 2)
		pygame.draw.polygon(panel, (255, 255, 255, 230), [
			(center[0], center[1] - 7),
			(center[0] - 4, center[1] + 4),
			(center[0] + 4, center[1] + 4),
		])
		title = self._hic._font_mono.render("SOTIF C+R", True, (255, 255, 255))
		panel.blit(title, (14, 2))
		panel.blit(self._hic._font_mono.render("L", True, (200, 200, 200)), (14, 14 + map_size // 2))
		panel.blit(self._hic._font_mono.render("R", True, (200, 200, 200)), (12 + map_size - 12, 14 + map_size // 2))
		panel.blit(self._hic._font_mono.render("F", True, (200, 200, 200)), (12 + map_size // 2 - 4, 14))
		self._cached_sotif_hud_surface = panel
		self._last_sotif_hud_frame = self._frame_index
		return panel

	def _render_risk_hud(self):
		if self._hic is None or self._hic._display is None:
			return
		margin = 20
		is_sotif = isinstance(self.boundary_risk_estimator, SOTIFRiskEstimator)
		if is_sotif:
			if not self.enable_sotif_heatmap_hud:
				return
			surface = self._render_sotif_heatmap_hud(size=220)
			if surface is None:
				return
			x = self._hic._display.get_width() - surface.get_width() - margin
			y = self._hic._display.get_height() - surface.get_height() - margin
			self._hic._display.blit(surface, (x, y))
		else:
			surface = self._render_risk_radar(size=220)
			if surface is None:
				return
			x = self._hic._display.get_width() - surface.get_width() - margin
			y = self._hic._display.get_height() - surface.get_height() - margin
			self._hic._display.blit(surface, (x, y))
			max_risk = float(np.max(self.last_boundary_risk_field)) if self.last_boundary_risk_field is not None else 0.0
			if max_risk >= 1.0:
				warn_text = self._hic._font_mono.render("WARNING: COLLISION RISK", True, (255, 80, 80))
				self._hic._display.blit(warn_text, (x - warn_text.get_width() - 10, y + 10))

	def _record_risk_sample(self, timestamp, max_risk, risk_level, risk_direction_deg, threat_info=None):
		self._risk_history['timestamp'].append(float(timestamp))
		self._risk_history['max_risk'].append(float(max_risk))
		self._risk_history['risk_level'].append(str(risk_level))
		if risk_direction_deg is None:
			self._risk_history['risk_direction_deg'].append(np.nan)
		else:
			self._risk_history['risk_direction_deg'].append(float(risk_direction_deg))
		risk_type = 'sotif' if isinstance(self.boundary_risk_estimator, SOTIFRiskEstimator) else 'boundary'
		self._risk_history['risk_type'].append(risk_type)

		primary = threat_info.get('primary_threat') if isinstance(threat_info, dict) else None
		if isinstance(primary, dict):
			self._risk_history['primary_object_id'].append(primary.get('object_id'))
			self._risk_history['primary_object_class'].append(str(primary.get('class_name', 'unknown')))
			self._risk_history['primary_object_class_id'].append(primary.get('object_class'))
			self._risk_history['primary_object_distance'].append(float(primary.get('distance', np.nan)))
			self._risk_history['primary_object_x'].append(float(primary.get('object_x', np.nan)))
			self._risk_history['primary_object_y'].append(float(primary.get('object_y', np.nan)))
		else:
			self._risk_history['primary_object_id'].append(None)
			self._risk_history['primary_object_class'].append('none')
			self._risk_history['primary_object_class_id'].append(None)
			self._risk_history['primary_object_distance'].append(np.nan)
			self._risk_history['primary_object_x'].append(np.nan)
			self._risk_history['primary_object_y'].append(np.nan)

		# Capture all risk-related objects for this frame from risky sectors.
		related_ids = set()
		related_classes = []
		if isinstance(threat_info, dict):
			risk_field = threat_info.get('risk_field')
			polar_boundary = getattr(self.boundary_risk_estimator, 'last_polar_boundary', None)
			if risk_field is not None and polar_boundary is not None:
				try:
					risk_threshold = float(getattr(self.boundary_risk_estimator, 'tau', 0.3))
					sector_threshold = max(0.05, min(0.3, 0.5 * risk_threshold))
					for risk_val, boundary_point in zip(risk_field, polar_boundary):
						if risk_val is None or float(risk_val) < sector_threshold:
							continue
						obj_id = boundary_point.get('object_id')
						obj_class = boundary_point.get('object_class')
						if obj_id is not None:
							related_ids.add(int(obj_id))
						if obj_class is not None:
							related_classes.append(int(obj_class))
				except Exception:
					pass
		self._risk_history['related_object_count'].append(int(len(related_ids)))
		self._risk_history['related_object_ids'].append(sorted(list(related_ids)))
		self._risk_history['related_object_classes'].append(related_classes)

	def _generate_post_simulation_risk_plots(self):
		if len(self._risk_history['timestamp']) == 0:
			print("[HumanAgent] No risk samples collected; skipping post-simulation plots.")
			return

		try:
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[HumanAgent] Matplotlib not available; cannot render risk plots: {exc}")
			return

		os.makedirs(self.risk_plot_output_dir, exist_ok=True)
		run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		ts = np.array(self._risk_history['timestamp'], dtype=np.float64)
		ts = ts - ts[0] if ts.size > 0 else ts
		max_risk = np.array(self._risk_history['max_risk'], dtype=np.float64)
		risk_dir = np.array(self._risk_history['risk_direction_deg'], dtype=np.float64)
		risk_levels = list(self._risk_history['risk_level'])
		risk_types = list(self._risk_history['risk_type'])
		estimator_label = risk_types[-1] if risk_types else 'unknown'
		estimator_tag = str(estimator_label).strip().lower().replace(' ', '_')
		primary_obj_ids = list(self._risk_history['primary_object_id'])
		primary_obj_classes = list(self._risk_history['primary_object_class'])
		primary_obj_dist = np.array(self._risk_history['primary_object_distance'], dtype=np.float64)
		primary_obj_x = np.array(self._risk_history['primary_object_x'], dtype=np.float64)
		primary_obj_y = np.array(self._risk_history['primary_object_y'], dtype=np.float64)
		related_count = np.array(self._risk_history['related_object_count'], dtype=np.int32)
		related_ids_series = list(self._risk_history['related_object_ids'])
		related_classes_series = list(self._risk_history['related_object_classes'])

		level_colors = {
			'SAFE': '#4caf50',
			'CAUTION': '#ffb300',
			'WARNING': '#ff7043',
			'CRITICAL': '#d32f2f',
			'UNKNOWN': '#607d8b',
			'DISABLED': '#9e9e9e',
			'NO_EGO': '#9e9e9e',
		}

		# 1) Risk value timeline
		plt.figure(figsize=(12, 4))
		plt.plot(ts, max_risk, color='#1976d2', linewidth=1.6, label='max_risk')
		plt.axhline(0.3, color='#ffa000', linestyle='--', linewidth=1, label='threshold ~0.3')
		plt.axhline(0.7, color='#d32f2f', linestyle='--', linewidth=1, label='threshold ~0.7')
		plt.xlabel('Time (s)')
		plt.ylabel('Max risk')
		plt.title('Post-Simulation Risk Value Timeline')
		plt.grid(alpha=0.25)
		plt.legend(loc='upper right')
		value_path = os.path.join(self.risk_plot_output_dir, f'risk_value_{estimator_tag}_{run_stamp}.png')
		plt.tight_layout()
		plt.savefig(value_path, dpi=150)
		plt.close()

		# 2) Risk type/level visualization (timeline + distribution)
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
		level_to_y = {'SAFE': 0, 'CAUTION': 1, 'WARNING': 2, 'CRITICAL': 3}
		y_vals = [level_to_y.get(lvl, -1) for lvl in risk_levels]
		point_colors = [level_colors.get(lvl, '#607d8b') for lvl in risk_levels]
		ax1.scatter(ts, y_vals, c=point_colors, s=10, alpha=0.8)
		ax1.set_yticks([0, 1, 2, 3])
		ax1.set_yticklabels(['SAFE', 'CAUTION', 'WARNING', 'CRITICAL'])
		ax1.set_xlabel('Time (s)')
		ax1.set_title('Risk Type (Level) Timeline')
		ax1.grid(alpha=0.25)

		counts = Counter(risk_levels)
		ordered_levels = ['SAFE', 'CAUTION', 'WARNING', 'CRITICAL', 'UNKNOWN']
		x_levels = [lvl for lvl in ordered_levels if counts.get(lvl, 0) > 0]
		x_counts = [counts[lvl] for lvl in x_levels]
		ax2.bar(x_levels, x_counts, color=[level_colors.get(lvl, '#607d8b') for lvl in x_levels], alpha=0.9)
		ax2.set_title(f'Risk Level Distribution (estimator={estimator_label})')
		ax2.set_ylabel('Frame count')
		ax2.grid(axis='y', alpha=0.25)
		type_path = os.path.join(self.risk_plot_output_dir, f'risk_type_{estimator_tag}_{run_stamp}.png')
		fig.tight_layout()
		fig.savefig(type_path, dpi=150)
		plt.close(fig)

		# 3) Risk direction timeline (only valid direction samples)
		valid_mask = ~np.isnan(risk_dir)
		plt.figure(figsize=(12, 4))
		if np.any(valid_mask):
			plt.scatter(ts[valid_mask], risk_dir[valid_mask], s=10, c='#8e24aa', alpha=0.8)
		plt.axhline(0.0, color='#455a64', linestyle='--', linewidth=1)
		plt.axhline(90.0, color='#90a4ae', linestyle=':', linewidth=1)
		plt.axhline(-90.0, color='#90a4ae', linestyle=':', linewidth=1)
		plt.xlabel('Time (s)')
		plt.ylabel('Direction (deg)')
		plt.ylim([-180, 180])
		plt.title('Risk Direction Timeline (0=front, +90=right, -90=left)')
		plt.grid(alpha=0.25)
		dir_path = os.path.join(self.risk_plot_output_dir, f'risk_direction_{estimator_tag}_{run_stamp}.png')
		plt.tight_layout()
		plt.savefig(dir_path, dpi=150)
		plt.close()

		# 4) Risk-related object plots (class timeline + distance timeline + XY location)
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
		class_counts = Counter(primary_obj_classes)
		class_order = [k for k, _v in class_counts.most_common() if k != 'none']
		class_to_idx = {name: idx for idx, name in enumerate(class_order)}
		valid_class_idx = [class_to_idx.get(name, -1) for name in primary_obj_classes]
		valid_class_mask = np.array([idx >= 0 for idx in valid_class_idx], dtype=bool)
		if np.any(valid_class_mask):
			ax1.scatter(ts[valid_class_mask], np.array(valid_class_idx, dtype=np.int32)[valid_class_mask], s=10, c='#26a69a', alpha=0.8)
		ax1.set_yticks(list(range(len(class_order))))
		ax1.set_yticklabels(class_order if class_order else ['none'])
		ax1.set_xlabel('Time (s)')
		ax1.set_title('Primary Risk Object Class Timeline')
		ax1.grid(alpha=0.25)

		valid_dist_mask = ~np.isnan(primary_obj_dist)
		if np.any(valid_dist_mask):
			ax2.plot(ts[valid_dist_mask], primary_obj_dist[valid_dist_mask], color='#5e35b1', linewidth=1.4)
			ax2.scatter(ts[valid_dist_mask], primary_obj_dist[valid_dist_mask], s=8, c='#7e57c2', alpha=0.75)
		ax2.set_xlabel('Time (s)')
		ax2.set_ylabel('Distance (m)')
		ax2.set_title('Primary Risk Object Distance Timeline')
		ax2.grid(alpha=0.25)

		valid_xy_mask = (~np.isnan(primary_obj_x)) & (~np.isnan(primary_obj_y))
		if np.any(valid_xy_mask):
			sc = ax3.scatter(primary_obj_x[valid_xy_mask], primary_obj_y[valid_xy_mask], c=max_risk[valid_xy_mask], cmap='inferno', s=12, alpha=0.85)
			fig.colorbar(sc, ax=ax3, label='Max risk')
		ax3.axhline(0.0, color='#90a4ae', linestyle=':', linewidth=1)
		ax3.axvline(0.0, color='#90a4ae', linestyle=':', linewidth=1)
		ax3.set_xlabel('Object x (m, forward)')
		ax3.set_ylabel('Object y (m, right)')
		ax3.set_title('Primary Risk Object Relative Position')
		ax3.grid(alpha=0.25)
		object_path = os.path.join(self.risk_plot_output_dir, f'risk_objects_{estimator_tag}_{run_stamp}.png')
		fig.tight_layout()
		fig.savefig(object_path, dpi=150)
		plt.close(fig)

		# 5) Related-objects timeline and distribution
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
		ax1.plot(ts, related_count, color='#00897b', linewidth=1.5)
		ax1.scatter(ts, related_count, s=8, c='#26a69a', alpha=0.8)
		ax1.set_xlabel('Time (s)')
		ax1.set_ylabel('Count')
		ax1.set_title('Risk-Related Object Count Timeline')
		ax1.grid(alpha=0.25)

		all_related_classes = []
		for cls_list in related_classes_series:
			all_related_classes.extend(cls_list)
		class_id_to_name = {0: 'vehicle', 1: 'pedestrian', 2: 'traffic_light', 3: 'stop_sign', 4: 'emergency'}
		class_counts = Counter([class_id_to_name.get(c, f'class{c}') for c in all_related_classes])
		if len(class_counts) == 0:
			class_counts = Counter({'none': 1})
		labels = list(class_counts.keys())
		values = [class_counts[k] for k in labels]
		ax2.bar(labels, values, color='#5c6bc0', alpha=0.9)
		ax2.set_ylabel('Occurrences in risky sectors')
		ax2.set_title('Risk-Related Object Class Distribution')
		ax2.grid(axis='y', alpha=0.25)
		related_path = os.path.join(self.risk_plot_output_dir, f'risk_related_objects_{estimator_tag}_{run_stamp}.png')
		fig.tight_layout()
		fig.savefig(related_path, dpi=150)
		plt.close(fig)

		# JSON summary for downstream analysis
		valid_obj_count = int(np.sum(np.array(primary_obj_classes) != 'none'))
		object_id_counts = Counter([oid for oid in primary_obj_ids if oid is not None])
		top_object_ids = [{'object_id': oid, 'frames_as_primary': cnt} for oid, cnt in object_id_counts.most_common(10)]
		related_object_id_counts = Counter()
		for ids in related_ids_series:
			for oid in ids:
				related_object_id_counts[int(oid)] += 1
		top_related_object_ids = [{'object_id': oid, 'frames_as_related': cnt}
		                          for oid, cnt in related_object_id_counts.most_common(15)]
		summary = {
			'frames': int(len(ts)),
			'estimator': estimator_label,
			'risk_level_counts': dict(Counter(risk_levels)),
			'primary_object_class_counts': dict(Counter([c for c in primary_obj_classes if c != 'none'])),
			'primary_object_frames': valid_obj_count,
			'top_primary_object_ids': top_object_ids,
			'mean_related_object_count': float(np.mean(related_count)) if len(related_count) > 0 else 0.0,
			'max_related_object_count': int(np.max(related_count)) if len(related_count) > 0 else 0,
			'top_related_object_ids': top_related_object_ids,
			'emergency_presence_frames': int(self._emergency_presence_frames),
			'emergency_records': int(len(self._emergency_bbox_records)),
		}
		summary_path = os.path.join(self.risk_plot_output_dir, f'risk_objects_summary_{estimator_tag}_{run_stamp}.json')
		with open(summary_path, 'w', encoding='utf-8') as f:
			json.dump(summary, f, indent=2)

		print("[HumanAgent] Post-simulation risk plots saved:")
		print(f"  - {value_path}")
		print(f"  - {type_path}")
		print(f"  - {dir_path}")
		print(f"  - {object_path}")
		print(f"  - {related_path}")
		print(f"  - {summary_path}")

class SteeringWheelControl(object):
	"""
	Keyboard control for the human agent
	"""

	def __init__(self, path_to_conf_file, hic):
		"""
		Init
		"""
		self._control = carla.VehicleControl()
		self._steer_cache = 0.0
		self._clock = pygame.time.Clock()
		self._hic = hic
		self.recording = False
		self.client = carla.Client('localhost', 2000)
		self._hic.map = self.client.get_world().get_map().name.split('/')[-1]
		self._hic.vehicle = 'Mercedes - Coupe 2020'

		self._notifications = hic._notifications
		self._mode = "normal"
		self._endpoint = None
		self._log_data = None
		self._index = 0
		self._control_list = []

		# Get the mode
		if path_to_conf_file and os.path.isfile(path_to_conf_file):
			try:
				with open(path_to_conf_file, "r", encoding="utf-8") as f:
					lines = [line.strip() for line in f.read().split("\n") if line.strip()]
				if len(lines) >= 2:
					self._mode = lines[0].split(" ")[1]
					self._endpoint = lines[1].split(" ")[1]
			except Exception as exc:  # pylint: disable=broad-except
				print(f"[HumanAgent] Invalid config file '{path_to_conf_file}', fallback to normal mode: {exc}")
				self._mode = "normal"
				self._endpoint = None

		# Get the needed vars
		if self._mode == "log":
			self._log_data = {'records': []}
		elif self._mode == "playback" and self._endpoint and os.path.isfile(self._endpoint):
			with open(self._endpoint, encoding="utf-8") as fd:
				try:
					self._records = json.load(fd)
					self._json_to_control()
				except json.JSONDecodeError:
					pass

		# start recording
		self.toggle_recording()


	def tick(self, passed_millis, fps, timestamp):
		self._notifications.tick(passed_millis)
		self._hic.tick(self._control, fps, timestamp)

	def _json_to_control(self):
		# transform strs into VehicleControl commands
		for entry in self._records['records']:
			control = carla.VehicleControl(throttle=entry['control']['throttle'],
			                               steer=entry['control']['steer'],
			                               brake=entry['control']['brake'],
			                               hand_brake=entry['control']['hand_brake'],
			                               reverse=entry['control']['reverse'],
			                               manual_gear_shift=entry['control']['manual_gear_shift'],
			                               gear=entry['control']['gear'])
			self._control_list.append(control)


	def parse_events(self, timestamp):
		"""
		Parse the keyboard events and set the vehicle controls accordingly
		"""
		# Move the vehicle
		if self._mode == "playback":
			self._parse_json_control()
		else:
			self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)

		# Record the control
		if self._mode == "log":
			self._record_control()

		return self._control


	def toggle_recording(self):
		self.recording = not self.recording
		text = "Start recording" if self.recording else "Stop recording"
		self._notifications.set_text(text, color=(255, 255, 255), seconds=2.0)
		if self.recording:
			print('Started recording - {}'.format(self._endpoint))
			if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
				os.mkdir(os.path.join(os.getcwd(), 'logs'))
			self.client.start_recorder(os.path.join(os.getcwd(), 'logs/log_{}.log'.format(datetime.now())))
		else:
			self.client.stop_recorder()

	def _parse_vehicle_keys(self, keys, milliseconds):
		"""
		Calculate new vehicle controls based on input keys
		"""

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return True
			elif event.type == pygame.KEYUP:
				if event.key == K_r:
					self.toggle_recording()
				elif event.key == K_q:
					self._control.gear = 1 if self._control.reverse else -1
					self._control.reverse = self._control.gear < 0

		steer_increment = 5e-4 * milliseconds
		if keys[K_LEFT] or keys[K_a]:
			self._steer_cache -= steer_increment
		elif keys[K_RIGHT] or keys[K_d]:
			self._steer_cache += steer_increment
		else:
			self._steer_cache *= 0.7
		self._steer_cache = max(-1.0, min(1.0, self._steer_cache))
		self._control.steer = round(self._steer_cache, 3)
		self._control.throttle = 1.0 if (keys[K_UP] or keys[K_w]) else 0.0
		self._control.brake = 1.0 if (keys[K_DOWN] or keys[K_s]) else 0.0
		self._control.hand_brake = bool(keys[K_SPACE])


	def _parse_json_control(self):
		if self._index < len(self._control_list):
			self._control = self._control_list[self._index]
			self._index += 1
		else:
			print("JSON file has no more entries")


	def _record_control(self):
		new_record = {
			'control': {
				'throttle': self._control.throttle,
				'steer': self._control.steer,
				'brake': self._control.brake,
				'hand_brake': self._control.hand_brake,
				'reverse': self._control.reverse,
				'manual_gear_shift': self._control.manual_gear_shift,
				'gear': self._control.gear
			}
		}

		self._log_data['records'].append(new_record)

	def __del__(self):
		if getattr(self, "recording", False):
			self.toggle_recording()

		# Get ready to log user commands

		if getattr(self, "_mode", None) == "log" and getattr(self, "_log_data", None) and getattr(self, "_endpoint", None):
			with open(self._endpoint, 'w', encoding="utf-8") as fd:
				json.dump(self._log_data, fd, indent=4, sort_keys=True)
