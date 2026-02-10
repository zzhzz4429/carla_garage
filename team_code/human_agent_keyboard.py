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

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from boundary_risk_estimator import BoundaryRiskEstimator
from sotif_risk_estimator import SOTIFRiskEstimator


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
		self._width = width * 3  # Tripled for three cameras side by side
		self._height = height
		self.dim = (width * 3, height)  # Updated dimensions
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
		
		# Double the window size while keeping same internal resolution
		display_width = self._width 
		display_height = self._height
		self._display = pygame.display.set_mode((display_width, display_height), 
											  pygame.HWSURFACE | 
											  pygame.DOUBLEBUF | 
											  pygame.SCALED)
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

		# Add these flags for better pygame performance
		pygame.display.set_mode((self._width, self._height), 
							  pygame.HWSURFACE | 
							  pygame.DOUBLEBUF | 
							  pygame.SCALED)  # Add SCALED flag
		
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
		Optimized version of run_interface
		"""
		# Clear the display once
		self._display.fill((0, 0, 0))
		
		# Process main cameras more efficiently
		for i, cam_id in enumerate(['Left', 'Center', 'Right']):
			if cam_id in input_data:
				# Direct slice and flip operations
				image = input_data[cam_id][1][:, :, :3][:, :, ::-1]
				
				# Create surface directly without extra processing
				surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
				
				# Calculate position
				x_pos = i * (self._width // 3)
				self._display.blit(surface, (x_pos, 0))

		# Process mirrors if needed (using same efficient approach)
		if 'LeftMirror' in input_data:
			image = input_data['LeftMirror'][1][:, :, :3][:, :, ::-1]
			surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
			self._display.blit(surface, (self._width // 3, 0))

		if 'RightMirror' in input_data:
			image = input_data['RightMirror'][1][:, :, :3][:, :, ::-1]
			surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
			right_x = (2 * self._width // 3) - int(self.camera_width * self._scale)
			self._display.blit(surface, (right_x, 0))

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

		# Boundary/SOTIF risk estimation config (GT actor based).
		self.use_boundary_risk = strtobool(os.environ.get('USE_BOUNDARY_RISK', 'True'))
		self.enable_sotif_heatmap_hud = strtobool(os.environ.get('DRAW_SOTIF_HEATMAP_HUD', 'True'))
		self.boundary_risk_estimator = None
		risk_estimator_type = str(os.environ.get('RISK_ESTIMATOR_TYPE', 'sotif')).strip().lower()

		if self.use_boundary_risk:
			angular_resolution = int(os.environ.get('BOUNDARY_ANGULAR_RESOLUTION', 72))
			max_range = float(os.environ.get('BOUNDARY_MAX_RANGE', 50.0))
			k_distance = float(os.environ.get('BOUNDARY_K_DISTANCE', -1.0))
			alpha_coeff = float(os.environ.get('BOUNDARY_ALPHA', 1.0))
			beta_coeff = float(os.environ.get('BOUNDARY_BETA', 0.5))
			risk_threshold = float(os.environ.get('BOUNDARY_RISK_THRESHOLD', 0.3))
			lateral_threshold = float(os.environ.get('BOUNDARY_LATERAL_THRESHOLD', 2.5))

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
			# Left camera - rotated 60 degrees left
			{'type': 'sensor.camera.rgb', 
			 'x': 1.4, 'y': 0.0, 'z': 1.1,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
			 'width': self.camera_width, 'height': self.camera_height,
			 'fov': 60, 'id': 'Left'},

			# Center camera
			{'type': 'sensor.camera.rgb',
			 'x': 1.4, 'y': 0.0, 'z': 1.1,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
			 'width': self.camera_width, 'height': self.camera_height,
			 'fov': 60, 'id': 'Center'},

			# Right camera - rotated 60 degrees right
			{'type': 'sensor.camera.rgb',
			 'x': 1.4, 'y': 0.0, 'z': 1.1,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
			 'width': self.camera_width, 'height': self.camera_height,
			 'fov': 60, 'id': 'Right'},

			# Left mirror - reduced resolution
			{'type': 'sensor.camera.rgb',
			 'x': 0.7, 'y': -1.0, 'z': 1.0,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 210.0,
			 'width': int(self.camera_width * self._side_scale),
			 'height': int(self.camera_height * self._side_scale),
			 'fov': 100, 'id': 'LeftMirror'},

			# Right mirror - reduced resolution
			{'type': 'sensor.camera.rgb',
			 'x': 0.7, 'y': 1.0, 'z': 1.0,
			 'roll': 0.0, 'pitch': 0.0, 'yaw': 150.0,
			 'width': int(self.camera_width * self._side_scale),
			 'height': int(self.camera_height * self._side_scale),
			 'fov': 100, 'id': 'RightMirror'},

			{'type': 'sensor.speedometer', 'id': 'speedometer'}
		]

		return sensors

	def run_step(self, input_data, timestamp):
		"""
		Execute one step of navigation.
		"""

		passed_milliseconds = self._clock.tick_busy_loop(20)
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
		self._hic.set_black_screen()
		self._hic._quit = True

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
				vehicles = self._world.get_actors().filter('*vehicle*')
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
		velocity_np = np.array([velocity.x, velocity.y, velocity.z])
		pitch_rad = np.deg2rad(transform.rotation.pitch)
		yaw_rad = np.deg2rad(transform.rotation.yaw)
		orientation_vector = np.array([
			np.cos(pitch_rad) * np.cos(yaw_rad),
			np.cos(pitch_rad) * np.sin(yaw_rad),
			np.sin(pitch_rad)
		])
		return float(np.dot(velocity_np, orientation_vector))

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
				float(pos[0]), float(pos[1]),
				float(extent[0]), float(extent[1]),
				yaw, speed, brake, int(obj_class),
				int(box.get('id')) if box.get('id') is not None else None
			])
		return formatted

	def _collect_gt_bounding_boxes(self, radius=60.0):
		if self._world is None or self._vehicle is None:
			return []
		results = []
		ego_transform = self._vehicle.get_transform()
		ego_matrix = np.array(ego_transform.get_matrix())
		ego_rotation = ego_transform.rotation
		ego_yaw = np.deg2rad(ego_rotation.yaw)
		ego_extent = self._vehicle.bounding_box.extent
		ego_speed = self._get_forward_speed(transform=ego_transform, velocity=self._vehicle.get_velocity())

		results.append({
			'class': 'ego_car',
			'extent': [ego_extent.x, ego_extent.y, ego_extent.z],
			'position': [0.0, 0.0, 0.0],
			'yaw': 0.0,
			'speed': ego_speed,
			'brake': self._vehicle.get_control().brake,
			'id': int(self._vehicle.id)
		})

		self._actors = self._world.get_actors()
		vehicle_list = self._actors.filter('*vehicle*')
		for actor in vehicle_list:
			if actor.id == self._vehicle.id:
				continue
			if actor.get_location().distance(self._vehicle.get_location()) > radius:
				continue
			transform = actor.get_transform()
			matrix = np.array(transform.get_matrix())
			yaw = np.deg2rad(transform.rotation.yaw)
			relative_yaw = normalize_angle(yaw - ego_yaw)
			relative_pos = get_relative_transform(ego_matrix, matrix)
			extent = actor.bounding_box.extent
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
				'type_id': actor.type_id
			})

		for actor in self._actors.filter('*walker*'):
			if actor.get_location().distance(self._vehicle.get_location()) > radius:
				continue
			transform = actor.get_transform()
			matrix = np.array(transform.get_matrix())
			yaw = np.deg2rad(transform.rotation.yaw)
			relative_yaw = normalize_angle(yaw - ego_yaw)
			relative_pos = get_relative_transform(ego_matrix, matrix)
			extent = actor.bounding_box.extent
			results.append({
				'class': 'walker',
				'extent': [extent.x, extent.y, extent.z],
				'position': [float(relative_pos[0]), float(relative_pos[1]), float(relative_pos[2])],
				'yaw': float(relative_yaw),
				'speed': self._get_forward_speed(transform=transform, velocity=actor.get_velocity()),
				'id': int(actor.id)
			})

		for actor in self._actors.filter('*traffic_light*'):
			if actor.get_location().distance(self._vehicle.get_location()) > radius:
				continue
			transform = actor.get_transform()
			matrix = np.array(transform.get_matrix())
			yaw = np.deg2rad(transform.rotation.yaw)
			relative_yaw = normalize_angle(yaw - ego_yaw)
			relative_pos = get_relative_transform(ego_matrix, matrix)
			extent = actor.bounding_box.extent
			results.append({
				'class': 'traffic_light',
				'extent': [extent.x, extent.y, extent.z],
				'position': [float(relative_pos[0]), float(relative_pos[1]), float(relative_pos[2])],
				'yaw': float(relative_yaw),
				'id': int(actor.id)
			})

		for actor in self._actors:
			if 'traffic.stop' not in actor.type_id:
				continue
			if actor.get_location().distance(self._vehicle.get_location()) > radius:
				continue
			transform = actor.get_transform()
			matrix = np.array(transform.get_matrix())
			yaw = np.deg2rad(transform.rotation.yaw)
			relative_yaw = normalize_angle(yaw - ego_yaw)
			relative_pos = get_relative_transform(ego_matrix, matrix)
			extent = actor.bounding_box.extent
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
			return
		self._ensure_world_and_ego()
		if self._vehicle is None:
			return
		try:
			self.boundary_risk_estimator.last_polar_boundary = None
		except Exception:
			pass
		ego_speed = self._get_forward_speed(
			transform=self._vehicle.get_transform(),
			velocity=self._vehicle.get_velocity()
		)
		self.last_ego_speed = ego_speed
		gt_boxes = self._collect_gt_bounding_boxes()
		bbs = self._gt_boxes_to_boundary_format(gt_boxes)
		if not bbs:
			return
		risk_field, _max_risk, threat_info = self.boundary_risk_estimator.calculate_boundary_risk(
			ego_speed, bbs, timestamp
		)
		threat_info['risk_field'] = risk_field
		self.last_boundary_risk_field = risk_field
		self.last_boundary_risk_info = threat_info

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
		heatmap = self.last_boundary_risk_info.get('heatmap')
		if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
			return None
		display_map = np.flipud(heatmap.T)
		ref = max(1e-8, float(np.percentile(display_map, 99.0)))
		normalized = np.clip(display_map / ref, 0.0, 1.0).astype(np.float32)
		r = (255.0 * normalized).astype(np.uint8)
		g = (200.0 * (1.0 - normalized)).astype(np.uint8)
		b = (45.0 * (1.0 - normalized)).astype(np.uint8)
		rgb = np.stack((r, g, b), axis=2)
		panel = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
		panel.fill((12, 12, 12, 185))
		map_size = size - 24
		heat_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
		heat_surface = pygame.transform.smoothscale(heat_surface, (map_size, map_size))
		panel.blit(heat_surface, (12, 12))
		pygame.draw.rect(panel, (220, 220, 220, 220), pygame.Rect(11, 11, map_size + 2, map_size + 2), 1)
		center = (12 + map_size // 2, 12 + map_size // 2)
		pygame.draw.polygon(panel, (255, 255, 255, 230), [
			(center[0], center[1] - 7),
			(center[0] - 4, center[1] + 4),
			(center[0] + 4, center[1] + 4),
		])
		title = self._hic._font_mono.render("SOTIF P*C", True, (255, 255, 255))
		panel.blit(title, (14, 2))
		return panel

	def _render_risk_hud(self):
		if self._hic is None or self._hic._display is None:
			return
		margin = 20
		# Place risk HUD on the bottom-right of the CENTER monitor.
		center_panel_right = int(self._hic.camera_width * 2)
		is_sotif = isinstance(self.boundary_risk_estimator, SOTIFRiskEstimator)
		if is_sotif:
			if not self.enable_sotif_heatmap_hud:
				return
			surface = self._render_sotif_heatmap_hud(size=220)
			if surface is None:
				return
			x = max(margin, center_panel_right - surface.get_width() - margin)
			y = self._hic._display.get_height() - surface.get_height() - margin
			self._hic._display.blit(surface, (x, y))
		else:
			surface = self._render_risk_radar(size=220)
			if surface is None:
				return
			x = max(margin, center_panel_right - surface.get_width() - margin)
			y = self._hic._display.get_height() - surface.get_height() - margin
			self._hic._display.blit(surface, (x, y))
			max_risk = float(np.max(self.last_boundary_risk_field)) if self.last_boundary_risk_field is not None else 0.0
			if max_risk >= 1.0:
				warn_text = self._hic._font_mono.render("WARNING: COLLISION RISK", True, (255, 80, 80))
				self._hic._display.blit(warn_text, (x - warn_text.get_width() - 10, y + 10))

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
