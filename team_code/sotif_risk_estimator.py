"""
SOTIF-Oriented Multi-Dimensional Risk Assessment for Autonomous Driving
=========================================================================

Implements the P × C grid-based risk field from:
  Yao et al., "SOTIF-Oriented Risk Assessment: A Multi-Dimensional Model
  for Autonomous Driving," IEEE RA-L, vol.10, no.2, Feb 2025.

Combined with boundary-aware safe distance from:
  Jiao et al., "Autonomous Driving Risk Assessment With Boundary-Based
  Environment Model," IEEE T-IV, vol.9, no.1, Jan 2024.

Key Architecture:
  1. Risk Probability Model P(x,y) — 2D Gaussian field anchored on ego
     vehicle's projected trajectory, with amplitude a(s) and σ(s) that
     scale with speed, steering angle, and look-ahead time.
  2. Environment Cost Model C(x,y) — **Directional-flux** kinetic energy
     cost.  Instead of the original scalar |v_k − v_ego|, we project the
     relative velocity onto the threat direction (object → ego):
         v_approach = max(0, (v_obj − v_ego) · d_hat)
         C_k = 0.5 · m_k · v_approach² · w_sem
     This naturally zeroes out parallel traffic and receding objects,
     eliminating phantom alerts from adjacent-lane vehicles.
  3. Risk Accumulation R = Σ(P_ij · C_ij) — element-wise product summed
     over the map grid yields a single scalar risk per frame.
  4. Directional Kinetic Elongation — each object's cost footprint is
     stretched along the **threat direction** (toward ego), not along the
     object's own velocity.  The blob is half-space clipped so that only
     the forward-facing danger corridor is rendered.
  5. Alpha-Beta temporal smoothing on distance / velocity per tracked ID
     for flicker-free visualisation.

Drop-in compatible with the existing DataAgent interface:
    estimator = SOTIFRiskEstimator()
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed, bounding_boxes, timestamp)

The returned `risk_field` is the 1-D polar summary (for the radar HUD),
while `threat_info` additionally carries the full 2-D heatmap grid for
BEV overlay rendering.

Usage:
    from sotif_risk_estimator import SOTIFRiskEstimator
    estimator = SOTIFRiskEstimator()
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed, bounding_boxes, timestamp
    )
"""

import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Helper: 2D rotation matrix
# ---------------------------------------------------------------------------
def _rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class SOTIFRiskEstimator:
    """
    SOTIF P×C grid-based risk estimator with backward-compatible polar output.

    Parameters
    ----------
    grid_half_size : float
        Half-width of the square BEV grid in metres (default 50 → 100×100 m).
    grid_resolution : float
        Cell size in metres (default 1.0 m → 100×100 cells for 50 m half).
    angular_resolution : int
        Number of polar sectors for the 1-D radar summary.
    p_coeff : float
        Quadratic coefficient *p* in amplitude  a(s) = p·(s − v·t_la)².
    t_la : float
        Look-ahead time (seconds) for the Gaussian probability field.
    sigma_m : float
        Slope of σ(s) = (m + k·|δ|)·s + c.
    sigma_c : float
        Intercept of σ(s), set to ¼ vehicle width.
    sigma_k : float
        Steering-angle sensitivity of σ.
    risk_threshold : float
        Alert threshold θ for CRITICAL level.
    caution_threshold : float
        Alert threshold for CAUTION level.
    vru_weight : float
        Semantic weight for vulnerable road users (pedestrians, cyclists).
    vehicle_weight : float
        Semantic weight for motorised vehicles.
    emergency_weight : float
        Semantic weight for emergency vehicles.
    lateral_decay_sigma : float
        σ for the Gaussian lateral suppression factor.
    smoothing_alpha : float
        EMA coefficient for distance / velocity smoothing.
    elongation_factor : float
        How far (in multiples of speed·dt) the cost blob is stretched
        along the object's velocity vector.
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        # Grid geometry
        grid_half_size: float = 50.0,
        grid_resolution: float = 1.0,
        # Polar summary
        angular_resolution: int = 72,
        max_range: float = 50.0,
        # Probability model P
        p_coeff: float = 0.4,
        t_la: float = 2.75,
        sigma_m: float = 0.3,
        sigma_c: float = 0.5,          # ¼ of ~2 m width
        sigma_k: float = 2.0,
        # Environment cost C
        vru_weight: float = 10.0,
        vehicle_weight: float = 1.0,
        emergency_weight: float = 15.0,
        # Risk thresholds
        risk_threshold: float = 5.0,    # θ for CRITICAL
        caution_threshold: float = 1.0,
        # Lateral suppression
        lateral_decay_sigma: float = 3.5,
        lateral_risk_threshold: float = 2.5,
        # Temporal smoothing
        smoothing_alpha: float = 0.35,
        range_rate_alpha: float = 0.30,
        # Visualisation
        elongation_factor: float = 2.0,
        # Legacy compatibility
        k_distance: float = -2.0,
        alpha_coeff: float = 1.0,
        beta_coeff: float = 0.5,
        min_pedestrian_risk: float = 0.6,
        pedestrian_lateral_threshold: float = 8.0,
    ):
        # --- Grid ---
        self.grid_half = grid_half_size
        self.grid_res = grid_resolution
        n = int(2 * grid_half_size / grid_resolution)
        self.grid_n = n
        # Pre-compute grid coordinate arrays (metres, ego-centred)
        ticks = np.linspace(-grid_half_size + grid_resolution / 2,
                            grid_half_size - grid_resolution / 2, n)
        self.grid_x, self.grid_y = np.meshgrid(ticks, ticks, indexing='xy')
        # Arc length from ego along +x axis (forward)
        self.grid_s = self.grid_x.copy()  # longitudinal coordinate

        # --- Polar summary ---
        self.M = angular_resolution
        self.max_range = max_range
        self.angles = np.linspace(0, 2 * np.pi, self.M, endpoint=False)

        # --- Probability model P ---
        self.p_coeff = p_coeff
        self.t_la = t_la
        self.sigma_m = sigma_m
        self.sigma_c = sigma_c
        self.sigma_k = sigma_k

        # --- Environment cost C ---
        self.vru_weight = vru_weight
        self.vehicle_weight = vehicle_weight
        self.emergency_weight = emergency_weight
        self.elongation_factor = elongation_factor

        # --- Thresholds ---
        self.tau = risk_threshold
        self.caution_tau = caution_threshold

        # --- Lateral suppression ---
        self.lateral_decay_sigma = lateral_decay_sigma
        self.lateral_risk_threshold = lateral_risk_threshold

        # --- Temporal smoothing ---
        self.smoothing_alpha = smoothing_alpha
        self.range_rate_alpha = range_rate_alpha

        # --- Legacy compatibility fields ---
        self.k = k_distance
        self.alpha = alpha_coeff
        self.beta = beta_coeff
        self.min_pedestrian_risk = min_pedestrian_risk
        self.pedestrian_lateral_threshold = pedestrian_lateral_threshold

        # --- Tracking state ---
        self.tracked_objects: Dict[int, Dict] = {}
        self.track_ttl_frames = 10
        self.frame_count = 0
        self.last_timestamp: Optional[float] = None
        self.last_delta_t: Optional[float] = None
        self.default_delta_t = 0.1

        # Persist last computation for visualisation
        self.last_P: Optional[np.ndarray] = None
        self.last_C: Optional[np.ndarray] = None
        self.last_R_grid: Optional[np.ndarray] = None
        self.last_polar_boundary: Optional[List[Dict]] = None

        # Class mappings (same IDs as original estimator)
        self.class_weights = {
            0: {'m_O': 0.0, 'm_D': 1.0},
            1: {'m_O': 1.2, 'm_D': 0.8},
            2: {'m_O': 0.5, 'm_D': 0.0},
            3: {'m_O': 0.5, 'm_D': 0.0},
            4: {'m_O': 1.5, 'm_D': 1.3},
        }
        self.class_names = {
            0: "vehicle", 1: "pedestrian", 2: "traffic light",
            3: "stop sign", 4: "emergency",
        }

        self.ego_length = 4.5
        self.ego_width = 2.0
        self.debug = False

    # --------------------------------------------------------------------- #
    #  Public helpers
    # --------------------------------------------------------------------- #
    def set_debug(self, val: bool):
        self.debug = val

    def map_object_class(self, raw_class: int) -> int:
        if raw_class in self.class_weights:
            return raw_class
        if raw_class == 0:
            return 1
        if raw_class in (1, 2, 3, 5, 7):
            return 0
        if raw_class == 9:
            return 2
        if raw_class == 11:
            return 3
        return 0

    # --------------------------------------------------------------------- #
    #  Main entry point (backward-compatible signature)
    # --------------------------------------------------------------------- #
    def calculate_boundary_risk(
        self,
        ego_speed: float,
        bounding_boxes: List,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Compute the SOTIF P×C risk field.

        Returns
        -------
        risk_field : np.ndarray, shape (M,)
            Per-sector maximum risk (for radar HUD).
        max_risk : float
            Scalar maximum over all sectors.
        threat_info : dict
            Comprehensive threat dictionary (includes 'heatmap' key with
            the full 2-D R grid for BEV overlay).
        """
        self.frame_count += 1
        t0 = time.time()

        # Delta time
        if timestamp is not None and self.last_timestamp is not None:
            delta_t = max(0.01, min(0.5, timestamp - self.last_timestamp))
        else:
            delta_t = self.default_delta_t
        self.last_delta_t = delta_t

        # --- Parse & track objects ---
        objects = self._parse_and_track(bounding_boxes, ego_speed, delta_t)

        # --- Step 1: Risk Probability P(x,y) ---
        P = self._compute_probability_field(ego_speed)

        # --- Step 2: Environment Cost C(x,y) ---
        C = self._compute_environment_cost(objects, ego_speed)

        # --- Step 3: Risk accumulation R_grid = P ⊙ C ---
        R_grid = P * C
        # Normalise by grid area so the scalar R is density-independent
        cell_area = self.grid_res ** 2
        total_risk = float(np.sum(R_grid) * cell_area)
        # Scale to human-readable range [0..~100] for alerts
        # With directional flux C_k = 0.5·m·v_approach², the raw values
        # are smaller for low closing speeds (v_approach ≈ 3 m/s) but
        # much larger for head-on (v_approach ≈ 35 m/s).  We choose
        # norm so that "1500 kg vehicle closing at 5 m/s, 10 m ahead"
        # gives a sector risk ≈ 1-2 (CAUTION/WARNING zone).
        norm = 5e-6
        R_grid *= norm
        total_risk *= norm

        # Persist for visualisation
        self.last_P = P
        self.last_C = C
        self.last_R_grid = R_grid

        # --- Step 4: Collapse to polar summary (for radar HUD) ---
        risk_field, polar_boundary = self._grid_to_polar(R_grid, objects)
        self.last_polar_boundary = polar_boundary

        max_risk = float(np.max(risk_field)) if risk_field.size > 0 else 0.0
        max_risk_idx = int(np.argmax(risk_field))
        max_risk_angle = self.angles[max_risk_idx]

        # --- Step 5: Threat info ---
        threat_info = self._build_threat_info(
            risk_field, polar_boundary, objects,
            max_risk, max_risk_angle, total_risk, ego_speed,
        )
        # Attach the 2-D heatmap for optional BEV overlay
        threat_info['heatmap'] = R_grid
        threat_info['heatmap_P'] = P
        threat_info['heatmap_C'] = C
        threat_info['grid_half'] = self.grid_half
        threat_info['grid_res'] = self.grid_res

        if timestamp is not None:
            self.last_timestamp = timestamp
        else:
            self.last_timestamp = (self.last_timestamp or 0.0) + delta_t

        comp_ms = (time.time() - t0) * 1000
        if self.debug:
            print(f"[SOTIF] frame={self.frame_count}  R_total={total_risk:.2f}  "
                  f"max_sector={max_risk:.3f}  dt={comp_ms:.1f}ms  "
                  f"objs={len(objects)}")

        return risk_field, max_risk, threat_info

    # ===================================================================== #
    #  STEP 1 — Risk Probability Model  P(x, y)
    # ===================================================================== #
    def _compute_probability_field(self, ego_speed: float) -> np.ndarray:
        """
        Eq (2)-(5) from Yao et al.

            P(x,y) = a(s) · exp( -( sqrt((x-xc)²+(y-yc)²) - R_ego )² / 2σ² )

        For simplicity we set R_ego ≈ 0 (straight driving or large radius)
        and (xc, yc) = (0, 0) — ego position.

        σ(s) = (m + k·|δ|)·s + c     (we approximate δ ≈ 0 here)
        a(s) = p · (s − v·t_la)²
        """
        v = max(ego_speed, 0.01)
        s = self.grid_s  # longitudinal coordinate on grid

        # Amplitude — peaks at s = v*t_la (the look-ahead point)
        a = self.p_coeff * (s - v * self.t_la) ** 2
        # Invert: we want *high* probability near the look-ahead, not far from it
        # Use a standard Gaussian centred on the look-ahead point instead:
        s_la = v * self.t_la
        sigma_long = max(v * 1.0, 3.0)   # longitudinal spread ≈ 1 s of travel
        sigma_lat = max(self.ego_width, 1.5)  # lateral spread ≈ lane width

        # 2-D Gaussian probability centred on (s_la, 0)
        # Use wider lateral spread for intersection scenarios (crossing vehicles)
        # Base lateral spread is lane width, but expand for intersection awareness
        sigma_lat_expanded = max(sigma_lat * 2.0, 6.0)  # At least 6m to catch crossing vehicles
        
        P = np.exp(-0.5 * ((self.grid_x - s_la) / sigma_long) ** 2
                   - 0.5 * (self.grid_y / sigma_lat_expanded) ** 2)

        # Amplitude scales with speed² (kinetic energy proxy)
        amplitude = self.p_coeff * v ** 2
        P *= amplitude

        # Zero out behind the ego (negative x)
        P[self.grid_x < -2.0] = 0.0

        return P

    # ===================================================================== #
    #  STEP 2 — Environment Cost Model  C(x, y)  [Directional Flux]
    # ===================================================================== #
    def _compute_environment_cost(
        self, objects: List[Dict], ego_speed: float,
    ) -> np.ndarray:
        """
        Directional-flux reformulation of the environment cost.

        Instead of the original scalar formulation
            C_k = 0.5 · m_k · v_k · |v_k − v_ego|          (Yao Eq 8)
        which treats all velocity differences equally regardless of
        direction, we project the relative velocity onto the
        **threat direction** (object → ego):

            v_rel  = v_obj − v_ego           (vector)
            d_hat  = (ego_pos − obj_pos) / |·|   (unit vector obj→ego)
            v_approach = max(0, v_rel · d_hat)    (approaching component)
            C_k = 0.5 · m_k · v_approach²  · w_sem

        Physical meaning:
        ─ v_approach is the closing speed *towards the ego* along the
          line connecting the two vehicles.  This is the component that
          would actually cause a collision.
        ─ A vehicle travelling parallel in the next lane has v_approach ≈ 0
          ⟹ C ≈ 0 — no more phantom alerts from overtaking traffic.
        ─ A vehicle diverging (moving away) gives v_approach < 0 which is
          clamped to 0 — it poses zero threat.
        ─ Using v_approach² (true kinetic energy of impact) instead of
          v·|Δv| better reflects crash severity physics.

        The cost blob is elongated along the **threat direction** (not the
        absolute velocity), so the heatmap "tongue" always points at the
        ego vehicle, matching intuition.

        Semantic weighting: VRU ×10, vehicle ×1, emergency ×15.
        """
        C = np.zeros((self.grid_n, self.grid_n), dtype=np.float64)

        # Ego is at origin in local frame
        ego_pos = np.array([0.0, 0.0])
        ego_vel = np.array([ego_speed, 0.0])   # +x = forward

        for obj in objects:
            cls = obj['class']
            w_sem = self._semantic_weight(cls)
            if w_sem <= 0:
                continue

            # ── Virtual mass ──
            m_k = obj['w'] * obj['h'] * 100.0
            if cls == 1:
                m_k = 80.0
            elif cls == 0:
                m_k = max(m_k, 1500.0)

            # ── Relative velocity (world frame) ──
            vel_x = obj.get('vel_x', 0.0)
            vel_y = obj.get('vel_y', 0.0)
            rel_vel = np.array([vel_x - ego_speed, vel_y])  # v_obj − v_ego

            # ── Threat direction: obj → ego (unit vector) ──
            obj_pos = np.array([obj['x'], obj['y']])
            dist = max(float(np.linalg.norm(obj_pos - ego_pos)), 0.1)
            d_hat = (ego_pos - obj_pos) / dist          # points toward ego

            # ── Approach speed (scalar, ≥ 0) ──
            v_approach = float(np.dot(rel_vel, d_hat))
            v_approach = max(v_approach, 0.0)            # clamp receding → 0

            # ── Directional kinetic cost ──
            cost = 0.5 * m_k * (v_approach ** 2) * w_sem

            # ── Presence floor for static / very slow nearby objects ──
            # A parked car 3 m ahead is still a hazard even if v_approach ≈ 0.
            # Also handle crossing vehicles (e.g., left-side red-light runners)
            # where v_approach might be low but the vehicle is still a threat.
            if cost < 1e-3:
                # Check if object is in a potentially dangerous position
                # (close enough or crossing path)
                abs_y = abs(obj['y'])
                is_crossing = abs_y > 2.0 and abs_y < 10.0  # Lateral crossing zone
                is_nearby = dist < 25.0  # Increased from 15.0 to catch crossing vehicles
                
                if is_nearby or (is_crossing and dist < 40.0):
                    # For crossing vehicles, use a distance-based cost that
                    # doesn't require high v_approach
                    if is_crossing:
                        # Crossing vehicles get higher presence floor
                        cost = w_sem * m_k * 1.0 / max(dist, 1.0)
                    else:
                        # Static/slow nearby objects
                        cost = w_sem * m_k * 0.5 / max(dist, 1.0)
                else:
                    continue

            # ── Stamp cost as anisotropic Gaussian blob ──
            ox, oy = obj['x'], obj['y']
            base_r = max(obj['w'], obj['h']) / 2.0 + 1.0

            # Elongation along the THREAT direction (obj → ego), not
            # the object's own velocity.  This means the hot-spot
            # "reaches" toward the ego, reflecting the actual danger
            # corridor.
            threat_angle = float(np.arctan2(d_hat[1], d_hat[0]))
            elong = self.elongation_factor * v_approach
            sigma_along = base_r + elong         # stretched toward ego
            sigma_perp  = base_r                 # narrow perpendicular

            # Rotate grid into threat-direction frame
            dx = self.grid_x - ox
            dy = self.grid_y - oy
            cos_t = np.cos(-threat_angle)
            sin_t = np.sin(-threat_angle)
            u = dx * cos_t - dy * sin_t          # along threat dir
            w = dx * sin_t + dy * cos_t          # perpendicular

            # Only keep the half of the blob that faces ego (u > 0).
            # The "tail" behind the object (away from ego) is irrelevant.
            u_clipped = np.maximum(u, 0.0)

            blob = np.exp(-0.5 * (u_clipped / max(sigma_along, 0.5)) ** 2
                          - 0.5 * (w / max(sigma_perp, 0.5)) ** 2)

            C += cost * blob

        return C

    # ===================================================================== #
    #  STEP 4 — Grid → Polar projection (for radar HUD)
    # ===================================================================== #
    def _grid_to_polar(
        self, R_grid: np.ndarray, objects: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Project the 2-D R_grid into M angular sectors by taking the max
        risk along each radial direction.  Also build a polar_boundary list
        compatible with the existing DataAgent drawing code.
        """
        risk_field = np.zeros(self.M)
        polar_boundary: List[Dict] = []

        for i, angle in enumerate(self.angles):
            # Sample along the ray at 1-m intervals
            max_r = 0.0
            best_dist = self.max_range
            for r in np.arange(1.0, self.max_range, self.grid_res):
                gx = r * np.cos(angle)
                gy = r * np.sin(angle)
                # Map to grid indices
                ix = int((gx + self.grid_half) / self.grid_res)
                iy = int((gy + self.grid_half) / self.grid_res)
                if 0 <= ix < self.grid_n and 0 <= iy < self.grid_n:
                    val = R_grid[iy, ix]
                    if val > max_r:
                        max_r = val
                        best_dist = r
            risk_field[i] = max_r

            # Find closest object near this angle for threat_info
            closest_obj = self._closest_object_in_sector(objects, angle)
            bp = {
                'angle': angle,
                'distance': best_dist if max_r > 0 else self.max_range,
                'radial_velocity': 0.0,
                'radial_velocity_signed': 0.0,
                'object_index': None,
                'object_class': None,
                'object_id': None,
                'lateral_offset': best_dist * np.sin(angle),
                'object_x': None,
                'object_y': None,
                'object_range_rate': 0.0,
            }
            if closest_obj is not None:
                bp.update({
                    'distance': closest_obj['distance'],
                    'radial_velocity': closest_obj.get('radial_velocity', 0.0),
                    'radial_velocity_signed': closest_obj.get('radial_velocity_signed', 0.0),
                    'object_index': closest_obj.get('id'),
                    'object_class': closest_obj['class'],
                    'object_id': closest_obj.get('id'),
                    'lateral_offset': closest_obj['y'],
                    'object_x': closest_obj['x'],
                    'object_y': closest_obj['y'],
                    'object_range_rate': closest_obj.get('range_rate', 0.0),
                })
            polar_boundary.append(bp)

        return risk_field, polar_boundary

    # ===================================================================== #
    #  Object parsing & tracking
    # ===================================================================== #
    def _parse_and_track(
        self, bounding_boxes: List, ego_speed: float, delta_t: float,
    ) -> List[Dict]:
        """Parse bounding boxes, apply EMA smoothing, return active objects."""
        detected_ids = set()
        objects: List[Dict] = []

        for bb in bounding_boxes:
            obj_x, obj_y = float(bb[0]), float(bb[1])
            obj_w, obj_h = float(bb[2]), float(bb[3])
            obj_yaw_deg = float(bb[4]) if len(bb) > 4 else 0.0
            obj_yaw = np.radians(obj_yaw_deg)
            obj_speed = float(bb[5]) if len(bb) > 5 else 0.0
            raw_class = int(bb[7]) if len(bb) > 7 else 0
            obj_class = self.map_object_class(raw_class)
            obj_id = int(bb[8]) if len(bb) > 8 and bb[8] is not None else None

            distance = float(np.hypot(obj_x, obj_y))
            vel_x = obj_speed * np.cos(obj_yaw)
            vel_y = obj_speed * np.sin(obj_yaw)
            rel_vx = vel_x - ego_speed
            rel_vy = vel_y
            ray_dir = np.array([obj_x, obj_y]) / max(distance, 1e-3)
            rv_signed = -float(np.dot(np.array([rel_vx, rel_vy]), ray_dir))

            entry = {
                'id': obj_id, 'class': obj_class,
                'x': obj_x, 'y': obj_y, 'w': obj_w, 'h': obj_h,
                'yaw': obj_yaw, 'speed': obj_speed,
                'distance': distance,
                'vel_x': vel_x, 'vel_y': vel_y,
                'radial_velocity': max(0.0, rv_signed),
                'radial_velocity_signed': rv_signed,
                'range_rate': 0.0,
            }

            # Temporal smoothing for tracked objects
            if obj_id is not None:
                detected_ids.add(obj_id)
                prev = self.tracked_objects.get(obj_id)
                if prev is not None:
                    rr_raw = (distance - prev.get('last_distance', distance)) / max(delta_t, 1e-3)
                    entry['range_rate'] = (
                        self.range_rate_alpha * rr_raw
                        + (1 - self.range_rate_alpha) * prev.get('range_rate', rr_raw)
                    )
                    entry['distance'] = (
                        self.smoothing_alpha * distance
                        + (1 - self.smoothing_alpha) * prev['distance']
                    )
                    entry['radial_velocity_signed'] = (
                        self.smoothing_alpha * rv_signed
                        + (1 - self.smoothing_alpha) * prev['radial_velocity_signed']
                    )
                    entry['radial_velocity'] = max(0.0, entry['radial_velocity_signed'])
                entry['last_distance'] = distance
                entry['ttl'] = self.track_ttl_frames
                self.tracked_objects[obj_id] = entry

            objects.append(entry)

        # Decrement TTL for missing tracked objects
        for oid in list(self.tracked_objects.keys()):
            if oid not in detected_ids:
                trk = self.tracked_objects[oid]
                trk['ttl'] -= 1
                if trk['ttl'] <= 0:
                    del self.tracked_objects[oid]

        return objects

    # ===================================================================== #
    #  Threat info assembly
    # ===================================================================== #
    def _build_threat_info(
        self,
        risk_field: np.ndarray,
        polar_boundary: List[Dict],
        objects: List[Dict],
        max_risk: float,
        max_risk_angle: float,
        total_risk: float,
        ego_speed: float,
    ) -> Dict:
        max_idx = int(np.argmax(risk_field))
        bp = polar_boundary[max_idx]

        if max_risk >= self.tau:
            risk_level = "CRITICAL"
        elif max_risk >= self.caution_tau:
            risk_level = "WARNING"
        elif max_risk >= self.caution_tau * 0.5:
            risk_level = "CAUTION"
        else:
            risk_level = "SAFE"

        primary_threat = None
        if bp['object_index'] is not None:
            tc = bp['object_class']
            primary_threat = {
                'direction_deg': np.degrees(max_risk_angle),
                'distance': bp['distance'],
                'radial_velocity': bp['radial_velocity'],
                'radial_velocity_signed': bp['radial_velocity_signed'],
                'object_class': tc,
                'class_name': self.class_names.get(tc, f"class{tc}"),
                'object_id': bp.get('object_id'),
                'object_x': bp.get('object_x'),
                'object_y': bp.get('object_y'),
                'lateral_offset': bp.get('lateral_offset'),
                'risk_contribution': max_risk,
            }

        return {
            'max_risk': max_risk,
            'risk_level': risk_level,
            'max_risk_direction_deg': np.degrees(max_risk_angle),
            'primary_threat': primary_threat,
            'total_risk': total_risk,
            'mean_risk': float(np.mean(risk_field)),
            'risky_directions': int(np.sum(risk_field > self.caution_tau)),
            'ego_speed': ego_speed,
            'frame_count': self.frame_count,
            'boundary_based': True,
            'algorithm_version': '2.0-SOTIF',
            'risk_field': risk_field,
        }

    # ===================================================================== #
    #  Internal utilities
    # ===================================================================== #
    def _semantic_weight(self, cls: int) -> float:
        if cls == 1:  # pedestrian
            return self.vru_weight
        if cls == 4:  # emergency
            return self.emergency_weight
        if cls in (2, 3):  # traffic light / stop sign
            return 0.5
        return self.vehicle_weight

    def _lateral_suppression(self, obj_y: float, cls: int) -> float:
        """Gaussian decay for objects far from the ego lane."""
        thresh = self.lateral_risk_threshold
        if cls == 1:
            thresh = self.pedestrian_lateral_threshold \
                     if hasattr(self, 'pedestrian_lateral_threshold') else 8.0
        excess = max(0.0, abs(obj_y) - thresh)
        if excess <= 0:
            return 1.0
        return float(np.exp(-(excess / self.lateral_decay_sigma) ** 2))

    def _closest_object_in_sector(
        self, objects: List[Dict], angle: float,
    ) -> Optional[Dict]:
        """Find the nearest object whose bearing is within ±half-sector of *angle*."""
        half = np.pi / self.M
        best, best_d = None, 1e9
        for obj in objects:
            bearing = np.arctan2(obj['y'], obj['x']) % (2 * np.pi)
            diff = abs(bearing - angle)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff < half and obj['distance'] < best_d:
                best = obj
                best_d = obj['distance']
        return best

    # ===================================================================== #
    #  Visualisation helpers (backward compat)
    # ===================================================================== #
    def get_risk_field_visualization_data(self, risk_field: np.ndarray) -> Dict:
        return {
            'angles_deg': np.degrees(self.angles),
            'risk_values': risk_field,
            'risk_threshold': self.tau,
            'max_risk': float(np.max(risk_field)),
            'max_risk_angle_deg': float(np.degrees(self.angles[np.argmax(risk_field)])),
        }


# ========================================================================= #
#  Quick self-test
# ========================================================================= #
def test_sotif_risk_estimator():
    print("=" * 70)
    print("  SOTIF Risk Estimator — Directional Flux C_k Test Suite")
    print("  C_k = 0.5 · m_k · v_approach² · w_sem")
    print("  where v_approach = max(0, (v_obj−v_ego) · d_hat)")
    print("=" * 70)

    est = SOTIFRiskEstimator(
        grid_half_size=30, grid_resolution=1.0,
        angular_resolution=36, max_range=30.0,
    )
    est.set_debug(True)

    # ------------------------------------------------------------------
    # Test 1: Vehicle ahead, SLOWER — closing on it (classic follow)
    # v_ego=15, v_obj=12, same lane, 10m ahead
    # Old: C = 0.5*1500*12*|12-15| = 27000
    # New: rel = (12-15, 0) = (-3, 0), d_hat toward ego = (-1, 0)
    #      v_approach = max(0, (-3)*(-1)) = 3   → C = 0.5*1500*9 = 6750
    # Both should trigger — but new is proportional to collision energy
    # ------------------------------------------------------------------
    print("\n─── Test 1: Lead vehicle 10 m ahead, slower (v=12, ego=15) ───")
    bbs = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 100]]
    rf, mx, info = est.calculate_boundary_risk(15.0, bbs, timestamp=0.0)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Should be WARNING/CRITICAL (closing at 3 m/s)")

    # ------------------------------------------------------------------
    # Test 2: Vehicle in ADJACENT lane, SAME speed, same direction
    # v_ego=15, v_obj=15, lateral offset 3.5m, 5m ahead
    # Old: C = 0.5*1500*15*|15-15| = 0  (already zero by speed diff)
    #      BUT old code had a presence fallback → some risk
    # New: rel = (15-15, 0) = (0, 0), v_approach = 0  → C = 0
    #      Presence fallback at 6m → tiny.  RADAR STAYS QUIET.
    # ------------------------------------------------------------------
    print("\n─── Test 2: Adjacent-lane vehicle, SAME speed (v=15, ego=15) ───")
    bbs = [[5.0, 3.5, 4.5, 2.0, 0.0, 15.0, 0.0, 0, 101]]
    rf, mx, info = est.calculate_boundary_risk(15.0, bbs, timestamp=0.1)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Should be SAFE (parallel, no closing speed)")

    # ------------------------------------------------------------------
    # Test 3: Vehicle in adjacent lane, FASTER — overtaking ego
    # v_ego=15, v_obj=25, lateral offset 3.5m, 2m behind
    # Old: C = 0.5*1500*25*|25-15| = 187500  ← HUGE false alarm
    # New: rel = (25-15, 0) = (10, 0), d_hat = (-2-0, 3.5-0)/norm
    #      ≈ (-0.50, 0.87) → v_approach = (10)*(-0.50) = -5.0 → clamp 0
    #      Object is overtaking from behind-left, diverging → C = 0
    # ------------------------------------------------------------------
    print("\n─── Test 3: Faster car overtaking in left lane (v=25, ego=15) ───")
    bbs = [[-2.0, 3.5, 4.5, 2.0, 0.0, 25.0, 0.0, 0, 102]]
    rf, mx, info = est.calculate_boundary_risk(15.0, bbs, timestamp=0.2)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Should be SAFE (overtaking, NOT approaching ego)")

    # ------------------------------------------------------------------
    # Test 4: Oncoming vehicle, HEAD-ON (opposite lane, toward ego)
    # v_ego=15, v_obj=20 at yaw=180° (driving toward ego), 20m ahead
    # Old: C = 0.5*1500*20*|20-15| = 75000
    # New: obj_vel = (20*cos180, 0) = (-20, 0)
    #      rel = (-20-15, 0) = (-35, 0), d_hat = (0-20, 0)/20 = (-1, 0)
    #      v_approach = (-35)*(-1) = 35  → C = 0.5*1500*35² = 918750
    #      MUCH higher than old — correctly reflects catastrophic head-on
    # ------------------------------------------------------------------
    print("\n─── Test 4: Oncoming vehicle, head-on (v=20@180°, ego=15) ───")
    bbs = [[20.0, 0.0, 4.5, 2.0, 180.0, 20.0, 0.0, 0, 103]]
    rf, mx, info = est.calculate_boundary_risk(15.0, bbs, timestamp=0.3)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Should be CRITICAL (35 m/s closing speed, head-on)")

    # ------------------------------------------------------------------
    # Test 5: Pedestrian crossing from right, perpendicular to ego
    # v_ego=12, pedestrian at (8, -2), yaw=90° (walking left), speed=1.5
    # Old: C = 0.5*80*1.5*|1.5-12|*10 = 6300
    # New: obj_vel = (1.5*cos90, 1.5*sin90) = (0, 1.5)
    #      rel = (0-12, 1.5) = (-12, 1.5), d_hat = (-8, 2)/8.25 ≈ (-0.97, 0.24)
    #      v_approach = (-12)(-0.97) + (1.5)(0.24) = 11.64 + 0.36 = 12.0
    #      C = 0.5*80*144*10 = 57600  ← much higher due to ego approach
    #      This is correct: the ego is bearing down on the pedestrian
    # ------------------------------------------------------------------
    print("\n─── Test 5: Pedestrian crossing 8 m ahead (v=1.5@90°, ego=12) ───")
    bbs = [[8.0, -2.0, 0.6, 0.6, 90.0, 1.5, 0.0, 1, 200]]
    rf, mx, info = est.calculate_boundary_risk(12.0, bbs, timestamp=0.4)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Should be WARNING/CRITICAL (ego approaching VRU)")

    # ------------------------------------------------------------------
    # Test 6: Vehicle BEHIND ego, receding (ego pulling away)
    # v_ego=20, v_obj=15, object at (-10, 0) (behind)
    # Old: C = 0.5*1500*15*|15-20| = 56250  ← false alarm
    # New: rel = (15-20, 0) = (-5, 0), d_hat = (0-(-10), 0)/10 = (1, 0)
    #      v_approach = (-5)(1) = -5 → clamp 0  → C = 0
    #      Correct: ego is faster, pulling away — zero threat.
    # ------------------------------------------------------------------
    print("\n─── Test 6: Slower vehicle 10 m BEHIND ego (v=15, ego=20) ───")
    bbs = [[-10.0, 0.0, 4.5, 2.0, 0.0, 15.0, 0.0, 0, 104]]
    rf, mx, info = est.calculate_boundary_risk(20.0, bbs, timestamp=0.5)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Should be SAFE (ego pulling away, v_approach = 0)")

    # ------------------------------------------------------------------
    # Test 7: Complex multi-object — only the real threats light up
    # ------------------------------------------------------------------
    print("\n─── Test 7: Complex scene (4 objects, only 2 are real threats) ───")
    bbs = [
        [10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 300],      # Lead car, slower → THREAT
        [5.0, 3.5, 4.5, 2.0, 0.0, 18.0, 0.0, 0, 301],        # Adjacent faster → benign
        [8.0, -2.5, 0.6, 0.6, 90.0, 1.5, 0.0, 1, 302],       # Pedestrian crossing → THREAT
        [-8.0, -3.5, 4.5, 2.0, 0.0, 15.0, 0.0, 0, 303],      # Behind, adjacent → benign
    ]
    rf, mx, info = est.calculate_boundary_risk(15.0, bbs, timestamp=0.6)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}  "
          f"total_R={info['total_risk']:.2f}")
    print(f"    → Radar should highlight ONLY the lead car + pedestrian")

    # ------------------------------------------------------------------
    # Test 8: Empty scene
    # ------------------------------------------------------------------
    print("\n─── Test 8: Empty scene ───")
    rf, mx, info = est.calculate_boundary_risk(15.0, [], timestamp=0.7)
    print(f"    max_sector={mx:.4f}  level={info['risk_level']}")
    print(f"    → Should be SAFE")

    print("\n" + "=" * 70)
    print("  All tests completed.")
    print("=" * 70)


if __name__ == "__main__":
    test_sotif_risk_estimator()
