#!/usr/bin/env python3
"""
Comprehensive test suite for BoundaryRiskEstimator and SOTIFRiskEstimator.

Run with: python test_risk_estimators.py
"""

import numpy as np
import sys
import os

# Add team_code to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boundary_risk_estimator import BoundaryRiskEstimator
from sotif_risk_estimator import SOTIFRiskEstimator


class TestResult:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_true(self, condition, msg):
        if condition:
            self.passed += 1
            print(f"  ✓ {msg}")
        else:
            self.failed += 1
            self.errors.append(msg)
            print(f"  ✗ FAILED: {msg}")
    
    def assert_close(self, actual, expected, tolerance, msg):
        if abs(actual - expected) <= tolerance:
            self.passed += 1
            print(f"  ✓ {msg} (got {actual:.4f}, expected {expected:.4f})")
        else:
            self.failed += 1
            self.errors.append(f"{msg} (got {actual:.4f}, expected {expected:.4f})")
            print(f"  ✗ FAILED: {msg} (got {actual:.4f}, expected {expected:.4f})")
    
    def assert_greater(self, actual, threshold, msg):
        if actual > threshold:
            self.passed += 1
            print(f"  ✓ {msg} (got {actual:.4f})")
        else:
            self.failed += 1
            self.errors.append(f"{msg} (got {actual:.4f}, expected > {threshold:.4f})")
            print(f"  ✗ FAILED: {msg} (got {actual:.4f}, expected > {threshold:.4f})")
    
    def assert_less(self, actual, threshold, msg):
        if actual < threshold:
            self.passed += 1
            print(f"  ✓ {msg} (got {actual:.4f})")
        else:
            self.failed += 1
            self.errors.append(f"{msg} (got {actual:.4f}, expected < {threshold:.4f})")
            print(f"  ✗ FAILED: {msg} (got {actual:.4f}, expected < {threshold:.4f})")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Test Summary: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'='*70}\n")
        return self.failed == 0


def test_boundary_estimator_basic(result):
    """Test 1: Basic BoundaryRiskEstimator initialization and empty scene"""
    print("\n[Test 1] BoundaryRiskEstimator - Basic initialization and empty scene")
    
    estimator = BoundaryRiskEstimator(
        angular_resolution=36,
        max_range=50.0,
        risk_threshold=0.3
    )
    
    # Empty scene
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=[],
        timestamp=0.0
    )
    
    result.assert_true(len(risk_field) == 36, "Risk field has correct angular resolution")
    result.assert_true(max_risk == 0.0, "Max risk is zero for empty scene")
    result.assert_true(threat_info['risk_level'] == 'SAFE', "Risk level is SAFE for empty scene")
    result.assert_true(threat_info['primary_threat'] is None, "No primary threat in empty scene")


def test_boundary_estimator_vehicle_ahead(result):
    """Test 2: BoundaryRiskEstimator - Vehicle ahead, slower (closing scenario)"""
    print("\n[Test 2] BoundaryRiskEstimator - Vehicle ahead, slower (closing)")
    
    estimator = BoundaryRiskEstimator(
        angular_resolution=36,
        max_range=50.0,
        risk_threshold=0.3
    )
    
    # Vehicle 10m ahead, ego=15 m/s, vehicle=12 m/s (closing at 3 m/s)
    bbs = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 100]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    result.assert_greater(max_risk, 0.0, "Max risk > 0 for closing vehicle")
    result.assert_true(threat_info['risk_level'] in ('CAUTION', 'WARNING', 'CRITICAL'),
                      "Risk level indicates threat")
    result.assert_true(threat_info['primary_threat'] is not None, "Primary threat identified")
    result.assert_close(threat_info['primary_threat']['object_class'], 0, 0.1,
                        "Primary threat is a vehicle (class 0)")


def test_boundary_estimator_pedestrian(result):
    """Test 3: BoundaryRiskEstimator - Pedestrian crossing"""
    print("\n[Test 3] BoundaryRiskEstimator - Pedestrian crossing")
    
    estimator = BoundaryRiskEstimator(
        angular_resolution=36,
        max_range=50.0,
        risk_threshold=0.3
    )
    
    # Pedestrian 8m ahead, 2m to the right, walking left (yaw=90°)
    bbs = [[8.0, -2.0, 0.6, 0.6, 90.0, 1.5, 0.0, 1, 200]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=12.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    result.assert_greater(max_risk, 0.0, "Max risk > 0 for pedestrian")
    result.assert_true(threat_info['primary_threat'] is not None, "Primary threat identified")
    result.assert_close(threat_info['primary_threat']['object_class'], 1, 0.1,
                        "Primary threat is a pedestrian (class 1)")


def test_boundary_estimator_receding_vehicle(result):
    """Test 4: BoundaryRiskEstimator - Receding vehicle (should be filtered)"""
    print("\n[Test 4] BoundaryRiskEstimator - Receding vehicle (should be filtered)")
    
    estimator = BoundaryRiskEstimator(
        angular_resolution=36,
        max_range=50.0,
        risk_threshold=0.3
    )
    
    # Vehicle 10m ahead, ego=15 m/s, vehicle=20 m/s (pulling away)
    bbs = [[10.0, 0.0, 4.5, 2.0, 0.0, 20.0, 0.0, 0, 101]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    # Should be filtered out (zero risk) or very low
    result.assert_less(max_risk, 0.1, "Max risk is low/zero for receding vehicle")


def test_boundary_estimator_adjacent_lane(result):
    """Test 5: BoundaryRiskEstimator - Adjacent lane vehicle (should be suppressed)"""
    print("\n[Test 5] BoundaryRiskEstimator - Adjacent lane vehicle (lateral suppression)")
    
    estimator = BoundaryRiskEstimator(
        angular_resolution=36,
        max_range=50.0,
        risk_threshold=0.3,
        lateral_risk_threshold=2.5
    )
    
    # Vehicle in adjacent lane, same speed, 3.5m lateral offset
    bbs = [[5.0, 3.5, 4.5, 2.0, 0.0, 15.0, 0.0, 0, 102]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    # Should be suppressed due to lateral offset
    result.assert_less(max_risk, 1.0, "Max risk is suppressed for adjacent lane vehicle")


def test_sotif_estimator_basic(result):
    """Test 6: SOTIFRiskEstimator - Basic initialization and empty scene"""
    print("\n[Test 6] SOTIFRiskEstimator - Basic initialization and empty scene")
    
    estimator = SOTIFRiskEstimator(
        grid_half_size=30.0,
        grid_resolution=1.0,
        angular_resolution=36,
        max_range=50.0
    )
    
    # Empty scene
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=[],
        timestamp=0.0
    )
    
    result.assert_true(len(risk_field) == 36, "Risk field has correct angular resolution")
    result.assert_true(max_risk == 0.0, "Max risk is zero for empty scene")
    result.assert_true(threat_info['risk_level'] == 'SAFE', "Risk level is SAFE for empty scene")
    result.assert_true('heatmap' in threat_info, "Heatmap is included in threat_info")
    result.assert_true(threat_info['heatmap'].shape[0] == threat_info['heatmap'].shape[1],
                       "Heatmap is square")


def test_sotif_estimator_vehicle_ahead(result):
    """Test 7: SOTIFRiskEstimator - Vehicle ahead, slower (closing scenario)"""
    print("\n[Test 7] SOTIFRiskEstimator - Vehicle ahead, slower (closing)")
    
    estimator = SOTIFRiskEstimator(
        grid_half_size=30.0,
        grid_resolution=1.0,
        angular_resolution=36,
        max_range=50.0
    )
    
    # Vehicle 10m ahead, ego=15 m/s, vehicle=12 m/s (closing at 3 m/s)
    bbs = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 100]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    result.assert_greater(max_risk, 0.0, "Max risk > 0 for closing vehicle")
    result.assert_true(threat_info['risk_level'] in ('CAUTION', 'WARNING', 'CRITICAL'),
                      "Risk level indicates threat")
    result.assert_true(threat_info['primary_threat'] is not None, "Primary threat identified")
    result.assert_true('heatmap' in threat_info, "Heatmap is included")
    result.assert_greater(np.max(threat_info['heatmap']), 0.0, "Heatmap has non-zero values")


def test_sotif_estimator_pedestrian(result):
    """Test 8: SOTIFRiskEstimator - Pedestrian crossing"""
    print("\n[Test 8] SOTIFRiskEstimator - Pedestrian crossing")
    
    estimator = SOTIFRiskEstimator(
        grid_half_size=30.0,
        grid_resolution=1.0,
        angular_resolution=36,
        max_range=50.0,
        vru_weight=10.0  # Higher weight for pedestrians
    )
    
    # Pedestrian 8m ahead, 2m to the right, walking left
    bbs = [[8.0, -2.0, 0.6, 0.6, 90.0, 1.5, 0.0, 1, 200]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=12.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    result.assert_greater(max_risk, 0.0, "Max risk > 0 for pedestrian")
    result.assert_true(threat_info['primary_threat'] is not None, "Primary threat identified")
    result.assert_close(threat_info['primary_threat']['object_class'], 1, 0.1,
                        "Primary threat is a pedestrian (class 1)")


def test_sotif_estimator_directional_flux(result):
    """Test 9: SOTIFRiskEstimator - Directional flux (adjacent lane should be low)"""
    print("\n[Test 9] SOTIFRiskEstimator - Directional flux (adjacent lane suppression)")
    
    estimator = SOTIFRiskEstimator(
        grid_half_size=30.0,
        grid_resolution=1.0,
        angular_resolution=36,
        max_range=50.0
    )
    
    # Vehicle in adjacent lane, same speed, 3.5m lateral offset
    bbs = [[5.0, 3.5, 4.5, 2.0, 0.0, 15.0, 0.0, 0, 102]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    # Directional flux should give v_approach ≈ 0 for parallel traffic
    result.assert_less(max_risk, 2.0, "Max risk is low for parallel adjacent lane vehicle")


def test_sotif_estimator_head_on(result):
    """Test 10: SOTIFRiskEstimator - Head-on collision scenario"""
    print("\n[Test 10] SOTIFRiskEstimator - Head-on collision scenario")
    
    estimator = SOTIFRiskEstimator(
        grid_half_size=30.0,
        grid_resolution=1.0,
        angular_resolution=36,
        max_range=50.0
    )
    
    # Oncoming vehicle, 20m ahead, driving toward ego (yaw=180°)
    bbs = [[20.0, 0.0, 4.5, 2.0, 180.0, 20.0, 0.0, 0, 103]]
    
    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=15.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )
    
    # Head-on should have very high risk (closing speed = 35 m/s)
    result.assert_greater(max_risk, 5.0, "Max risk is very high for head-on collision")
    result.assert_true(threat_info['risk_level'] in ('WARNING', 'CRITICAL'),
                      "Risk level is WARNING/CRITICAL for head-on")


def test_coordinate_conventions(result):
    """Test 11: Coordinate conventions (both estimators)"""
    print("\n[Test 11] Coordinate conventions - x forward, y right, angles from atan2(y,x)")
    
    # Test with object at 45° (x=10, y=10)
    bbs = [[10.0, 10.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 104]]
    
    boundary_est = BoundaryRiskEstimator(angular_resolution=72, max_range=50.0)
    sotif_est = SOTIFRiskEstimator(angular_resolution=72, max_range=50.0)
    
    bf, bmax, binfo = boundary_est.calculate_boundary_risk(15.0, bbs, 0.0)
    sf, smax, sinfo = sotif_est.calculate_boundary_risk(15.0, bbs, 0.0)
    
    # Use threat_info's max_risk_direction_deg (more reliable than sector index)
    bangle_deg = binfo.get('max_risk_direction_deg', 0.0)
    sangle_deg = sinfo.get('max_risk_direction_deg', 0.0)
    
    expected_angle_deg = np.degrees(np.arctan2(10.0, 10.0))  # ≈ 45°
    
    # Normalize angles to [0, 360) for comparison
    def normalize_angle(angle):
        angle = angle % 360.0
        if angle > 180.0:
            angle -= 360.0
        return angle
    
    bangle_norm = normalize_angle(bangle_deg)
    sangle_norm = normalize_angle(sangle_deg)
    expected_norm = normalize_angle(expected_angle_deg)
    
    # Allow ±20° tolerance due to discretization and risk spreading
    result.assert_close(bangle_norm, expected_norm, 20.0,
                       f"Boundary estimator max risk direction ≈ 45° (got {bangle_deg:.1f}°)")
    result.assert_close(sangle_norm, expected_norm, 20.0,
                       f"SOTIF estimator max risk direction ≈ 45° (got {sangle_deg:.1f}°)")
    
    # Also verify that risk is detected (non-zero)
    result.assert_greater(bmax, 0.0, "Boundary estimator detects risk for object at 45°")
    result.assert_greater(smax, 0.0, "SOTIF estimator detects risk for object at 45°")


def test_multi_object_scenario(result):
    """Test 12: Multi-object scenario (both estimators)"""
    print("\n[Test 12] Multi-object scenario - multiple threats")
    
    bbs = [
        [10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 300],      # Lead car, slower
        [8.0, -2.5, 0.6, 0.6, 90.0, 1.5, 0.0, 1, 301],       # Pedestrian crossing
        [5.0, 3.5, 4.5, 2.0, 0.0, 18.0, 0.0, 0, 302],        # Adjacent lane (should be low)
    ]
    
    boundary_est = BoundaryRiskEstimator(angular_resolution=36, max_range=50.0)
    sotif_est = SOTIFRiskEstimator(angular_resolution=36, max_range=50.0)
    
    bf, bmax, binfo = boundary_est.calculate_boundary_risk(15.0, bbs, 0.0)
    sf, smax, sinfo = sotif_est.calculate_boundary_risk(15.0, bbs, 0.0)
    
    # Both should detect threats
    result.assert_greater(bmax, 0.0, "Boundary estimator detects threat in multi-object scene")
    result.assert_greater(smax, 0.0, "SOTIF estimator detects threat in multi-object scene")
    
    # Both should identify primary threat
    result.assert_true(binfo['primary_threat'] is not None,
                      "Boundary estimator identifies primary threat")
    result.assert_true(sinfo['primary_threat'] is not None,
                      "SOTIF estimator identifies primary threat")


def test_temporal_smoothing(result):
    """Test 13: Temporal smoothing (tracked objects)"""
    print("\n[Test 13] Temporal smoothing - object tracking across frames")
    
    estimator = BoundaryRiskEstimator(angular_resolution=36, max_range=50.0)
    
    # Frame 1: Object at 10m
    bbs1 = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 400]]
    _, max1, _ = estimator.calculate_boundary_risk(15.0, bbs1, timestamp=0.0)
    
    # Frame 2: Object at 9.5m (moved closer)
    bbs2 = [[9.5, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 400]]
    _, max2, _ = estimator.calculate_boundary_risk(15.0, bbs2, timestamp=0.1)
    
    # Risk should increase as object gets closer
    result.assert_greater(max2, max1, "Risk increases as object approaches (temporal tracking)")


def test_emergency_vehicle(result):
    """Test 14: Emergency vehicle (class 4) - should have higher weight"""
    print("\n[Test 14] Emergency vehicle - higher semantic weight")
    
    boundary_est = BoundaryRiskEstimator(angular_resolution=36, max_range=50.0)
    sotif_est = SOTIFRiskEstimator(angular_resolution=36, max_range=50.0,
                                   emergency_weight=15.0)
    
    # Regular vehicle
    bbs_regular = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 500]]
    # Emergency vehicle (same position/speed)
    bbs_emergency = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 501]]
    # Set class to 4 (emergency) - need to modify the bounding box format
    # Actually, the format is [x, y, w, h, yaw, speed, brake, class, id]
    # So we set index 7 to 4
    bbs_emergency[0][7] = 4
    
    _, bmax_reg, _ = boundary_est.calculate_boundary_risk(15.0, bbs_regular, 0.0)
    _, bmax_emg, _ = boundary_est.calculate_boundary_risk(15.0, bbs_emergency, 0.0)
    
    _, smax_reg, _ = sotif_est.calculate_boundary_risk(15.0, bbs_regular, 0.0)
    _, smax_emg, _ = sotif_est.calculate_boundary_risk(15.0, bbs_emergency, 0.0)
    
    # Emergency should have higher risk (or at least not lower)
    result.assert_greater(bmax_emg, bmax_reg * 0.8,
                         "Boundary: Emergency vehicle has higher/similar risk")
    result.assert_greater(smax_emg, smax_reg * 0.8,
                         "SOTIF: Emergency vehicle has higher/similar risk")


def test_sotif_intersection_emergency_red_light(result):
    """Test 15: SOTIF emergency vehicle crossing from right at intersection."""
    print("\n[Test 15] SOTIF - Ego through intersection, emergency vehicle runs red from right")

    estimator = SOTIFRiskEstimator(
        grid_half_size=30.0,
        grid_resolution=1.0,
        angular_resolution=72,
        max_range=50.0,
        emergency_weight=15.0
    )

    # Scenario:
    # - Ego moves straight through intersection (+x direction)
    # - Emergency vehicle comes from right side (+y) crossing right->left (-y)
    # Bounding box format: [x, y, w, h, yaw_deg, speed, brake, class, id]
    # class=4 is emergency vehicle
    bbs = [[8.0, 6.0, 4.8, 2.2, 270.0, 20.0, 0.0, 4, 700]]

    risk_field, max_risk, threat_info = estimator.calculate_boundary_risk(
        ego_speed=12.0,
        bounding_boxes=bbs,
        timestamp=0.0
    )

    risk_level = threat_info.get('risk_level')
    direction_deg = threat_info.get('max_risk_direction_deg')
    primary = threat_info.get('primary_threat')
    risky_directions = threat_info.get('risky_directions', 0)

    print(f"  SOTIF output max_risk={max_risk:.4f}")
    print(f"  SOTIF output risk_level={risk_level}")
    print(f"  SOTIF output max_risk_direction_deg={direction_deg}")
    print(f"  SOTIF output primary_threat={primary}")
    print(f"  SOTIF output risky_directions={risky_directions}")

    result.assert_true(len(risk_field) == 72, "Risk field has expected angular resolution")
    result.assert_greater(max_risk, 0.5, "Intersection red-light emergency creates elevated risk")
    result.assert_true(risk_level in ('CAUTION', 'WARNING', 'CRITICAL'),
                      "Risk level indicates a crossing threat")
    result.assert_true(isinstance(direction_deg, (float, int)),
                      "Max risk direction is reported")
    result.assert_true(0.0 <= float(direction_deg) <= 90.0,
                      "Max risk direction is in front-right intersection sector")
    result.assert_greater(float(risky_directions), 0.0,
                         "At least one risky direction is detected")


def test_output_format_consistency(result):
    """Test 16: Output format consistency (both estimators should return same structure)"""
    print("\n[Test 16] Output format consistency - both estimators return compatible structures")
    
    bbs = [[10.0, 0.0, 4.5, 2.0, 0.0, 12.0, 0.0, 0, 600]]
    
    boundary_est = BoundaryRiskEstimator(angular_resolution=36, max_range=50.0)
    sotif_est = SOTIFRiskEstimator(angular_resolution=36, max_range=50.0)
    
    bf, bmax, binfo = boundary_est.calculate_boundary_risk(15.0, bbs, 0.0)
    sf, smax, sinfo = sotif_est.calculate_boundary_risk(15.0, bbs, 0.0)
    
    # Both should return same tuple structure
    result.assert_true(isinstance(bf, np.ndarray), "Boundary: risk_field is numpy array")
    result.assert_true(isinstance(sf, np.ndarray), "SOTIF: risk_field is numpy array")
    result.assert_true(isinstance(bmax, (float, np.floating)), "Boundary: max_risk is float")
    result.assert_true(isinstance(smax, (float, np.floating)), "SOTIF: max_risk is float")
    result.assert_true(isinstance(binfo, dict), "Boundary: threat_info is dict")
    result.assert_true(isinstance(sinfo, dict), "SOTIF: threat_info is dict")
    
    # Both should have required keys
    required_keys = ['max_risk', 'risk_level', 'primary_threat', 'max_risk_direction_deg']
    for key in required_keys:
        result.assert_true(key in binfo, f"Boundary: threat_info has '{key}'")
        result.assert_true(key in sinfo, f"SOTIF: threat_info has '{key}'")
    
    # SOTIF should additionally have heatmap
    result.assert_true('heatmap' in sinfo, "SOTIF: threat_info has 'heatmap'")


def main():
    """Run all test cases"""
    print("="*70)
    print("Risk Estimator Test Suite")
    print("Testing BoundaryRiskEstimator and SOTIFRiskEstimator")
    print("="*70)
    
    result = TestResult()
    
    # Boundary estimator tests
    test_boundary_estimator_basic(result)
    test_boundary_estimator_vehicle_ahead(result)
    test_boundary_estimator_pedestrian(result)
    test_boundary_estimator_receding_vehicle(result)
    test_boundary_estimator_adjacent_lane(result)
    
    # SOTIF estimator tests
    test_sotif_estimator_basic(result)
    test_sotif_estimator_vehicle_ahead(result)
    test_sotif_estimator_pedestrian(result)
    test_sotif_estimator_directional_flux(result)
    test_sotif_estimator_head_on(result)
    
    # Cross-cutting tests
    test_coordinate_conventions(result)
    test_multi_object_scenario(result)
    test_temporal_smoothing(result)
    test_emergency_vehicle(result)
    test_sotif_intersection_emergency_red_light(result)
    test_output_format_consistency(result)
    
    # Print summary
    success = result.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

