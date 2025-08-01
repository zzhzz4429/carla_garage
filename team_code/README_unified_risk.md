# Unified Risk Management System

A comprehensive risk assessment system that combines external TTC (Time-To-Collision) risk with internal driver state risk for autonomous driving applications.

## Overview

The system integrates three main components:

1. **External Risk Assessment**: TTC-based collision risk calculation using CARLA sensor data
2. **Internal Risk Assessment**: Driver state monitoring via UDP from AGX Orin device
3. **Unified Risk Management**: Adaptive fusion of both risk factors with multi-modal alerts

## System Architecture

```
┌─────────────────┐    UDP    ┌──────────────────────┐
│   AGX Orin      │──────────▶│  Unified Risk        │
│ Driver Detection│           │     Manager          │
└─────────────────┘           │                      │
                              │  ┌─────────────────┐ │    ┌─────────────┐
┌─────────────────┐           │  │ Safety Evaluator│ │───▶│ Audio/Visual│
│ CARLA Sensors   │──────────▶│  │  (TTC Risk)     │ │    │   Alerts    │
│  (Perception)   │           │  └─────────────────┘ │    └─────────────┘
└─────────────────┘           │                      │
                              │  ┌─────────────────┐ │    ┌─────────────┐
                              │  │ Risk Fusion &   │ │───▶│ Control     │
                              │  │ Alert Logic     │ │    │ Modifications│
                              │  └─────────────────┘ │    └─────────────┘
                              └──────────────────────┘
```

## Files Description

### Core Components

- **`unified_risk_manager.py`**: Main risk fusion system
- **`safety_evaluator.py`**: External TTC risk calculation (existing)
- **`test_agx_orin_simulator.py`**: AGX Orin driver detection simulator
- **`unified_risk_integration_example.py`**: Integration example with CARLA

### Key Features

#### Driver State Recognition
- **safe_driving**: Risk = 0.0
- **sleepy**: Risk = 0.7
- **reaching_back**: Risk = 0.5
- **using_phone**: Risk = 0.8

#### Risk Levels
- **safe** (< 0.2): No alerts
- **caution** (0.2-0.4): Gentle alerts every 10s
- **warning** (0.4-0.6): Moderate alerts every 5s
- **critical** (0.6-0.8): Strong alerts every 2s
- **emergency** (> 0.8): Immediate intervention every 1s

#### Alert Systems
- **Visual**: Console messages with emojis and color coding
- **Audio**: Pygame-based sound alerts with fallback to system beeps
- **TTS**: Text-to-speech for critical situations
- **Control**: Automatic throttle/brake adjustments

## Usage Instructions

### 1. Basic Testing

Start the unified risk manager:
```bash
cd team_code
python unified_risk_manager.py
```

In another terminal, start the AGX Orin simulator:
```bash
python test_agx_orin_simulator.py --mode scenario --scenario mixed_risks
```

### 2. Integration Example

Run the integration example:
```bash
python unified_risk_integration_example.py
```

### 3. AGX Orin Simulator Options

#### Continuous Simulation
```bash
# Run for 60 seconds at 2Hz
python test_agx_orin_simulator.py --mode simulation --duration 60 --frequency 2.0

# Run indefinitely
python test_agx_orin_simulator.py --mode simulation --duration 0
```

#### Predefined Scenarios
```bash
# Normal driving
python test_agx_orin_simulator.py --mode scenario --scenario normal_driving

# Driver getting sleepy
python test_agx_orin_simulator.py --mode scenario --scenario getting_sleepy

# Phone usage
python test_agx_orin_simulator.py --mode scenario --scenario phone_usage

# Mixed risks
python test_agx_orin_simulator.py --mode scenario --scenario mixed_risks
```

#### Manual Testing
```bash
python test_agx_orin_simulator.py --mode manual
# Then enter states like: sleepy 0.8, using_phone 0.9, safe_driving 0.95
```

### 4. UDP Message Format

The AGX Orin device should send JSON messages to UDP port 9999:

```json
{
  "driver_state": "sleepy",
  "confidence": 0.85,
  "timestamp": 1234567890.123
}
```

### 5. Integration with Your CARLA Agent

Replace the placeholder functions in `unified_risk_integration_example.py`:

```python
def get_ego_speed(self, input_data):
    # Extract speed from your sensor data
    velocity = input_data.ego_vehicle.get_velocity()
    return np.sqrt(velocity.x**2 + velocity.y**2)

def get_detected_objects(self, input_data):
    # Extract bounding boxes from your perception system
    # Return format: [x, y, z, extent_x, extent_y, extent_z, rotation, class]
    return your_perception_output
```

## Risk Fusion Algorithm

### Adaptive Weighting
- **Normal conditions**: 60% external, 40% internal
- **High external risk**: 75% external, 25% internal  
- **High internal risk**: 40% external, 60% internal
- **Stale driver data**: 80% external, 20% internal

### Synergistic Amplification
When both external and internal risks are > 0.5:
```
amplified_risk = base_risk × (1.0 + external_risk × internal_risk × 0.5)
```

### Emergency Vehicle Handling
- 20% risk boost when driver is inattentive + emergency vehicle present
- Automatic yielding behavior in control commands

## Audio Requirements

Install audio dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-pygame espeak

# For better TTS (optional)
sudo apt-get install festival
```

## Research Applications

This system is designed for research papers combining:
- External collision risk assessment
- Internal driver state monitoring  
- Adaptive multi-modal risk fusion
- Real-time intervention strategies

### Experimental Metrics
- Risk correlation analysis
- Intervention effectiveness
- Driver behavior adaptation
- Multi-modal alert comparison

### Data Collection
The system logs:
- Unified risk scores over time
- Driver state transitions
- Critical event triggers
- Alert response patterns

## Troubleshooting

### Common Issues

1. **UDP not receiving data**:
   - Check firewall settings
   - Verify port 9999 is available
   - Test with manual mode first

2. **Audio not working**:
   - Install pygame: `pip install pygame`
   - Test audio system: `pygame.mixer.init()`
   - Use fallback beep patterns

3. **TTC calculations unstable**:
   - Check delta_t timing
   - Verify object detection format
   - Use simplified TTC method

### Debug Mode

Enable detailed logging:
```python
config.debug = True
```

This will show:
- Frame-by-frame TTC calculations
- Object tracking details
- Risk fusion steps
- Alert triggers

## Configuration

Key parameters in `UnifiedRiskManager`:

```python
# Risk thresholds
self.risk_thresholds = {
    'safe': 0.2,
    'caution': 0.4,
    'warning': 0.6,
    'critical': 0.8,
    'emergency': 0.95
}

# Driver state timeout
self.driver_state_timeout = 5.0  # seconds

# Alert frequencies (Hz)
self.alert_config = {
    'caution': {'frequency': 10},
    'warning': {'frequency': 5},
    'critical': {'frequency': 2}, 
    'emergency': {'frequency': 1}
}
```

## Future Extensions

- **Haptic feedback**: Steering wheel vibration alerts
- **Vehicle communication**: V2V emergency coordination
- **Predictive modeling**: Machine learning for risk prediction
- **Personalization**: Adaptive thresholds per driver
- **Multi-sensor fusion**: Camera + radar + lidar integration

## Citation

If you use this system in research, please cite:
```
[Your paper title on unified external-internal risk assessment]
[Conference/Journal, Year]
``` 