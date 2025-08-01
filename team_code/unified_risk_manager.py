import socket
import json
import threading
import time
import numpy as np
import os
from typing import Dict, Optional, Tuple
import subprocess
import pygame
from safety_evaluator import SafetyEvaluator

class UnifiedRiskManager:
    def __init__(self, config, udp_port=9999, audio_enabled=True, disable_external_risk=False):
        """
        Unified Risk Manager combining external (TTC) and internal (driver state) risks
        
        Args:
            config: Configuration object
            udp_port: UDP port to listen for AGX Orin driver state data
            audio_enabled: Whether to enable audio alerts
            disable_external_risk: Set to True to disable external risk for experiments
        """
        self.config = config
        self.udp_port = udp_port
        self.audio_enabled = audio_enabled
        
        # EXPERIMENT CONTROL: Disable external risk if environment variable OR parameter is set
        env_disabled = os.getenv('DISABLE_EXTERNAL_RISK', 'false').lower() == 'true'
        self.external_risk_enabled = not (disable_external_risk or env_disabled)
        
        if not self.external_risk_enabled:
            disable_reason = "parameter" if disable_external_risk else "DISABLE_EXTERNAL_RISK environment variable"
            print(f"🔇 EXTERNAL RISK DISABLED via {disable_reason}")
        
        # Initialize safety evaluator for external risk
        self.safety_evaluator = SafetyEvaluator(config)
        
        # Driver state risk mapping
        self.driver_state_risks = {
            'safe_driving': 0.0,    # No internal risk
            'sleepy': 0.7,          # High internal risk
            'reaching_back': 0.5,   # Medium internal risk  
            'using_phone': 0.8      # Very high internal risk
        }
        
        # Risk thresholds for different alert levels
        self.risk_thresholds = {
            'safe': 0.2,        # Below this = safe
            'caution': 0.4,     # Caution level
            'warning': 0.6,     # Warning level
            'critical': 0.8,    # Critical level
            'emergency': 0.95   # Emergency intervention
        }
        
        # Alert configuration
        self.alert_config = {
            'safe': {'audio': None, 'frequency': 0},
            'caution': {'audio': 'caution.wav', 'frequency': 10},     # Every 10 seconds
            'warning': {'audio': 'warning.wav', 'frequency': 5},      # Every 5 seconds
            'critical': {'audio': 'critical.wav', 'frequency': 10},    # Every 2 seconds
            'emergency': {'audio': 'emergency.wav', 'frequency': 10}   # Every 1 second
        }
        
        # Internal state variables
        self.current_driver_state = 'safe_driving'
        self.driver_state_confidence = 0.0
        self.last_driver_update = 0.0
        self.driver_state_timeout = 5.0  # seconds - consider driver state stale after 5s
        
        # Risk history for temporal analysis
        self.risk_history = []
        self.max_history_length = 50  # Keep 5 seconds at 10Hz
        
        # Alert timing
        self.last_alert_time = 0.0
        self.alert_suppression_time = 0.5  # Minimum time between alerts
        
        # UDP listener thread
        self.udp_thread = None
        self.running = False
        
        # Initialize audio system
        if self.audio_enabled:
            try:
                pygame.mixer.init()
                print("Audio system initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize audio system: {e}")
                self.audio_enabled = False
        
        # TTS process management for non-blocking speech
        self.tts_process = None
        self.tts_lock = threading.Lock()
        
        # Start UDP listener
        self.start_udp_listener()
        
    def start_udp_listener(self):
        """Start the UDP listener thread for AGX Orin driver state data"""
        self.running = True
        self.udp_thread = threading.Thread(target=self._udp_listener_worker, daemon=True)
        self.udp_thread.start()
        print(f"UDP listener started on port {self.udp_port}")
        
    def stop_udp_listener(self):
        """Stop the UDP listener thread"""
        self.running = False
        if self.udp_thread:
            self.udp_thread.join(timeout=2.0)
        print("UDP listener stopped")
        
    def _udp_listener_worker(self):
        """Worker function for UDP listener thread"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)  # 1 second timeout for responsive shutdown
        
        try:
            sock.bind(('', self.udp_port))
            print(f"UDP socket bound to port {self.udp_port}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)  # Buffer size 1024 bytes
                    self._process_driver_state_data(data, addr)
                except socket.timeout:
                    continue  # Normal timeout, check if still running
                except Exception as e:
                    print(f"UDP listener error: {e}")
                    
        except Exception as e:
            print(f"Failed to bind UDP socket: {e}")
        finally:
            sock.close()
            
    def _process_driver_state_data(self, data, addr):
        """Process incoming driver state data from AGX Orin"""
        try:
            # Parse JSON data
            message = json.loads(data.decode('utf-8'))
            
            # Expected format:
            # {
            #   "driver_state": "sleepy",  # or "safe_driving", "reaching_back", "using_phone"
            #   "confidence": 0.85,       # Confidence score 0-1
            #   "timestamp": 1234567890.123
            # }
            
            driver_state = message.get('driver_state', 'safe_driving')
            confidence = message.get('confidence', 0.0)
            timestamp = message.get('timestamp', time.time())
            
            # Validate driver state
            if driver_state not in self.driver_state_risks:
                print(f"Warning: Unknown driver state '{driver_state}', defaulting to 'safe_driving'")
                driver_state = 'safe_driving'
                
            # Update internal state
            self.current_driver_state = driver_state
            self.driver_state_confidence = confidence
            self.last_driver_update = timestamp
            
            print(f"Driver state update: {driver_state} (confidence: {confidence:.2f}) from {addr[0]}")
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse driver state JSON: {e}")
        except Exception as e:
            print(f"Error processing driver state data: {e}")
            
    def get_driver_state_risk(self) -> Tuple[float, bool]:
        """
        Get current driver state risk with staleness check
        
        Returns:
            tuple: (risk_value, is_stale) where risk_value is 0-1 and is_stale indicates outdated data
        """
        current_time = time.time()
        is_stale = (current_time - self.last_driver_update) > self.driver_state_timeout
        
        if is_stale:
            # If driver state data is stale, use conservative approach
            print(f"Warning: Driver state data is stale ({current_time - self.last_driver_update:.1f}s old)")
            return 0.3, True  # Moderate risk when uncertain
        
        # Get risk based on current driver state and confidence
        base_risk = self.driver_state_risks[self.current_driver_state]
        confidence_adjusted_risk = base_risk * self.driver_state_confidence
        
        return confidence_adjusted_risk, False
        
    def calculate_adaptive_weights(self, external_risk: float, internal_risk: float, 
                                 is_driver_data_stale: bool) -> Tuple[float, float]:
        """
        Calculate adaptive weights for external vs internal risk based on context
        
        Args:
            external_risk: TTC-based external risk (0-1)
            internal_risk: Driver state internal risk (0-1) 
            is_driver_data_stale: Whether driver data is outdated
            
        Returns:
            tuple: (external_weight, internal_weight) that sum to 1.0
        """
        # Base weights
        base_external_weight = 0.6  # External risk is primary in normal conditions
        base_internal_weight = 0.4  # Internal risk is secondary
        
        # Adaptive adjustments
        if is_driver_data_stale:
            # If driver data is stale, rely more on external risk
            external_weight = 0.8
            internal_weight = 0.2
        elif external_risk > 0.7:
            # High external risk - prioritize external factors
            external_weight = 0.75
            internal_weight = 0.25
        elif internal_risk > 0.6:
            # High internal risk - prioritize driver state
            external_weight = 0.4
            internal_weight = 0.6
        else:
            # Normal conditions - use base weights
            external_weight = base_external_weight
            internal_weight = base_internal_weight
            
        return external_weight, internal_weight
        
    def calculate_unified_risk(self, ego_speed: float, bounding_boxes: list) -> Dict:
        """
        Calculate unified risk combining external and internal factors
        
        Args:
            ego_speed: Current ego vehicle speed (m/s)
            bounding_boxes: Detected objects for external risk assessment
            
        Returns:
            dict: Comprehensive risk assessment data
        """
        current_time = time.time()
        
        # Store current ego speed for alert filtering
        self._current_ego_speed = ego_speed
        
        # 1. Calculate external risk (TTC-based) - ALWAYS for measurement, but disable for alerts if needed
        external_risk_raw, external_breakdown = self.safety_evaluator.calculate_external_risk_unified(
            ego_speed, bounding_boxes
        )
        
        if self.external_risk_enabled:
            # Normal mode: Use calculated external risk for alerts
            external_risk = external_risk_raw
            print("✅ External risk: Calculated and USED for alerts")
        else:
            # EXPERIMENT MODE: Calculate for measurement but disable for alerts
            external_risk = 0.0
            external_breakdown['disabled_for_alerts'] = True
            print(f"📊 External risk: Calculated ({external_risk_raw:.3f}) but DISABLED for alerts")
        
        # 2. Calculate internal risk (driver state)
        internal_risk, is_driver_data_stale = self.get_driver_state_risk()
        
        # 3. Calculate adaptive weights
        external_weight, internal_weight = self.calculate_adaptive_weights(
            external_risk, internal_risk, is_driver_data_stale
        )
        
        # 4. Combine risks with adaptive weighting
        base_unified_risk = (external_weight * external_risk + 
                           internal_weight * internal_risk)
        
        # 5. Apply synergistic risk amplification
        # When both external and internal risks are high, the combined risk should be higher
        synergy_factor = 1.0
        if external_risk > 0.5 and internal_risk > 0.5:
            # Amplify risk when both conditions are concerning
            synergy_multiplier = 1.0 + (external_risk * internal_risk * 0.5)
            synergy_factor = synergy_multiplier
            base_unified_risk *= synergy_multiplier
            
        # 6. Emergency vehicle special handling
        emergency_boost = 0.0
        if external_breakdown:
            primary_threat = external_breakdown.get('primary_threat')
            if primary_threat and primary_threat.get('is_emergency', False):
                if internal_risk > 0.3:  # Driver not fully attentive
                    emergency_boost = 0.2  # Significant boost for emergency + inattentive driver
                    base_unified_risk += emergency_boost
                
        # Cap at 1.0
        unified_risk = min(base_unified_risk, 1.0)
        
        # 7. Determine risk level
        if unified_risk >= self.risk_thresholds['emergency']:
            risk_level = 'emergency'
        elif unified_risk >= self.risk_thresholds['critical']:
            risk_level = 'critical'
        elif unified_risk >= self.risk_thresholds['warning']:
            risk_level = 'warning'
        elif unified_risk >= self.risk_thresholds['caution']:
            risk_level = 'caution'
        else:
            risk_level = 'safe'
            
        # 8. Add to risk history for temporal analysis
        risk_entry = {
            'timestamp': current_time,
            'unified_risk': unified_risk,
            'external_risk': external_risk,
            'internal_risk': internal_risk,
            'risk_level': risk_level
        }
        
        self.risk_history.append(risk_entry)
        if len(self.risk_history) > self.max_history_length:
            self.risk_history.pop(0)
            
        # 9. Compile comprehensive result
        result = {
            'unified_risk': unified_risk,
            'external_risk': external_risk,  # Used for alerts (may be 0.0 if disabled)
            'external_risk_raw': external_risk_raw,  # Always calculated for measurement
            'internal_risk': internal_risk,
            'risk_level': risk_level,
            'experiment_mode': {
                'external_risk_enabled': self.external_risk_enabled,
                'external_calculated_but_disabled': not self.external_risk_enabled
            },
            'weights': {
                'external_weight': external_weight,
                'internal_weight': internal_weight
            },
            'driver_state': {
                'current_state': self.current_driver_state,
                'confidence': self.driver_state_confidence,
                'is_stale': is_driver_data_stale,
                'last_update': self.last_driver_update
            },
            'risk_factors': {
                'base_unified_risk': base_unified_risk,
                'synergy_factor': synergy_factor,
                'emergency_boost': emergency_boost
            },
            'external_breakdown': external_breakdown,
            'temporal_context': self._analyze_risk_trend(),
            'timestamp': current_time
        }
        
        # 10. Trigger alerts based on risk level
        self._handle_risk_alert(result)
        
        # Debug output
        print(f"=== UNIFIED RISK ASSESSMENT ===")
        if not self.external_risk_enabled:
            print(f"External Risk: {external_risk:.3f} (calculated: {external_risk_raw:.3f}, DISABLED for alerts)")
        else:
            print(f"External Risk: {external_risk:.3f} (weight: {external_weight:.2f})")
        print(f"Internal Risk: {internal_risk:.3f} (weight: {internal_weight:.2f})")
        print(f"Driver State: {self.current_driver_state} (conf: {self.driver_state_confidence:.2f})")
        print(f"Unified Risk: {unified_risk:.3f}")
        print(f"Risk Level: {risk_level.upper()}")
        if synergy_factor > 1.0:
            print(f"Synergy Amplification: {synergy_factor:.2f}x")
        if emergency_boost > 0:
            print(f"Emergency Boost: +{emergency_boost:.2f}")
        print("===============================")
        
        return result
        
    def _analyze_risk_trend(self) -> Dict:
        """Analyze temporal risk trends for predictive insights"""
        if len(self.risk_history) < 3:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'volatility': 0.0}
            
        # Calculate risk trend over recent history
        recent_risks = [entry['unified_risk'] for entry in self.risk_history[-10:]]
        recent_times = [entry['timestamp'] for entry in self.risk_history[-10:]]
        
        # Simple linear regression for trend
        n = len(recent_risks)
        sum_x = sum(range(n))
        sum_y = sum(recent_risks)
        sum_xy = sum(i * risk for i, risk in enumerate(recent_risks))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if n * sum_x2 - sum_x * sum_x != 0 else 0
        
        # Risk volatility (standard deviation)
        mean_risk = sum_y / n
        volatility = np.sqrt(sum((risk - mean_risk) ** 2 for risk in recent_risks) / n)
        
        # Determine trend
        if slope > 0.02:
            trend = 'increasing'
        elif slope < -0.02:
            trend = 'decreasing'
        else:
            trend = 'stable'
            
        return {
            'trend': trend,
            'slope': slope,
            'volatility': volatility,
            'recent_mean': mean_risk
        }
        
    def _handle_risk_alert(self, risk_assessment: Dict):
        """Handle alert generation based on unified risk assessment with speed-based filtering"""
        current_time = time.time()
        risk_level = risk_assessment['risk_level']
        unified_risk = risk_assessment['unified_risk']
        external_risk = risk_assessment['external_risk']
        internal_risk = risk_assessment['internal_risk']
        
        # Get current ego speed (stored during risk calculation)
        ego_speed_ms = getattr(self, '_current_ego_speed', 0.0)  # m/s
        ego_speed_kmh = ego_speed_ms * 3.6  # Convert to km/h
        
        # Check if we should trigger an alert
        alert_config = self.alert_config[risk_level]
        alert_frequency = alert_config['frequency']
        
        if alert_frequency == 0:  # No alerts for safe level
            return
            
        # SPEED-BASED INTERNAL ALERT SUPPRESSION
        # If speed ≤ 30 km/h AND risk is primarily internal (not external), suppress alert
        speed_threshold_kmh = 30.0
        
        if ego_speed_kmh <= speed_threshold_kmh:
            # Check if this is primarily an internal risk alert
            if internal_risk > external_risk and external_risk < 0.25:
                print(f"🔇 Alert suppressed: Low speed ({ego_speed_kmh:.1f} km/h) + internal risk dominant")
                print(f"   Risk breakdown: External={external_risk:.3f}, Internal={internal_risk:.3f}")
                return  # Suppress internal-only alerts at low speeds
            elif external_risk < 0.15:
                print(f"🔇 Alert suppressed: Low speed ({ego_speed_kmh:.1f} km/h) + minimal external risk")
                print(f"   Risk breakdown: External={external_risk:.3f}, Internal={internal_risk:.3f}")
                return  # Suppress all alerts if very low external risk at low speeds
            else:
                print(f"⚠️ Alert allowed: Low speed but significant external risk ({external_risk:.3f})")
                print(f"   Speed: {ego_speed_kmh:.1f} km/h, External: {external_risk:.3f}, Internal: {internal_risk:.3f}")
        
        # Check timing constraints
        time_since_last_alert = current_time - self.last_alert_time
        min_interval = 1.0 / alert_frequency if alert_frequency > 0 else float('inf')
        
        if time_since_last_alert < min_interval or time_since_last_alert < self.alert_suppression_time:
            return  # Too soon for next alert
            
        # Generate alert
        self._generate_alert(risk_assessment)
        self.last_alert_time = current_time
        
    def _generate_alert(self, risk_assessment: Dict):
        """Generate multi-modal alert (audio, visual, haptic) with logging"""
        risk_level = risk_assessment['risk_level']
        unified_risk = risk_assessment['unified_risk']
        driver_state = risk_assessment['driver_state']['current_state']
        external_risk = risk_assessment['external_risk']
        
        # SHORT, CLEAR ALERT MESSAGES for TTS
        # Different messages based on context for maximum clarity
        
        # Check for emergency vehicle context
        external_breakdown = risk_assessment.get('external_breakdown')
        primary_threat = external_breakdown.get('primary_threat') if external_breakdown else None
        is_emergency_vehicle = primary_threat and primary_threat.get('is_emergency', False)
        
        # Generate context-aware short messages
        if risk_level == 'critical':
            if is_emergency_vehicle:
                full_message = "Alert! Emergency vehicle!"
            elif driver_state in ['sleepy', 'using_phone']:
                full_message = "Alert! Pay attention!"
            else:
                full_message = "Alert! Vehicle ahead!"
                
        elif risk_level == 'emergency':
            if is_emergency_vehicle:
                full_message = "Alert! Alert! Emergency vehicle!"
            else:
                full_message = "Alert! Alert! Brake now!"
        else:
            # Fallback for warning/caution (though these use beeps)
            full_message = "Alert!"
            
        print(f"🚨 ALERT [{risk_level.upper()}]: {full_message}")
        
        # Log alert to experiment logger if available
        self._log_alert_to_experiment_logger(risk_level, risk_assessment, driver_state)
        
        # Audio alert strategy based on risk level
        if self.audio_enabled:
            if risk_level in ['caution', 'warning']:
                # Use beeps for caution and warning
                self._play_beep_alert(risk_level)
            elif risk_level in ['critical', 'emergency']:
                # Use text-to-speech for critical and emergency
                self._speak_alert_async(full_message)
                
                # For emergency level, add extra urgency with repetition
                if risk_level == 'emergency':
                    # Add a brief pause then repeat once more for maximum attention
                    import threading
                    def _repeat_emergency():
                        time.sleep(0.8)  # Brief pause
                        self._speak_alert_async("Brake now!")
                    threading.Thread(target=_repeat_emergency, daemon=True).start()
            
    def set_experiment_logger_callback(self, callback_func):
        """Set callback function for experiment logging"""
        self._experiment_logger_callback = callback_func
        
    def _get_current_trigger_distance(self, risk_assessment: Dict) -> float:
        """Get current trigger distance for logging"""
        external_breakdown = risk_assessment.get('external_breakdown', {})
        primary_threat = external_breakdown.get('primary_threat')
        return primary_threat.get('distance', float('inf')) if primary_threat else float('inf')
        
    def _log_alert_to_experiment_logger(self, risk_level: str, risk_assessment: Dict, driver_state: str):
        """Log alert event to experiment logger if available"""
        if hasattr(self, 'experiment_logger') and self.experiment_logger:
            current_time = time.time()
            
            # Determine alert type
            alert_type = 'beep' if risk_level in ['caution', 'warning'] else 'tts'
            
            # Get current TTC and distance from risk assessment
            external_breakdown = risk_assessment.get('external_breakdown', {})
            primary_threat = external_breakdown.get('primary_threat')
            
            current_ttc = primary_threat.get('ttc') if primary_threat else None
            current_distance = primary_threat.get('distance', float('inf')) if primary_threat else float('inf')
            
            # Log to experiment logger
            try:
                self.experiment_logger.log_alert_event(
                    timestamp=current_time,
                    alert_level=risk_level,
                    alert_type=alert_type,
                    current_ttc=current_ttc,
                    current_distance=current_distance,
                    driver_state=driver_state
                )
            except Exception as e:
                print(f"Warning: Failed to log alert event: {e}")
                
    def set_experiment_logger(self, logger):
        """Set the experiment logger for alert tracking"""
        self.experiment_logger = logger
        print(f"📊 Experiment logger connected to Unified Risk Manager")
        
    def _play_beep_alert(self, risk_level: str):
        """Play beep alerts for caution and warning levels (non-blocking)"""
        def _beep_worker():
            """Worker function for beeps in separate thread"""
            print(f"Playing beep alert for {risk_level} level")
            
            # Always use 3 beeps for both caution and warning as requested
            beep_count = 3
            
            # Different frequencies for caution vs warning
            if risk_level == 'caution':
                frequency = 800    # Lower frequency for caution
                duration = 150     # Shorter duration
            else:  # warning
                frequency = 1200   # Higher frequency for warning
                duration = 200     # Longer duration
                
            try:
                # Try system beep first
                for i in range(beep_count):
                    subprocess.run(['beep', '-f', str(frequency), '-l', str(duration)], 
                                 check=False, capture_output=True)
                    if i < beep_count - 1:
                        time.sleep(0.15)  # Short pause between beeps
                        
                print(f"System beep: {beep_count} beeps at {frequency}Hz for {risk_level}")
                
            except Exception as e:
                print(f"System beep failed: {e}, trying pygame fallback")
                # Fallback to pygame-generated tones
                self._generate_pygame_beeps(beep_count, frequency, duration)
        
        # Start beep worker in daemon thread (non-blocking)
        beep_thread = threading.Thread(target=_beep_worker, daemon=True)
        beep_thread.start()
        print(f"Started async beep alert for {risk_level}")
        
    def _generate_pygame_beeps(self, count: int, frequency: int, duration_ms: int):
        """Generate beep tones using pygame when system beep is not available"""
        try:
            import numpy as np  # Make sure numpy is imported
            
            sample_rate = 44100
            duration_sec = duration_ms / 1000.0
            frames = int(duration_sec * sample_rate)
            
            # Generate sine wave
            arr = np.zeros((frames, 2), dtype=np.int16)
            for i in range(frames):
                sample = int(16000 * np.sin(2 * np.pi * frequency * i / sample_rate))
                arr[i] = [sample, sample]  # Stereo
                
            # Convert to pygame sound
            sound = pygame.sndarray.make_sound(arr)
            
            # Play the beeps
            for i in range(count):
                sound.play()
                time.sleep(duration_sec + 0.15)  # Wait for sound + pause
                
            print(f"Generated pygame beeps: {count} beeps at {frequency}Hz")
            
        except Exception as e:
            print(f"Pygame beep generation failed: {e}")
            # Ultimate fallback - system bell
            for i in range(count):
                print('\a', end='', flush=True)
                time.sleep(0.2)

    def _play_audio_alert(self, risk_level: str, driver_state: str):
        """Legacy method - now redirects to appropriate alert type"""
        if risk_level in ['caution', 'warning']:
            self._play_beep_alert(risk_level)
        elif risk_level in ['critical', 'emergency']:
            # This shouldn't be called for critical/emergency since we use TTS directly
            print(f"Note: _play_audio_alert called for {risk_level} - using TTS instead")
        else:
            print(f"Unknown risk level: {risk_level}")
            
    def _speak_alert_async(self, message: str):
        """Use non-blocking text-to-speech for critical alerts"""
        def _tts_worker():
            """Worker function for TTS in separate thread"""
            try:
                # Use espeak with optimized parameters for alerts
                # -s 180: Faster speech (180 words per minute)
                # -a 200: Higher amplitude (louder)
                # -p 60: Higher pitch for urgency
                subprocess.run(['espeak', '-s', '180', '-a', '200', '-p', '60', '-v', 'en', message], 
                             check=False, capture_output=True, timeout=5)
            except subprocess.TimeoutExpired:
                print(f"TTS timeout: {message[:30]}...")
            except FileNotFoundError:
                try:
                    # Alternative: use festival
                    subprocess.run(['echo', message, '|', 'festival', '--tts'], 
                                 shell=True, check=False, capture_output=True, timeout=10)
                except:
                    print(f"TTS FALLBACK: {message}")
            except Exception as e:
                print(f"TTS error: {e} - Message: {message[:30]}...")
        
        # Manage TTS process to prevent overlapping speech
        with self.tts_lock:
            # Kill any existing TTS process
            if self.tts_process and self.tts_process.is_alive():
                print("Stopping previous TTS to speak new alert")
                # Don't wait for it to finish, just start new one
                
            # Start new TTS in daemon thread (won't block shutdown)
            self.tts_process = threading.Thread(target=_tts_worker, daemon=True)
            self.tts_process.start()
            
    def _speak_alert(self, message: str):
        """Legacy blocking TTS method - now redirects to async version"""
        self._speak_alert_async(message)
        
    def get_risk_statistics(self) -> Dict:
        """Get comprehensive risk statistics for analysis"""
        if not self.risk_history:
            return {'error': 'No risk history available'}
            
        risks = [entry['unified_risk'] for entry in self.risk_history]
        external_risks = [entry['external_risk'] for entry in self.risk_history]
        internal_risks = [entry['internal_risk'] for entry in self.risk_history]
        
        return {
            'unified_risk': {
                'current': risks[-1] if risks else 0.0,
                'mean': np.mean(risks),
                'max': np.max(risks),
                'std': np.std(risks)
            },
            'external_risk': {
                'current': external_risks[-1] if external_risks else 0.0,
                'mean': np.mean(external_risks),
                'max': np.max(external_risks)
            },
            'internal_risk': {
                'current': internal_risks[-1] if internal_risks else 0.0,
                'mean': np.mean(internal_risks),
                'max': np.max(internal_risks)
            },
            'driver_state': {
                'current': self.current_driver_state,
                'confidence': self.driver_state_confidence,
                'last_update': self.last_driver_update
            },
            'alert_history': {
                'total_history_points': len(self.risk_history),
                'high_risk_episodes': len([r for r in risks if r > 0.6]),
                'critical_episodes': len([r for r in risks if r > 0.8])
            }
        }
        
    def shutdown(self):
        """Clean shutdown of the unified risk manager"""
        print("Shutting down Unified Risk Manager...")
        self.stop_udp_listener()
        
        if self.audio_enabled:
            try:
                pygame.mixer.quit()
            except:
                pass
                
        # Finalize any plots
        if hasattr(self.safety_evaluator, 'finalize_plots'):
            self.safety_evaluator.finalize_plots()
            
        # Stop any ongoing TTS
        with self.tts_lock:
            if self.tts_process and self.tts_process.is_alive():
                print("Stopping TTS process...")
                # Daemon thread will be cleaned up automatically
        
        print("Unified Risk Manager shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    # Mock config for testing
    class MockConfig:
        def __init__(self):
            self.debug = True
            
    config = MockConfig()
    
    # Initialize unified risk manager
    risk_manager = UnifiedRiskManager(config, udp_port=9999, audio_enabled=True)
    
    try:
        print("Unified Risk Manager started. Send driver state data to UDP port 9999")
        print("Expected JSON format:")
        print('{"driver_state": "sleepy", "confidence": 0.85, "timestamp": 1234567890.123}')
        print("Valid driver states: safe_driving, sleepy, reaching_back, using_phone")
        print("Press Ctrl+C to stop")
        
        # Simulate some test scenarios
        test_scenarios = [
            # Scenario 1: Normal driving
            {
                'ego_speed': 15.0,  # 15 m/s (~50 km/h)
                'bounding_boxes': [[20.0, 0.5, 0, 0, 0, 0, 0, 0]],  # Vehicle 20m ahead
                'description': 'Normal driving scenario'
            },
            # Scenario 2: Close vehicle + sleepy driver
            {
                'ego_speed': 20.0,
                'bounding_boxes': [[8.0, 0.0, 0, 0, 0, 0, 0, 0]],  # Vehicle 8m ahead
                'description': 'Close vehicle scenario'
            },
            # Scenario 3: Emergency vehicle
            {
                'ego_speed': 15.0,
                'bounding_boxes': [[15.0, 0.0, 0, 0, 0, 0, 0, 4]],  # Emergency vehicle
                'description': 'Emergency vehicle scenario'
            }
        ]
        
        # Run test scenarios
        for i, scenario in enumerate(test_scenarios):
            print(f"\n--- Test Scenario {i+1}: {scenario['description']} ---")
            
            result = risk_manager.calculate_unified_risk(
                scenario['ego_speed'], 
                scenario['bounding_boxes']
            )
            
            print(f"Unified Risk: {result['unified_risk']:.3f}")
            print(f"Risk Level: {result['risk_level']}")
            
            time.sleep(3)  # Wait between scenarios
            
        # Keep running to listen for UDP data
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        risk_manager.shutdown()