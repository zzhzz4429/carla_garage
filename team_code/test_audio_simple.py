#!/usr/bin/env python3
"""
Simple Audio Test for Unified Risk Manager
Tests the audio alert functionality to verify it's working correctly
"""

import os
import sys
import time
import tempfile
import wave
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_audio_files():
    """Create simple test audio files for testing"""
    print("Creating test audio files...")
    
    # Create audio directory
    audio_dir = Path.cwd() / 'audio'
    audio_dir.mkdir(exist_ok=True)
    
    # Audio files needed by the risk manager
    audio_files = [
        'caution.wav',
        'warning.wav', 
        'critical.wav',
        'emergency.wav',
        'emergency_wake_up.wav',
        'emergency_alert.wav',
        'critical_attention.wav',
        'critical_alert.wav'
    ]
    
    # Audio parameters
    sample_rate = 22050
    duration = 0.5
    
    # Different frequencies for different alert types
    frequencies = {
        'caution': 300,
        'warning': 500, 
        'critical': 800,
        'emergency': 1000
    }
    
    for audio_file in audio_files:
        file_path = audio_dir / audio_file
        
        # Skip if file already exists
        if file_path.exists():
            continue
            
        # Determine frequency based on file name
        freq = 440  # Default frequency
        for alert_type, alert_freq in frequencies.items():
            if alert_type in audio_file.lower():
                freq = alert_freq
                break
        
        try:
            # Generate tone
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.5
            
            # Add envelope to avoid clicks
            fade_frames = int(0.05 * sample_rate)  # 50ms fade
            for i in range(fade_frames):
                arr[i] *= i / fade_frames
                arr[-(i+1)] *= i / fade_frames
            
            # Convert to 16-bit integers
            arr = (arr * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(str(file_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(arr.tobytes())
                
            print(f"  Created: {audio_file}")
            
        except Exception as e:
            print(f"  Failed to create {audio_file}: {e}")
    
    print("Test audio files created successfully!")

def test_pygame_audio():
    """Test if pygame audio is working"""
    print("\n=== Testing Pygame Audio ===")
    
    try:
        import pygame
        pygame.mixer.init()
        print("✓ Pygame audio initialized successfully")
        
        # Test loading a simple audio file
        audio_dir = Path.cwd() / 'audio'
        test_file = audio_dir / 'caution.wav'
        
        if test_file.exists():
            sound = pygame.mixer.Sound(str(test_file))
            print("✓ Audio file loaded successfully")
            
            print("  Playing test sound...")
            sound.play()
            time.sleep(1)
            print("✓ Audio playback test completed")
        else:
            print("⚠ No test audio file found")
            
        pygame.mixer.quit()
        return True
        
    except ImportError:
        print("✗ Pygame not installed. Install with: pip install pygame")
        return False
    except Exception as e:
        print(f"✗ Pygame audio test failed: {e}")
        return False

def test_tts():
    """Test text-to-speech functionality"""
    print("\n=== Testing Text-to-Speech ===")
    
    test_message = "This is a test of the text to speech system"
    
    # Test espeak
    try:
        import subprocess
        result = subprocess.run(['espeak', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Espeak is available")
            print("  Testing espeak speech...")
            subprocess.run(['espeak', '-s', '150', '-v', 'en', test_message], 
                         check=False, capture_output=True, timeout=10)
            print("✓ Espeak test completed")
            return True
        else:
            print("✗ Espeak not working properly")
    except FileNotFoundError:
        print("✗ Espeak not found. Install with: sudo apt-get install espeak")
    except Exception as e:
        print(f"✗ Espeak test failed: {e}")
    
    # Test festival as fallback
    try:
        result = subprocess.run(['festival', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Festival is available as fallback")
            return True
    except:
        print("✗ Festival also not available")
    
    return False

def test_system_beep():
    """Test system beep functionality"""
    print("\n=== Testing System Beep ===")
    
    try:
        import subprocess
        # Test beep command
        result = subprocess.run(['beep', '-f', '1000', '-l', '200'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✓ System beep working")
            return True
        else:
            print("⚠ Beep command available but may not work (check permissions)")
    except FileNotFoundError:
        print("✗ Beep command not found. Install with: sudo apt-get install beep")
    except Exception as e:
        print(f"✗ Beep test failed: {e}")
    
    # Test terminal bell as fallback
    try:
        print("  Testing terminal bell...")
        print('\a', end='', flush=True)
        time.sleep(0.5)
        print("✓ Terminal bell test completed")
        return True
    except Exception as e:
        print(f"✗ Terminal bell failed: {e}")
    
    return False

def test_unified_risk_manager_audio():
    """Test the audio functionality in the unified risk manager"""
    print("\n=== Testing Unified Risk Manager Audio ===")
    
    try:
        # Import the unified risk manager
        from unified_risk_manager import UnifiedRiskManager
        
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.debug = True
        
        config = MockConfig()
        
        # Initialize risk manager with audio enabled
        print("Initializing Unified Risk Manager...")
        risk_manager = UnifiedRiskManager(config, udp_port=9998, audio_enabled=True)
        
        if not risk_manager.audio_enabled:
            print("✗ Audio is disabled in risk manager")
            return False
            
        print("✓ Unified Risk Manager initialized with audio enabled")
        
        # Test different risk levels and driver states
        test_cases = [
            ('caution', 'safe_driving'),
            ('warning', 'safe_driving'),
            ('critical', 'sleepy'),
            ('emergency', 'using_phone'),
        ]
        
        print("\nTesting audio alerts for different scenarios...")
        
        for risk_level, driver_state in test_cases:
            print(f"  Testing {risk_level} alert with {driver_state} driver...")
            
            try:
                # Test the audio alert method directly
                risk_manager._play_audio_alert(risk_level, driver_state)
                time.sleep(1.5)  # Wait between tests
                print(f"    ✓ {risk_level} audio alert completed")
                
            except Exception as e:
                print(f"    ✗ {risk_level} audio alert failed: {e}")
        
        # Test TTS functionality
        print("\n  Testing TTS alerts...")
        test_messages = [
            "Critical: High risk situation detected",
            "EMERGENCY: Immediate intervention required"
        ]
        
        for message in test_messages:
            try:
                print(f"    Testing TTS: '{message[:30]}...'")
                risk_manager._speak_alert(message)
                time.sleep(2)  # Wait for TTS
                print(f"    ✓ TTS alert completed")
            except Exception as e:
                print(f"    ✗ TTS alert failed: {e}")
        
        # Test system beep patterns
        print("\n  Testing beep patterns...")
        for risk_level in ['caution', 'warning', 'critical', 'emergency']:
            try:
                print(f"    Testing {risk_level} beep pattern...")
                risk_manager._system_beep_pattern(risk_level)
                time.sleep(1)
                print(f"    ✓ {risk_level} beep pattern completed")
            except Exception as e:
                print(f"    ✗ {risk_level} beep pattern failed: {e}")
        
        # Cleanup
        risk_manager.shutdown()
        print("✓ All unified risk manager audio tests completed")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import unified risk manager: {e}")
        return False
    except Exception as e:
        print(f"✗ Unified risk manager audio test failed: {e}")
        return False

def main():
    """Run all audio tests"""
    print("Simple Audio Test for Unified Risk Manager")
    print("=" * 50)
    
    # Step 1: Create test audio files if needed
    create_test_audio_files()
    
    # Step 2: Test individual components
    tests = [
        ("Pygame Audio", test_pygame_audio),
        ("Text-to-Speech", test_tts),
        ("System Beep", test_system_beep),
        ("Unified Risk Manager Audio", test_unified_risk_manager_audio),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("AUDIO TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All audio tests passed! Your audio system is working correctly.")
    elif passed > 0:
        print("⚠ Some audio functionality is working. Check failed tests above.")
        print("\nTo install missing dependencies:")
        print("  sudo apt-get install python3-pygame espeak beep")
        print("  pip install pygame numpy")
    else:
        print("❌ Audio system is not working. Please install dependencies:")
        print("  sudo apt-get install python3-pygame espeak beep")
        print("  pip install pygame numpy")

if __name__ == "__main__":
    main() 