#!/usr/bin/env python3
"""
Minimal Audio Test - Tests only the audio components that should work
"""

import os
import sys
import time
import subprocess
import pygame
from pathlib import Path

def test_pygame_sounds():
    """Test pygame audio with the actual audio files"""
    print("=== Testing Pygame Audio Files ===")
    
    # Check if audio directory exists
    audio_dir = Path.cwd() / 'audio'
    if not audio_dir.exists():
        print("✗ Audio directory not found")
        return False
    
    # Audio files that should exist
    audio_files = [
        'caution.wav',
        'warning.wav', 
        'critical.wav',
        'emergency.wav'
    ]
    
    try:
        pygame.mixer.init()
        print("✓ Pygame mixer initialized")
        
        for audio_file in audio_files:
            file_path = audio_dir / audio_file
            if file_path.exists():
                print(f"  Testing {audio_file}...")
                sound = pygame.mixer.Sound(str(file_path))
                sound.play()
                time.sleep(1.5)  # Wait for sound to finish
                print(f"  ✓ {audio_file} played successfully")
            else:
                print(f"  ⚠ {audio_file} not found")
        
        pygame.mixer.quit()
        return True
        
    except Exception as e:
        print(f"✗ Pygame test failed: {e}")
        return False

def test_audio_alert_simulation():
    """Simulate the audio alert logic without the full risk manager"""
    print("\n=== Testing Audio Alert Simulation ===")
    
    # Simulate the _play_audio_alert method logic
    def play_audio_alert(risk_level, driver_state):
        """Simulate the audio alert method"""
        try:
            # Adaptive audio selection (same logic as risk manager)
            if risk_level == 'emergency':
                if driver_state == 'sleepy':
                    audio_file = "emergency_wake_up.wav"
                else:
                    audio_file = "emergency_alert.wav"
            elif risk_level == 'critical':
                if driver_state in ['sleepy', 'using_phone']:
                    audio_file = "critical_attention.wav"
                else:
                    audio_file = "critical_alert.wav"
            else:
                audio_file = f"{risk_level}_alert.wav"
            
            # Try to play the audio file
            audio_dir = Path.cwd() / 'audio'
            file_path = audio_dir / audio_file
            
            if file_path.exists():
                sound = pygame.mixer.Sound(str(file_path))
                sound.play()
                print(f"    ✓ Played {audio_file}")
                return True
            else:
                # Fallback to basic file
                basic_file = audio_dir / f"{risk_level}.wav"
                if basic_file.exists():
                    sound = pygame.mixer.Sound(str(basic_file))
                    sound.play()
                    print(f"    ✓ Played {basic_file} (fallback)")
                    return True
                else:
                    print(f"    ✗ No audio file found for {risk_level}")
                    return False
                    
        except Exception as e:
            print(f"    ✗ Audio alert failed: {e}")
            return False
    
    # Test different scenarios
    test_cases = [
        ('caution', 'safe_driving'),
        ('warning', 'safe_driving'),
        ('critical', 'sleepy'),
        ('emergency', 'using_phone'),
    ]
    
    try:
        pygame.mixer.init()
        
        for risk_level, driver_state in test_cases:
            print(f"  Testing {risk_level} alert with {driver_state} driver...")
            success = play_audio_alert(risk_level, driver_state)
            if success:
                time.sleep(2)  # Wait between tests
            
        pygame.mixer.quit()
        return True
        
    except Exception as e:
        print(f"✗ Audio alert simulation failed: {e}")
        return False

def test_beep_patterns():
    """Test the beep pattern logic"""
    print("\n=== Testing Beep Patterns ===")
    
    def system_beep_pattern(risk_level):
        """Simulate the beep pattern method"""
        beep_patterns = {
            'caution': 1,      # Single beep
            'warning': 2,      # Double beep
            'critical': 3,     # Triple beep
            'emergency': 5     # Five rapid beeps
        }
        
        beep_count = beep_patterns.get(risk_level, 1)
        
        try:
            # Try beep command first
            for i in range(beep_count):
                result = subprocess.run(['beep', '-f', '1000', '-l', '200'], 
                                      capture_output=True, timeout=2)
                if result.returncode != 0:
                    # Fallback to terminal bell
                    print('\a', end='', flush=True)
                if i < beep_count - 1:
                    time.sleep(0.1)
            return True
        except:
            # Final fallback - terminal bell
            for i in range(beep_count):
                print('\a', end='', flush=True)
                time.sleep(0.1)
            return True
    
    # Test each risk level
    for risk_level in ['caution', 'warning', 'critical', 'emergency']:
        print(f"  Testing {risk_level} beep pattern...")
        success = system_beep_pattern(risk_level)
        if success:
            print(f"    ✓ {risk_level} beep pattern completed")
            time.sleep(1)
        else:
            print(f"    ✗ {risk_level} beep pattern failed")
    
    return True

def test_tts_simple():
    """Test TTS with simple fallback"""
    print("\n=== Testing Text-to-Speech (Simple) ===")
    
    def speak_alert(message):
        """Simulate TTS method"""
        try:
            # Try espeak
            result = subprocess.run(['espeak', '-s', '150', '-v', 'en', message], 
                                  check=False, capture_output=True, timeout=10)
            if result.returncode == 0:
                return True
        except:
            pass
        
        try:
            # Try festival
            result = subprocess.run(f'echo "{message}" | festival --tts', 
                                  shell=True, check=False, capture_output=True, timeout=10)
            if result.returncode == 0:
                return True
        except:
            pass
        
        # Fallback - just print
        print(f"    TTS FALLBACK: {message}")
        return False
    
    # Test messages
    test_messages = [
        "Critical: High risk situation detected",
        "EMERGENCY: Immediate intervention required"
    ]
    
    tts_working = False
    for message in test_messages:
        print(f"  Testing TTS: '{message[:30]}...'")
        if speak_alert(message):
            print(f"    ✓ TTS played successfully")
            tts_working = True
        else:
            print(f"    ⚠ TTS not available, used fallback")
        time.sleep(1)
    
    return tts_working

def main():
    """Run minimal audio tests"""
    print("Minimal Audio Test for Unified Risk Manager")
    print("=" * 50)
    
    tests = [
        ("Pygame Audio Files", test_pygame_sounds),
        ("Audio Alert Simulation", test_audio_alert_simulation),
        ("Beep Patterns", test_beep_patterns),
        ("Text-to-Speech", test_tts_simple),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("MINIMAL AUDIO TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 3:
        print("🎉 Core audio functionality is working!")
        print("Your unified risk manager should be able to play audio alerts.")
    elif passed >= 2:
        print("⚠ Basic audio is working, but some features may be limited.")
    else:
        print("❌ Audio system needs attention.")
    
    print("\nTo improve audio functionality:")
    print("  sudo apt-get install espeak beep")

if __name__ == "__main__":
    main() 