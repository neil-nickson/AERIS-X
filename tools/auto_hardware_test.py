#!/usr/bin/env python3
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.camera import AerisCamera
from aeris.config import ConfigManager

def test_camera_index(index, name="Camera"):
    print(f"\n--- Testing {name} (Index {index}) ---")
    cfg_mgr = ConfigManager()
    cfg_mgr.config["camera"]["index"] = index
    cfg_mgr.config["camera"]["device_index"] = index
    cfg_mgr.config["camera"]["source"] = "iphone" if "iPhone" in name else "mac"
    
    cam = AerisCamera(cfg_mgr)
    if not cam.initialize():
        print(f"FAILED: Could not open {name} at index {index}")
        return False
        
    print(f"SUCCESS: Opened {name}. Health: {cam.get_health()}")
    
    # Capture some frames
    frames_captured = 0
    for _ in range(50):
        success, _ = cam.read()
        if success:
            frames_captured += 1
        time.sleep(0.01)
        
    stats = cam.get_diagnostic_stats()
    print(f"Frames Captured: {frames_captured}/50")
    print(f"Measured FPS: {stats['measured_fps']:.1f}")
    print(f"Read Failures: {stats['read_failures']}")
    print(f"Dropped Frames: {stats['dropped_frames']}")
    
    cam.release()
    print(f"SUCCESS: Clean shutdown for {name}.")
    
    if "iPhone" in name:
        # Re-open test
        print("Testing reopen...")
        cam2 = AerisCamera(cfg_mgr)
        if not cam2.initialize():
            print("FAILED: Reopen failed")
            return False
        cam2.read()
        cam2.release()
        print("SUCCESS: Reopen and clean shutdown passed.")
        
    return True

def main():
    print("AERIS-X AUTOMATED HARDWARE DETECTION")
    print("====================================")
    
    devices = AerisCamera.discover_cameras(max_indices=5)
    
    mac_index = None
    iphone_index = None
    
    # Simple heuristic to guess based on discover_cameras
    # If there is an index 0, it's usually mac.
    # If there are multiple, maybe index 1 is iPhone.
    # But since we can't reliably read the name, we just test index 0 and 1.
    
    available = [d for d in devices if d['status'] == 'AVAILABLE']
    
    if len(available) == 0:
        print("NO CAMERAS AVAILABLE")
    elif len(available) == 1:
        mac_index = available[0]['index']
    elif len(available) >= 2:
        mac_index = available[0]['index']
        iphone_index = available[1]['index']
        
    print(f"\nMac Camera Index Identified: {mac_index}")
    if mac_index is not None:
        test_camera_index(mac_index, "Mac Camera")
        
    print(f"\niPhone Continuity Camera Index Identified: {iphone_index}")
    if iphone_index is not None:
        test_camera_index(iphone_index, "iPhone Continuity Camera")
    else:
        print("\nPHYSICAL IPHONE NOT AVAILABLE FOR AUTOMATED HARDWARE TEST")

if __name__ == "__main__":
    main()
