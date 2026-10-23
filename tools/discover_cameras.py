#!/usr/bin/env python3
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.camera import AerisCamera

def main():
    print("AERIS-X HARDWARE CAMERA DISCOVERY")
    print("=================================")
    print("Probing macOS/OpenCV camera indices 0-5...\n")
    
    devices = AerisCamera.discover_cameras(max_indices=5)
    
    if not devices:
        print("NO CAMERAS DETECTED.")
        return
        
    print(f"{'INDEX':<7} | {'STATUS':<14} | {'RESOLUTION':<12} | {'FPS':<6} | {'NAME'}")
    print("-" * 80)
    for d in devices:
        fps_str = f"{d['reported_fps']:.1f}" if d['reported_fps'] else "N/A"
        print(f"{d['index']:<7} | {d['status']:<14} | {d['resolution']:<12} | {fps_str:<6} | {d['name']}")
        
    print("\nNote: iPhone Continuity Camera may appear as index 0, 1, or 2.")
    print("To use the iPhone, set the following in config/config.json:")
    print('  "camera": {')
    print('      "source": "iphone",')
    print('      "device_index": <INDEX>')
    print('  }')

if __name__ == "__main__":
    main()
