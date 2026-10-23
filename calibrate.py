import argparse
import time
from aeris.config import ConfigManager
from aeris.serial_link import SerialLink
from aeris.sensor_parser import SensorParser
from aeris.calibration import Calibrator

def main():
    print("==========================================")
    print("       AERIS-X HARDWARE CALIBRATOR")
    print("==========================================")
    
    parser_args = argparse.ArgumentParser(description="AERIS-X Calibration Script")
    parser_args.add_argument('--duration', type=int, default=30, help="Duration of calibration in seconds")
    args = parser_args.parse_args()

    config_mgr = ConfigManager()
    link = SerialLink(config_mgr)
    sensor_parser = SensorParser(config_mgr)
    
    if not link.connect():
        print("Failed to connect to Arduino for calibration. Exiting.")
        return

    calibrator = Calibrator(config_mgr, link, sensor_parser)
    success = calibrator.run_calibration(duration=args.duration)
    
    link.close()
    
    if success:
        # aeris/calibration.py prints "Calibration successful and saved at {timestamp}." natively.
        print("\nCalibration successful and saved.")
        
        # Verify persistence and display timestamp
        config_mgr_reload = ConfigManager()
        ts = config_mgr_reload.config.get("calibration", {}).get("timestamp", "UNKNOWN")
        print(f"Verified Active Calibration Timestamp: {ts}")
        print("Phase 1 is now ready to use this baseline.")
    else:
        print("\nCalibration failed.")

if __name__ == "__main__":
    main()
