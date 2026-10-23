import argparse
import time
import math
import os
import csv
from datetime import datetime
from aeris.config import ConfigManager
from aeris.serial_link import SerialLink
from aeris.sensor_parser import SensorParser
from aeris.calibration import Calibrator
from aeris.health_monitor import HealthMonitor

LOG_FILE = "logs/experiment_log.csv"

def init_csv():
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Temp", "Hum", "Light_Raw", "AX", "AY", "AZ", "Rain", 
                "Temp_Dev", "Light_Dev_Pct", "Vib_Mag", "Health_Status"
            ])

def log_data(data, dev_data, health_status):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.fromtimestamp(data["timestamp"]).isoformat(),
            data.get("TEMP"), data.get("HUM"), data.get("LIGHT"),
            data.get("AX"), data.get("AY"), data.get("AZ"), data.get("RAIN"),
            f"{dev_data.get('temp_dev', 0):.2f}",
            f"{dev_data.get('light_dev_pct', 0):.2f}",
            f"{dev_data.get('vibration', 0):.2f}",
            health_status
        ])

def main():
    parser = argparse.ArgumentParser(description="AERIS-X Phase 1 - Sensor Acquisition & Health")
    parser.add_argument('--calibrate', action='store_true', help="Force a new baseline calibration")
    args = parser.parse_args()

    config_mgr = ConfigManager()
    link = SerialLink(config_mgr)
    sensor_parser = SensorParser(config_mgr)
    health_mon = HealthMonitor()
    
    init_csv()

    if not link.connect():
        print("Failed to connect to Arduino. Exiting.")
        return

    if args.calibrate or not config_mgr.get_baseline("TEMP"):
        calibrator = Calibrator(config_mgr, link, sensor_parser)
        success = calibrator.run_calibration(duration=30)
        if not success:
            print("Calibration failed. Exiting.")
            return

    # Load baselines
    base_temp = config_mgr.get_baseline("TEMP")
    base_light = config_mgr.get_baseline("LIGHT")
    base_x = config_mgr.get_baseline("AX")
    base_y = config_mgr.get_baseline("AY")
    base_z = config_mgr.get_baseline("AZ")

    print("\n==========================================")
    print("AERIS-X PHASE 1 LIVE MONITOR")
    print("==========================================")
    print("Logging to:", LOG_FILE)
    print("Press Ctrl+C to stop.")

    try:
        while True:
            line = link.read_line()
            
            if line is None:
                health_mon.update(None)
                status = health_mon.get_health_status()
                if status == "COMMUNICATION LOST":
                    print("\n[!] SENSOR HEALTH: COMMUNICATION LOST. Attempting reconnect...")
                    link.close()
                    time.sleep(2)
                    link.connect()
                continue
                
            parsed = sensor_parser.parse(line)
            if not parsed:
                continue
                
            health_mon.update(parsed)
            health_status = health_mon.get_health_status()
            
            if health_status != "NOMINAL":
                print(f"\n[!] SENSOR HEALTH: {health_status}")
                if "FAIL" in health_status or "ERROR" in health_status:
                    continue # Skip processing bad data

            # Calculate deviations based on dynamically measured baselines
            t_dev = abs(parsed["TEMP"] - base_temp)
            l_dev_pct = (abs(parsed["LIGHT"] - base_light) / base_light) * 100 if base_light > 0 else 0
            
            dx = parsed.get("AX", 0) - base_x
            dy = parsed.get("AY", 0) - base_y
            dz = parsed.get("AZ", 0) - base_z
            vib_mag = math.sqrt(dx*dx + dy*dy + dz*dz) if health_status == "NOMINAL" else 0.0
            
            dev_data = {
                "temp_dev": t_dev,
                "light_dev_pct": l_dev_pct,
                "vibration": vib_mag
            }
            
            log_data(parsed, dev_data, health_status)
            
            # Print cleanly separated information
            print("\n------------------------------------------")
            print("1. SENSOR HEALTH")
            print(f"Status        : {health_status}")
            print()
            print("2. RAW SENSOR DATA")
            print(f"Temperature   : {parsed['TEMP']:.1f} °C")
            print(f"Humidity      : {parsed['HUM']:.1f} %")
            print(f"Light         : {parsed['LIGHT']:.0f}")
            print(f"ADXL335 (X,Y,Z): {parsed['AX']}, {parsed['AY']}, {parsed['AZ']}")
            print(f"Rain detected : {'YES' if parsed['RAIN_DETECTED'] else 'NO'}")
            print()
            print("3. CALIBRATED BASELINE (from config.json)")
            print(f"Base Temp     : {base_temp:.2f} °C")
            print(f"Base Light    : {base_light:.2f}")
            print(f"Base ADXL(X,Y,Z): {base_x:.1f}, {base_y:.1f}, {base_z:.1f}")
            print()
            print("4. BASELINE-RELATIVE DEVIATION")
            print(f"Temp Deviation: {t_dev:.2f} °C")
            print(f"Light Deviat. : {l_dev_pct:.1f} %")
            print(f"Vibration     : {vib_mag:.2f} (Calculated from Base ADXL)")
            print("------------------------------------------")

    except KeyboardInterrupt:
        print("\n\nStopped by user. Phase 1 execution complete.")
    finally:
        link.close()

if __name__ == "__main__":
    main()
