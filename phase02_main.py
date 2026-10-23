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
from aeris.environment_engine import EnvironmentEngine

LOG_FILE = "logs/experiment_log.csv"

def init_csv():
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Temp", "Hum", "Light_Raw", "AX", "AY", "AZ", "Rain", 
                "Temp_Dev", "Light_Dev_Pct", "Vib_Mag", "Health_Status",
                "Env_State", "Active_Conditions"
            ])

def log_data(data, dev_data, env_result):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.fromtimestamp(data["timestamp"]).isoformat(),
            data.get("TEMP"), data.get("HUM"), data.get("LIGHT"),
            data.get("AX"), data.get("AY"), data.get("AZ"), data.get("RAIN"),
            f"{dev_data.get('temp_dev', 0):.2f}",
            f"{dev_data.get('light_dev_pct', 0):.2f}",
            f"{dev_data.get('vibration', 0):.2f}",
            env_result["health"],
            env_result["state"],
            "|".join(env_result["active_conditions"])
        ])

def main():
    parser = argparse.ArgumentParser(description="AERIS-X Phase 2 - Environment Engine")
    parser.add_argument('--calibrate', action='store_true', help="Force a new baseline calibration")
    args = parser.parse_args()

    config_mgr = ConfigManager()
    link = SerialLink(config_mgr)
    sensor_parser = SensorParser(config_mgr)
    health_mon = HealthMonitor()
    env_engine = EnvironmentEngine(config_mgr)
    
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
    print("AERIS-X PHASE 2 LIVE ENGINE")
    print("==========================================")
    print("Logging to:", LOG_FILE)
    print("Press Ctrl+C to stop.")

    try:
        while True:
            line = link.read_line()
            
            if line is None:
                health_mon.update(None)
                status = health_mon.get_health_status()
                # Run environment engine with None data to handle health faults
                env_res = env_engine.evaluate(None, status, None)
                if status == "COMMUNICATION LOST":
                    print(f"\n[!] SENSOR HEALTH: COMMUNICATION LOST. Env State: {env_res['state']}. Attempting reconnect...")
                    link.close()
                    time.sleep(2)
                    link.connect()
                continue
                
            parsed = sensor_parser.parse(line)
            if not parsed:
                continue
                
            health_mon.update(parsed)
            health_status = health_mon.get_health_status()

            # Process Dev Data even if health is slightly off, but engine will handle it
            t_dev = abs(parsed.get("TEMP", 0) - base_temp)
            l_ratio = (parsed.get("LIGHT", 1) / base_light) if base_light > 0 else 1.0
            
            dx = parsed.get("AX", 0) - base_x
            dy = parsed.get("AY", 0) - base_y
            dz = parsed.get("AZ", 0) - base_z
            vib_mag = math.sqrt(dx*dx + dy*dy + dz*dz) if health_status == "NOMINAL" else 0.0
            
            dev_data = {
                "temp_abs_delta": t_dev,
                "light_ratio": l_ratio,
                "vibration": vib_mag
            }

            env_res = env_engine.evaluate(parsed, health_status, dev_data)
            
            if health_status != "NOMINAL":
                if "FAIL" in health_status or "ERROR" in health_status:
                    print(f"\n[!] SENSOR HEALTH: {health_status} | Env State: {env_res['state']}")
                    continue # Skip normal processing, wait for recovery
            
            log_data(parsed, dev_data, env_res)
            
            # Print requested CLI output format
            print("\n------------------------------------------")
            print("ENVIRONMENT")
            print("-----------")
            print(f"State              : {env_res['state']}")
            print(f"Active Conditions  : {', '.join(env_res['active_conditions']) if env_res['active_conditions'] else 'NONE'}")
            print(f"Temperature Delta  : {dev_data['temp_abs_delta']:.2f} °C")
            print(f"Light Ratio        : {dev_data['light_ratio']:.2f}")
            print(f"Vibration Index    : {dev_data['vibration']:.2f}")
            print(f"Rain               : {'YES' if parsed.get('RAIN_DETECTED') else 'NO'}")
            print(f"Sensor Health      : {health_status}")
            print("------------------------------------------")

    except KeyboardInterrupt:
        print("\n\nStopped by user. Phase 2 execution complete.")
    finally:
        link.close()

if __name__ == "__main__":
    main()
