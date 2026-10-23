import time
import math
import statistics
from datetime import datetime

class Calibrator:
    def __init__(self, config_manager, serial_link, sensor_parser):
        self.config = config_manager
        self.link = serial_link
        self.parser = sensor_parser

    def run_calibration(self, duration=30):
        print(f"\n--- STARTING {duration}s BASELINE CALIBRATION ---")
        print("Keep the sensor completely stationary.")
        
        start_time = time.time()
        samples = []
        
        total_lines = 0
        parsed_samples = 0
        rejected_samples = 0
        malformed_samples = 0
        
        while time.time() - start_time < duration:
            line = self.link.read_line()
            if line is None:
                continue
                
            total_lines += 1
            parsed = self.parser.parse(line)
            
            if not parsed:
                malformed_samples += 1
                continue
                
            parsed_samples += 1
            
            if parsed.get("TEMP") == -999.0: # Ignore DHT errors during cal
                rejected_samples += 1
            else:
                samples.append(parsed)
                
        actual_duration = time.time() - start_time
        hz = parsed_samples / actual_duration if actual_duration > 0 else 0
        
        print("\n--- DIAGNOSTICS ---")
        print(f"Total Serial Lines Received : {total_lines}")
        print(f"Successfully Parsed Samples : {parsed_samples}")
        print(f"Malformed Packets           : {malformed_samples}")
        print(f"DHT Rejected Samples        : {rejected_samples}")
        print(f"Effective Baud/Sample Rate  : {hz:.2f} Hz")
        print("-------------------")
                
        min_samples = self.config.config["thresholds"]["min_calibration_samples"]
        if len(samples) < min_samples:
            print(f"[WARNING] Calibration failed: Not enough samples ({len(samples)} < {min_samples}).")
            print("Check serial connection and sensor health.")
            return False
            
        print(f"Collected {len(samples)} valid samples.")
        
        # Calculate statistics
        keys = ["TEMP", "HUM", "LIGHT", "AX", "AY", "AZ"]
        cal_data = {}
        
        print("\n--- CALIBRATION RESULTS ---")
        for k in keys:
            vals = [s[k] for s in samples]
            cal_data[k] = {
                "mean": statistics.mean(vals),
                "min": min(vals),
                "max": max(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0
            }
            print(f"{k:5} -> Mean: {cal_data[k]['mean']:.2f}, Min: {cal_data[k]['min']:.2f}, Max: {cal_data[k]['max']:.2f}, Std: {cal_data[k]['std']:.2f}")
            
        # Calculate vibration based on the new AX/AY/AZ means
        ax_mean = cal_data["AX"]["mean"]
        ay_mean = cal_data["AY"]["mean"]
        az_mean = cal_data["AZ"]["mean"]
        
        vib_vals = []
        for s in samples:
            dx = s["AX"] - ax_mean
            dy = s["AY"] - ay_mean
            dz = s["AZ"] - az_mean
            mag = math.sqrt(dx*dx + dy*dy + dz*dz)
            vib_vals.append(mag)
            
        cal_data["VIBRATION"] = {
            "mean": statistics.mean(vib_vals),
            "min": min(vib_vals),
            "max": max(vib_vals),
            "std": statistics.stdev(vib_vals) if len(vib_vals) > 1 else 0.0
        }
        print(f"VIBE  -> Mean: {cal_data['VIBRATION']['mean']:.2f}, Min: {cal_data['VIBRATION']['min']:.2f}, Max: {cal_data['VIBRATION']['max']:.2f}, Std: {cal_data['VIBRATION']['std']:.2f}")
        print("---------------------------")
        
        # Warnings
        if cal_data["VIBRATION"]["mean"] > 10.0:
            print("[WARNING] Accelerometer noise is unusually high! Is it stationary?")
        if cal_data["TEMP"]["mean"] < 0 or cal_data["TEMP"]["mean"] > 50:
            print("[WARNING] Unusual temperature baseline detected.")
            
        timestamp = datetime.now().isoformat()
        self.config.update_calibration(cal_data, timestamp)
        
        print(f"Calibration successful and saved at {timestamp}.")
        return True
