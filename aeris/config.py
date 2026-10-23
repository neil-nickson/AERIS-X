import json
import os
from pathlib import Path

CONFIG_FILE = Path("config/config.json")

DEFAULT_CONFIG = {
    "serial": {
        "port": "auto",
        "baud_rate": 9600
    },
    "sensors": {
        "rain_active_low": True
    },
    "thresholds": {
        "temp_deviation": 5.0,
        "low_light_ratio": 0.5,
        "vibration_threshold": 15.0,
        "min_calibration_samples": 50,
        "stability_samples": 5,
        "prototype_envelope_min_temp": 0.0,
        "prototype_envelope_max_temp": 45.0
    },
    "calibration": {
        "timestamp": None,
        "TEMP": {},
        "HUM": {},
        "LIGHT": {},
        "AX": {},
        "AY": {},
        "AZ": {},
        "VIBRATION": {}
    },
    "camera": {
        "source": "mac",
        "device_index": 0,
        "index": 0,
        "width": 1280,
        "height": 720,
        "target_fps": 30,
        "display": True,
        "recording_enabled": False
    },
    "detector": {
        "model_path": "models/drone_yolov8n.pt",
        "target_class": "quadcopter",
        "confidence_threshold": 0.40,
        "device": "cpu"
    },
    "tracker": {
        "association_iou_threshold": 0.3,
        "max_center_distance_ratio": 0.3,
        "max_missed_frames": 10,
        "min_detection_confidence": 0.40,
        "trajectory_length": 30,
        "target_selection_policy": "highest_confidence"
    },
    "fusion": {
        "max_timestamp_delta_ms": 500
    },
    "experiment": {
        "stabilization_seconds": 3.0,
        "recording_seconds": 10.0,
        "minimum_tracking_frames": 10
    },
    "analysis": {
        "minimum_correlation_trials": 5
    }
}

class ConfigManager:
    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = config_path
        self.config = self.load()

    def load(self):
        if not self.config_path.exists():
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                
                # Merge missing defaults (shallow merge for simplicity)
                for key in DEFAULT_CONFIG:
                    if key not in data:
                        data[key] = DEFAULT_CONFIG[key]
                return data
        except Exception as e:
            print(f"Warning: Failed to load config ({e}). Using defaults.")
            return DEFAULT_CONFIG.copy()

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error: Failed to save config to {self.config_path}: {e}")

    def update_calibration(self, new_cal_data, timestamp):
        self.config["calibration"] = new_cal_data
        self.config["calibration"]["timestamp"] = timestamp
        self.save()

    def get_baseline(self, sensor_key):
        return self.config["calibration"].get(sensor_key, {}).get("mean", None)
