import time
from typing import Optional, Dict

class SensorParser:
    def __init__(self, config_manager):
        self.config = config_manager

    def parse(self, line: str) -> Optional[Dict]:
        """Parses a serial line into a dictionary with a timestamp"""
        if not line or not line.startswith("TEMP="):
            return None
            
        values = {}
        try:
            parts = line.split(",")
            for part in parts:
                key, value_str = part.split("=")
                values[key] = float(value_str)
                
            # Apply Rain logic mapping
            # rain sensor gives 0 or 1. Active low means 0 is Rain, 1 is dry.
            active_low = self.config.config["sensors"]["rain_active_low"]
            raw_rain = values.get("RAIN", 1)
            
            if active_low:
                values["RAIN_DETECTED"] = (raw_rain == 0)
            else:
                values["RAIN_DETECTED"] = (raw_rain == 1)
                
            values["timestamp"] = time.time()
            return values
        except Exception as e:
            # Drop malformed packets silently but safely
            return None
