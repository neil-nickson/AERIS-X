import time

class HealthMonitor:
    def __init__(self):
        self.last_packet_time = 0
        self.status = "UNKNOWN"
        self.dht_health = "OK"
        self.adxl_health = "OK"
        
    def update(self, parsed_data):
        if not parsed_data:
            return
            
        self.last_packet_time = time.time()
        
        # Check DHT health
        if parsed_data.get("TEMP") == -999.0 or parsed_data.get("HUM") == -999.0:
            self.dht_health = "FAIL (Invalid Reading)"
        else:
            self.dht_health = "OK"
            
        # Check ADXL health
        ax = parsed_data.get("AX")
        ay = parsed_data.get("AY")
        az = parsed_data.get("AZ")
        
        if ax is None or ay is None or az is None:
            self.adxl_health = "FAIL (Missing Data)"
        elif ax <= 10 or ax >= 1000 or ay <= 10 or ay >= 1000 or az <= 10 or az >= 1000:
            self.adxl_health = "FAIL (Invalid Reading)"
        else:
            self.adxl_health = "OK"
            
    def get_health_status(self):
        time_since_last = time.time() - self.last_packet_time
        
        if self.last_packet_time == 0:
            return "WAITING FOR DATA"
            
        if time_since_last > 3.0:
            return "COMMUNICATION LOST"
            
        if self.dht_health != "OK":
            return "DHT SENSOR ERROR"
            
        if getattr(self, 'adxl_health', 'OK') != "OK":
            return "ADXL SENSOR ERROR"
            
        return "NOMINAL"
