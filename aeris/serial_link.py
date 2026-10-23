import serial
import serial.tools.list_ports
import time

class SerialLink:
    def __init__(self, config_manager):
        self.config = config_manager
        self.conn = None
        self.port = self.config.config["serial"]["port"]
        self.baud = self.config.config["serial"]["baud_rate"]

    def discover_port(self):
        """Automatically discovers the most likely Arduino port on macOS/Linux/Windows"""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Common patterns for Arduino on Mac/Linux
            if "usbmodem" in p.device or "usbserial" in p.device or "ttyACM" in p.device or "ttyUSB" in p.device:
                return p.device
            # Common for Windows
            if "COM" in p.device:
                return p.device
                
        if ports:
            return ports[0].device # fallback to first available
        return None

    def connect(self):
        target_port = self.port
        if target_port == "auto" or not target_port:
            target_port = self.discover_port()
            
        if not target_port:
            print("Error: No serial port found during auto-discovery.")
            return False

        try:
            self.conn = serial.Serial(target_port, self.baud, timeout=1)
            time.sleep(2) # wait for Arduino reset
            print(f"Successfully connected to Arduino on {target_port}")
            return True
        except Exception as e:
            print(f"Error connecting to {target_port}: {e}")
            self.conn = None
            return False

    def read_line(self):
        """Reads a line, handles disconnection cleanly"""
        if not self.conn:
            return None
            
        try:
            line = self.conn.readline().decode('utf-8', errors='ignore').strip()
            return line if line else None
        except serial.SerialException:
            print("\n[WARNING] Serial connection lost. Arduino may have been disconnected.")
            self.conn.close()
            self.conn = None
            return None
        except Exception:
            return None

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
