import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.health_monitor import HealthMonitor
from aeris.environment_engine import EnvironmentEngine
from aeris.config import ConfigManager

class TestVibrationHealth(unittest.TestCase):
    def setUp(self):
        self.config = ConfigManager()
        self.health = HealthMonitor()
        self.env = EnvironmentEngine(self.config)

    def _eval(self, parsed, dev_data):
        self.health.update(parsed)
        status = self.health.get_health_status()
        return self.env.evaluate(parsed, status, dev_data)

    def test_01_valid_stationary(self):
        # ax=330, vib=0
        res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 330.0, "AY": 330.0, "AZ": 330.0}, {"vibration": 0.0})
        self.assertEqual(res["health"], "NOMINAL")
        self.assertEqual(res["state"], "NORMAL")

    def test_02_genuine_high_vibration(self):
        # ax=500, vib=100
        res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 500.0, "AY": 500.0, "AZ": 500.0}, {"vibration": 100.0})
        # Wait for debounce (5 samples)
        for i in range(5):
            res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 500.0, "AY": 500.0, "AZ": 500.0}, {"vibration": 100.0})
        self.assertEqual(res["health"], "NOMINAL")
        self.assertEqual(res["state"], "HIGH-VIBRATION")

    def test_03_missing_axes(self):
        res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 330.0}, {"vibration": 571.0})
        self.assertEqual(res["health"], "ADXL SENSOR ERROR")
        self.assertEqual(res["state"], "UNKNOWN")

    def test_04_invalid_non_numeric(self):
        # Let's say parser failed entirely or put None
        res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": None, "AY": None, "AZ": None}, {"vibration": 571.0})
        self.assertEqual(res["health"], "ADXL SENSOR ERROR")
        self.assertEqual(res["state"], "UNKNOWN")

    def test_05_accelerometer_unavailable(self):
        # 0V floating
        res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 0.0, "AY": 0.0, "AZ": 0.0}, {"vibration": 571.0})
        self.assertEqual(res["health"], "ADXL SENSOR ERROR")
        self.assertEqual(res["state"], "UNKNOWN")

    def test_06_stale_accelerometer_data(self):
        self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 330.0, "AY": 330.0, "AZ": 330.0}, {"vibration": 0.0})
        self.health.last_packet_time -= 5.0 # force stale
        status = self.health.get_health_status()
        self.assertEqual(status, "COMMUNICATION LOST")
        res = self.env.evaluate({"TEMP": 25.0, "HUM": 50.0, "AX": 330.0, "AY": 330.0, "AZ": 330.0}, status, {"vibration": 0.0})
        self.assertEqual(res["state"], "UNKNOWN")

    def test_07_recovery_from_high_vibration(self):
        # Trigger it
        for i in range(5):
            res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 500.0, "AY": 500.0, "AZ": 500.0}, {"vibration": 100.0})
        self.assertEqual(res["state"], "HIGH-VIBRATION")
        # Recover
        for i in range(5):
            res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 330.0, "AY": 330.0, "AZ": 330.0}, {"vibration": 0.0})
        self.assertEqual(res["state"], "NORMAL")

    def test_08_invalid_never_becomes_high_vibration(self):
        for i in range(10):
            res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 0.0, "AY": 0.0, "AZ": 0.0}, {"vibration": 571.0})
        self.assertEqual(res["health"], "ADXL SENSOR ERROR")
        self.assertEqual(res["state"], "UNKNOWN")

    def test_09_invalid_never_silently_becomes_normal(self):
        res = self._eval({"TEMP": 25.0, "HUM": 50.0, "AX": 1023.0, "AY": 1023.0, "AZ": 1023.0}, {"vibration": 0.0})
        self.assertEqual(res["health"], "ADXL SENSOR ERROR")
        self.assertEqual(res["state"], "UNKNOWN")

if __name__ == '__main__':
    unittest.main()
