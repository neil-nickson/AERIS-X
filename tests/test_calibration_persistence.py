import unittest
import os
import sys
import json
import math
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.config import ConfigManager
from aeris.calibration import Calibrator

class TestCalibrationPersistence(unittest.TestCase):
    def setUp(self):
        # We will use a temporary config file to prevent destroying actual user config
        self.temp_config = Path("config/test_config.json")
        if self.temp_config.exists():
            os.remove(self.temp_config)
            
        self.config_mgr = ConfigManager(self.temp_config)
        self.config_mgr.config["thresholds"]["min_calibration_samples"] = 2
        
        # Mock SerialLink and SensorParser
        self.mock_link = MagicMock()
        self.mock_parser = MagicMock()
        
        # Set up a synthetic data stream simulating 200ms stationary reads
        self.synthetic_reads = [
            {"TEMP": 24.0, "HUM": 50.0, "LIGHT": 300, "AX": 341.0, "AY": 358.0, "AZ": 397.0, "RAIN": 1, "RAIN_DETECTED": False},
            {"TEMP": 24.1, "HUM": 50.1, "LIGHT": 301, "AX": 341.5, "AY": 358.5, "AZ": 397.5, "RAIN": 1, "RAIN_DETECTED": False},
            {"TEMP": 24.0, "HUM": 50.0, "LIGHT": 300, "AX": 341.0, "AY": 358.0, "AZ": 397.0, "RAIN": 1, "RAIN_DETECTED": False},
            {"TEMP": 24.2, "HUM": 50.2, "LIGHT": 302, "AX": 340.5, "AY": 357.5, "AZ": 396.5, "RAIN": 1, "RAIN_DETECTED": False},
            {"TEMP": 24.0, "HUM": 50.0, "LIGHT": 300, "AX": 341.0, "AY": 358.0, "AZ": 397.0, "RAIN": 1, "RAIN_DETECTED": False},
            {"TEMP": 24.0, "HUM": 50.0, "LIGHT": 300, "AX": 341.0, "AY": 358.0, "AZ": 397.0, "RAIN": 1, "RAIN_DETECTED": False}
        ]
        
        self.mock_parser.parse.side_effect = self.synthetic_reads
        
        self.calibrator = Calibrator(self.config_mgr, self.mock_link, self.mock_parser)

    def tearDown(self):
        if self.temp_config.exists():
            os.remove(self.temp_config)

    def test_A_calibration_values_are_persisted(self):
        # Run calibration fast using only the mock reads
        # We need duration > 0, but time.time() in calibrator will block.
        # We can patch time.time if needed, but it's simpler to run a 1-second calib with side_effect.
        import time
        real_time = time.time
        def fake_time():
            if not hasattr(fake_time, 't'): fake_time.t = 0
            fake_time.t += 0.5
            return fake_time.t
            
        with unittest.mock.patch('time.time', fake_time):
            success = self.calibrator.run_calibration(duration=2)
            
        self.assertTrue(success)
        
        # Verify JSON file is written
        self.assertTrue(self.temp_config.exists())
        
        with open(self.temp_config, 'r') as f:
            data = json.load(f)
            self.assertIn("calibration", data)
            self.assertIn("timestamp", data["calibration"])
            self.assertAlmostEqual(data["calibration"]["AX"]["mean"], 341.17, places=1)
            self.assertAlmostEqual(data["calibration"]["TEMP"]["mean"], 24.03, places=1)

    def test_B_fresh_config_manager_loads_persisted_values(self):
        # Pre-populate temp config
        self.config_mgr.update_calibration({"AX": {"mean": 341.0}, "AY": {"mean": 358.0}, "AZ": {"mean": 397.0}}, "2026-09-11T12:00:00")
        
        # Instantiate a completely fresh manager
        fresh_mgr = ConfigManager(self.temp_config)
        self.assertEqual(fresh_mgr.get_baseline("AX"), 341.0)
        self.assertEqual(fresh_mgr.get_baseline("AY"), 358.0)
        self.assertEqual(fresh_mgr.get_baseline("AZ"), 397.0)

    def test_C_identical_readings_produce_zero_vibration(self):
        self.config_mgr.update_calibration({"AX": {"mean": 341.0}, "AY": {"mean": 358.0}, "AZ": {"mean": 397.0}}, "2026-09-11T12:00:00")
        
        # Simulating Phase 1 vib logic
        base_x = self.config_mgr.get_baseline("AX")
        base_y = self.config_mgr.get_baseline("AY")
        base_z = self.config_mgr.get_baseline("AZ")
        
        reading = {"AX": 341.0, "AY": 358.0, "AZ": 397.0}
        
        dx = reading["AX"] - base_x
        dy = reading["AY"] - base_y
        dz = reading["AZ"] - base_z
        vib_mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        self.assertEqual(vib_mag, 0.0)

    def test_D_phase1_loads_newly_persisted_values(self):
        # We assert that ConfigManager.get_baseline() correctly pulls from disk
        self.config_mgr.update_calibration({"TEMP": {"mean": 99.9}}, "2026-09-11T12:00:00")
        
        fresh_mgr = ConfigManager(self.temp_config)
        self.assertEqual(fresh_mgr.get_baseline("TEMP"), 99.9)

    def test_E_stale_baseline_mismatch_is_detectable(self):
        # Simulate an old baseline
        self.config_mgr.update_calibration({"AX": {"mean": 100.0}, "AY": {"mean": 100.0}, "AZ": {"mean": 100.0}}, "2025-01-01T12:00:00")
        
        fresh_mgr = ConfigManager(self.temp_config)
        base_x = fresh_mgr.get_baseline("AX")
        
        reading = {"AX": 341.0, "AY": 358.0, "AZ": 397.0} # Current physical reading
        dx = reading["AX"] - base_x
        
        # The deviation is huge, which detects a mismatch vs reality
        self.assertTrue(abs(dx) > 100.0)

if __name__ == '__main__':
    unittest.main()
