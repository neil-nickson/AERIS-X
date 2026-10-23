import unittest
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.fusion import AerisFusionEngine, FusionRecord
from aeris.tracking_metrics import TrackingSample
from aeris.config import ConfigManager

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.config = ConfigManager()
        self.config.config = {"fusion": {"max_timestamp_delta_ms": 500}}
        self.fusion = AerisFusionEngine(self.config)
        
        self.dummy_env_data = {"TEMP": 25.0, "HUM": 50.0, "LIGHT": 200, "AX": 370, "AY": 350, "AZ": 390, "RAIN_DETECTED": 1}
        self.dummy_env_result = {"state": "DEGRADED", "active_conditions": ["HIGH-VIBRATION", "RAIN"]}
        self.dummy_dev_data = {"vibration": 5.2}
        
        self.dummy_vision_sample = TrackingSample(
            timestamp=0.0, frame_index=1, track_id=1, status="TRACKING", center_x=100.0, center_y=100.0,
            bbox_x1=90, bbox_y1=90, bbox_x2=110, bbox_y2=110, confidence=0.8, displacement=10.0, jitter=2.0,
            camera_fps=30.0, detector_fps=20.0, tracking_fps=None, position_error=None
        )

    def test_01_synchronized(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp + 0.1 # 100ms delta
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.data_quality, "SYNCHRONIZED")
        
    def test_02_100ms_delta(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp + 0.1 # 100ms delta
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertLessEqual(rec.timestamp_delta_ms, 100.001)

    def test_03_stale_data(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp + 0.6 # 600ms delta > 500ms
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.data_quality, "STALE_SENSOR_DATA")
        
    def test_04_no_sensor_data(self):
        # Did not call update_environment
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.data_quality, "NO_SENSOR_DATA")
        self.assertIsNone(rec.temperature)
        
    def test_05_communication_lost(self):
        self.fusion.update_environment(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.data_quality, "SENSOR_COMMUNICATION_LOST")
        
    def test_06_env_state_preservation(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.environment_state, "DEGRADED")
        
    def test_07_active_conditions_preservation(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.active_conditions, "HIGH-VIBRATION|RAIN")

    def test_08_tracking_waiting_state(self):
        self.dummy_vision_sample.status = "WAITING"
        self.dummy_vision_sample.track_id = None
        self.dummy_vision_sample.center_x = None
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.tracking_status, "WAITING")
        self.assertIsNone(rec.track_id)
        self.assertIsNone(rec.center_x)

    def test_09_tracking_state(self):
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.tracking_status, "TRACKING")
        self.assertEqual(rec.track_id, 1)

    def test_10_missed_state(self):
        self.dummy_vision_sample.status = "MISSED"
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.tracking_status, "MISSED")

    def test_11_lost_state(self):
        self.dummy_vision_sample.status = "LOST"
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(rec.tracking_status, "LOST")

    def test_12_timestamp_delta_exact(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        # Manually force time for exact math
        self.fusion.latest_env_timestamp = 100.180
        self.dummy_vision_sample.timestamp = 100.250
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertAlmostEqual(rec.timestamp_delta_ms, 70.0, places=1)

    def test_13_wall_clock_separation(self):
        self.fusion.latest_env_timestamp = 100.180
        self.dummy_vision_sample.timestamp = 100.250
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        # Should contain ISO-like string
        self.assertIsInstance(rec.wall_clock_timestamp, str)
        self.assertTrue("-" in rec.wall_clock_timestamp and ":" in rec.wall_clock_timestamp)

    def test_14_invalid_sensor_fields(self):
        self.fusion.update_environment({"TEMP": None}, "NOMINAL", {"state": "NORMAL", "active_conditions": []}, {})
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertIsNone(rec.temperature)
        self.assertEqual(rec.environment_state, "NORMAL")

    def test_15_invalid_tracking_fields(self):
        self.dummy_vision_sample.center_x = None
        self.dummy_vision_sample.center_y = None
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertIsNone(rec.center_x)

    def test_16_bounded_sensor_buffer(self):
        # We only store one latest element. Let's verify len(records)
        self.fusion.fuse_vision(self.dummy_vision_sample)
        self.assertEqual(len(self.fusion.records), 1)

    def test_17_csv_serialization(self):
        self.fusion.update_environment(self.dummy_env_data, "NOMINAL", self.dummy_env_result, self.dummy_dev_data)
        self.dummy_vision_sample.timestamp = self.fusion.latest_env_timestamp
        rec = self.fusion.fuse_vision(self.dummy_vision_sample)
        
        self.assertEqual(rec.active_conditions, "HIGH-VIBRATION|RAIN")
        self.assertEqual(rec.rain, 1)

    def test_18_summary_counters(self):
        self.fusion.fuse_vision(self.dummy_vision_sample) # NO_SENSOR_DATA
        self.assertEqual(self.fusion.summary.missing_sensor_records, 1)
        self.assertEqual(self.fusion.summary.tracking_records, 1)

    def test_19_empty_session(self):
        summary = self.fusion.summary
        self.assertEqual(summary.total_fused_records, 0)

    def test_20_shutdown_flush(self):
        self.fusion.fuse_vision(self.dummy_vision_sample)
        csv_p, json_p = self.fusion.export_logs("scratch/test_logs")
        self.assertTrue(os.path.exists(csv_p))
        self.assertTrue(os.path.exists(json_p))

if __name__ == '__main__':
    unittest.main()
