import unittest
import numpy as np
from aeris.dashboard import AerisDashboard

class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.dashboard = AerisDashboard()

    def test_dashboard_initializes(self):
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.canvas.shape, (800, 1280, 3))

    def test_missing_camera(self):
        # Should not crash if camera_frame is None
        state = {
            "mode": "SIMULATION",
            "camera_frame": None,
            "environment": {"TEMP": 25.0},
            "metrics": {},
            "performance": {},
            "degradation": {},
            "fusion": {},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_missing_sensor_data(self):
        # Should handle missing environment data
        state = {
            "mode": "SIMULATION",
            "camera_frame": None,
            "environment": {"state": "UNKNOWN", "health": "COMMUNICATION LOST"},
            "metrics": {},
            "performance": {},
            "degradation": {},
            "fusion": {"status": "SENSOR_COMMUNICATION_LOST"},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_stale_fusion_data(self):
        state = {
            "mode": "SIMULATION",
            "camera_frame": None,
            "environment": {"state": "NORMAL"},
            "metrics": {},
            "performance": {},
            "degradation": {},
            "fusion": {"status": "STALE_SENSOR_DATA", "delta_ms": 1500.0},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_no_detection(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "mode": "SIMULATION",
            "camera_frame": frame,
            "environment": {},
            "metrics": {"status": "WAITING"},
            "performance": {},
            "degradation": {},
            "fusion": {},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_active_tracking(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "mode": "SIMULATION",
            "camera_frame": frame,
            "environment": {},
            "metrics": {
                "status": "TRACKING",
                "center_x": 320,
                "center_y": 240,
                "bbox_x1": 300,
                "bbox_y1": 200,
                "bbox_x2": 340,
                "bbox_y2": 280,
                "confidence": 0.88,
                "mean_jitter": 2.5
            },
            "performance": {},
            "degradation": {},
            "fusion": {},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_unknown_performance(self):
        state = {
            "mode": "SIMULATION",
            "camera_frame": None,
            "environment": {},
            "metrics": {},
            "performance": {"performance_state": "UNKNOWN", "reason_codes": ["NO_SENSOR_DATA"]},
            "degradation": {"degradation_state": "UNKNOWN"},
            "fusion": {},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_adaptation_accepted(self):
        state = {
            "mode": "SIMULATION",
            "camera_frame": None,
            "environment": {},
            "metrics": {},
            "performance": {},
            "degradation": {},
            "fusion": {},
            "adaptation": {"state": "ADAPTATION_ACCEPTED", "strategy": "VIBRATION_SMOOTHING", "status": "ACCEPTED"}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)
        
    def test_adaptation_reverted(self):
        state = {
            "mode": "SIMULATION",
            "camera_frame": None,
            "environment": {},
            "metrics": {},
            "performance": {},
            "degradation": {},
            "fusion": {},
            "adaptation": {"state": "ADAPTATION_REJECTED", "strategy": "NONE", "status": "REVERTED"}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

    def test_severe_degradation(self):
        state = {
            "mode": "SIMULATION",
            "scenario": "SEVERE DEGRADATION",
            "camera_frame": None,
            "environment": {"state": "HIGH-VIBRATION"},
            "metrics": {},
            "performance": {"performance_state": "DEGRADED", "reason_codes": ["HIGH_JITTER", "LOW_TRACKING_RATIO"]},
            "degradation": {"degradation_state": "SEVERE_DEGRADATION"},
            "fusion": {},
            "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)
        
    def test_live_camera_state(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "mode": "LIVE MODE",
            "camera_frame": frame,
            "camera_source": "MAC CAMERA",
            "environment": {}, "metrics": {}, "performance": {}, "degradation": {}, "fusion": {}, "adaptation": {}
        }
        ui = self.dashboard.render(state)
        self.assertIsNotNone(ui)

if __name__ == '__main__':
    unittest.main()
