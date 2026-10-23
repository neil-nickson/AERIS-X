import unittest
import os
import sys
from unittest.mock import MagicMock, patch
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.camera import AerisCamera
from aeris.config import ConfigManager

class MockConfigManager:
    def __init__(self, overrides=None):
        self.config = {
            "camera": {
                "source": "mac",
                "device_index": 0,
                "index": 0,
                "width": 1280,
                "height": 720,
                "target_fps": 30,
                "display": False,
                "recording_enabled": False
            }
        }
        if overrides:
            self.config["camera"].update(overrides)

class TestCamera(unittest.TestCase):
    def setUp(self):
        self.config = MockConfigManager()
        
    def test_01_config_parsing_mac(self):
        cam = AerisCamera(self.config)
        self.assertEqual(cam.source, "mac")
        self.assertEqual(cam.camera_index, 0)
        self.assertEqual(cam.width, 1280)
        self.assertEqual(cam.height, 720)
        self.assertEqual(cam.health_status, "CAMERA_UNAVAILABLE")

    def test_02_config_parsing_iphone(self):
        config2 = MockConfigManager({"source": "iphone", "device_index": 2})
        cam = AerisCamera(config2)
        self.assertEqual(cam.source, "iphone")
        self.assertEqual(cam.camera_index, 2)
        
    def test_03_config_fallback_index(self):
        config3 = MockConfigManager()
        del config3.config["camera"]["device_index"]
        config3.config["camera"]["index"] = 5
        cam = AerisCamera(config3)
        self.assertEqual(cam.camera_index, 5)

    @patch('aeris.camera.cv2.VideoCapture')
    def test_04_initialization_success(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        res = cam.initialize()
        
        self.assertTrue(res)
        self.assertEqual(cam.health_status, "CAMERA_NOMINAL")
        self.assertIsNotNone(cam.start_time)

    @patch('aeris.camera.cv2.VideoCapture')
    def test_05_initialization_failure(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        res = cam.initialize()
        
        self.assertFalse(res)
        self.assertEqual(cam.health_status, "CAMERA_UNAVAILABLE")

    @patch('aeris.camera.cv2.VideoCapture')
    def test_06_frame_read_success(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        cam.initialize()
        
        success, frame = cam.read()
        self.assertTrue(success)
        self.assertEqual(cam.health_status, "CAMERA_NOMINAL")
        self.assertIsNotNone(cam.last_frame_timestamp)
        self.assertEqual(cam.dropped_frames, 0)
        self.assertEqual(cam.read_failures, 0)

    @patch('aeris.camera.cv2.VideoCapture')
    def test_07_frame_read_failure(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        cam.initialize()
        
        success, frame = cam.read()
        self.assertFalse(success)
        self.assertEqual(cam.health_status, "CAMERA_READ_ERROR")
        self.assertEqual(cam.read_failures, 1)

    @patch('aeris.camera.cv2.VideoCapture')
    def test_08_empty_frame_drop(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.array([]))
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        cam.initialize()
        
        success, frame = cam.read()
        self.assertFalse(success)
        self.assertEqual(cam.dropped_frames, 1)

    @patch('aeris.camera.cv2.VideoCapture')
    @patch('aeris.camera.time.time')
    def test_09_fps_measurement(self, mock_time, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        
        # 1 call in initialize, 2 calls per read loop
        # Total calls = 1 + (30 * 2) = 61
        time_values = [0.0] + [0.0]*59 + [1.5]
        mock_time.side_effect = time_values
        
        cam.initialize()
        for _ in range(30):
            cam.read()
            
        fps = cam.get_fps()
        self.assertEqual(fps, 20.0)

    def test_10_latency_string(self):
        cam = AerisCamera(self.config)
        self.assertEqual(cam.get_latency_string(), "End-to-End Latency: NOT MEASURED")

    @patch('aeris.camera.cv2.VideoCapture')
    def test_11_clean_shutdown(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_vc.return_value = mock_cap
        
        cam = AerisCamera(self.config)
        cam.initialize()
        
        cam.release()
        self.assertEqual(cam.health_status, "CAMERA_UNAVAILABLE")
        self.assertIsNone(cam.cap)
        self.assertTrue(mock_cap.release.called)

    def test_12_get_diagnostic_stats(self):
        cam = AerisCamera(self.config)
        stats = cam.get_diagnostic_stats()
        self.assertEqual(stats["source"], "mac")
        self.assertEqual(stats["index"], 0)
        self.assertEqual(stats["read_failures"], 0)

if __name__ == '__main__':
    unittest.main()
