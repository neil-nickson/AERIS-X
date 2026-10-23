import unittest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.detector import AerisDroneDetector

class MockConfigManager:
    def __init__(self, overrides=None):
        self.config = {
            "detector": {
                "model_path": "models/fake_model.pt",
                "target_class": "quadcopter",
                "confidence_threshold": 0.40,
                "device": "cpu"
            }
        }
        if overrides:
            self.config["detector"].update(overrides)

class MockBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [xyxy]
        self.conf = [conf]

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.config = MockConfigManager()

    @patch('aeris.detector.ULTRALYTICS_AVAILABLE', True)
    @patch('aeris.detector.os.path.exists')
    def test_missing_model(self, mock_exists):
        mock_exists.return_value = False
        det = AerisDroneDetector(self.config)
        success = det.initialize()
        
        self.assertFalse(success)
        self.assertEqual(det.get_health(), "MODEL_UNAVAILABLE")

    @patch('aeris.detector.ULTRALYTICS_AVAILABLE', True)
    @patch('aeris.detector.os.path.exists')
    @patch('aeris.detector.YOLO', create=True)
    def test_missing_target_class(self, mock_yolo, mock_exists):
        mock_exists.return_value = True
        
        mock_model = MagicMock()
        # Mock class mapping that DOES NOT have 'drone'
        mock_model.names = {0: "person", 1: "car"}
        mock_yolo.return_value = mock_model
        
        det = AerisDroneDetector(self.config)
        success = det.initialize()
        
        self.assertFalse(success)
        self.assertEqual(det.get_health(), "MODEL_LOAD_ERROR")

    @patch('aeris.detector.ULTRALYTICS_AVAILABLE', True)
    @patch('aeris.detector.os.path.exists')
    @patch('aeris.detector.YOLO', create=True)
    def test_successful_initialization(self, mock_yolo, mock_exists):
        mock_exists.return_value = True
        
        mock_model = MagicMock()
        # Mock class mapping that DOES have 'drone'
        mock_model.names = {0: "person", 1: "quadcopter"}
        mock_yolo.return_value = mock_model
        
        det = AerisDroneDetector(self.config)
        success = det.initialize()
        
        self.assertTrue(success)
        self.assertEqual(det.get_health(), "DETECTOR_NOMINAL")
        self.assertEqual(det.target_class_id, 1)

    @patch('aeris.detector.ULTRALYTICS_AVAILABLE', True)
    @patch('aeris.detector.os.path.exists')
    @patch('aeris.detector.YOLO', create=True)
    def test_detection_parsing(self, mock_yolo, mock_exists):
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.names = {0: "quadcopter"}
        
        # Mock a prediction returning 2 boxes: one high conf, one low conf
        box1 = MockBox([10, 20, 100, 200], 0.95)
        box2 = MockBox([30, 40, 50, 60], 0.10) # below 0.40 threshold
        
        mock_result = MockResult([box1, box2])
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        det = AerisDroneDetector(self.config)
        det.initialize()
        
        detections = det.detect("fake_frame")
        
        # Only box1 should survive thresholding
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["confidence"], 0.95)
        self.assertEqual(detections[0]["bbox"], (10, 20, 100, 200))

if __name__ == '__main__':
    unittest.main()
