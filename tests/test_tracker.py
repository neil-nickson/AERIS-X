import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.tracker import AerisTracker

class MockConfigManager:
    def __init__(self, overrides=None):
        self.config = {
            "tracker": {
                "association_iou_threshold": 0.3,
                "max_center_distance_ratio": 0.3,
                "max_missed_frames": 2,
                "min_detection_confidence": 0.40,
                "trajectory_length": 5,
                "target_selection_policy": "highest_confidence"
            }
        }
        if overrides:
            self.config["tracker"].update(overrides)

class TestTracker(unittest.TestCase):
    def setUp(self):
        self.config = MockConfigManager()
        self.tracker = AerisTracker(self.config)
        self.w, self.h = 1280, 720

    def test_new_track_creation(self):
        detections = [{"bbox": (100, 100, 160, 160), "confidence": 0.8}]
        track = self.tracker.update(detections, self.w, self.h)
        
        self.assertIsNotNone(track)
        self.assertEqual(track.status, "NEW")
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.center_x, 130.0)
        self.assertEqual(track.center_y, 130.0)
        self.assertEqual(len(track.trajectory), 1)

    def test_tracking_association(self):
        # Frame 1
        d1 = [{"bbox": (100, 100, 160, 160), "confidence": 0.8}]
        self.tracker.update(d1, self.w, self.h)
        
        # Frame 2 (Moved slightly)
        d2 = [{"bbox": (110, 105, 170, 165), "confidence": 0.85}]
        track = self.tracker.update(d2, self.w, self.h)
        
        self.assertEqual(track.status, "TRACKING")
        self.assertEqual(track.track_id, 1)
        self.assertEqual(len(track.trajectory), 2)
        self.assertEqual(track.center_x, 140.0)

    def test_missed_and_lost(self):
        # Frame 1
        self.tracker.update([{"bbox": (100, 100, 160, 160), "confidence": 0.8}], self.w, self.h)
        
        # Frame 2 (No detection)
        track = self.tracker.update([], self.w, self.h)
        self.assertEqual(track.status, "MISSED")
        self.assertEqual(track.missed_frames, 1)
        self.assertEqual(track.track_id, 1)
        
        # Frame 3 (No detection)
        track = self.tracker.update([], self.w, self.h)
        self.assertEqual(track.status, "MISSED")
        self.assertEqual(track.missed_frames, 2)
        
        # Frame 4 (No detection, exceeds max_missed_frames=2)
        track = self.tracker.update([], self.w, self.h)
        self.assertEqual(track.status, "LOST")
        
        # Frame 5 (New detection should get ID 2)
        track = self.tracker.update([{"bbox": (200, 200, 260, 260), "confidence": 0.9}], self.w, self.h)
        self.assertEqual(track.status, "NEW")
        self.assertEqual(track.track_id, 2)

    def test_multiple_detections_highest_confidence(self):
        detections = [
            {"bbox": (10, 10, 20, 20), "confidence": 0.5},
            {"bbox": (100, 100, 160, 160), "confidence": 0.9},
            {"bbox": (50, 50, 60, 60), "confidence": 0.7}
        ]
        track = self.tracker.update(detections, self.w, self.h)
        self.assertEqual(track.confidence, 0.9)
        self.assertEqual(track.bbox, (100, 100, 160, 160))

if __name__ == '__main__':
    unittest.main()
