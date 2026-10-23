import unittest
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.tracker import TargetTrack
from aeris.tracking_metrics import TrackingMetrics

class TestTrackingMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = TrackingMetrics()

    def test_01_empty_session(self):
        summary = self.metrics.get_summary()
        self.assertEqual(summary.total_frames, 0)
        self.assertEqual(summary.tracking_ratio, 0.0)
        self.assertIsNone(summary.mean_displacement)
        self.assertIsNone(summary.mean_jitter)
        self.assertIsNone(summary.mean_position_error)

    def test_02_consecutive_positions_displacement(self):
        t1 = TargetTrack(1, (90, 90, 110, 110), 100, 100, 20, 20, 0.9, 1, 0, "TRACKING")
        t2 = TargetTrack(1, (100, 90, 120, 110), 110, 100, 20, 20, 0.9, 2, 0, "TRACKING")
        t3 = TargetTrack(1, (110, 90, 130, 110), 120, 100, 20, 20, 0.9, 3, 0, "TRACKING")
        
        self.metrics.update(t1)
        self.metrics.update(t2)
        s3 = self.metrics.update(t3)
        
        self.assertEqual(s3.displacement, 10.0)
        
        summary = self.metrics.get_summary()
        self.assertEqual(summary.mean_displacement, 10.0)

    def test_03_jitter_calculation(self):
        # 0: center(100, 100)
        # 1: center(110, 100) -> displacement (10, 0)
        # 2: center(120, 110) -> displacement (10, 10) -> jitter = sqrt((10-10)^2 + (10-0)^2) = 10
        t1 = TargetTrack(1, (0, 0, 0, 0), 100, 100, 20, 20, 0.9, 1, 0, "TRACKING")
        t2 = TargetTrack(1, (0, 0, 0, 0), 110, 100, 20, 20, 0.9, 2, 0, "TRACKING")
        t3 = TargetTrack(1, (0, 0, 0, 0), 120, 110, 20, 20, 0.9, 3, 0, "TRACKING")
        
        self.metrics.update(t1)
        self.metrics.update(t2)
        s3 = self.metrics.update(t3)
        
        self.assertEqual(s3.jitter, 10.0)
        summary = self.metrics.get_summary()
        self.assertEqual(summary.mean_jitter, 10.0)

    def test_04_tracking_ratio(self):
        t_track = TargetTrack(1, (0, 0, 0, 0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        t_miss = TargetTrack(1, (0, 0, 0, 0), 10, 10, 10, 10, 0.9, 1, 1, "MISSED")
        
        for _ in range(8):
            self.metrics.update(t_track)
        for _ in range(2):
            self.metrics.update(t_miss)
            
        summary = self.metrics.get_summary()
        self.assertEqual(summary.total_frames, 10)
        self.assertEqual(summary.tracking_ratio, 0.8)
        self.assertEqual(summary.missed_ratio, 0.2)

    def test_05_lost_event_counting(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        t2 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 2, 1, "MISSED")
        t3 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 3, 2, "LOST")
        t4 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 4, 3, "LOST") # Continued lost shouldn't recount
        
        self.metrics.update(t1)
        self.metrics.update(t2)
        self.metrics.update(t3)
        self.metrics.update(t4)
        
        summary = self.metrics.get_summary()
        self.assertEqual(summary.lost_events, 1)

    def test_06_multiple_lost_events(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        l1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 3, 2, "LOST")
        t2 = TargetTrack(2, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        l2 = TargetTrack(2, (0,0,0,0), 10, 10, 10, 10, 0.9, 3, 2, "LOST")
        
        self.metrics.update(t1)
        self.metrics.update(l1)
        self.metrics.update(t2)
        self.metrics.update(l2)
        
        self.assertEqual(self.metrics.get_summary().lost_events, 2)

    def test_07_recovery(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        m1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 2, 1, "MISSED")
        
        self.metrics.update(t1)
        self.metrics.update(m1)
        # Fake sleep wouldn't affect time.monotonic during tests effectively unless we mock it, 
        # but the recovery list should just have length 1.
        self.metrics.update(t1) # Recovered
        
        self.assertEqual(len(self.metrics.recoveries), 1)

    def test_08_no_false_recovery(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        l1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 3, 2, "LOST")
        t2 = TargetTrack(2, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        
        self.metrics.update(t1)
        self.metrics.update(l1)
        self.metrics.update(t2)
        
        # ID changed, no recovery recorded
        self.assertEqual(len(self.metrics.recoveries), 0)

    def test_09_ground_truth_position_error(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        # Track is at (10,10), GT is at (13,14) -> dx=3, dy=4 -> error=5
        self.metrics.update(t1, ground_truth_center=(13, 14))
        
        summary = self.metrics.get_summary()
        self.assertTrue(summary.ground_truth_available)
        self.assertEqual(summary.mean_position_error, 5.0)
        self.assertEqual(summary.peak_position_error, 5.0)

    def test_10_missing_ground_truth(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        self.metrics.update(t1, ground_truth_center=None)
        
        summary = self.metrics.get_summary()
        self.assertFalse(summary.ground_truth_available)
        self.assertIsNone(summary.mean_position_error)

    def test_11_bounding_box_stability(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 20, 40, 0.9, 1, 0, "TRACKING")
        self.metrics.update(t1)
        summary = self.metrics.get_summary()
        self.assertEqual(summary.mean_bbox_width, 20.0)
        self.assertEqual(summary.mean_bbox_height, 40.0)

    def test_12_fps_handling(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        sample = self.metrics.update(t1, camera_fps=None, detector_fps=None)
        self.assertIsNone(sample.camera_fps)
        self.assertIsNone(sample.detector_fps)

    def test_13_single_sample_jitter(self):
        t1 = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        sample = self.metrics.update(t1)
        self.assertIsNone(sample.jitter)
        self.assertIsNone(sample.displacement)

    def test_14_invalid_input(self):
        # Passing None for track should not crash
        sample = self.metrics.update(None)
        self.assertEqual(sample.status, "WAITING")
        self.assertIsNone(sample.track_id)

    def test_15_req_a_empty_camera(self):
        # A. Empty camera/no detections for 100 frames
        for _ in range(100):
            sample = self.metrics.update(None)
            self.assertEqual(sample.status, "WAITING")
        
        s = self.metrics.get_summary()
        self.assertEqual(s.missed_frames, 0)
        self.assertEqual(s.lost_events, 0)

    def test_16_req_b_tracking_to_lost(self):
        # B. TRACKING -> 10 missed frames -> LOST (exactly one LOST event)
        t = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "TRACKING")
        self.metrics.update(t)
        
        # 10 missed frames
        for i in range(1, 11):
            m = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 1+i, i, "MISSED")
            self.metrics.update(m)
            
        # LOST frame
        l = TargetTrack(1, (0,0,0,0), 10, 10, 10, 10, 0.9, 12, 11, "LOST")
        self.metrics.update(l)
        
        s = self.metrics.get_summary()
        self.assertEqual(s.lost_events, 1)

    def test_17_req_c_after_lost_empty(self):
        # C. After LOST, 100 more empty frames (no additional missed/lost)
        self.test_16_req_b_tracking_to_lost()
        
        for _ in range(100):
            self.metrics.update(None)
            
        s = self.metrics.get_summary()
        self.assertEqual(s.lost_events, 1)
        self.assertEqual(s.missed_frames, 10) # from the 10 missed frames earlier
        
    def test_18_req_d_new_detection_after_lost(self):
        # D. New detection after LOST (new track ID, no recovery)
        self.test_16_req_b_tracking_to_lost()
        
        t2 = TargetTrack(2, (0,0,0,0), 10, 10, 10, 10, 0.9, 1, 0, "NEW")
        self.metrics.update(t2)
        
        s = self.metrics.get_summary()
        self.assertEqual(len(self.metrics.recoveries), 0)

if __name__ == '__main__':
    unittest.main()
