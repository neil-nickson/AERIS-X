import unittest
import os
import sys
import time
import json
import math
import csv
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.experiment_runner import ExperimentSession, ExperimentState, ConditionVerification, ExperimentConditions
from aeris.fusion import FusionRecord
from aeris.config import ConfigManager

class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.config = ConfigManager()
        self.config.config = {"experiment": {"stabilization_seconds": 0.1, "recording_seconds": 0.2, "minimum_tracking_frames": 2}}
        self.session = ExperimentSession(self.config, "test_exp_123")
        
        self.dummy_record = FusionRecord(
            wall_clock_timestamp="2026", monotonic_timestamp=0.0, frame_index=1,
            sensor_timestamp=0.0, vision_timestamp=0.0, timestamp_delta_ms=0.0, data_quality="SYNCHRONIZED",
            temperature=25.0, humidity=50.0, light=200, ax=300, ay=300, az=300, vibration_index=5.0, rain=0,
            environment_state="NORMAL", active_conditions="HIGH-VIBRATION", sensor_health="NOMINAL",
            track_id=1, tracking_status="TRACKING", center_x=100, center_y=100,
            bbox_x1=90, bbox_y1=90, bbox_x2=110, bbox_y2=110, confidence=0.8,
            displacement=2.0, jitter=1.0, camera_fps=30.0, detector_fps=20.0, tracking_fps=25.0
        )

    def test_01_new_session(self):
        self.assertEqual(self.session.state, ExperimentState.NOT_STARTED)

    def test_02_baseline_creation(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.assertEqual(self.session.state, ExperimentState.PREPARING)
        self.assertEqual(self.session.current_trial_number, 1)

    def test_03_trial_numbering(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.stop_trial()
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.assertEqual(self.session.current_trial_number, 2)
        self.session.start_trial(ExperimentConditions.LOW_LIGHT)
        self.assertEqual(self.session.current_trial_number, 1)

    def test_04_trial_lifecycle(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.update(self.dummy_record) # PREP -> STAB
        self.assertEqual(self.session.state, ExperimentState.STABILIZING)
        time.sleep(0.15)
        self.session.update(self.dummy_record) # STAB -> REC
        self.assertEqual(self.session.state, ExperimentState.RECORDING)
        self.session.update(self.dummy_record) # 1 frame
        self.session.update(self.dummy_record) # 2 frames
        time.sleep(0.25)
        self.session.update(self.dummy_record) # REC -> STOP
        self.assertEqual(self.session.state, ExperimentState.COMPLETED)

    def test_05_failed_trial(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.update(self.dummy_record)
        time.sleep(0.15)
        self.session.update(self.dummy_record)
        # Stop early, no minimum tracking frames (needs 2, got 0)
        self.session.stop_trial()
        self.assertEqual(self.session.trials[0].status, ExperimentState.FAILED)

    def test_06_stabilization_failure(self):
        # We need HIGH_VIBRATION but dummy_record says active_conditions="LOW-LIGHT"
        self.session.start_trial(ExperimentConditions.HIGH_VIBRATION)
        self.dummy_record.active_conditions = "LOW-LIGHT"
        self.session.update(self.dummy_record) # STAB
        time.sleep(0.15)
        self.session.update(self.dummy_record) # Verifies -> fails -> FAILED
        self.assertEqual(self.session.state, ExperimentState.FAILED)
        self.assertEqual(self.session.trials[0].verification, ConditionVerification.FAILED)

    def test_07_condition_verification_success(self):
        self.session.start_trial(ExperimentConditions.HIGH_VIBRATION)
        self.dummy_record.active_conditions = "HIGH-VIBRATION"
        self.session.update(self.dummy_record) # STAB
        time.sleep(0.15)
        self.session.update(self.dummy_record) # REC
        self.assertEqual(self.session.state, ExperimentState.RECORDING)

    def test_08_condition_verification_failure(self):
        # Same as 6
        self.test_06_stabilization_failure()

    def test_09_requested_vs_observed_distinction(self):
        self.session.start_trial(ExperimentConditions.LOW_LIGHT)
        self.dummy_record.active_conditions = "NORMAL"
        self.session.update(self.dummy_record) # STAB
        time.sleep(0.15)
        self.session.update(self.dummy_record)
        # It should fail validation
        self.assertEqual(self.session.trials[0].verification, ConditionVerification.FAILED)

    def test_10_minimum_tracking_frame(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.update(self.dummy_record)
        time.sleep(0.15)
        self.session.update(self.dummy_record) # REC starts
        self.session.update(self.dummy_record) # 1
        self.session.update(self.dummy_record) # 2
        self.session.stop_trial()
        self.assertEqual(self.session.trials[0].status, ExperimentState.COMPLETED)

    def test_11_no_real_drone_available(self):
        c, j = self.session.export_logs("scratch")
        with open(j, 'r') as f:
            d = json.load(f)
        self.assertFalse(d["real_drone_available"])

    def test_12_ground_truth_unavailable(self):
        c, j = self.session.export_logs("scratch")
        with open(j, 'r') as f:
            d = json.load(f)
        self.assertFalse(d["ground_truth_available"])

    def test_13_repeated_trials(self):
        self.test_03_trial_numbering() # generates multiple
        self.assertEqual(len(self.session.trial_counts), 2)

    def test_14_summary_aggregation(self):
        self.test_10_minimum_tracking_frame() # 1 success
        c, j = self.session.export_logs("scratch")
        with open(j, 'r') as f:
            d = json.load(f)
        self.assertEqual(d["successful_trials"], 1)

    def test_15_missing_sensor_fields(self):
        self.dummy_record.temperature = None
        self.test_10_minimum_tracking_frame()
        self.assertIsNone(self.session.trials[0].mean_temperature)

    def test_16_missing_tracking_fields(self):
        self.dummy_record.displacement = None
        self.test_10_minimum_tracking_frame()
        self.assertIsNone(self.session.trials[0].mean_displacement)

    def test_17_csv_serialization(self):
        self.test_10_minimum_tracking_frame()
        c, j = self.session.export_logs("scratch")
        self.assertTrue(os.path.exists(c))

    def test_18_json_summary_generation(self):
        self.test_10_minimum_tracking_frame()
        c, j = self.session.export_logs("scratch")
        self.assertTrue(os.path.exists(j))

    def test_19_descriptive_baseline_difference(self):
        self.test_10_minimum_tracking_frame() # Creates a BASELINE
        self.session.start_trial(ExperimentConditions.HIGH_VIBRATION)
        self.dummy_record.active_conditions = "HIGH-VIBRATION"
        self.dummy_record.jitter = 5.0 # Jitter was 1.0 in baseline
        
        self.session.update(self.dummy_record)
        time.sleep(0.15)
        self.session.update(self.dummy_record)
        self.session.update(self.dummy_record)
        self.session.update(self.dummy_record)
        self.session.stop_trial()
        
        c, j = self.session.export_logs("scratch")
        with open(j, 'r') as f:
            d = json.load(f)
            
        t2 = d["trials"][1]
        self.assertEqual(t2["descriptive_comparison"]["baseline_difference_mean_jitter"], 4.0)

    def test_20_no_division_by_zero(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.stop_trial()
        self.session.export_logs("scratch")
        # should not throw

    def test_21_safe_stop(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.stop_safely()
        self.assertEqual(self.session.state, ExperimentState.COMPLETED)

    def test_22_invalid_experiment_condition(self):
        # Runner doesn't strict check the enum during start, but verification will fail
        self.session.start_trial("NONEXISTENT")
        self.session.update(self.dummy_record)
        time.sleep(0.15)
        self.session.update(self.dummy_record)
        self.assertEqual(self.session.trials[0].verification, ConditionVerification.FAILED)

    def test_23_multi_condition_naming(self):
        self.session.start_trial(ExperimentConditions.MULTI_CONDITION)
        self.assertEqual(self.session.current_condition, ExperimentConditions.MULTI_CONDITION)

    def test_24_incomplete_trial_handling(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.update(self.dummy_record)
        self.session.start_trial(ExperimentConditions.LOW_LIGHT) # aborts first
        self.assertEqual(self.session.trials[0].status, ExperimentState.FAILED)

    def test_25_session_restart_behavior(self):
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.assertEqual(self.session.current_trial_number, 1)

    def test_26_trial_csv_export(self):
        # Remove experiments directory to test directory creation
        import shutil
        if os.path.exists("experiments"):
            shutil.rmtree("experiments")
            
        # No-row behavior
        self.session.start_trial(ExperimentConditions.BASELINE)
        # We don't advance past STABILIZING, so records is empty
        self.session.stop_trial()
        self.assertFalse(os.path.exists("experiments")) # Should not create dir/file for 0 rows
        
        # Collect at least one sample
        self.session.start_trial(ExperimentConditions.BASELINE)
        self.session.update(self.dummy_record) # STABILIZING
        time.sleep(0.15)
        self.session.update(self.dummy_record) # Enters RECORDING
        self.session.update(self.dummy_record) # Appends to trial_stats.records
        
        # Stopping the trial -> CSV creation, file close/flush
        self.session.stop_trial()
        
        # Check if any CSV file was created in experiments/
        self.assertTrue(os.path.exists("experiments"))
        files = os.listdir("experiments")
        csv_files = [f for f in files if f.endswith(".csv") and f.startswith("phase8_")]
        self.assertTrue(len(csv_files) >= 1)
        
        # Check CSV header correctness and row preservation
        csv_path = os.path.join("experiments", csv_files[0])
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            self.assertIn("experiment_id", headers)
            self.assertIn("condition", headers)
            self.assertIn("trial_number", headers)
            self.assertIn("temperature", headers)
            
            rows = list(reader)
            self.assertEqual(len(rows), 1) # One record was appended
            self.assertEqual(rows[0]["condition"], ExperimentConditions.BASELINE)
            self.assertEqual(rows[0]["trial_phase"], ExperimentState.RECORDING)

        # Cleanup generated test CSV and ensure .gitkeep is preserved
        if os.path.exists(csv_path):
            os.remove(csv_path)
        gitkeep = os.path.join("experiments", ".gitkeep")
        if not os.path.exists(gitkeep):
            with open(gitkeep, "w") as f:
                f.write("# Keep directory\n")

if __name__ == '__main__':
    unittest.main()
