import unittest
from aeris.performance_assessment import PerformanceAssessor
from aeris.config import ConfigManager

class TestPerformanceAssessor(unittest.TestCase):
    def setUp(self):
        self.config_mgr = ConfigManager()
        self.assessor = PerformanceAssessor(self.config_mgr)
        
    def test_normal_assessment(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.95},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.performance_assessment["performance_state"], "NORMAL")
        self.assertEqual(report.degradation_assessment["degradation_state"], "NORMAL")
        self.assertIn("NORMAL_TRACKING", report.performance_assessment["reason_codes"])

    def test_mild_degradation(self):
        # Assuming jitter threshold is 5.0
        report = self.assessor.assess(
            env_data={"state": "HIGH-VIBRATION"},
            tracking_metrics={"mean_jitter": 6.0, "lost_events": 1, "tracking_ratio": 0.85},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.performance_assessment["performance_state"], "DEGRADED")
        self.assertEqual(report.degradation_assessment["degradation_state"], "MILD_DEGRADATION")
        self.assertIn("HIGH_JITTER", report.performance_assessment["reason_codes"])
        
    def test_moderate_degradation(self):
        report = self.assessor.assess(
            env_data={"state": "HIGH-VIBRATION"},
            tracking_metrics={"mean_jitter": 12.0, "lost_events": 2, "tracking_ratio": 0.85},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.degradation_assessment["degradation_state"], "MODERATE_DEGRADATION")
        
    def test_severe_degradation_lost_events(self):
        # lost events > 3 * 3 = 9
        report = self.assessor.assess(
            env_data={"state": "PRECIPITATION"},
            tracking_metrics={"mean_jitter": 2.0, "lost_events": 10, "tracking_ratio": 0.60},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.degradation_assessment["degradation_state"], "SEVERE_DEGRADATION")
        self.assertIn("HIGH_LOST_EVENTS", report.performance_assessment["reason_codes"])

    def test_unknown_environment(self):
        report = self.assessor.assess(
            env_data={"state": "UNKNOWN"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.95},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.performance_assessment["performance_state"], "UNKNOWN")
        self.assertIn("UNKNOWN_ENVIRONMENT", report.performance_assessment["reason_codes"])

    def test_stale_data(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.95},
            baseline_metrics={},
            data_quality={"status": "STALE_SENSOR_DATA", "stale_sensor_data": True},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.performance_assessment["performance_state"], "UNKNOWN")
        self.assertIn("STALE_SENSOR_DATA", report.performance_assessment["reason_codes"])

    def test_insufficient_tracking_data(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": None, "lost_events": None, "tracking_ratio": None},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertEqual(report.performance_assessment["performance_state"], "UNKNOWN")
        self.assertIn("INSUFFICIENT_TRACKING_DATA", report.performance_assessment["reason_codes"])

    def test_low_tracking_ratio(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.20},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"}
        )
        self.assertIn("LOW_TRACKING_RATIO", report.performance_assessment["reason_codes"])
        
    def test_adaptation_accepted(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.95},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "ACCEPTED"}
        )
        self.assertIn("ADAPTATION_ACCEPTED", report.performance_assessment["reason_codes"])
        
    def test_adaptation_reverted(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.95},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "REVERTED"}
        )
        self.assertIn("ADAPTATION_REVERTED", report.performance_assessment["reason_codes"])

    def test_limitations_included(self):
        report = self.assessor.assess(
            env_data={"state": "NORMAL"},
            tracking_metrics={"mean_jitter": 1.0, "lost_events": 0, "tracking_ratio": 0.95},
            baseline_metrics={},
            data_quality={"status": "SYNCHRONIZED", "stale_sensor_data": False},
            adaptation_data={"status": "NOT_ATTEMPTED"},
            is_simulation=True
        )
        # Check simulation tag
        self.assertTrue(any("PHYSICAL EXPERIMENTAL VALIDATION: PENDING" in l for l in report.limitations))
        # Check confounder
        self.assertTrue(any("TARGET MOTION CONFOUNDER" in l for l in report.limitations))
        # Check stateless/scoreless
        self.assertTrue(any("SCORELESS ASSESSMENT" in l for l in report.limitations))

if __name__ == '__main__':
    unittest.main()
