import unittest
from unittest.mock import MagicMock
from aeris.adaptive_controller import AdaptiveController, AdaptiveState

class TestAdaptiveController(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.config = {
            "adaptive_tracking": {
                "enabled": True,
                "minimum_improvement": 10.0,
                "validation_window": 5,
                "max_adaptation_attempts": 3,
                "cooldown_frames": 5,
                "degradation_jitter_threshold": 5.0,
                "degradation_lost_threshold": 2
            }
        }
        self.controller = AdaptiveController(self.mock_config)

    def test_initial_state(self):
        self.assertEqual(self.controller.state, AdaptiveState.NORMAL)
        
    def test_degraded_state(self):
        for _ in range(6):
            self.controller.update_tracking_metrics(jitter=6.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.update_environment("NORMAL", "NONE", "NOMINAL")
        self.controller.evaluate()
        # It goes from NORMAL -> DEGRADED -> ADAPTING or INSUFFICIENT_EVIDENCE
        # Since NONE has no evidence, it goes to INSUFFICIENT_EVIDENCE. Let's check that.
        self.assertEqual(self.controller.state, AdaptiveState.INSUFFICIENT_EVIDENCE)

    def test_insufficient_evidence(self):
        # Setup degraded
        self.controller.evidence_cache = {} # Force empty to guarantee insufficient evidence
        for _ in range(6):
            self.controller.update_tracking_metrics(jitter=6.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.update_environment("NORMAL", "LOW-LIGHT", "NOMINAL")
        self.controller.evaluate() # Moves to degraded
        self.controller.evaluate() # Moves to insufficient evidence
        self.assertEqual(self.controller.state, AdaptiveState.INSUFFICIENT_EVIDENCE)
        
        self.controller.evaluate() # Cooldown active
        self.assertTrue(self.controller.cooldown_counter > 0)

    def test_sensor_communication_loss(self):
        for _ in range(6):
            self.controller.update_tracking_metrics(jitter=6.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.update_environment("UNKNOWN", "UNKNOWN", "SENSOR_COMMUNICATION_LOST")
        self.controller.evaluate() # Moves to degraded
        self.controller.evaluate() # Tries to adapt
        self.assertEqual(self.controller.state, AdaptiveState.UNKNOWN)

    def test_valid_adaptation_accepted(self):
        # Mock evidence
        self.controller.evidence_cache["LOW-LIGHT"] = True
        
        for _ in range(6):
            self.controller.update_tracking_metrics(jitter=10.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.update_environment("NORMAL", "LOW-LIGHT", "NOMINAL")
        self.controller.evaluate() # NORMAL -> DEGRADED -> ADAPTING
        self.assertEqual(self.controller.state, AdaptiveState.ADAPTING)
        self.assertEqual(self.controller.current_strategy, "LOW_LIGHT_PREPROCESSING")
        
        self.controller.update_tracking_metrics(jitter=2.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.evaluate() # ADAPTING -> VALIDATING
        self.assertEqual(self.controller.state, AdaptiveState.VALIDATING)
        
        # Now provide better metrics during validation
        for _ in range(5):
            self.controller.update_tracking_metrics(jitter=2.0, displacement=1.0, tracking_status="TRACKING")
            self.controller.evaluate()
            
        # Should be accepted because 2.0 is much better than 10.0
        self.assertEqual(self.controller.state, AdaptiveState.ADAPTATION_ACCEPTED)

    def test_invalid_adaptation_rejected(self):
        # Mock evidence
        self.controller.evidence_cache["LOW-LIGHT"] = True
        
        for _ in range(6):
            self.controller.update_tracking_metrics(jitter=10.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.update_environment("NORMAL", "LOW-LIGHT", "NOMINAL")
        self.controller.evaluate() # NORMAL -> DEGRADED -> ADAPTING
        
        self.controller.update_tracking_metrics(jitter=11.0, displacement=1.0, tracking_status="TRACKING")
        self.controller.evaluate() # ADAPTING -> VALIDATING
        
        # Provide worse metrics during validation
        for _ in range(5):
            self.controller.update_tracking_metrics(jitter=11.0, displacement=1.0, tracking_status="TRACKING")
            self.controller.evaluate()
            
        self.assertEqual(self.controller.state, AdaptiveState.ADAPTATION_REJECTED)
        self.assertIsNone(self.controller.current_strategy) # Rollback

    def test_cooldown_and_hysteresis(self):
        self.controller.state = AdaptiveState.ADAPTATION_REJECTED
        self.controller.cooldown_counter = 2
        
        self.controller.evaluate()
        self.assertEqual(self.controller.cooldown_counter, 1)
        self.assertEqual(self.controller.state, AdaptiveState.ADAPTATION_REJECTED)
        
        self.controller.evaluate()
        self.assertEqual(self.controller.cooldown_counter, 0)
        self.assertEqual(self.controller.state, AdaptiveState.NORMAL)

    def test_export_report(self):
        import os
        import json
        self.controller.evidence_cache = {"LOW-LIGHT": True}
        self.controller.state = AdaptiveState.ADAPTATION_ACCEPTED
        self.controller.current_strategy = "LOW_LIGHT_PREPROCESSING"
        self.controller._log_decision("Test log entry")
        
        report_path = "experiments/test_phase10_report.json"
        self.controller.export_report(report_path)
        
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data["final_state"], "ADAPTATION_ACCEPTED")
        self.assertEqual(data["final_strategy"], "LOW_LIGHT_PREPROCESSING")
        self.assertIn("LOW-LIGHT", data["evidence_sufficiency"])
        self.assertEqual(len(data["logs"]), 1)
        
        os.remove(report_path)

if __name__ == '__main__':
    unittest.main()
