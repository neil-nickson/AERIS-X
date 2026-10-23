import unittest
import os
import sys

# Ensure aeris can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aeris.environment_engine import EnvironmentEngine

class MockConfigManager:
    def __init__(self):
        self.config = {
            "thresholds": {
                "temp_deviation": 5.0,
                "low_light_ratio": 0.5,
                "vibration_threshold": 15.0,
                "stability_samples": 3,
                "prototype_envelope_min_temp": 0.0,
                "prototype_envelope_max_temp": 45.0
            }
        }

class TestEnvironmentEngine(unittest.TestCase):
    def setUp(self):
        self.config = MockConfigManager()
        self.engine = EnvironmentEngine(self.config)
        self.base_parsed = {"TEMP": 25.0, "RAIN_DETECTED": False}
        self.base_dev = {"temp_abs_delta": 0.0, "light_ratio": 1.0, "vibration": 0.0}

    def _feed_samples(self, count, parsed, dev, health="NOMINAL"):
        res = None
        for _ in range(count):
            res = self.engine.evaluate(parsed, health, dev)
        return res

    def test_normal(self):
        res = self._feed_samples(3, self.base_parsed, self.base_dev)
        self.assertEqual(res["state"], "NORMAL")
        self.assertEqual(len(res["active_conditions"]), 0)

    def test_low_light(self):
        # Trigger low light (ratio < 0.5)
        dev = dict(self.base_dev, light_ratio=0.4)
        # Should not trigger immediately because stability_samples=3
        res = self._feed_samples(1, self.base_parsed, dev)
        self.assertEqual(res["state"], "NORMAL")
        # Complete debouncing
        res = self._feed_samples(2, self.base_parsed, dev)
        self.assertEqual(res["state"], "LOW-LIGHT")
        self.assertIn("LOW-LIGHT", res["active_conditions"])

    def test_high_light_not_low_light(self):
        # High light (ratio > 1.0)
        dev = dict(self.base_dev, light_ratio=2.0)
        res = self._feed_samples(3, self.base_parsed, dev)
        self.assertEqual(res["state"], "NORMAL")

    def test_high_vibration(self):
        dev = dict(self.base_dev, vibration=16.0)
        res = self._feed_samples(3, self.base_parsed, dev)
        self.assertEqual(res["state"], "HIGH-VIBRATION")
        self.assertIn("HIGH-VIBRATION", res["active_conditions"])

    def test_precipitation(self):
        parsed = dict(self.base_parsed, RAIN_DETECTED=True)
        res = self._feed_samples(3, parsed, self.base_dev)
        self.assertEqual(res["state"], "PRECIPITATION")

    def test_temperature_deviation(self):
        dev = dict(self.base_dev, temp_abs_delta=6.0)
        res = self._feed_samples(3, self.base_parsed, dev)
        self.assertEqual(res["state"], "TEMPERATURE-DEVIATION")

    def test_out_of_validated_envelope(self):
        # Exceeds prototype envelope max 45.0
        parsed = dict(self.base_parsed, TEMP=46.0)
        res = self._feed_samples(3, parsed, self.base_dev)
        self.assertEqual(res["state"], "OUT-OF-VALIDATED-ENVELOPE")

    def test_multi_condition(self):
        parsed = dict(self.base_parsed, RAIN_DETECTED=True)
        dev = dict(self.base_dev, vibration=16.0)
        res = self._feed_samples(3, parsed, dev)
        self.assertEqual(res["state"], "MULTI-CONDITION")
        self.assertIn("HIGH-VIBRATION", res["active_conditions"])
        self.assertIn("PRECIPITATION", res["active_conditions"])
        self.assertEqual(len(res["active_conditions"]), 2)

    def test_sensor_health_override(self):
        # If health is bad, state becomes UNKNOWN regardless of sensor inputs
        parsed = dict(self.base_parsed, RAIN_DETECTED=True)
        dev = dict(self.base_dev, vibration=16.0)
        res = self._feed_samples(3, parsed, dev, health="COMMUNICATION LOST")
        self.assertEqual(res["state"], "UNKNOWN")

    def test_recovery_debouncing(self):
        dev = dict(self.base_dev, vibration=16.0)
        res = self._feed_samples(3, self.base_parsed, dev)
        self.assertEqual(res["state"], "HIGH-VIBRATION")

        # Now go back to normal
        res = self._feed_samples(1, self.base_parsed, self.base_dev)
        # Should still be HIGH-VIBRATION because it needs 3 normal samples to recover
        self.assertEqual(res["state"], "HIGH-VIBRATION")
        
        # Complete recovery
        res = self._feed_samples(2, self.base_parsed, self.base_dev)
        self.assertEqual(res["state"], "NORMAL")

if __name__ == '__main__':
    unittest.main()
