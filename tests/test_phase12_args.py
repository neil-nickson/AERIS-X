import unittest
from phase12_main import parse_args

class TestPhase12Args(unittest.TestCase):
    def test_default_mode(self):
        # python3 phase12_main.py => selects LIVE mode
        args = parse_args([])
        self.assertEqual(args.mode, "live")
        
    def test_live_mode(self):
        # --mode live => selects LIVE mode
        args = parse_args(["--mode", "live"])
        self.assertEqual(args.mode, "live")
        
    def test_simulation_mode(self):
        # --mode simulation => selects SIMULATION mode
        args = parse_args(["--mode", "simulation"])
        self.assertEqual(args.mode, "simulation")
        
    def test_replay_mode(self):
        # --mode replay => selects REPLAY mode
        args = parse_args(["--mode", "replay"])
        self.assertEqual(args.mode, "replay")

if __name__ == "__main__":
    unittest.main()
