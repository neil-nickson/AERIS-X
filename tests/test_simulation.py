import unittest
import threading
import time
import sys
import os

from phase12_main import run_simulation
from aeris.config import ConfigManager

class TestSimulationTermination(unittest.TestCase):
    def test_simulation_mode_terminates(self):
        """
        Verify that the simulation execution path returns within a reasonable bounded time 
        and does not leave background threads running.
        """
        # Patch cv2.waitKey to return immediately to speed up test
        import cv2
        original_waitkey = cv2.waitKey
        
        def mock_waitkey(delay=0):
            return -1 # Simulate no key press

        cv2.waitKey = mock_waitkey
        
        try:
            initial_thread_count = threading.active_count()
            
            # Run simulation
            config_mgr = ConfigManager()
            
            start_time = time.time()
            run_simulation(config_mgr)
            end_time = time.time()
            
            runtime = end_time - start_time
            
            # The test should finish extremely fast because we mocked waitKey
            self.assertTrue(runtime < 5.0, f"Simulation took too long: {runtime} seconds")
            
            # Check thread count
            final_thread_count = threading.active_count()
            self.assertEqual(
                initial_thread_count, 
                final_thread_count, 
                f"Thread leak detected! Initial: {initial_thread_count}, Final: {final_thread_count}"
            )
            
        finally:
            cv2.waitKey = original_waitkey

if __name__ == '__main__':
    unittest.main()
