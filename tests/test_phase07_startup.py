import unittest
import os
import sys
import threading
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import phase07_main
from aeris.config import ConfigManager

class TestPhase7Startup(unittest.TestCase):
    def setUp(self):
        self.config_mgr = ConfigManager()
        self.config_mgr.config["serial"]["port"] = "COM3"
        self.mock_fusion = MagicMock()
        
    @patch('aeris.serial_link.serial.Serial')
    @patch('phase07_main.SerialLink.read_line')
    def test_startup_valid_sensor_data(self, mock_read, mock_serial):
        # We need to simulate the thread processing some lines, then stopping.
        # side_effect can be a list or a function that sets the stop event.
        def fake_read():
            fake_read.calls += 1
            if fake_read.calls == 1:
                return "TEMP=25.0,HUM=50.0,LIGHT=300,AX=330,AY=330,AZ=330,RAIN=0"
            elif fake_read.calls == 2:
                # Setting stop_event will exit the loop cleanly
                phase07_main.stop_event.set()
                return "TEMP=25.0,HUM=50.0,LIGHT=300,AX=330,AY=330,AZ=330,RAIN=0"
            return None
        
        fake_read.calls = 0
        mock_read.side_effect = fake_read
        
        # Ensure stop_event is clear
        phase07_main.stop_event.clear()
        
        try:
            with patch('time.sleep', return_value=None):
                phase07_main.serial_thread_func(self.config_mgr, self.mock_fusion)
        except NameError as e:
            self.fail(f"NameError was raised: {e}")
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")
            
        # Verify fusion_engine was updated with the parsed data
        self.assertTrue(self.mock_fusion.update_environment.called)
        
    @patch('aeris.serial_link.serial.Serial')
    @patch('phase07_main.SerialLink.read_line')
    def test_startup_missing_sensor_data(self, mock_read, mock_serial):
        def fake_read():
            fake_read.calls += 1
            if fake_read.calls == 1:
                return None
            elif fake_read.calls == 2:
                phase07_main.stop_event.set()
                return "TEMP=25.0,HUM=50.0,LIGHT=300,AX=330,AY=330,AZ=330,RAIN=0"
            return None
        
        fake_read.calls = 0
        mock_read.side_effect = fake_read
        phase07_main.stop_event.clear()
        
        try:
            with patch('time.sleep', return_value=None):
                phase07_main.serial_thread_func(self.config_mgr, self.mock_fusion)
        except NameError as e:
            self.fail(f"NameError was raised: {e}")
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")
            
    @patch('phase07_main.SerialLink.connect')
    @patch('phase07_main.SerialLink.read_line')
    def test_startup_communication_lost(self, mock_read, mock_connect):
        def fake_connect():
            fake_connect.calls += 1
            if fake_connect.calls >= 2:
                phase07_main.stop_event.set()
            return False
            
        fake_connect.calls = 0
        mock_connect.side_effect = fake_connect
        
        phase07_main.stop_event.clear()
        
        # To avoid time.sleep(2) in the test, we mock sleep
        with patch('time.sleep', return_value=None):
            phase07_main.serial_thread_func(self.config_mgr, self.mock_fusion)
            
        # Verify fusion_engine was told COMM LOST
        self.mock_fusion.update_environment.assert_any_call(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)

if __name__ == '__main__':
    unittest.main()
