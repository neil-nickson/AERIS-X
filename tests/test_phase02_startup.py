import unittest
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import main from phase02
from phase02_main import main

class TestPhase2Startup(unittest.TestCase):
    @patch('phase02_main.SerialLink.connect')
    @patch('phase02_main.SerialLink.read_line')
    @patch('sys.argv', ['phase02_main.py'])
    def test_startup_valid_sensor_data(self, mock_read, mock_connect):
        mock_connect.return_value = True
        
        # First yield a valid data line, then KeyboardInterrupt to exit the infinite loop
        mock_read.side_effect = [
            "TEMP=25.0,HUM=50.0,LIGHT=300,AX=330,AY=330,AZ=330,RAIN=0",
            KeyboardInterrupt()
        ]
        
        # This should run without UnboundLocalError
        try:
            main()
        except UnboundLocalError as e:
            self.fail(f"UnboundLocalError was raised: {e}")
        except Exception as e:
            # We expect KeyboardInterrupt to be caught gracefully by main(), 
            # so no exceptions should bubble up.
            self.fail(f"Unexpected exception raised: {e}")

    @patch('phase02_main.SerialLink.connect')
    @patch('phase02_main.SerialLink.read_line')
    @patch('sys.argv', ['phase02_main.py'])
    def test_startup_missing_sensor_data(self, mock_read, mock_connect):
        mock_connect.return_value = True
        
        # First yield None (missing data), then valid, then exit
        mock_read.side_effect = [
            None,
            "TEMP=25.0,HUM=50.0,LIGHT=300,AX=330,AY=330,AZ=330,RAIN=0",
            KeyboardInterrupt()
        ]
        
        try:
            main()
        except UnboundLocalError as e:
            self.fail(f"UnboundLocalError was raised on missing data path: {e}")
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

if __name__ == '__main__':
    unittest.main()
