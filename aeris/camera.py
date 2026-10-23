import cv2
import time
import os
from datetime import datetime

class AerisCamera:
    def __init__(self, config_manager):
        self.config = config_manager
        cfg = self.config.config.get("camera", {})
        
        self.source = cfg.get("source", "mac")
        # Backwards compatibility: use device_index if present, else index
        if "device_index" in cfg:
            self.camera_index = cfg.get("device_index", 0)
        else:
            self.camera_index = cfg.get("index", 0)
            
        self.width = cfg.get("width", 1280)
        self.height = cfg.get("height", 720)
        self.target_fps = cfg.get("target_fps", 30)
        self.recording_enabled = cfg.get("recording_enabled", False)
        
        self.cap = None
        self.writer = None
        
        self.health_status = "CAMERA_UNAVAILABLE"
        
        # Diagnostics
        self.dropped_frames = 0
        self.read_failures = 0
        self.last_frame_timestamp = None
        
        # FPS calculation state
        self.frame_count = 0
        self.start_time = None
        self.measured_fps = 0.0
        self.fps_calc_interval = 1.0 # seconds

    def initialize(self) -> bool:
        """Attempts to open the camera and set parameters."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                if self.camera_index != 0:
                    print(f"Camera index {self.camera_index} failed, falling back to 0")
                    self.cap = cv2.VideoCapture(0)
                    self.camera_index = 0
                
                if not self.cap.isOpened():
                    self.health_status = "CAMERA_UNAVAILABLE"
                    self.cap = None
                    return False
            
            # Request specific resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.health_status = "CAMERA_NOMINAL"
            self.start_time = time.time()
            
            if self.recording_enabled:
                self._init_recording()
                
            return True
            
        except Exception as e:
            self.health_status = "CAMERA_UNAVAILABLE"
            self.cap = None
            return False

    def _init_recording(self):
        """Initializes OpenCV VideoWriter for optional recording."""
        os.makedirs("experiments", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiments/aeris_phase3_{timestamp}.avi"
        
        # We read one frame to get the actual resolution the camera gave us, 
        # which might differ from requested width/height.
        success, frame = self.cap.read()
        if success:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(filename, fourcc, self.target_fps, (w, h))

    def read(self):
        """Reads a frame, calculates FPS, and optionally writes to video."""
        if not self.cap or not self.cap.isOpened():
            self.health_status = "CAMERA_UNAVAILABLE"
            return False, None
            
        success, frame = self.cap.read()
        if not success:
            self.health_status = "CAMERA_READ_ERROR"
            self.read_failures += 1
            return False, None
            
        # Optional: verify frame is not empty
        if frame is None or frame.size == 0:
            self.dropped_frames += 1
            return False, None
            
        self.last_frame_timestamp = time.time()
        self.health_status = "CAMERA_NOMINAL"
        
        # FPS Calculation
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.start_time
        if elapsed >= self.fps_calc_interval:
            self.measured_fps = self.frame_count / elapsed
            self.start_time = now
            self.frame_count = 0
            
        # Optional recording
        if self.writer is not None:
            self.writer.write(frame)
            
        return True, frame

    def get_fps(self) -> float:
        """Returns the last measured rolling FPS."""
        return self.measured_fps
        
    def get_health(self) -> str:
        """Returns the current camera health string."""
        return self.health_status

    def get_latency_string(self) -> str:
        """Returns explicitly that end-to-end optical latency is not measured."""
        return "End-to-End Latency: NOT MEASURED"
        
    def get_diagnostic_stats(self) -> dict:
        """Returns a dictionary of software acquisition diagnostics."""
        return {
            "source": self.source,
            "index": self.camera_index,
            "measured_fps": self.measured_fps,
            "read_failures": self.read_failures,
            "dropped_frames": self.dropped_frames,
            "last_frame_timestamp": self.last_frame_timestamp
        }
        
    @staticmethod
    def discover_cameras(max_indices=5):
        """
        Probes reasonable macOS/OpenCV camera indices.
        Returns a list of dicts with discovery info.
        """
        devices = []
        for idx in range(max_indices):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    # Attempt to get reported FPS
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    devices.append({
                        "index": idx,
                        "status": "AVAILABLE",
                        "resolution": f"{w}x{h}",
                        "reported_fps": fps,
                        "name": "Hardware name unavailable through OpenCV"
                    })
                else:
                    devices.append({
                        "index": idx,
                        "status": "READ_FAILURE",
                        "resolution": "N/A",
                        "reported_fps": 0,
                        "name": "Hardware name unavailable through OpenCV"
                    })
                cap.release()
            else:
                pass # not available
        return devices

    def release(self):
        """Releases camera and writer resources safely."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        self.health_status = "CAMERA_UNAVAILABLE"

    def draw_overlay(self, frame, x=20, y=30) -> int:
        """
        Draws standard camera diagnostics on the frame.
        Returns the next available Y coordinate.
        """
        source_name = "iPhone Continuity Camera" if self.source == "iphone" else "Mac Camera"
        
        lines = [
            f"Camera Source: {source_name}",
            f"Health: {self.health_status} | Res: {self.width}x{self.height} | FPS: {self.measured_fps:.1f}",
            f"Dropped: {self.dropped_frames} | Read Failures: {self.read_failures}",
            self.get_latency_string()
        ]
        
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 30
            
        return y
