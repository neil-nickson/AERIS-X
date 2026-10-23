import os
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class AerisDroneDetector:
    def __init__(self, config_manager):
        self.config = config_manager
        cfg = self.config.config.get("detector", {})
        
        self.model_path = cfg.get("model_path", "models/drone_model.pt")
        self.target_class_name = cfg.get("target_class", "drone")
        self.confidence_threshold = cfg.get("confidence_threshold", 0.40)
        self.device = cfg.get("device", "cpu")
        
        self.model = None
        self.target_class_id = None
        self.health_status = "MODEL_UNAVAILABLE"
        
        # Performance metrics
        self.inference_time_ms = 0.0
        self.frame_count = 0
        self.start_time = None
        self.measured_fps = 0.0
        self.fps_calc_interval = 1.0

    def initialize(self) -> bool:
        """Attempts to load the configured YOLO model and validate the target class."""
        if not ULTRALYTICS_AVAILABLE:
            self.health_status = "MODEL_UNAVAILABLE"
            print("[!] Detector: Ultralytics not installed.")
            return False
            
        if not os.path.exists(self.model_path):
            self.health_status = "MODEL_UNAVAILABLE"
            print(f"[!] Detector: Model file not found at {self.model_path}")
            return False
            
        try:
            self.model = YOLO(self.model_path)
            
            # Validate target class exists in the model
            names = self.model.names
            class_id = None
            for idx, name in names.items():
                if name.lower() == self.target_class_name.lower():
                    class_id = idx
                    break
                    
            if class_id is None:
                self.health_status = "MODEL_LOAD_ERROR"
                self.model = None
                print(f"[!] Detector: Target class '{self.target_class_name}' not found in model classes: {list(names.values())}")
                return False
                
            self.target_class_id = class_id
            self.health_status = "DETECTOR_NOMINAL"
            self.start_time = time.time()
            return True
            
        except Exception as e:
            self.health_status = "MODEL_LOAD_ERROR"
            self.model = None
            print(f"[!] Detector: Failed to load model. {str(e)}")
            return False

    def detect(self, frame):
        """Runs inference on a frame and returns structured valid detections."""
        if self.health_status != "DETECTOR_NOMINAL" or self.model is None:
            return []
            
        try:
            t0 = time.time()
            
            # Run inference
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=self.confidence_threshold,
                classes=[self.target_class_id],
                verbose=False
            )
            
            t1 = time.time()
            self.inference_time_ms = (t1 - t0) * 1000.0
            
            # Calculate FPS
            self.frame_count += 1
            elapsed = t1 - self.start_time
            if elapsed >= self.fps_calc_interval:
                self.measured_fps = self.frame_count / elapsed
                self.start_time = t1
                self.frame_count = 0
            
            # Parse detections
            detections = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Confidence check (Ultralytics usually filters this for us due to conf= arg, but safety check)
                    conf = float(box.conf[0])
                    if conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            "class": self.target_class_name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2)
                        })
                        
            return detections
            
        except Exception as e:
            self.health_status = "INFERENCE_ERROR"
            return []

    def get_health(self) -> str:
        return self.health_status
        
    def get_inference_time_ms(self) -> float:
        return self.inference_time_ms
        
    def get_inference_fps(self) -> float:
        return self.measured_fps
