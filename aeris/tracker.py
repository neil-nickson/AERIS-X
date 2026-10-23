import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class TargetTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    age: int = 0
    missed_frames: int = 0
    status: str = "NEW"
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

class AerisTracker:
    def __init__(self, config_manager):
        self.config = config_manager
        cfg = self.config.config.get("tracker", {})
        
        self.iou_threshold = cfg.get("association_iou_threshold", 0.3)
        self.max_center_distance_ratio = cfg.get("max_center_distance_ratio", 0.3)
        self.max_missed_frames = cfg.get("max_missed_frames", 10)
        self.min_confidence = cfg.get("min_detection_confidence", 0.40)
        self.max_trajectory = cfg.get("trajectory_length", 30)
        self.selection_policy = cfg.get("target_selection_policy", "highest_confidence")
        
        self.active_track: Optional[TargetTrack] = None
        self.next_track_id = 1

    def _calculate_iou(self, boxA, boxB) -> float:
        """Calculates Intersection over Union for two boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _get_center_and_size(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1)

    def _select_best_detection(self, detections: List[dict]) -> Optional[dict]:
        """Applies selection policy (e.g. highest_confidence) to valid detections."""
        valid_detections = [d for d in detections if d.get("confidence", 0) >= self.min_confidence]
        if not valid_detections:
            return None
            
        if self.selection_policy == "highest_confidence":
            # Sort by confidence descending
            valid_detections.sort(key=lambda x: x["confidence"], reverse=True)
            return valid_detections[0]
            
        return valid_detections[0] # Fallback to first

    def update(self, detections: List[dict], frame_width: int, frame_height: int) -> Optional[TargetTrack]:
        """Updates the active track based on incoming frame detections."""
        
        # 1. If we have NO active track
        if self.active_track is None or self.active_track.status == "LOST":
            best_det = self._select_best_detection(detections)
            if best_det:
                cx, cy, w, h = self._get_center_and_size(best_det["bbox"])
                self.active_track = TargetTrack(
                    track_id=self.next_track_id,
                    bbox=best_det["bbox"],
                    center_x=cx,
                    center_y=cy,
                    width=w,
                    height=h,
                    confidence=best_det["confidence"],
                    status="NEW"
                )
                self.active_track.trajectory.append((cx, cy))
                self.next_track_id += 1
                return self.active_track
            return None

        # 2. We HAVE an active track. We need to associate.
        # For this prototype, we greedily look for the best matching detection.
        best_match = None
        best_iou = -1.0
        
        for det in detections:
            if det.get("confidence", 0) < self.min_confidence:
                continue
                
            iou = self._calculate_iou(self.active_track.bbox, det["bbox"])
            
            # Center distance check
            cx, cy, _, _ = self._get_center_and_size(det["bbox"])
            dist = math.hypot(cx - self.active_track.center_x, cy - self.active_track.center_y)
            max_dist = self.max_center_distance_ratio * math.hypot(frame_width, frame_height)
            
            if iou >= self.iou_threshold or dist <= max_dist:
                # Give priority to highest IoU, or if 0, closest distance
                if iou > best_iou:
                    best_iou = iou
                    best_match = det

        # 3. Apply updates based on association
        if best_match:
            # Associated successfully
            cx, cy, w, h = self._get_center_and_size(best_match["bbox"])
            self.active_track.bbox = best_match["bbox"]
            self.active_track.center_x = cx
            self.active_track.center_y = cy
            self.active_track.width = w
            self.active_track.height = h
            self.active_track.confidence = best_match["confidence"]
            self.active_track.missed_frames = 0
            self.active_track.age += 1
            self.active_track.status = "TRACKING"
            
            self.active_track.trajectory.append((cx, cy))
            if len(self.active_track.trajectory) > self.max_trajectory:
                self.active_track.trajectory.pop(0)
        else:
            # Missed
            self.active_track.missed_frames += 1
            self.active_track.age += 1
            
            if self.active_track.missed_frames > self.max_missed_frames:
                self.active_track.status = "LOST"
            else:
                self.active_track.status = "MISSED"

        return self.active_track
