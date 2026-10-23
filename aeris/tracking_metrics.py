import math
import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from aeris.tracker import TargetTrack

@dataclass
class TrackingSample:
    timestamp: float
    frame_index: int
    track_id: Optional[int]
    status: str
    center_x: Optional[float]
    center_y: Optional[float]
    bbox_x1: Optional[int]
    bbox_y1: Optional[int]
    bbox_x2: Optional[int]
    bbox_y2: Optional[int]
    confidence: Optional[float]
    displacement: Optional[float]
    jitter: Optional[float]
    camera_fps: Optional[float]
    detector_fps: Optional[float]
    tracking_fps: Optional[float]
    position_error: Optional[float]

@dataclass
class TrackingSummary:
    total_frames: int = 0
    tracked_frames: int = 0
    missed_frames: int = 0
    lost_frames: int = 0
    tracking_ratio: float = 0.0
    missed_ratio: float = 0.0
    lost_events: int = 0
    tracking_duration: float = 0.0
    mean_displacement: Optional[float] = None
    mean_jitter: Optional[float] = None
    rms_jitter: Optional[float] = None
    peak_jitter: Optional[float] = None
    mean_position_error: Optional[float] = None
    rms_position_error: Optional[float] = None
    peak_position_error: Optional[float] = None
    mean_bbox_width: Optional[float] = None
    mean_bbox_height: Optional[float] = None
    ground_truth_available: bool = False

class TrackingMetrics:
    def __init__(self):
        self.samples: List[TrackingSample] = []
        self.total_frames = 0
        self.tracked_frames = 0
        self.missed_frames = 0
        self.lost_frames = 0
        self.lost_events = 0
        
        # State tracking for metrics
        self._last_track_id = None
        self._last_status = None
        self._last_center = None
        self._last_displacement = None
        
        # Recovery timing
        self._first_missed_time = None
        self.recoveries: List[float] = []

        # Session tracking
        self.session_start_time = None
        self.session_end_time = None
        
        # Jitter lists
        self._jitters = []
        self._displacements = []
        self._position_errors = []
        self._bbox_widths = []
        self._bbox_heights = []
        self._last_dx_dy = None

    def update(self, 
               track: Optional[TargetTrack], 
               camera_fps: Optional[float] = None, 
               detector_fps: Optional[float] = None, 
               tracking_fps: Optional[float] = None, 
               ground_truth_center: Optional[Tuple[float, float]] = None):
        
        ts = time.monotonic()
        if self.session_start_time is None and track and track.status == "TRACKING":
            self.session_start_time = ts
            
        if track and track.status == "TRACKING":
            self.session_end_time = ts

        self.total_frames += 1
        
        # Extract variables safely
        track_id = track.track_id if track else None
        status = track.status if track else "WAITING"
        cx = track.center_x if track and track.status != "LOST" else None
        cy = track.center_y if track and track.status != "LOST" else None
        conf = track.confidence if track and track.status != "LOST" else None
        
        bbox = track.bbox if track and track.status != "LOST" else None
        x1 = int(bbox[0]) if bbox else None
        y1 = int(bbox[1]) if bbox else None
        x2 = int(bbox[2]) if bbox else None
        y2 = int(bbox[3]) if bbox else None
        w = float(track.width) if track and track.status != "LOST" else None
        h = float(track.height) if track and track.status != "LOST" else None

        if w is not None:
            self._bbox_widths.append(w)
        if h is not None:
            self._bbox_heights.append(h)

        # Basic counts
        if status == "TRACKING" or status == "NEW":
            self.tracked_frames += 1
        elif status == "MISSED":
            self.missed_frames += 1
        elif status == "LOST":
            self.lost_frames += 1

        # Track transitions
        if self._last_status != "LOST" and status == "LOST":
            self.lost_events += 1

        # Recovery Timing
        if status == "MISSED" and self._last_status in ("TRACKING", "NEW"):
            self._first_missed_time = ts
        
        if status == "TRACKING" and self._last_status == "MISSED" and self._last_track_id == track_id:
            if self._first_missed_time is not None:
                self.recoveries.append(ts - self._first_missed_time)
            self._first_missed_time = None
            
        if status == "LOST" or track_id != self._last_track_id:
            self._first_missed_time = None

        # Position Error
        pos_error = None
        if ground_truth_center and cx is not None and cy is not None:
            gx, gy = ground_truth_center
            pos_error = math.hypot(cx - gx, cy - gy)
            self._position_errors.append(pos_error)

        # Displacement and Jitter
        displacement = None
        jitter = None
        dx_dy = None
        
        if cx is not None and cy is not None and self._last_center is not None and self._last_track_id == track_id:
            dx = cx - self._last_center[0]
            dy = cy - self._last_center[1]
            dx_dy = (dx, dy)
            displacement = math.hypot(dx, dy)
            self._displacements.append(displacement)
            
            if self._last_dx_dy is not None:
                ddx = dx - self._last_dx_dy[0]
                ddy = dy - self._last_dx_dy[1]
                jitter = math.hypot(ddx, ddy)
                self._jitters.append(jitter)

        # Create Sample
        sample = TrackingSample(
            timestamp=ts,
            frame_index=self.total_frames,
            track_id=track_id,
            status=status,
            center_x=cx,
            center_y=cy,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            confidence=conf,
            displacement=displacement,
            jitter=jitter,
            camera_fps=camera_fps,
            detector_fps=detector_fps,
            tracking_fps=tracking_fps,
            position_error=pos_error
        )
        self.samples.append(sample)

        # Update last state
        if status != "LOST" and cx is not None:
            self._last_center = (cx, cy)
            self._last_track_id = track_id
        else:
            self._last_center = None
            self._last_track_id = None
            self._last_dx_dy = None
            
        if dx_dy is not None:
            self._last_dx_dy = dx_dy
            
        self._last_status = status
        
        return sample

    def _calc_mean_rms_peak(self, data: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not data:
            return None, None, None
        mean = sum(data) / len(data)
        rms = math.sqrt(sum(d*d for d in data) / len(data))
        peak = max(data)
        return mean, rms, peak

    def get_summary(self) -> TrackingSummary:
        mean_disp, _, _ = self._calc_mean_rms_peak(self._displacements)
        mean_j, rms_j, peak_j = self._calc_mean_rms_peak(self._jitters)
        mean_pe, rms_pe, peak_pe = self._calc_mean_rms_peak(self._position_errors)
        mean_w, _, _ = self._calc_mean_rms_peak(self._bbox_widths)
        mean_h, _, _ = self._calc_mean_rms_peak(self._bbox_heights)
        
        t_ratio = self.tracked_frames / self.total_frames if self.total_frames > 0 else 0.0
        m_ratio = self.missed_frames / self.total_frames if self.total_frames > 0 else 0.0
        
        duration = 0.0
        if self.session_start_time is not None and self.session_end_time is not None:
            duration = max(0.0, self.session_end_time - self.session_start_time)
            
        return TrackingSummary(
            total_frames=self.total_frames,
            tracked_frames=self.tracked_frames,
            missed_frames=self.missed_frames,
            lost_frames=self.lost_frames,
            tracking_ratio=t_ratio,
            missed_ratio=m_ratio,
            lost_events=self.lost_events,
            tracking_duration=duration,
            mean_displacement=mean_disp,
            mean_jitter=mean_j,
            rms_jitter=rms_j,
            peak_jitter=peak_j,
            mean_position_error=mean_pe,
            rms_position_error=rms_pe,
            peak_position_error=peak_pe,
            mean_bbox_width=mean_w,
            mean_bbox_height=mean_h,
            ground_truth_available=len(self._position_errors) > 0
        )

    def export_logs(self, output_dir: str = "experiments"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"phase6_{ts_str}.csv")
        json_path = os.path.join(output_dir, f"phase6_{ts_str}_summary.json")
        
        # Write CSV
        if self.samples:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.samples[0].__dict__.keys())
                writer.writeheader()
                for s in self.samples:
                    writer.writerow(s.__dict__)
                    
        # Write JSON
        summary = self.get_summary()
        with open(json_path, 'w') as f:
            json.dump(summary.__dict__, f, indent=4)
            
        return csv_path, json_path
