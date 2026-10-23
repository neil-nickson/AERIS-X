import time
import math
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from aeris.tracking_metrics import TrackingSample

@dataclass
class FusionRecord:
    # Timing
    wall_clock_timestamp: str
    monotonic_timestamp: float
    frame_index: int
    sensor_timestamp: Optional[float]
    vision_timestamp: float
    timestamp_delta_ms: Optional[float]
    data_quality: str

    # Environment
    temperature: Optional[float]
    humidity: Optional[float]
    light: Optional[float]
    ax: Optional[float]
    ay: Optional[float]
    az: Optional[float]
    vibration_index: Optional[float]
    rain: Optional[int]
    environment_state: Optional[str]
    active_conditions: Optional[str] # | separated string
    sensor_health: Optional[str]

    # Tracking metrics (from TrackingSample)
    track_id: Optional[int]
    tracking_status: str
    center_x: Optional[float]
    center_y: Optional[float]
    bbox_x1: Optional[int]
    bbox_y1: Optional[int]
    bbox_x2: Optional[int]
    bbox_y2: Optional[int]
    confidence: Optional[float]
    displacement: Optional[float]
    jitter: Optional[float]
    
    # Session tracking stats per frame snapshot
    camera_fps: Optional[float]
    detector_fps: Optional[float]
    tracking_fps: Optional[float]

@dataclass
class FusionSummary:
    total_fused_records: int = 0
    synchronized_records: int = 0
    stale_sensor_records: int = 0
    missing_sensor_records: int = 0
    communication_loss_records: int = 0
    tracking_records: int = 0
    waiting_records: int = 0
    missed_records: int = 0
    lost_records: int = 0
    session_duration: float = 0.0

class AerisFusionEngine:
    def __init__(self, config_manager):
        self.config = config_manager
        cfg = self.config.config.get("fusion", {})
        self.max_timestamp_delta_ms = cfg.get("max_timestamp_delta_ms", 500)
        
        # Last known environment cache
        self.latest_env_data: Optional[dict] = None
        self.latest_env_status: Optional[str] = None
        self.latest_env_result: Optional[dict] = None
        self.latest_env_timestamp: Optional[float] = None
        self.latest_env_vibration: Optional[float] = None

        self.records: List[FusionRecord] = []
        
        # Summary Counters
        self.summary = FusionSummary()
        self.session_start = time.monotonic()

    def update_environment(self, parsed_data: Optional[dict], health_status: str, env_result: dict, dev_data: Optional[dict]):
        """Caches the most recent environmental state from Phase 2"""
        self.latest_env_timestamp = time.monotonic()
        self.latest_env_data = parsed_data
        self.latest_env_status = health_status
        self.latest_env_result = env_result
        self.latest_env_vibration = dev_data.get("vibration") if dev_data else None

    def fuse_vision(self, vision_sample: TrackingSample) -> FusionRecord:
        """Takes a tracking sample and fuses it with the latest environmental data."""
        vision_ts = vision_sample.timestamp
        wall_clock = time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        
        # Determine Data Quality
        delta_ms = None
        quality = "NO_SENSOR_DATA"
        
        if self.latest_env_status == "COMMUNICATION LOST":
            quality = "SENSOR_COMMUNICATION_LOST"
        elif self.latest_env_timestamp is not None:
            delta_ms = abs(vision_ts - self.latest_env_timestamp) * 1000.0
            if delta_ms <= self.max_timestamp_delta_ms:
                quality = "SYNCHRONIZED"
            else:
                quality = "STALE_SENSOR_DATA"

        # Update Summary Quality Counters
        self.summary.total_fused_records += 1
        if quality == "SYNCHRONIZED":
            self.summary.synchronized_records += 1
        elif quality == "STALE_SENSOR_DATA":
            self.summary.stale_sensor_records += 1
        elif quality == "NO_SENSOR_DATA":
            self.summary.missing_sensor_records += 1
        elif quality == "SENSOR_COMMUNICATION_LOST":
            self.summary.communication_loss_records += 1
            
        # Update Summary Tracking Counters
        t_status = vision_sample.status
        if t_status == "WAITING":
            self.summary.waiting_records += 1
        elif t_status in ("TRACKING", "NEW"):
            self.summary.tracking_records += 1
        elif t_status == "MISSED":
            self.summary.missed_records += 1
        elif t_status == "LOST":
            self.summary.lost_records += 1

        # Extract environment variables safely
        temp = self.latest_env_data.get("TEMP") if self.latest_env_data else None
        hum = self.latest_env_data.get("HUM") if self.latest_env_data else None
        light = self.latest_env_data.get("LIGHT") if self.latest_env_data else None
        ax = self.latest_env_data.get("AX") if self.latest_env_data else None
        ay = self.latest_env_data.get("AY") if self.latest_env_data else None
        az = self.latest_env_data.get("AZ") if self.latest_env_data else None
        rain = self.latest_env_data.get("RAIN_DETECTED") if self.latest_env_data else None
        
        # In a real run, env_result handles its own string values.
        env_state = self.latest_env_result.get("state") if self.latest_env_result else None
        active_cond_list = self.latest_env_result.get("active_conditions", []) if self.latest_env_result else []
        active_conds = "|".join(active_cond_list) if active_cond_list else None

        record = FusionRecord(
            wall_clock_timestamp=wall_clock,
            monotonic_timestamp=vision_ts,
            frame_index=vision_sample.frame_index,
            sensor_timestamp=self.latest_env_timestamp,
            vision_timestamp=vision_ts,
            timestamp_delta_ms=delta_ms,
            data_quality=quality,
            temperature=temp,
            humidity=hum,
            light=light,
            ax=ax,
            ay=ay,
            az=az,
            vibration_index=self.latest_env_vibration,
            rain=rain,
            environment_state=env_state,
            active_conditions=active_conds,
            sensor_health=self.latest_env_status,
            track_id=vision_sample.track_id,
            tracking_status=vision_sample.status,
            center_x=vision_sample.center_x,
            center_y=vision_sample.center_y,
            bbox_x1=vision_sample.bbox_x1,
            bbox_y1=vision_sample.bbox_y1,
            bbox_x2=vision_sample.bbox_x2,
            bbox_y2=vision_sample.bbox_y2,
            confidence=vision_sample.confidence,
            displacement=vision_sample.displacement,
            jitter=vision_sample.jitter,
            camera_fps=vision_sample.camera_fps,
            detector_fps=vision_sample.detector_fps,
            tracking_fps=vision_sample.tracking_fps
        )
        
        self.records.append(record)
        return record

    def export_logs(self, output_dir: str = "experiments"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.summary.session_duration = time.monotonic() - self.session_start
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"phase7_{ts_str}.csv")
        json_path = os.path.join(output_dir, f"phase7_{ts_str}_summary.json")
        
        # Write CSV
        if self.records:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.records[0].__dict__.keys())
                writer.writeheader()
                for r in self.records:
                    writer.writerow(r.__dict__)
                    
        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(self.summary.__dict__, f, indent=4)
            
        return csv_path, json_path
