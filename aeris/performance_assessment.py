import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

@dataclass
class AssessmentReport:
    session_id: str
    timestamp: float
    mode: str
    metadata: Dict[str, Any]
    environment: Dict[str, Any]
    tracking_metrics: Dict[str, Any]
    data_quality: Dict[str, Any]
    baseline_reference: Dict[str, Any]
    performance_assessment: Dict[str, Any]
    degradation_assessment: Dict[str, Any]
    adaptation: Dict[str, Any]
    limitations: List[str]

class PerformanceAssessor:
    def __init__(self, config_manager):
        self.config = config_manager
        cfg = self.config.config.get("adaptive_tracking", {})
        
        # PROTOTYPE / EXPERIMENTAL ASSESSMENT THRESHOLDS
        # These are NOT scientifically validated, universal, military, high-altitude, or safety thresholds.
        self.degradation_jitter_threshold = cfg.get("degradation_jitter_threshold", 5.0)
        self.degradation_lost_threshold = cfg.get("degradation_lost_threshold", 3)
        self.min_tracking_ratio = 0.5
        
    def assess(self, env_data: dict, tracking_metrics: dict, baseline_metrics: dict, data_quality: dict, adaptation_data: dict, is_simulation: bool = False) -> AssessmentReport:
        mode = "DETERMINISTIC SIMULATION" if is_simulation else "REAL DATA"
        
        # Default states
        perf_state = "UNKNOWN"
        deg_state = "UNKNOWN"
        reason_codes = []
        
        # 1. Data Quality Check
        quality = data_quality.get("status", "UNKNOWN")
        stale_flag = data_quality.get("stale_sensor_data", False)
        
        if quality == "NO_SENSOR_DATA":
            reason_codes.append("NO_SENSOR_DATA")
        elif quality == "SENSOR_COMMUNICATION_LOST":
            reason_codes.append("SENSOR_COMMUNICATION_LOST")
        elif stale_flag or quality == "STALE_SENSOR_DATA":
            reason_codes.append("STALE_SENSOR_DATA")
            
        env_state = env_data.get("state", "UNKNOWN")
        if env_state == "UNKNOWN":
            reason_codes.append("UNKNOWN_ENVIRONMENT")
            
        if len(reason_codes) > 0:
            # If any data quality issues exist, we cannot confidently assess performance.
            perf_state = "UNKNOWN"
            deg_state = "UNKNOWN"
        else:
            # 2. Tracking Metrics Check
            mean_jitter = tracking_metrics.get("mean_jitter")
            lost_events = tracking_metrics.get("lost_events")
            tracking_ratio = tracking_metrics.get("tracking_ratio")
            
            if mean_jitter is None or lost_events is None or tracking_ratio is None:
                reason_codes.append("INSUFFICIENT_TRACKING_DATA")
                perf_state = "UNKNOWN"
                deg_state = "UNKNOWN"
            else:
                # 3. Categorical Degradation Assessment (PROTOTYPE/EXPERIMENTAL)
                if tracking_ratio < self.min_tracking_ratio:
                    reason_codes.append("LOW_TRACKING_RATIO")
                    
                j_ratio = mean_jitter / self.degradation_jitter_threshold if self.degradation_jitter_threshold > 0 else 0
                l_ratio = lost_events / self.degradation_lost_threshold if self.degradation_lost_threshold > 0 else 0
                
                max_ratio = max(j_ratio, l_ratio)
                
                if max_ratio < 1.0:
                    deg_state = "NORMAL"
                    perf_state = "NORMAL"
                    reason_codes.append("NORMAL_TRACKING")
                elif max_ratio < 2.0:
                    deg_state = "MILD_DEGRADATION"
                    perf_state = "DEGRADED"
                    if j_ratio >= 1.0: reason_codes.append("HIGH_JITTER")
                    if l_ratio >= 1.0: reason_codes.append("HIGH_LOST_EVENTS")
                elif max_ratio < 3.0:
                    deg_state = "MODERATE_DEGRADATION"
                    perf_state = "DEGRADED"
                    if j_ratio >= 2.0: reason_codes.append("HIGH_JITTER")
                    if l_ratio >= 2.0: reason_codes.append("HIGH_LOST_EVENTS")
                else:
                    deg_state = "SEVERE_DEGRADATION"
                    perf_state = "DEGRADED"
                    if j_ratio >= 3.0: reason_codes.append("HIGH_JITTER")
                    if l_ratio >= 3.0: reason_codes.append("HIGH_LOST_EVENTS")
                    
        # 4. Adaptation Integration
        adapt_status = adaptation_data.get("status", "NOT_ATTEMPTED")
        if adapt_status == "ACCEPTED":
            reason_codes.append("ADAPTATION_ACCEPTED")
        elif adapt_status in ("REJECTED", "REVERTED"):
            reason_codes.append("ADAPTATION_REVERTED")
            
        limitations = [
            "PROTOTYPE / EXPERIMENTAL ASSESSMENT THRESHOLDS: Thresholds are not scientifically validated, universal, military, high-altitude, or safety limits.",
            "ENVIRONMENTAL CAUSALITY: Degradation state does not imply environmental causality.",
            "TARGET MOTION CONFOUNDER: Displacement and jitter may be heavily confounded by autonomous target motion since independent ground-truth drone motion is absent.",
            "SCORELESS ASSESSMENT: Using categorical assessment. No 0-100 arbitrary scores are provided."
        ]
        
        if is_simulation:
            limitations.insert(0, "PHYSICAL EXPERIMENTAL VALIDATION: PENDING")

        report = AssessmentReport(
            session_id=f"session_{int(time.time())}",
            timestamp=time.time(),
            mode=mode,
            metadata={"assessor_version": "1.0", "thresholds_source": "config.json"},
            environment=env_data,
            tracking_metrics=tracking_metrics,
            data_quality=data_quality,
            baseline_reference=baseline_metrics,
            performance_assessment={
                "performance_state": perf_state,
                "reason_codes": list(set(reason_codes))
            },
            degradation_assessment={
                "degradation_state": deg_state,
                "thresholds_used_EXPERIMENTAL": {
                    "jitter": self.degradation_jitter_threshold,
                    "lost_events": self.degradation_lost_threshold
                }
            },
            adaptation=adaptation_data,
            limitations=limitations
        )
        
        return report
