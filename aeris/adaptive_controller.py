import os
import glob
import json
import time
from enum import Enum, auto

class AdaptiveState(Enum):
    NORMAL = auto()
    DEGRADED = auto()
    ADAPTING = auto()
    VALIDATING = auto()
    ADAPTATION_ACCEPTED = auto()
    ADAPTATION_REJECTED = auto()
    INSUFFICIENT_EVIDENCE = auto()
    UNKNOWN = auto()

class AdaptiveController:
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Load adaptive configuration
        cfg = self.config.config.get("adaptive_tracking", {})
        self.enabled = cfg.get("enabled", False)
        self.min_improvement = cfg.get("minimum_improvement", 10.0) # percentage improvement required
        self.validation_window = cfg.get("validation_window", 30) # frames
        self.max_attempts = cfg.get("max_adaptation_attempts", 3)
        self.cooldown_frames = cfg.get("cooldown_frames", 100)
        self.degradation_jitter_threshold = cfg.get("degradation_jitter_threshold", 5.0)
        self.degradation_lost_threshold = cfg.get("degradation_lost_threshold", 3)
        self.evidence_thresholds = cfg.get("evidence_thresholds", {
            "min_independent_trials": 2,
            "require_correlation": True
        })
        
        self.state = AdaptiveState.NORMAL
        self.current_strategy = None
        self.validation_frames = 0
        self.cooldown_counter = 0
        self.adaptation_attempts = 0
        
        self.baseline_metrics = None
        self.current_metrics = None
        
        # Environment state
        self.environment_state = "UNKNOWN"
        self.active_conditions = "UNKNOWN"
        self.sensor_health = "UNKNOWN"
        
        # Metrics window
        self.recent_metrics = []
        
        self.evidence_cache = self._load_phase9_evidence()
        
        self.logs = []
        
        self.strategies = {
            "LOW-LIGHT": "LOW_LIGHT_PREPROCESSING",
            "HIGH-VIBRATION": "VIBRATION_SMOOTHING",
            "PRECIPITATION": "RAIN_DENOISING",
            "TEMPERATURE-DEVIATION": "THERMAL_COMPENSATION",
            "MULTI-CONDITION": "COMPOSITE_ADAPTATION",
            "GENERAL": "GENERAL_PARAMETER_TUNING"
        }
        
    def _load_phase9_evidence(self):
        """Dynamically load Phase 9 evidence from the latest analysis JSON."""
        evidence = {}
        files = glob.glob("experiments/phase9_*_analysis.json")
        if not files:
            return evidence
            
        latest = max(files, key=os.path.getmtime)
        try:
            with open(latest, 'r') as f:
                data = json.load(f)
                
            min_trials = self.evidence_thresholds.get("min_independent_trials", 2)
            require_corr = self.evidence_thresholds.get("require_correlation", True)
            
            # Identify which conditions have correlations
            corr_conds = set()
            correlations = data.get("correlations", [])
            for corr in correlations:
                if corr.get("status") == "CALCULATED":
                    var = corr.get("x_variable")
                    if var == "mean_light": corr_conds.add("LOW-LIGHT")
                    if var == "mean_vibration": corr_conds.add("HIGH-VIBRATION")
                
            # Parse comparisons and gate by minimum trials
            comparisons = data.get("condition_comparisons", [])
            for comp in comparisons:
                cond = comp.get("condition")
                trials = comp.get("condition_trials", 0)
                if cond and "difference_mean_jitter" in comp:
                    if trials >= min_trials:
                        if not require_corr or cond in corr_conds or cond not in ["LOW-LIGHT", "HIGH-VIBRATION"]:
                            evidence[cond] = True
                    
        except Exception:
            pass
            
        return evidence

    def _has_sufficient_evidence(self, condition):
        """Check if Phase 9 provides enough evidence to adapt to this condition."""
        # For prototype Phase 10: Even if evidence cache had something, we gate it strictly
        # But per requirements, load it dynamically.
        return self.evidence_cache.get(condition, False)

    def _log_decision(self, message, candidate="NONE", decision="NONE", validation_result="N/A"):
        log_entry = {
            "timestamp": time.time(),
            "environment_state": self.environment_state,
            "active_conditions": self.active_conditions,
            "sensor_health": self.sensor_health,
            "state": self.state.name,
            "candidate": candidate,
            "decision": decision,
            "validation": validation_result,
            "message": message
        }
        self.logs.append(log_entry)

    def update_environment(self, environment_state, active_conditions, sensor_health):
        self.environment_state = environment_state
        self.active_conditions = active_conditions
        self.sensor_health = sensor_health
        
    def update_tracking_metrics(self, jitter, displacement, tracking_status):
        self.recent_metrics.append({
            "jitter": jitter,
            "displacement": displacement,
            "status": tracking_status
        })
        if len(self.recent_metrics) > self.validation_window:
            self.recent_metrics.pop(0)
            
    def _aggregate_metrics(self):
        if not self.recent_metrics:
            return None
        jitters = [m["jitter"] for m in self.recent_metrics if m["jitter"] is not None]
        lost_count = sum(1 for m in self.recent_metrics if m["status"] == "LOST")
        
        return {
            "mean_jitter": sum(jitters) / len(jitters) if jitters else 0.0,
            "lost_count": lost_count
        }

    def evaluate(self):
        if not self.enabled:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0 and self.state in [AdaptiveState.ADAPTATION_ACCEPTED, AdaptiveState.ADAPTATION_REJECTED, AdaptiveState.INSUFFICIENT_EVIDENCE, AdaptiveState.UNKNOWN]:
                 self.state = AdaptiveState.NORMAL
            return

        metrics = self._aggregate_metrics()
        if not metrics:
            return

        # SENSE & ASSESS
        if self.state in [AdaptiveState.NORMAL]:
            is_degraded = (metrics["mean_jitter"] > self.degradation_jitter_threshold) or \
                          (metrics["lost_count"] >= self.degradation_lost_threshold)
            
            if is_degraded:
                self.state = AdaptiveState.DEGRADED
                self.baseline_metrics = metrics
                
        # ADAPT
        if self.state == AdaptiveState.DEGRADED:
            if self.sensor_health in ["NO_SENSOR_DATA", "SENSOR_COMMUNICATION_LOST", "STALE_SENSOR_DATA", "UNKNOWN"] or \
               self.environment_state == "UNKNOWN":
                self.state = AdaptiveState.UNKNOWN
                self._log_decision("Cannot adapt: Sensor data invalid/unknown.", decision="KEEP_BASELINE")
                self.cooldown_counter = self.cooldown_frames
                return
                
            condition = self.active_conditions
            if condition in ["NONE", "NORMAL", ""]:
                condition = "GENERAL"
                
            if not self._has_sufficient_evidence(condition):
                self.state = AdaptiveState.INSUFFICIENT_EVIDENCE
                self._log_decision(f"Insufficient Phase 9 experimental evidence for {condition}.", candidate=self.strategies.get(condition, "NONE"), decision="INSUFFICIENT_EVIDENCE")
                self.cooldown_counter = self.cooldown_frames
                return
                
            self.current_strategy = self.strategies.get(condition, "GENERAL_PARAMETER_TUNING")
            self.state = AdaptiveState.ADAPTING
            self.validation_frames = 0
            self.recent_metrics.clear()
            self._log_decision(f"Applying adaptation for {condition}.", candidate=self.current_strategy, decision="APPLYING")
            
        elif self.state == AdaptiveState.ADAPTING:
            self.state = AdaptiveState.VALIDATING
            
        # MEASURE & VALIDATE
        elif self.state == AdaptiveState.VALIDATING:
            self.validation_frames += 1
            if self.validation_frames >= self.validation_window:
                self.current_metrics = self._aggregate_metrics()
                
                improvement_pct = 0.0
                if self.baseline_metrics and self.baseline_metrics["mean_jitter"] > 0:
                    diff = self.baseline_metrics["mean_jitter"] - self.current_metrics["mean_jitter"]
                    improvement_pct = (diff / self.baseline_metrics["mean_jitter"]) * 100.0
                    
                if improvement_pct >= self.min_improvement:
                    self.state = AdaptiveState.ADAPTATION_ACCEPTED
                    self._log_decision("Adaptation improved metrics.", candidate=self.current_strategy, decision="ACCEPTED", validation_result=f"IMPROVED_BY_{improvement_pct:.1f}%")
                else:
                    self.state = AdaptiveState.ADAPTATION_REJECTED
                    self._log_decision("Adaptation failed to meet improvement criteria. Reverting.", candidate=self.current_strategy, decision="REVERTED", validation_result=f"CHANGE_{improvement_pct:.1f}%")
                    self.current_strategy = None
                    
                self.cooldown_counter = self.cooldown_frames

    def export_report(self, filepath):
        report = {
            "session_metadata": {
                "timestamp": time.time(),
                "enabled": self.enabled
            },
            "evidence_source": "Phase 9 Analysis JSON",
            "evidence_sufficiency": self.evidence_cache,
            "evidence_thresholds": self.evidence_thresholds,
            "final_state": self.state.name,
            "final_strategy": self.current_strategy,
            "limitations": [
                "PHYSICAL EXPERIMENTAL VALIDATION: PENDING",
                "Simulations evaluate logic but do not alter a physical drone."
            ],
            "logs": self.logs
        }
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
