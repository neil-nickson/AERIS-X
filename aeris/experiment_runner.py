import time
import os
import csv
import json
import math
from typing import List, Dict, Optional
from aeris.fusion import FusionRecord

class ExperimentState:
    NOT_STARTED = "NOT_STARTED"
    PREPARING = "PREPARING"
    STABILIZING = "STABILIZING"
    RECORDING = "RECORDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_PERFORMED = "NOT_PERFORMED"

class ConditionVerification:
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

class ExperimentConditions:
    BASELINE = "BASELINE"
    HIGH_VIBRATION = "HIGH_VIBRATION"
    LOW_LIGHT = "LOW_LIGHT"
    PRECIPITATION = "PRECIPITATION"
    TEMPERATURE_DEVIATION = "TEMPERATURE_DEVIATION"
    MULTI_CONDITION = "MULTI_CONDITION"

class TrialStats:
    def __init__(self, condition: str, trial_number: int):
        self.condition = condition
        self.trial_number = trial_number
        self.start_time = time.monotonic()
        self.end_time = None
        self.records: List[dict] = []
        
        self.status = ExperimentState.COMPLETED
        self.verification = ConditionVerification.UNKNOWN

        # Summary values
        self.sample_count = 0
        self.tracking_frames = 0
        self.missed_frames = 0
        self.lost_events = 0
        
        self.synchronized_records = 0
        self.stale_records = 0
        self.missing_sensor_records = 0
        self.communication_loss_records = 0
        
        self.mean_displacement = None
        self.mean_jitter = None
        self.peak_jitter = None
        self.mean_confidence = None
        
        self.camera_fps = 0.0
        self.detector_fps = 0.0
        self.tracking_fps = 0.0
        
        self.mean_temperature = None
        self.min_temperature = None
        self.max_temperature = None
        self.mean_humidity = None
        self.min_humidity = None
        self.max_humidity = None
        self.mean_light = None
        self.min_light = None
        self.max_light = None
        self.mean_vibration = None
        self.max_vibration = None
        self.rain_observed = False

class ExperimentSession:
    def __init__(self, config_manager, experiment_id: Optional[str] = None):
        self.config = config_manager
        cfg = self.config.config.get("experiment", {})
        self.stabilization_seconds = cfg.get("stabilization_seconds", 3.0)
        self.recording_seconds = cfg.get("recording_seconds", 10.0)
        self.min_tracking_frames = cfg.get("minimum_tracking_frames", 10)
        
        self.experiment_id = experiment_id or time.strftime("%Y%m%d_%H%M%S")
        self.state = ExperimentState.NOT_STARTED
        
        self.current_condition = None
        self.current_trial_number = 0
        self.trials: List[TrialStats] = []
        
        self.trial_counts: Dict[str, int] = {}
        
        # Timing
        self.state_start_time = None
        self.current_trial_stats: Optional[TrialStats] = None
        
        # Removed combined csv_records buffer


    def start_trial(self, condition: str):
        if self.state in [ExperimentState.STABILIZING, ExperimentState.RECORDING, ExperimentState.PREPARING]:
            # Clean up incomplete trial
            if self.current_trial_stats:
                self.current_trial_stats.status = ExperimentState.FAILED
                self.trials.append(self.current_trial_stats)
                
        self.current_condition = condition
        self.trial_counts[condition] = self.trial_counts.get(condition, 0) + 1
        self.current_trial_number = self.trial_counts[condition]
        
        self.state = ExperimentState.PREPARING
        self.state_start_time = time.monotonic()
        
        self.current_trial_stats = TrialStats(condition, self.current_trial_number)

    def _verify_condition(self, env_state: Optional[str], active_conditions: Optional[str]) -> str:
        if not self.current_condition or self.current_condition == ExperimentConditions.BASELINE:
            return ConditionVerification.VERIFIED
            
        act_conds = active_conditions.split("|") if active_conditions else []
        
        if self.current_condition == ExperimentConditions.HIGH_VIBRATION:
            if "HIGH-VIBRATION" in act_conds:
                return ConditionVerification.VERIFIED
                
        if self.current_condition == ExperimentConditions.LOW_LIGHT:
            if "LOW-LIGHT" in act_conds:
                return ConditionVerification.VERIFIED
                
        if self.current_condition == ExperimentConditions.PRECIPITATION:
            if "RAIN" in act_conds:
                return ConditionVerification.VERIFIED
                
        if self.current_condition == ExperimentConditions.TEMPERATURE_DEVIATION:
            if "HIGH-TEMP" in act_conds or "LOW-TEMP" in act_conds or "TEMP-DEVIATION" in act_conds:
                return ConditionVerification.VERIFIED
                
        return ConditionVerification.FAILED

    def update(self, record: FusionRecord):
        if self.state == ExperimentState.NOT_STARTED or self.state in [ExperimentState.COMPLETED, ExperimentState.FAILED]:
            return
            
        ts = time.monotonic()
        elapsed = ts - self.state_start_time
        
        if self.state == ExperimentState.PREPARING:
            self.state = ExperimentState.STABILIZING
            self.state_start_time = ts
            elapsed = 0
            
        if self.state == ExperimentState.STABILIZING:
            if elapsed >= self.stabilization_seconds:
                # Need to verify if the env is stable in the requested condition
                verification = self._verify_condition(record.environment_state, record.active_conditions)
                self.current_trial_stats.verification = verification
                
                if verification == ConditionVerification.FAILED and self.current_condition != ExperimentConditions.BASELINE:
                    self.state = ExperimentState.FAILED
                    self.current_trial_stats.status = ExperimentState.FAILED
                    self.trials.append(self.current_trial_stats)
                    self.current_trial_stats = None
                    return
                else:
                    self.state = ExperimentState.RECORDING
                    self.state_start_time = ts
                    return # Start recording next frame
                    
        if self.state == ExperimentState.RECORDING:
            # Build flat dictionary
            flat_rec = {
                "experiment_id": self.experiment_id,
                "condition": self.current_condition,
                "trial_number": self.current_trial_number,
                "trial_phase": self.state,
                "requested_condition": self.current_condition,
                "observed_condition": record.active_conditions or "NORMAL",
                "condition_verification": self.current_trial_stats.verification,
                "camera_source": self.config.config.get("camera", {}).get("source", "mac")
            }
            flat_rec.update(record.__dict__)
            self.current_trial_stats.records.append(flat_rec)
            
            if elapsed >= self.recording_seconds:
                self.stop_trial()

    def _export_trial_csv(self, trial_stats):
        if not trial_stats.records:
            print(f"\n[!] Phase 8 Trial {trial_stats.trial_number} ({trial_stats.condition}) completed with 0 recorded rows. No CSV exported.")
            return None
            
        output_dir = "experiments"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"phase8_{ts}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trial_stats.records[0].keys())
            writer.writeheader()
            for row in trial_stats.records:
                writer.writerow(row)
                
        # Verification
        if not os.path.exists(csv_path):
            print("PHASE 8 CSV EXPORT FAILED: File does not exist")
            return None
        if os.path.getsize(csv_path) == 0:
            print("PHASE 8 CSV EXPORT FAILED: File is empty")
            return None
            
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                if not headers:
                    print("PHASE 8 CSV EXPORT FAILED: Missing headers")
                    return None
                    
                required = ["experiment_id", "condition", "trial_number", "data_quality", "temperature", "light", "vibration_index"]
                missing = [c for c in required if c not in headers]
                if missing:
                    print(f"PHASE 8 CSV EXPORT FAILED: Missing required columns: {missing}")
                    return None
                    
                rows = list(reader)
                if len(rows) < 1:
                    print("PHASE 8 CSV EXPORT FAILED: No data rows found")
                    return None
        except Exception as e:
            print(f"PHASE 8 CSV EXPORT FAILED: Parse error - {e}")
            return None
                
        print(f"\nPHASE 8 CSV SAVED: {csv_path}")
        return csv_path

    def stop_trial(self):
        if self.state == ExperimentState.RECORDING and self.current_trial_stats:
            self.current_trial_stats.end_time = time.monotonic()
            
            # Analyze
            recs = self.current_trial_stats.records
            self.current_trial_stats.sample_count = len(recs)
            
            tracked = 0
            missed = 0
            lost_e = 0
            
            disps = []
            jitters = []
            confs = []
            
            temps = []
            hums = []
            lights = []
            vibs = []
            
            last_status = None
            
            for r in recs:
                st = r.get("tracking_status")
                if st in ("TRACKING", "NEW"):
                    tracked += 1
                elif st == "MISSED":
                    missed += 1
                    
                if last_status != "LOST" and st == "LOST":
                    lost_e += 1
                last_status = st
                
                if r.get("displacement") is not None: disps.append(r["displacement"])
                if r.get("jitter") is not None: jitters.append(r["jitter"])
                if r.get("confidence") is not None: confs.append(r["confidence"])
                
                if r.get("temperature") is not None: temps.append(r["temperature"])
                if r.get("humidity") is not None: hums.append(r["humidity"])
                if r.get("light") is not None: lights.append(r["light"])
                if r.get("vibration_index") is not None: vibs.append(r["vibration_index"])
                if r.get("rain") == 1: self.current_trial_stats.rain_observed = True
                
                dq = r.get("data_quality")
                if dq == "SYNCHRONIZED":
                    self.current_trial_stats.synchronized_records += 1
                elif dq == "STALE_SENSOR_DATA":
                    self.current_trial_stats.stale_records += 1
                elif dq == "NO_SENSOR_DATA":
                    self.current_trial_stats.missing_sensor_records += 1
                elif dq == "SENSOR_COMMUNICATION_LOST":
                    self.current_trial_stats.communication_loss_records += 1
                
            self.current_trial_stats.tracking_frames = tracked
            self.current_trial_stats.missed_frames = missed
            self.current_trial_stats.lost_events = lost_e
            
            if tracked < self.min_tracking_frames:
                self.current_trial_stats.status = ExperimentState.FAILED
                self.current_trial_stats.verification = ConditionVerification.FAILED
            else:
                self.current_trial_stats.status = ExperimentState.COMPLETED
                
            if disps: self.current_trial_stats.mean_displacement = sum(disps)/len(disps)
            if jitters: 
                self.current_trial_stats.mean_jitter = sum(jitters)/len(jitters)
                self.current_trial_stats.peak_jitter = max(jitters)
            if confs: self.current_trial_stats.mean_confidence = sum(confs)/len(confs)
            
            if temps:
                self.current_trial_stats.mean_temperature = sum(temps)/len(temps)
                self.current_trial_stats.min_temperature = min(temps)
                self.current_trial_stats.max_temperature = max(temps)
                
            if hums:
                self.current_trial_stats.mean_humidity = sum(hums)/len(hums)
                self.current_trial_stats.min_humidity = min(hums)
                self.current_trial_stats.max_humidity = max(hums)
                
            if lights:
                self.current_trial_stats.mean_light = sum(lights)/len(lights)
                self.current_trial_stats.min_light = min(lights)
                self.current_trial_stats.max_light = max(lights)
                
            if vibs:
                self.current_trial_stats.mean_vibration = sum(vibs)/len(vibs)
                self.current_trial_stats.max_vibration = max(vibs)
                
            if recs:
                last_rec = recs[-1]
                self.current_trial_stats.camera_fps = last_rec.get("camera_fps") or 0.0
                self.current_trial_stats.detector_fps = last_rec.get("detector_fps") or 0.0
                self.current_trial_stats.tracking_fps = last_rec.get("tracking_fps") or 0.0

            self._export_trial_csv(self.current_trial_stats)
            self.trials.append(self.current_trial_stats)
            self.state = ExperimentState.COMPLETED
            self.current_trial_stats = None
        else:
            if self.current_trial_stats:
                self._export_trial_csv(self.current_trial_stats)
                self.current_trial_stats.status = ExperimentState.FAILED
                self.trials.append(self.current_trial_stats)
                self.current_trial_stats = None
            self.state = ExperimentState.COMPLETED

    def stop_safely(self):
        self.stop_trial()
        return self.export_logs()

    def export_logs(self, output_dir: str = "experiments"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        csv_path = os.path.join(output_dir, f"phase8_{self.experiment_id}.csv")
        json_path = os.path.join(output_dir, f"phase8_{self.experiment_id}_summary.json")
        
        all_records = []
        for t in self.trials:
            all_records.extend(t.records)
            
        with open(csv_path, 'w', newline='') as f:
            if all_records:
                writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
                writer.writeheader()
                for row in all_records:
                    writer.writerow(row)

                    
        # Summary Aggregation
        summary = {
            "experiment_id": self.experiment_id,
            "conditions_performed": list(self.trial_counts.keys()),
            "trials_per_condition": self.trial_counts,
            "successful_trials": sum(1 for t in self.trials if t.status == ExperimentState.COMPLETED),
            "failed_trials": sum(1 for t in self.trials if t.status == ExperimentState.FAILED),
            "stabilization_failures": sum(1 for t in self.trials if t.status == ExperimentState.FAILED and t.sample_count == 0),
            "condition_verification_failures": sum(1 for t in self.trials if t.verification == ConditionVerification.FAILED),
            "camera_source": self.config.config.get("camera", {}).get("source", "mac"),
            "real_drone_available": False,
            "ground_truth_available": False,
            "trials": []
        }
        
        baseline_mean_jitter = None
        baseline_mean_disp = None
        baseline_tracking_ratio = None
        
        # Calculate descriptive baseline values
        baselines = [t for t in self.trials if t.condition == ExperimentConditions.BASELINE and t.status == ExperimentState.COMPLETED]
        if baselines:
            if any(t.mean_jitter is not None for t in baselines):
                baseline_mean_jitter = sum(t.mean_jitter for t in baselines if t.mean_jitter is not None) / len([t for t in baselines if t.mean_jitter is not None])
            if any(t.mean_displacement is not None for t in baselines):
                baseline_mean_disp = sum(t.mean_displacement for t in baselines if t.mean_displacement is not None) / len([t for t in baselines if t.mean_displacement is not None])
            
            t_ratios = [t.tracking_frames/t.sample_count for t in baselines if t.sample_count > 0]
            if t_ratios:
                baseline_tracking_ratio = sum(t_ratios)/len(t_ratios)

        for t in self.trials:
            t_ratio = (t.tracking_frames / t.sample_count) if t.sample_count > 0 else 0.0
            
            # Descriptive comparison
            delta_mean_jitter = None
            if t.mean_jitter is not None and baseline_mean_jitter is not None:
                delta_mean_jitter = t.mean_jitter - baseline_mean_jitter
                
            delta_tracking_ratio = None
            if baseline_tracking_ratio is not None:
                delta_tracking_ratio = t_ratio - baseline_tracking_ratio
                
            delta_mean_displacement = None
            if t.mean_displacement is not None and baseline_mean_disp is not None:
                delta_mean_displacement = t.mean_displacement - baseline_mean_disp

            t_dict = {
                "condition": t.condition,
                "trial_number": t.trial_number,
                "status": t.status,
                "verification": t.verification,
                "duration": (t.end_time - t.start_time) if t.end_time else 0.0,
                "sample_count": t.sample_count,
                "synchronized_records": t.synchronized_records,
                "stale_records": t.stale_records,
                "communication_loss_records": t.communication_loss_records,
                "missing_sensor_records": t.missing_sensor_records,
                "tracking_frames": t.tracking_frames,
                "missed_frames": t.missed_frames,
                "lost_events": t.lost_events,
                "tracking_ratio": t_ratio,
                "mean_displacement": t.mean_displacement,
                "mean_jitter": t.mean_jitter,
                "peak_jitter": t.peak_jitter,
                "mean_confidence": t.mean_confidence,
                "camera_fps": t.camera_fps,
                "detector_fps": t.detector_fps,
                "tracking_fps": t.tracking_fps,
                
                "environment": {
                    "mean_temperature": t.mean_temperature,
                    "min_temperature": t.min_temperature,
                    "max_temperature": t.max_temperature,
                    "mean_humidity": t.mean_humidity,
                    "min_humidity": t.min_humidity,
                    "max_humidity": t.max_humidity,
                    "mean_light": t.mean_light,
                    "min_light": t.min_light,
                    "max_light": t.max_light,
                    "mean_vibration": t.mean_vibration,
                    "max_vibration": t.max_vibration,
                    "rain_observed": t.rain_observed
                },
                "descriptive_comparison": {
                    "baseline_difference_mean_jitter": delta_mean_jitter,
                    "baseline_difference_tracking_ratio": delta_tracking_ratio,
                    "baseline_difference_mean_displacement": delta_mean_displacement
                }
            }
            summary["trials"].append(t_dict)
            
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        return csv_path, json_path
