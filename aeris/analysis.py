import os
import csv
import json
import math
import glob
import statistics
from datetime import datetime

class TrialAnalysis:
    def __init__(self, experiment_id, requested_condition, trial_number):
        self.experiment_id = experiment_id
        self.requested_condition = requested_condition
        self.condition = requested_condition
        self.observed_condition = "UNKNOWN"
        self.trial_number = trial_number
        self.records = []
        self.camera_source = "mac"
        
        # Quality
        self.total_records = 0
        self.synchronized_records = 0
        self.stale_records = 0
        self.missing_sensor_records = 0
        
        # Computed
        self.tracked_frame_count = 0
        self.missed_frame_count = 0
        self.lost_event_count = 0
        
        self.tracking_ratio = None
        self.missed_ratio = None
        self.mean_displacement = None
        self.median_displacement = None
        self.rms_displacement = None
        self.peak_displacement = None
        
        self.mean_jitter = None
        self.median_jitter = None
        self.rms_jitter = None
        self.peak_jitter = None
        
        self.mean_confidence = None
        self.median_confidence = None
        
        self.mean_camera_fps = None
        self.mean_detector_fps = None
        self.mean_tracking_fps = None
        
        self.mean_temperature = None
        self.median_temperature = None
        self.min_temperature = None
        self.max_temperature = None
        self.std_temperature = None
        
        self.mean_humidity = None
        self.min_humidity = None
        self.max_humidity = None
        
        self.mean_light = None
        self.min_light = None
        self.max_light = None
        
        self.mean_vibration = None
        self.max_vibration = None
        
        self.rain_observed = False
        self.condition_verification = "UNKNOWN"
        self.real_drone_available = False
        self.ground_truth_available = False

    def aggregate(self):
        self.total_records = len(self.records)
        if self.total_records == 0:
            return
            
        disps, jitters, confs = [], [], []
        temps, hums, lights, vibs = [], [], [], []
        
        cam_fps, det_fps, trk_fps = [], [], []
        
        for r in self.records:
            dq = r.get("data_quality") or r.get("fusion_data_quality")
            if dq == "SYNCHRONIZED": self.synchronized_records += 1
            elif dq == "STALE_SENSOR_DATA": self.stale_records += 1
            elif dq in ("NO_SENSOR_DATA", "SENSOR_COMMUNICATION_LOST"): self.missing_sensor_records += 1
            
            st = r.get("tracking_status")
            if st in ("TRACKING", "NEW"): self.tracked_frame_count += 1
            elif st == "MISSED": self.missed_frame_count += 1
            
            v = r.get("condition_verification")
            if v and v != "UNKNOWN": self.condition_verification = v
            
            cs = r.get("camera_source")
            if cs: self.camera_source = cs
            
            obs = r.get("observed_condition") or r.get("active_conditions")
            if obs: self.observed_condition = obs
            
            d = self._float_or_none(r.get("displacement"))
            j = self._float_or_none(r.get("jitter"))
            c = self._float_or_none(r.get("confidence"))
            
            if d is not None: disps.append(d)
            if j is not None: jitters.append(j)
            if c is not None: confs.append(c)
            
            cf = self._float_or_none(r.get("camera_fps"))
            df = self._float_or_none(r.get("detector_fps"))
            tf = self._float_or_none(r.get("tracking_fps"))
            
            if cf is not None: cam_fps.append(cf)
            if df is not None: det_fps.append(df)
            if tf is not None: trk_fps.append(tf)
            
            t = self._float_or_none(r.get("temperature"))
            h = self._float_or_none(r.get("humidity"))
            l = self._float_or_none(r.get("light"))
            vb = self._float_or_none(r.get("vibration_index"))
            
            if t is not None: temps.append(t)
            if h is not None: hums.append(h)
            if l is not None: lights.append(l)
            if vb is not None: vibs.append(vb)
            
            if r.get("rain") == "1": self.rain_observed = True
            
        # Lost events calculation
        last_status = None
        for r in self.records:
            st = r.get("tracking_status")
            if last_status != "LOST" and st == "LOST":
                self.lost_event_count += 1
            last_status = st
            
        self.tracking_ratio = self.tracked_frame_count / self.total_records if self.total_records > 0 else None
        self.missed_ratio = self.missed_frame_count / self.total_records if self.total_records > 0 else None
        
        self.mean_displacement = self._mean(disps)
        self.median_displacement = self._median(disps)
        self.rms_displacement = self._rms(disps)
        self.peak_displacement = max(disps) if disps else None
        
        self.mean_jitter = self._mean(jitters)
        self.median_jitter = self._median(jitters)
        self.rms_jitter = self._rms(jitters)
        self.peak_jitter = max(jitters) if jitters else None
        
        self.mean_confidence = self._mean(confs)
        self.median_confidence = self._median(confs)
        
        self.mean_camera_fps = self._mean(cam_fps)
        self.mean_detector_fps = self._mean(det_fps)
        self.mean_tracking_fps = self._mean(trk_fps)
        
        self.mean_temperature = self._mean(temps)
        self.median_temperature = self._median(temps)
        self.min_temperature = min(temps) if temps else None
        self.max_temperature = max(temps) if temps else None
        self.std_temperature = self._std(temps)
        
        self.mean_humidity = self._mean(hums)
        self.min_humidity = min(hums) if hums else None
        self.max_humidity = max(hums) if hums else None
        
        self.mean_light = self._mean(lights)
        self.min_light = min(lights) if lights else None
        self.max_light = max(lights) if lights else None
        
        self.mean_vibration = self._mean(vibs)
        self.max_vibration = max(vibs) if vibs else None

        if not self.observed_condition or self.observed_condition in ("NORMAL", "NONE", ""):
            if self.requested_condition == "BASELINE":
                self.condition = "BASELINE"
            else:
                self.condition = "NORMAL"
        else:
            self.condition = self.observed_condition

    def _float_or_none(self, val):
        try:
            return float(val) if val and val != "None" and val != "N/A" else None
        except ValueError:
            return None

    def _mean(self, arr):
        return sum(arr)/len(arr) if arr else None

    def _median(self, arr):
        return statistics.median(arr) if arr else None
        
    def _rms(self, arr):
        return math.sqrt(sum(x*x for x in arr)/len(arr)) if arr else None
        
    def _std(self, arr):
        if not arr or len(arr) < 2: return 0.0
        m = self._mean(arr)
        return math.sqrt(sum((x - m)**2 for x in arr)/(len(arr)-1))

class ExperimentAnalyzer:
    def __init__(self, config_manager):
        self.config = config_manager
        cfg = self.config.config.get("analysis", {})
        self.min_correlation_trials = cfg.get("minimum_correlation_trials", 5)
        
        self.trials = {} # (experiment_id, requested_condition, trial_number) -> TrialAnalysis
        self.baselines = []
        
        self.input_file = None
        self.real_drone_available = False
        self.ground_truth_available = False
        
        self.discovered_files = 0
        self.included_files = []
        self.excluded_files = []
        
        self.total_records = 0
        self.synchronized_records = 0
        self.stale_records = 0
        self.missing_sensor_records = 0
        
        self.limitations = []
        self.comparisons = []
        self.correlations = []

    def load_csvs(self, file_paths):
        for f in file_paths:
            self.load_csv(f)

    def load_csv(self, file_path):
        self.input_file = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
            
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.total_records += 1
                dq = row.get("data_quality") or row.get("fusion_data_quality")
                if dq == "SYNCHRONIZED": self.synchronized_records += 1
                elif dq == "STALE_SENSOR_DATA": self.stale_records += 1
                elif dq in ("NO_SENSOR_DATA", "SENSOR_COMMUNICATION_LOST"): self.missing_sensor_records += 1
                
                exp_id = row.get("experiment_id") or "UNKNOWN"
                if exp_id.startswith("test_") or exp_id == "test_exp_123":
                    # Exclude internal test fixtures from records completely
                    continue
                
                # We only analyze synchronized data by default
                if dq != "SYNCHRONIZED":
                    continue
                    
                cond = row.get("condition")
                t_num = row.get("trial_number")
                if not cond or not t_num:
                    continue
                
                try:
                    t_num = int(t_num)
                except ValueError:
                    continue
                    
                key = (exp_id, cond, t_num)
                if key not in self.trials:
                    self.trials[key] = TrialAnalysis(exp_id, cond, t_num)
                    
                self.trials[key].records.append(row)
                
        # Also check JSON for drone availability
        json_path = file_path.replace(".csv", "_summary.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    d = json.load(f)
                    self.real_drone_available = d.get("real_drone_available", False)
                    self.ground_truth_available = d.get("ground_truth_available", False)
            except:
                pass

    def aggregate_trials(self):
        for t in self.trials.values():
            t.aggregate()
            
            if t.requested_condition == "BASELINE":
                if t.observed_condition not in ("NORMAL", "NONE", ""):
                    self.limitations.append(f"CONFOUNDED BASELINE: Trial {t.trial_number} in {t.experiment_id} requested BASELINE but observed '{t.observed_condition}'. Excluded from baseline aggregation.")
                elif t.condition_verification != "FAILED":
                    self.baselines.append(t)
                
    def _mean_of_trials(self, trial_list, attr):
        vals = [getattr(t, attr) for t in trial_list if getattr(t, attr) is not None]
        return sum(vals)/len(vals) if vals else None
        
    def _safe_pct(self, val, base):
        if val is None or base is None or base == 0:
            return None
        return ((val - base) / abs(base)) * 100

    def compare_to_baseline(self):
        if not self.baselines:
            self.limitations.append("No valid BASELINE trials found. Cannot compute differences.")
            return
            
        base_mean_jitter = self._mean_of_trials(self.baselines, "mean_jitter")
        base_mean_disp = self._mean_of_trials(self.baselines, "mean_displacement")
        base_tracking_ratio = self._mean_of_trials(self.baselines, "tracking_ratio")
        base_lost_events = self._mean_of_trials(self.baselines, "lost_event_count")
        
        conditions = set(t.condition for t in self.trials.values() if t.condition != "BASELINE")
        
        for c in conditions:
            c_trials = [t for t in self.trials.values() if t.condition == c]
            c_mean_jitter = self._mean_of_trials(c_trials, "mean_jitter")
            c_mean_disp = self._mean_of_trials(c_trials, "mean_displacement")
            c_tracking_ratio = self._mean_of_trials(c_trials, "tracking_ratio")
            c_lost_events = self._mean_of_trials(c_trials, "lost_event_count")
            
            comp = {
                "condition": c,
                "baseline_trials": len(self.baselines),
                "condition_trials": len(c_trials),
            }
            
            if c_mean_jitter is not None and base_mean_jitter is not None:
                comp["mean_jitter_baseline"] = base_mean_jitter
                comp["mean_jitter_condition"] = c_mean_jitter
                comp["difference_mean_jitter"] = c_mean_jitter - base_mean_jitter
                comp["pct_diff_mean_jitter"] = self._safe_pct(c_mean_jitter, base_mean_jitter)
                comp["direction_mean_jitter"] = "INCREASED" if comp["difference_mean_jitter"] > 0 else ("DECREASED" if comp["difference_mean_jitter"] < 0 else "UNCHANGED")
                
            if c_tracking_ratio is not None and base_tracking_ratio is not None:
                comp["mean_tracking_ratio_baseline"] = base_tracking_ratio
                comp["mean_tracking_ratio_condition"] = c_tracking_ratio
                comp["difference_tracking_ratio"] = c_tracking_ratio - base_tracking_ratio
                comp["direction_tracking_ratio"] = "INCREASED" if comp["difference_tracking_ratio"] > 0 else ("DECREASED" if comp["difference_tracking_ratio"] < 0 else "UNCHANGED")
                
            self.comparisons.append(comp)

    def calculate_correlations(self, method="pearson"):
        # We need paired trials
        valid_trials = [t for t in self.trials.values() if t.total_records > 0]
        n = len(valid_trials)
        
        pairs = [
            ("mean_vibration", "mean_jitter"),
            ("mean_vibration", "tracking_ratio"),
            ("mean_light", "mean_jitter"),
            ("mean_light", "tracking_ratio")
        ]
        
        for env_attr, track_attr in pairs:
            x_vals = []
            y_vals = []
            
            for t in valid_trials:
                x = getattr(t, env_attr)
                y = getattr(t, track_attr)
                if x is not None and y is not None:
                    x_vals.append(x)
                    y_vals.append(y)
                    
            count = len(x_vals)
            res = {
                "x_variable": env_attr,
                "y_variable": track_attr,
                "n": count,
                "method": method
            }
            
            if count < self.min_correlation_trials:
                res["correlation"] = "N/A"
                res["status"] = "INSUFFICIENT_DATA"
            else:
                res["status"] = "CALCULATED"
                if method == "pearson":
                    res["correlation"] = self._pearson(x_vals, y_vals)
                elif method == "spearman":
                    res["correlation"] = self._spearman(x_vals, y_vals)
                    
            self.correlations.append(res)
            
    def _pearson(self, x, y):
        n = len(x)
        if n < 2: return None
        mx, my = sum(x)/n, sum(y)/n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den_x = sum((xi - mx)**2 for xi in x)
        den_y = sum((yi - my)**2 for yi in y)
        if den_x == 0 or den_y == 0: return 0.0
        return num / math.sqrt(den_x * den_y)
        
    def _spearman(self, x, y):
        n = len(x)
        if n < 2: return None
        def get_ranks(arr):
            indexed = list(enumerate(arr))
            indexed.sort(key=lambda v: v[1])
            ranks = [0] * n
            for rank, (idx, _) in enumerate(indexed):
                ranks[idx] = rank + 1
            return ranks
            
        rx = get_ranks(x)
        ry = get_ranks(y)
        return self._pearson(rx, ry)

    def detect_confounders(self):
        if not self.real_drone_available:
            self.limitations.append("REAL DRONE DATA NOT AVAILABLE: Results derived from software validation target, not a physical drone.")
        if not self.ground_truth_available:
            self.limitations.append("GROUND TRUTH NOT AVAILABLE: Positional accuracy cannot be independently verified.")
            
        if self.missing_sensor_records > 0 or self.stale_records > 0:
            self.limitations.append(f"Data quality issues present: {self.stale_records} stale records, {self.missing_sensor_records} missing sensor records ignored.")
            
        durs = [t.total_records for t in self.trials.values()]
        if durs and max(durs) - min(durs) > (0.5 * max(durs)):
            self.limitations.append("TRIAL DURATIONS VARY SIGNIFICANTLY: This may confound cumulative metrics like lost_event_count.")
            
        if any(t.tracked_frame_count < 10 for t in self.trials.values()):
            self.limitations.append("INSUFFICIENT TRACKING FRAMES: Some trials have fewer than 10 tracked frames.")
            
        conds = set(t.condition for t in self.trials.values())
        counts = [len([t for t in self.trials.values() if t.condition == c]) for c in conds]
        if counts and max(counts) != min(counts):
            self.limitations.append("UNEQUAL TRIAL COUNTS: Statistical comparisons may be unbalanced.")

    def generate_outputs(self, output_dir="experiments"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(output_dir, f"phase9_{ts}")
        
        # 1. JSON
        j_data = {
            "analysis_metadata": {
                "analysis_timestamp": ts,
                "input_file": self.input_file,
                "discovered_files": self.discovered_files,
                "included_files": self.included_files,
                "excluded_files": self.excluded_files,
                "real_drone_available": self.real_drone_available,
                "ground_truth_available": self.ground_truth_available
            },
            "data_quality": {
                "total_records": self.total_records,
                "synchronized_records": self.synchronized_records,
                "stale_records": self.stale_records,
                "missing_sensor_records": self.missing_sensor_records
            },
            "condition_comparisons": self.comparisons,
            "correlations": self.correlations,
            "limitations": self.limitations
        }
        json_path = f"{prefix}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(j_data, f, indent=4)
            
        # 2. CSV
        csv_path = f"{prefix}_trial_summary.csv"
        if self.trials:
            with open(csv_path, 'w', newline='') as f:
                keys = ["experiment_id", "camera_source", "requested_condition", "condition", "trial_number", "total_records", "tracking_ratio", "mean_jitter", "mean_displacement", "mean_vibration", "mean_light"]
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for t in self.trials.values():
                    row = {k: getattr(t, k, None) for k in keys}
                    writer.writerow(row)
                    
        # 3. TXT Report
        txt_path = f"{prefix}_report.txt"
        with open(txt_path, 'w') as f:
            f.write("AERIS-X PHASE 9 ANALYSIS\n")
            f.write("========================\n\n")
            f.write("1. Dataset Discovery\n")
            f.write(f"DISCOVERED PHASE 8 FILES: {self.discovered_files}\n")
            f.write(f"VALID REAL EXPERIMENT FILES: {len(self.included_files)}\n")
            f.write(f"EXCLUDED FILES: {len(self.excluded_files)}\n\n")
            
            f.write("Included Files:\n")
            for vf in self.included_files:
                f.write(f"  - {vf}\n")
            if self.excluded_files:
                f.write("\nExcluded Files:\n")
                for vf, reason in self.excluded_files:
                    f.write(f"  - {vf} ({reason})\n")
                    
            cond_counts = {}
            for t in self.trials.values():
                c = t.condition
                cond_counts[c] = cond_counts.get(c, 0) + 1
                
            f.write("\nVALID TRIALS BY OBSERVED CONDITION:\n")
            for c, count in cond_counts.items():
                f.write(f"{c}: {count}\n")
            f.write(f"CLEAN BASELINES: {len(self.baselines)}\n\n")
            
            f.write("2. Data Quality\n")
            f.write(f"Total rows: {self.total_records}\n")
            f.write(f"Synchronized: {self.synchronized_records}\n")
            f.write(f"Stale: {self.stale_records}\n")
            f.write(f"Missing: {self.missing_sensor_records}\n\n")
            
            f.write("3. Condition Comparisons\n")
            for c in self.comparisons:
                f.write(f"Condition: {c['condition']} vs BASELINE\n")
                if "difference_mean_jitter" in c:
                    f.write(f"  Mean Jitter: Associated change of {c['difference_mean_jitter']:.2f} px ({c['direction_mean_jitter']})\n")
                if "difference_tracking_ratio" in c:
                    f.write(f"  Tracking Ratio: Associated change of {c['difference_tracking_ratio']:.3f} ({c['direction_tracking_ratio']})\n")
            f.write("\n")
            
            f.write("4. Correlation Results\n")
            for cr in self.correlations:
                f.write(f"{cr['x_variable']} vs {cr['y_variable']} (n={cr['n']}): ")
                if cr['status'] == "INSUFFICIENT_DATA":
                    f.write("INSUFFICIENT_DATA\n")
                else:
                    f.write(f"{cr['correlation']:.3f} ({cr['method']})\n")
            f.write("\n")
            
            f.write("5. Interpretation Limitations (Confounders)\n")
            for l in self.limitations:
                f.write(f"- {l}\n")
            f.write("\n")
            
            f.write("6. Conclusions Supported by Data\n")
            if not self.real_drone_available:
                f.write("NO EXPERIMENTAL CONCLUSION GENERATED. (Lack of real drone target)\n")
            elif not self.comparisons:
                f.write("NO EXPERIMENTAL CONCLUSION GENERATED. (Insufficient baseline comparisons)\n")
            else:
                for c in self.comparisons:
                    if "difference_mean_jitter" in c:
                        diff = c["difference_mean_jitter"]
                        dir_str = "higher" if diff > 0 else "lower"
                        f.write(f"Within this prototype dataset, mean image-space jitter was {dir_str} during the measured {c['condition']} condition than during baseline.\n")
            
            f.write("\nNote: Phase 9 analyzes associations and descriptive differences within the collected prototype dataset. It does not by itself establish environmental causality.\n")
            
        return txt_path, csv_path, json_path
