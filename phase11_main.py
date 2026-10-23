import os
import json
import time
import argparse
from dataclasses import asdict
from aeris.config import ConfigManager
from aeris.performance_assessment import PerformanceAssessor

def run_simulation(config_mgr):
    print("============================================================")
    print("AERIS-X PHASE 11: PERFORMANCE ASSESSMENT (SIMULATION MODE)")
    print("============================================================")
    
    assessor = PerformanceAssessor(config_mgr)
    os.makedirs("experiments", exist_ok=True)
    
    simulations = [
        {
            "name": "1. Normal tracking",
            "env": {"state": "NORMAL"},
            "metrics": {"mean_jitter": 2.0, "lost_events": 0, "tracking_ratio": 0.95},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {"mean_jitter": 1.5, "lost_events": 0, "tracking_ratio": 0.98},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "2. Mild degradation",
            "env": {"state": "HIGH-VIBRATION"},
            "metrics": {"mean_jitter": 6.0, "lost_events": 1, "tracking_ratio": 0.85},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "3. Moderate degradation",
            "env": {"state": "HIGH-VIBRATION"},
            "metrics": {"mean_jitter": 12.0, "lost_events": 4, "tracking_ratio": 0.70},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "4. Severe degradation",
            "env": {"state": "MULTI-CONDITION"},
            "metrics": {"mean_jitter": 18.0, "lost_events": 10, "tracking_ratio": 0.40},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "5. Insufficient tracking data",
            "env": {"state": "LOW-LIGHT"},
            "metrics": {"mean_jitter": None, "lost_events": None, "tracking_ratio": None},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "6. Sensor communication lost",
            "env": {"state": "UNKNOWN"},
            "metrics": {"mean_jitter": 2.0, "lost_events": 0, "tracking_ratio": 0.95},
            "quality": {"status": "SENSOR_COMMUNICATION_LOST", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "7. Stale environmental data",
            "env": {"state": "NORMAL"},
            "metrics": {"mean_jitter": 2.0, "lost_events": 0, "tracking_ratio": 0.95},
            "quality": {"status": "STALE_SENSOR_DATA", "stale_sensor_data": True},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "8. Unknown environment",
            "env": {"state": "UNKNOWN"},
            "metrics": {"mean_jitter": 2.0, "lost_events": 0, "tracking_ratio": 0.95},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "NOT_ATTEMPTED"}
        },
        {
            "name": "9. Adaptation accepted",
            "env": {"state": "LOW-LIGHT"},
            "metrics": {"mean_jitter": 3.0, "lost_events": 0, "tracking_ratio": 0.90},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "ACCEPTED"}
        },
        {
            "name": "10. Adaptation rejected/reverted",
            "env": {"state": "LOW-LIGHT"},
            "metrics": {"mean_jitter": 8.0, "lost_events": 2, "tracking_ratio": 0.70},
            "quality": {"status": "SYNCHRONIZED", "stale_sensor_data": False},
            "baseline": {},
            "adapt": {"status": "REJECTED"}
        }
    ]
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = f"experiments/phase11_{timestamp}_assessment.json"
    txt_path = f"experiments/phase11_{timestamp}_report.txt"
    
    reports = []
    
    with open(txt_path, 'w') as f_txt:
        f_txt.write("============================================================\n")
        f_txt.write("AERIS-X PHASE 11: PERFORMANCE & DEGRADATION ASSESSMENT\n")
        f_txt.write(f"MODE: DETERMINISTIC SIMULATION\n")
        f_txt.write(f"TIMESTAMP: {timestamp}\n")
        f_txt.write("============================================================\n\n")
        
        for sim in simulations:
            report = assessor.assess(
                env_data=sim["env"],
                tracking_metrics=sim["metrics"],
                baseline_metrics=sim["baseline"],
                data_quality=sim["quality"],
                adaptation_data=sim["adapt"],
                is_simulation=True
            )
            
            reports.append(asdict(report))
            
            f_txt.write(f"--- {sim['name']} ---\n")
            f_txt.write(f"PERFORMANCE STATE: {report.performance_assessment['performance_state']}\n")
            f_txt.write(f"DEGRADATION STATE: {report.degradation_assessment['degradation_state']}\n")
            f_txt.write(f"REASON CODES:      {', '.join(report.performance_assessment['reason_codes'])}\n\n")
            
            print(f"Simulated: {sim['name']}")
            print(f"  -> {report.performance_assessment['performance_state']} | {report.degradation_assessment['degradation_state']}")
            
        f_txt.write("============================================================\n")
        f_txt.write("LIMITATIONS & DISCLAIMERS\n")
        for lim in reports[0]["limitations"]:
            f_txt.write(f"- {lim}\n")

    with open(json_path, 'w') as f_json:
        json.dump(reports, f_json, indent=4)
        
    print(f"\nGenerated simulation reports:")
    print(f" - {json_path}")
    print(f" - {txt_path}")
    print("Phase 11 simulation completed cleanly.")

def main():
    parser = argparse.ArgumentParser(description="AERIS-X Phase 11 - Performance Assessment")
    parser.add_argument("--simulate", action="store_true", help="Run deterministic simulation mode")
    args = parser.parse_args()
    
    config_mgr = ConfigManager()
    
    if args.simulate:
        run_simulation(config_mgr)
    else:
        print("REAL DATA MODE not fully integrated into a live runner in this prototype script.")
        print("Please use --simulate for deterministic simulation.")

if __name__ == "__main__":
    main()
