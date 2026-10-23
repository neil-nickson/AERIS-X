import sys
import os
import time
import argparse
import threading
import json
import csv
import cv2

from aeris.config import ConfigManager
from aeris.serial_link import SerialLink
from aeris.sensor_parser import SensorParser
from aeris.health_monitor import HealthMonitor
from aeris.environment_engine import EnvironmentEngine
from aeris.camera import AerisCamera
from aeris.detector import AerisDroneDetector
from aeris.tracker import AerisTracker
from aeris.tracking_metrics import TrackingMetrics
from aeris.fusion import AerisFusionEngine
from aeris.adaptive_controller import AdaptiveController
from aeris.performance_assessment import PerformanceAssessor
from aeris.dashboard import AerisDashboard

stop_event = threading.Event()

def serial_thread_func(config_mgr, fusion_engine, state_cache, link=None):
    if link is None:
        link = SerialLink(config_mgr)
        if not link.connect():
            fusion_engine.update_environment(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)
            state_cache['env_health'] = "COMMUNICATION LOST"
        else:
            state_cache['env_health'] = "NOMINAL"
    else:
        if link.conn is None:
            state_cache['env_health'] = "COMMUNICATION LOST"
        else:
            state_cache['env_health'] = "NOMINAL"
        
    parser = SensorParser(config_mgr)
    health = HealthMonitor()
    env_engine = EnvironmentEngine(config_mgr)
    
    while not stop_event.is_set():
        if link.conn is None:
            time.sleep(2)
            if not link.connect():
                fusion_engine.update_environment(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)
                state_cache['env_health'] = "COMMUNICATION LOST"
                continue
            else:
                state_cache['env_health'] = "NOMINAL"
                
        line = link.read_line()
        if line is None:
            health.update(None)
            status = health.get_health_status()
            env_res = env_engine.evaluate(None, status, None)
            if status == "COMMUNICATION LOST":
                link.close()
            fusion_engine.update_environment(None, status, env_res, None)
            state_cache['env_health'] = status
            continue
            
        parsed = parser.parse(line)
        if not parsed:
            continue
            
        health.update(parsed)
        status = health.get_health_status()
        state_cache['env_health'] = status
        
        # Simple extraction for deviation assuming baselines are not perfectly calibrated here for live demo
        base_temp, base_light, base_x, base_y, base_z = 25.0, 500, 330, 330, 330
        
        t_dev = abs(parsed.get("TEMP", 0) - base_temp)
        l_ratio = (parsed.get("LIGHT", 1) / base_light) if base_light > 0 else 1.0
        dx = parsed.get("AX", 0) - base_x
        dy = parsed.get("AY", 0) - base_y
        dz = parsed.get("AZ", 0) - base_z
        vib_mag = (dx*dx + dy*dy + dz*dz)**0.5 if status == "NOMINAL" else 0.0
        
        dev_data = {
            "temp_abs_delta": t_dev,
            "light_ratio": l_ratio,
            "vibration": vib_mag
        }
        
        env_res = env_engine.evaluate(parsed, status, dev_data)
        fusion_engine.update_environment(parsed, status, env_res, dev_data)
        
    if link.conn:
        link.close()

def run_live(config_mgr):
    print("==========================================")
    print("PHASE 12 LIVE STARTUP")
    print("==========================================\n")
    
    print("Camera index:")
    print(config_mgr.config.get("camera", {}).get("index", 0))
    cam = AerisCamera(config_mgr)
    opened = cam.initialize()
    print(f"Camera opened: {opened}")
    if opened and cam.cap:
        print(f"Camera resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n")
    else:
        print("Camera resolution: N/A\n")
        
    link = SerialLink(config_mgr)
    print(f"Serial port: {link.port if link.port != 'auto' else 'AUTO-DISCOVER'}")
    
    arduino_connected = False
    try:
        arduino_connected = link.connect()
        if arduino_connected:
            print("Serial opened: YES")
        else:
            print("Serial opened: FAILED")
    except Exception as e:
        print(f"Serial opened: FAILED ({e})")
        
    first_packet = None
    if arduino_connected:
        start_wait = time.time()
        while time.time() - start_wait < 3.0:
            line = link.read_line()
            if line:
                first_packet = line
                break
            time.sleep(0.1)
    
    print(f"First sensor packet received: {first_packet if first_packet else 'NONE'}")
    print(f"Sensor health: {'NOMINAL' if first_packet else 'UNKNOWN'}\n")
    
    det = AerisDroneDetector(config_mgr)
    print("Detector:")
    try:
        det.initialize()
        print("Detector loaded: SUCCESS\n")
    except Exception as e:
        print(f"Detector loaded: FAILED ({e})\n")
        
    tracker = AerisTracker(config_mgr)
    metrics = TrackingMetrics()
    
    fusion_engine = AerisFusionEngine(config_mgr)
    print("Fusion:")
    print("Fusion status: INITIALIZING\n")
    
    dashboard = AerisDashboard()
    adaptive_controller = AdaptiveController(config_mgr)
    assessor = PerformanceAssessor(config_mgr)
    
    state_cache = {'env_health': 'UNKNOWN'}
    
    serial_thread = threading.Thread(target=serial_thread_func, args=(config_mgr, fusion_engine, state_cache, link), daemon=True)
    serial_thread.start()
    
    window_name = "AERIS-X Final Demonstration"

    os.makedirs("experiments/phase12", exist_ok=True)
    recording = False
    csv_writer = None
    csv_file = None
    
    try:
        while True:
            # 1. Sense Camera
            if cam.cap and cam.cap.isOpened():
                ret, frame = cam.read()
                if not ret:
                    frame = None
            else:
                frame = None
                
            # 2. Detect & Track
            if frame is not None:
                h, w = frame.shape[:2]
                detections = det.detect(frame)
                track = tracker.update(detections, w, h)
                sample = metrics.update(track=track, camera_fps=cam.get_fps(), detector_fps=det.get_inference_fps(), tracking_fps=None)
            else:
                sample = metrics.update(track=None)

            # 3. Fuse
            fusion_record = fusion_engine.fuse_vision(sample)
            
            # 4. Adapt
            if frame is not None:
                adaptive_controller.update_environment(
                    fusion_record.environment_state or "UNKNOWN", 
                    fusion_record.active_conditions or "NONE", 
                    fusion_record.sensor_health or "UNKNOWN"
                )
                adaptive_controller.update_tracking_metrics(
                    jitter=fusion_record.jitter, 
                    displacement=fusion_record.displacement, 
                    tracking_status=fusion_record.tracking_status
                )
                adaptive_controller.evaluate()
            
            # 5. Assess
            env_data = {
                "TEMP": fusion_record.temperature,
                "HUM": fusion_record.humidity,
                "LIGHT": fusion_record.light,
                "vibration": fusion_record.vibration_index,
                "RAIN_DETECTED": fusion_record.rain,
                "state": fusion_record.environment_state or "UNKNOWN",
                "health": state_cache['env_health']
            }
            
            summary = metrics.get_summary()
            tracking_data = {
                "status": fusion_record.tracking_status,
                "track_id": fusion_record.track_id,
                "confidence": fusion_record.confidence,
                "center_x": fusion_record.center_x,
                "center_y": fusion_record.center_y,
                "bbox_x1": fusion_record.bbox_x1,
                "bbox_y1": fusion_record.bbox_y1,
                "bbox_x2": fusion_record.bbox_x2,
                "bbox_y2": fusion_record.bbox_y2,
                "mean_jitter": summary.mean_jitter,
                "mean_displacement": summary.mean_displacement,
                "tracking_ratio": summary.tracking_ratio,
                "lost_events": summary.lost_events
            }
            
            qual_data = {
                "status": fusion_record.data_quality,
                "stale_sensor_data": fusion_record.data_quality == "STALE_SENSOR_DATA",
                "delta_ms": fusion_record.timestamp_delta_ms
            }
            
            adapt_data = {
                "state": adaptive_controller.state.name,
                "strategy": adaptive_controller.current_strategy or "NONE",
                "status": "ACCEPTED" if adaptive_controller.state.name == "ADAPTATION_ACCEPTED" else ("REVERTED" if adaptive_controller.state.name == "ADAPTATION_REJECTED" else "PENDING"),
                "evidence": "INSUFFICIENT" if adaptive_controller.state.name == "INSUFFICIENT_EVIDENCE" else "SUFFICIENT"
            }

            report = assessor.assess(env_data, tracking_data, {}, qual_data, adapt_data, is_simulation=False)

            # Draw Dashboard
            state_data = {
                "mode": "LIVE MODE",
                "camera_frame": frame,
                "environment": env_data,
                "metrics": tracking_data,
                "performance": report.performance_assessment,
                "degradation": report.degradation_assessment,
                "fusion": qual_data,
                "adaptation": adapt_data,
                "camera_source": "IPHONE CONTINUITY CAMERA" if cam.source == "iphone" else "MAC CAMERA"
            }
            
            ui = dashboard.render(state_data)
            
            if recording:
                cv2.circle(ui, (1200, 40), 10, (0, 0, 255), -1)
                cv2.putText(ui, "REC", (1220, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if csv_writer:
                    csv_writer.writerow(fusion_record.__dict__)
            
            cv2.imshow(window_name, ui)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    recording = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    csv_file = open(f"experiments/phase12/phase12_{ts}.csv", "w", newline='')
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fusion_record.__dict__.keys())
                    csv_writer.writeheader()
                    print("Recording started.")
                else:
                    recording = False
                    if csv_file: csv_file.close()
                    print("Recording stopped.")
                    
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping Phase 12 safely...")
        stop_event.set()
        serial_thread.join(timeout=2.0)
        if cam.cap: cam.release()
        cv2.destroyAllWindows()
        if recording and csv_file: csv_file.close()

def run_simulation(config_mgr):
    print("Starting AERIS-X Phase 12 - SIMULATION MODE")
    dashboard = AerisDashboard()
    
    # Deterministic mock sequences
    stages = [
        {"desc": "Normal tracking", "env_state": "NORMAL", "q_state": "SYNCHRONIZED", "j": 1.2, "l": 0, "r": 0.98},
        {"desc": "Mild degradation (Vibration)", "env_state": "HIGH-VIBRATION", "q_state": "SYNCHRONIZED", "j": 6.5, "l": 1, "r": 0.85},
        {"desc": "Moderate degradation", "env_state": "HIGH-VIBRATION", "q_state": "SYNCHRONIZED", "j": 12.0, "l": 2, "r": 0.70},
        {"desc": "Adaptation Accepted", "env_state": "HIGH-VIBRATION", "q_state": "SYNCHRONIZED", "j": 3.0, "l": 0, "r": 0.90, "adapt": "ACCEPTED"},
        {"desc": "Sensor Communication Lost", "env_state": "UNKNOWN", "q_state": "SENSOR_COMMUNICATION_LOST", "j": 3.0, "l": 0, "r": 0.90}
    ]
    
    window_name = "AERIS-X Final Demonstration"
    
    for stage in stages:
        vib_val = 60.0 if stage["env_state"] == "HIGH-VIBRATION" else 1.0
        env_data = {
            "TEMP": 25.0, 
            "HUM": 45.0, 
            "LIGHT": 600, 
            "vibration": vib_val, 
            "RAIN_DETECTED": False, 
            "state": stage["env_state"], 
            "active_conditions": [stage["env_state"]] if stage["env_state"] != "NORMAL" else [],
            "health": "NOMINAL" if stage["q_state"] == "SYNCHRONIZED" else "COMMUNICATION LOST"
        }
        tracking_data = {"status": "TRACKING", "track_id": 1, "confidence": 0.85, "center_x": 400, "center_y": 300, "bbox_x1": 350, "bbox_y1": 250, "bbox_x2": 450, "bbox_y2": 350, "mean_jitter": stage["j"], "mean_displacement": stage["j"]*1.5, "tracking_ratio": stage["r"], "lost_events": stage["l"]}
        qual_data = {"status": stage["q_state"], "stale_sensor_data": False, "delta_ms": 10.0}
        adapt_data = {"state": "ADAPTATION_ACCEPTED" if stage.get("adapt") == "ACCEPTED" else "INACTIVE", "strategy": "VIBRATION_SMOOTHING" if stage.get("adapt") else "NONE", "status": stage.get("adapt", "PENDING"), "evidence": "SUFFICIENT"}
        
        assessor = PerformanceAssessor(config_mgr)
        report = assessor.assess(env_data, tracking_data, {}, qual_data, adapt_data, is_simulation=True)
        
        state_data = {
            "mode": "SIMULATION MODE",
            "scenario": stage["desc"].upper(),
            "camera_frame": None,
            "environment": env_data,
            "metrics": tracking_data,
            "performance": report.performance_assessment,
            "degradation": report.degradation_assessment,
            "fusion": qual_data,
            "adaptation": adapt_data,
            "camera_source": "SIMULATED DATA"
        }
        
        ui = dashboard.render(state_data)
        cv2.imshow(window_name, ui)
        
        print(f"Simulating: {stage['desc']}")
        # Wait 3 seconds per stage, break on 'q'
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    
    print("PHASE 12 SIMULATION COMPLETE")
    print("All deterministic scenarios executed.")
    print("Dashboard cleanup complete.")
    print("Process exiting.")

def run_replay(config_mgr):
    print("Starting AERIS-X Phase 12 - REPLAY MODE")
    print("Not fully implemented in this script version. Use --mode simulation or --mode live.")

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="AERIS-X Phase 12 - Final Demonstration\n\nDEFAULT:\nLIVE\n\nOPTIONAL:\n--mode live\n--mode simulation\n--mode replay",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", choices=["live", "simulation", "replay"], default="live", help="Run mode (default: live)")
    return parser.parse_args(args)

def main():
    args = parse_args()
    
    config_mgr = ConfigManager()
    
    if args.mode == "live":
        run_live(config_mgr)
    elif args.mode == "simulation":
        run_simulation(config_mgr)
    elif args.mode == "replay":
        run_replay(config_mgr)

if __name__ == "__main__":
    main()
