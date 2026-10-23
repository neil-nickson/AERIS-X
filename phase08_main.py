import cv2
import sys
import time
import math
import threading
from aeris.config import ConfigManager
from aeris.camera import AerisCamera
from aeris.detector import AerisDroneDetector
from aeris.tracker import AerisTracker
from aeris.tracking_metrics import TrackingMetrics
from aeris.serial_link import SerialLink
from aeris.sensor_parser import SensorParser
from aeris.health_monitor import HealthMonitor
from aeris.environment_engine import EnvironmentEngine
from aeris.fusion import AerisFusionEngine
from aeris.experiment_runner import ExperimentSession, ExperimentConditions, ExperimentState, ConditionVerification

stop_event = threading.Event()

def serial_thread_func(config_mgr, fusion_engine):
    link = SerialLink(config_mgr)
    if not link.connect():
        fusion_engine.update_environment(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)
    
    parser = SensorParser(config_mgr)
    health = HealthMonitor()
    env_engine = EnvironmentEngine(config_mgr)
    
    base_temp = config_mgr.get_baseline("TEMP") or 25.0
    base_light = config_mgr.get_baseline("LIGHT") or 500
    base_x = config_mgr.get_baseline("AX") or 330
    base_y = config_mgr.get_baseline("AY") or 330
    base_z = config_mgr.get_baseline("AZ") or 330
    
    while not stop_event.is_set():
        if link.conn is None:
            time.sleep(2)
            if not link.connect():
                fusion_engine.update_environment(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)
                continue
                
        line = link.read_line()
        if line is None:
            health.update(None)
            status = health.get_health_status()
            env_res = env_engine.evaluate(None, status, None)
            if status == "COMMUNICATION LOST":
                link.close()
            fusion_engine.update_environment(None, status, env_res, None)
            continue
            
        parsed = parser.parse(line)
        if not parsed:
            continue
            
        health.update(parsed)
        status = health.get_health_status()
        
        t_dev = abs(parsed.get("TEMP", 0) - base_temp)
        l_ratio = (parsed.get("LIGHT", 1) / base_light) if base_light > 0 else 1.0
        dx = parsed.get("AX", 0) - base_x
        dy = parsed.get("AY", 0) - base_y
        dz = parsed.get("AZ", 0) - base_z
        vib_mag = math.sqrt(dx*dx + dy*dy + dz*dz) if status == "NOMINAL" else 0.0
        
        dev_data = {
            "temp_abs_delta": t_dev,
            "light_ratio": l_ratio,
            "vibration": vib_mag
        }
        
        env_res = env_engine.evaluate(parsed, status, dev_data)
        fusion_engine.update_environment(parsed, status, env_res, dev_data)
        
    link.close()

def main():
    print("==========================================")
    print("AERIS-X PHASE 8: EXPERIMENT RUNNER")
    print("==========================================")
    print("CONTROLS:")
    print(" b = BASELINE Trial")
    print(" v = HIGH_VIBRATION Trial")
    print(" l = LOW_LIGHT Trial")
    print(" r = PRECIPITATION Trial")
    print(" t = TEMPERATURE_DEVIATION Trial")
    print(" m = MULTI_CONDITION Trial")
    print(" s = STOP Trial")
    print(" q = QUIT Session Safely")
    print("==========================================")

    config_mgr = ConfigManager()
    fusion_engine = AerisFusionEngine(config_mgr)
    session = ExperimentSession(config_mgr)
    
    serial_thread = threading.Thread(target=serial_thread_func, args=(config_mgr, fusion_engine), daemon=True)
    serial_thread.start()

    cam = AerisCamera(config_mgr)
    if not cam.initialize():
        print("Camera Failed.")
        stop_event.set()
        sys.exit(1)

    det = AerisDroneDetector(config_mgr)
    det.initialize()
    
    tracker = AerisTracker(config_mgr)
    metrics = TrackingMetrics()
    
    window_name = "AERIS-X Phase 8"

    try:
        consecutive_drops = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                consecutive_drops += 1
                if consecutive_drops > 30:
                    print("Camera feed lost. Exiting.")
                    break
                time.sleep(0.03)
                continue
            
            consecutive_drops = 0
            h, w = frame.shape[:2]
            detections = det.detect(frame)
            track = tracker.update(detections, w, h)
            
            sample = metrics.update(
                track=track,
                camera_fps=cam.get_fps(),
                detector_fps=det.get_inference_fps(),
                tracking_fps=None, 
                ground_truth_center=None
            )
            
            fusion_record = fusion_engine.fuse_vision(sample)
            session.update(fusion_record)

            # Rendering
            for d in detections:
                dx1, dy1, dx2, dy2 = d["bbox"]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (100, 100, 100), 1)
            
            if track and track.status != "LOST":
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.status in ("NEW", "TRACKING") else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cx, cy = int(track.center_x), int(track.center_y)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                
            # Overlay
            cv2.putText(frame, "PHASE 8 EXPERIMENT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_next = cam.draw_overlay(frame, x=20, y=60)
            
            cv2.putText(frame, f"TRACK: {fusion_record.tracking_status}", (20, y_next), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"ENV: {fusion_record.environment_state or 'UNKNOWN'}", (20, y_next+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"REAL DRONE: NOT AVAILABLE", (20, y_next+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Experiment Runner State
            y_off = y_next + 70
            cv2.putText(frame, "--- EXPERIMENT RUNNER ---", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"ID: {session.experiment_id}", (20, y_off+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"STATE: {session.state}", (20, y_off+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if session.current_condition:
                cv2.putText(frame, f"COND: {session.current_condition} (TRIAL {session.current_trial_number})", (20, y_off+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if session.current_trial_stats:
                    cv2.putText(frame, f"VERIFICATION: {session.current_trial_stats.verification}", (20, y_off+85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                session.start_trial(ExperimentConditions.BASELINE)
            elif key == ord('v'):
                session.start_trial(ExperimentConditions.HIGH_VIBRATION)
            elif key == ord('l'):
                session.start_trial(ExperimentConditions.LOW_LIGHT)
            elif key == ord('r'):
                session.start_trial(ExperimentConditions.PRECIPITATION)
            elif key == ord('t'):
                session.start_trial(ExperimentConditions.TEMPERATURE_DEVIATION)
            elif key == ord('m'):
                session.start_trial(ExperimentConditions.MULTI_CONDITION)
            elif key == ord('s'):
                session.stop_trial()

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping Phase 8 safely...")
        stop_event.set()
        serial_thread.join(timeout=2.0)
        cam.release()
        cv2.destroyAllWindows()
        csv_path, json_path = session.stop_safely()
        print(f"Exported Phase 8 Logs:")
        print(f" - {csv_path}")
        print(f" - {json_path}")
        print("Phase 8 stopped cleanly.")

if __name__ == "__main__":
    main()
