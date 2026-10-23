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

stop_event = threading.Event()

def serial_thread_func(config_mgr, fusion_engine):
    print("[Serial Thread] Starting...")
    link = SerialLink(config_mgr)
    if not link.connect():
        print("[Serial Thread] Failed to connect to Arduino. Will mark COMM LOST.")
        fusion_engine.update_environment(None, "COMMUNICATION LOST", {"state": "UNKNOWN", "active_conditions": []}, None)
    
    parser = SensorParser(config_mgr)
    health = HealthMonitor()
    env_engine = EnvironmentEngine(config_mgr)
    
    base_temp = config_mgr.get_baseline("TEMP")
    base_light = config_mgr.get_baseline("LIGHT")
    base_x = config_mgr.get_baseline("AX")
    base_y = config_mgr.get_baseline("AY")
    base_z = config_mgr.get_baseline("AZ")
    
    if not base_temp:
        print("[!] No baseline calibration found. Values may be inaccurate.")
        base_temp = 25.0
        base_light = 500
        base_x = base_y = base_z = 330
    
    while not stop_event.is_set():
        if link.conn is None:
            # Retry connection slowly
            time.sleep(2)
            if link.connect():
                print("[Serial Thread] Reconnected.")
            else:
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
    print("[Serial Thread] Stopped.")

def main():
    print("==========================================")
    print("AERIS-X PHASE 7: TIMESTAMPED DATA FUSION")
    print("==========================================")

    config_mgr = ConfigManager()
    
    fusion_engine = AerisFusionEngine(config_mgr)
    
    # Start serial thread
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
    
    window_name = "AERIS-X Phase 7 Fusion"

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

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
            
            # FUSE
            fusion_record = fusion_engine.fuse_vision(sample)

            # Draw Vision
            for d in detections:
                dx1, dy1, dx2, dy2 = d["bbox"]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (100, 100, 100), 1)
            
            if track and track.status != "LOST":
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.status in ("NEW", "TRACKING") else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cx, cy = int(track.center_x), int(track.center_y)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                
            # Draw UI
            cv2.putText(frame, "AERIS-X PHASE 7 FUSION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_next = cam.draw_overlay(frame, x=20, y=60)
            
            cv2.putText(frame, f"DET FPS: {sample.detector_fps or 0:.1f}", (20, y_next), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"TRACK STATUS: {fusion_record.tracking_status} (ID: {fusion_record.track_id})", (20, y_next+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            env_st = fusion_record.environment_state or "UNKNOWN"
            cond = fusion_record.active_conditions or "NONE"
            cv2.putText(frame, f"ENVIRONMENT STATE: {env_st}", (20, y_next+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(frame, f"ACTIVE CONDITIONS: {cond}", (20, y_next+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            temp_str = f"{fusion_record.temperature:.1f}" if fusion_record.temperature is not None else "N/A"
            light_str = f"{fusion_record.light:.1f}" if fusion_record.light is not None else "N/A"
            vib_str = f"{fusion_record.vibration_index:.1f}" if fusion_record.vibration_index is not None else "N/A"
            rain_str = "YES" if fusion_record.rain else "NO"
            
            cv2.putText(frame, f"T:{temp_str} | L:{light_str} | V:{vib_str} | R:{rain_str}", (20, y_next+115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            q_color = (0, 255, 0) if fusion_record.data_quality == "SYNCHRONIZED" else (0, 0, 255)
            cv2.putText(frame, f"FUSION: {fusion_record.data_quality}", (20, y_next+155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 2)
            
            delta_str = f"{fusion_record.timestamp_delta_ms:.1f}ms" if fusion_record.timestamp_delta_ms is not None else "N/A"
            cv2.putText(frame, f"DELTA: {delta_str}", (20, y_next+180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping Phase 7...")
        stop_event.set()
        serial_thread.join(timeout=2.0)
        cam.release()
        cv2.destroyAllWindows()
        csv_path, json_path = fusion_engine.export_logs()
        print(f"Exported Phase 7 Logs:")
        print(f" - {csv_path}")
        print(f" - {json_path}")
        print("Phase 7 stopped cleanly.")

if __name__ == "__main__":
    main()
