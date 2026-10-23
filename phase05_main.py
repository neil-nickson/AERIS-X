# pyrefly: ignore [missing-import]
import cv2
import sys
from aeris.config import ConfigManager
from aeris.camera import AerisCamera
from aeris.detector import AerisDroneDetector
from aeris.tracker import AerisTracker

def main():
    print("==========================================")
    print("AERIS-X PHASE 5 VISUAL TRACKING")
    print("==========================================")

    config_mgr = ConfigManager()
    
    # Initialize Camera
    cam = AerisCamera(config_mgr)
    if not cam.initialize():
        print(f"Camera: {cam.get_health()}")
        print("Phase 5 stopped cleanly (Camera Failed).")
        sys.exit(1)

    # Initialize Detector
    det = AerisDroneDetector(config_mgr)
    det.initialize()
    
    # Initialize Tracker
    tracker = AerisTracker(config_mgr)
    
    print("\nPhase 5 Initialization:")
    print(f"Camera   : {cam.get_health()}")
    print(f"Detector : {det.get_health()}")
    print("Tracker  : TRACKER_NOMINAL")
    print("------------------------------------------")
    print("Press 'q' or 'Ctrl+C' to stop.")
    
    window_name = "AERIS-X Phase 5"

    try:
        while True:
            # 1. Read Frame
            ret, frame = cam.read()
            if not ret:
                print(f"\n[!] Camera: {cam.get_health()}")
                break

            h, w = frame.shape[:2]

            # 2. Run Inference
            detections = det.detect(frame)
            
            # 3. Update Tracker
            track = tracker.update(detections, w, h)
            
            # 4. Visualization
            # First, optionally draw raw detections in dull color to show what the detector saw
            for d in detections:
                dx1, dy1, dx2, dy2 = d["bbox"]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (100, 100, 100), 1)
            
            if track:
                if track.status != "LOST":
                    x1, y1, x2, y2 = track.bbox
                    
                    # Colors based on status
                    color = (0, 255, 0) if track.status in ("NEW", "TRACKING") else (0, 165, 255)
                    
                    # Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"ID: {track.track_id} | {track.status} | Conf: {track.confidence:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Center Dot
                    cx, cy = int(track.center_x), int(track.center_y)
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    
                    # Trajectory
                    if len(track.trajectory) > 1:
                        for i in range(1, len(track.trajectory)):
                            pt1 = (int(track.trajectory[i-1][0]), int(track.trajectory[i-1][1]))
                            pt2 = (int(track.trajectory[i][0]), int(track.trajectory[i][1]))
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                            
                    if track.status == "MISSED":
                        cv2.putText(frame, f"MISSED: {track.missed_frames}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Minimal UI Overlay
            cam_fps = cam.get_fps()
            inf_fps = det.get_inference_fps()
            track_status = track.status if track else "WAITING"
            
            cv2.putText(frame, "AERIS-X PHASE 5", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_next = cam.draw_overlay(frame, x=20, y=60)
            cv2.putText(frame, f"Detector: {det.get_health()} | Inf FPS: {inf_fps:.1f}", (20, y_next), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Track Status: {track_status}", (20, y_next+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    finally:
        print("Stopping Phase 5...")
        cam.release()
        cv2.destroyAllWindows()
        print("Phase 5 stopped cleanly.")

if __name__ == "__main__":
    main()
