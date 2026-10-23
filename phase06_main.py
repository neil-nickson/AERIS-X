# pyrefly: ignore [missing-import]
import cv2
import sys
from aeris.config import ConfigManager
from aeris.camera import AerisCamera
from aeris.detector import AerisDroneDetector
from aeris.tracker import AerisTracker
from aeris.tracking_metrics import TrackingMetrics

def main():
    print("==========================================")
    print("AERIS-X PHASE 6 PERFORMANCE MEASUREMENT")
    print("==========================================")

    config_mgr = ConfigManager()
    
    cam = AerisCamera(config_mgr)
    if not cam.initialize():
        print("Camera Failed.")
        sys.exit(1)

    det = AerisDroneDetector(config_mgr)
    det.initialize()
    
    tracker = AerisTracker(config_mgr)
    metrics = TrackingMetrics()
    
    window_name = "AERIS-X Phase 6"

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            detections = det.detect(frame)
            track = tracker.update(detections, w, h)
            
            # Record metrics
            # Note: tracking_fps can be added via time differences if needed, using det.get_inference_fps() for now
            sample = metrics.update(
                track=track,
                camera_fps=cam.get_fps(),
                detector_fps=det.get_inference_fps(),
                tracking_fps=None, 
                ground_truth_center=None
            )

            # Vis (same as Phase 5, + Phase 6 Dashboard)
            for d in detections:
                dx1, dy1, dx2, dy2 = d["bbox"]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (100, 100, 100), 1)
            
            if track and track.status != "LOST":
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.status in ("NEW", "TRACKING") else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cx, cy = int(track.center_x), int(track.center_y)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                
            # Phase 6 Dashboard
            cv2.putText(frame, "PHASE 6 METRICS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_next = cam.draw_overlay(frame, x=20, y=60)
            
            cv2.putText(frame, f"DET FPS: {sample.detector_fps or 0:.1f}", (20, y_next), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"TRACK STATUS: {sample.status} (ID: {sample.track_id})", (20, y_next+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            summary = metrics.get_summary()
            if summary:
                cv2.putText(frame, f"TRACKED: {summary.tracked_frames} | MISSED: {summary.missed_frames} | LOST: {summary.lost_events}", (20, y_next+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"RATIO: {summary.tracking_ratio:.2f}", (20, y_next+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                disp_str = f"{summary.mean_displacement:.1f}px" if summary.mean_displacement else "N/A"
                jit_str = f"{summary.mean_jitter:.1f}px" if summary.mean_jitter else "N/A"
                cv2.putText(frame, f"MEAN DISP: {disp_str} | MEAN JITTER: {jit_str}", (20, y_next+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "GROUND TRUTH: NOT AVAILABLE", (20, y_next+125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "POSITION ERROR: N/A", (20, y_next+150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping Phase 6...")
        cam.release()
        cv2.destroyAllWindows()
        csv_path, json_path = metrics.export_logs()
        print(f"Exported Phase 6 Logs:")
        print(f" - {csv_path}")
        print(f" - {json_path}")
        print("Phase 6 stopped cleanly.")

if __name__ == "__main__":
    main()
