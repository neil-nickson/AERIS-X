import cv2
import sys
from aeris.config import ConfigManager
from aeris.camera import AerisCamera
from aeris.detector import AerisDroneDetector

def main():
    print("==========================================")
    print("AERIS-X PHASE 4 DRONE DETECTION")
    print("==========================================")

    config_mgr = ConfigManager()
    
    # Initialize Camera
    cam = AerisCamera(config_mgr)
    cam_success = cam.initialize()

    if not cam_success:
        print(f"Camera: {cam.get_health()}")
        print("Phase 4 stopped cleanly (Camera Failed).")
        sys.exit(1)

    # Initialize Detector
    det = AerisDroneDetector(config_mgr)
    det_success = det.initialize()
    
    print("\nPhase 4 Initialization:")
    print(f"Camera   : {cam.get_health()}")
    print(f"Detector : {det.get_health()}")
    print("------------------------------------------")
    print("Press 'q' or 'Ctrl+C' to stop.")
    
    window_name = "AERIS-X Phase 4"

    try:
        while True:
            # 1. Read Frame
            ret, frame = cam.read()
            if not ret:
                print(f"\n[!] Camera: {cam.get_health()}")
                break

            # 2. Run Inference
            detections = det.detect(frame)
            
            # 3. Visualization
            h, w = frame.shape[:2]
            
            # Draw Detections
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                conf = d["confidence"]
                
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Label
                label = f"DRONE Conf: {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Minimal UI Overlay
            cam_fps = cam.get_fps()
            cam_health = cam.get_health()
            det_health = det.get_health()
            inf_fps = det.get_inference_fps()
            
            cv2.putText(frame, "AERIS-X PHASE 4", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_next = cam.draw_overlay(frame, x=20, y=60)
            cv2.putText(frame, f"Detector: {det_health} | Inf FPS: {inf_fps:.1f}", (20, y_next), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Drone Detections: {len(detections)}", (20, y_next+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    finally:
        print("Stopping Phase 4...")
        cam.release()
        cv2.destroyAllWindows()
        print("Camera released.")
        print("Phase 4 stopped cleanly.")

if __name__ == "__main__":
    main()
