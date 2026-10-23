import cv2
import sys
from aeris.config import ConfigManager
from aeris.camera import AerisCamera

def main():
    print("==========================================")
    print("AERIS-X PHASE 3 CAMERA FOUNDATION")
    print("==========================================")

    config_mgr = ConfigManager()
    cfg = config_mgr.config.get("camera", {})
    
    print(f"Camera index : {cfg.get('index', 0)}")
    print(f"Resolution   : {cfg.get('width', 1280)}x{cfg.get('height', 720)}")
    print(f"Target FPS   : {cfg.get('target_fps', 30)}")
    print(f"Recording    : {'ENABLED' if cfg.get('recording_enabled') else 'DISABLED'}")
    print("------------------------------------------")

    cam = AerisCamera(config_mgr)
    success = cam.initialize()

    if not success:
        print(f"Camera: {cam.get_health()}")
        print(f"Unable to open camera index {cfg.get('index', 0)}.")
        print("Phase 3 stopped cleanly.")
        sys.exit(1)

    print("Camera: CONNECTED")
    print("Press 'q' or 'Ctrl+C' to stop.")
    
    display_enabled = cfg.get("display", True)
    window_name = "AERIS-X Preview"

    try:
        while True:
            ret, frame = cam.read()
            
            if not ret:
                print(f"\n[!] Camera: {cam.get_health()}")
                break

            if display_enabled:
                # Get actual resolution
                h, w = frame.shape[:2]
                
                # Overlay diagnostics
                fps = cam.get_fps()
                health = cam.get_health()
                
                cv2.putText(frame, "AERIS-X PHASE 3", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cam.draw_overlay(frame, x=20, y=70)

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    finally:
        print("Stopping Phase 3...")
        cam.release()
        cv2.destroyAllWindows()
        print("Camera released.")
        print("Phase 3 stopped cleanly.")

if __name__ == "__main__":
    main()
