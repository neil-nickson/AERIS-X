import cv2
import numpy as np
import time

class AerisDashboard:
    def __init__(self, width=1280, height=800):
        self.width = width
        self.height = height
        self.bg_color = (15, 15, 15)
        self.text_color = (0, 255, 0)
        self.warning_color = (0, 165, 255)
        self.error_color = (0, 0, 255)
        self.header_color = (255, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas[:] = self.bg_color

    def draw_text(self, img, text, x, y, color=None, scale=0.5, thickness=1):
        c = color if color else self.text_color
        cv2.putText(img, text, (x, y), self.font, scale, c, thickness, cv2.LINE_AA)

    def draw_panel(self, x, y, w, h, title):
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), (50, 50, 50), 2)
        cv2.rectangle(self.canvas, (x, y), (x+w, y+30), (50, 50, 50), -1)
        self.draw_text(self.canvas, title, x+10, y+20, color=self.header_color, scale=0.6, thickness=2)

    def render(self, state_data):
        self.canvas[:] = self.bg_color
        
        # Header
        self.draw_text(self.canvas, "AERIS-X | Adaptive Environmental Resilience & Intelligent Sensing", 20, 30, color=self.header_color, scale=0.7, thickness=2)
        
        mode_text = state_data.get('mode', 'LIVE MODE')
        mode_str = f"MODE: {mode_text}"
        (mode_w, mode_h), _ = cv2.getTextSize(mode_str, self.font, 0.6, 2)
        self.draw_text(self.canvas, mode_str, self.width - mode_w - 20, 25, color=self.warning_color, scale=0.6, thickness=2)
        
        if 'scenario' in state_data:
            scenario_str = f"SCENARIO: {state_data['scenario']}"
            scale = 0.6
            (scen_w, scen_h), _ = cv2.getTextSize(scenario_str, self.font, scale, 2)
            if scen_w > 500:
                scale = 0.45
                (scen_w, scen_h), _ = cv2.getTextSize(scenario_str, self.font, scale, 2)
            self.draw_text(self.canvas, scenario_str, self.width - scen_w - 20, 50, color=self.header_color, scale=scale, thickness=2)
        
        # 1. Environment Panel (Top Left)
        env = state_data.get('environment', {})
        self.draw_panel(20, 60, 300, 250, "ENVIRONMENT")
        
        def safe_fmt(val, fmt="{:.2f}", default="UNKNOWN"):
            if val is None: return default
            return fmt.format(val)
        
        y = 110
        self.draw_text(self.canvas, f"Temperature: {safe_fmt(env.get('TEMP'))}", 30, y); y += 25
        self.draw_text(self.canvas, f"Humidity: {safe_fmt(env.get('HUM'))}", 30, y); y += 25
        self.draw_text(self.canvas, f"Light: {safe_fmt(env.get('LIGHT'))}", 30, y); y += 25
        self.draw_text(self.canvas, f"Vibration: {safe_fmt(env.get('vibration_index', env.get('vibration')))}", 30, y); y += 25
        self.draw_text(self.canvas, f"Rain: {'YES' if env.get('RAIN_DETECTED') else 'NO'}", 30, y); y += 25
        
        e_state = env.get("state", "UNKNOWN")
        color_e = self.text_color if e_state == "NORMAL" else self.warning_color
        if e_state == "UNKNOWN": color_e = self.error_color
        self.draw_text(self.canvas, f"State: {e_state}", 30, y+10, color=color_e, scale=0.6, thickness=2)

        # 2. Performance Panel (Mid Left)
        perf = state_data.get('performance', {})
        metrics = state_data.get('metrics', {})
        self.draw_panel(20, 330, 300, 220, "TRACKING PERFORMANCE")
        
        y = 380
        self.draw_text(self.canvas, f"Track Status: {metrics.get('status', 'WAITING')}", 30, y); y += 25
        self.draw_text(self.canvas, f"Track ID: {metrics.get('track_id', 'N/A')}", 30, y); y += 25
        self.draw_text(self.canvas, f"Mean Jitter: {safe_fmt(metrics.get('mean_jitter'), default='N/A')}", 30, y); y += 25
        self.draw_text(self.canvas, f"Mean Displacement: {safe_fmt(metrics.get('mean_displacement'), default='N/A')}", 30, y); y += 25
        self.draw_text(self.canvas, f"Tracking Ratio: {safe_fmt(metrics.get('tracking_ratio'), default='N/A')}", 30, y); y += 25
        self.draw_text(self.canvas, f"Lost Events: {metrics.get('lost_events', 'N/A')}", 30, y); y += 25

        # 3. Degradation Panel (Bottom Left)
        self.draw_panel(20, 570, 300, 160, "DEGRADATION ASSESSMENT")
        y = 620
        p_state = perf.get('performance_state', 'UNKNOWN')
        d_state = state_data.get('degradation', {}).get('degradation_state', 'UNKNOWN')
        rc = perf.get('reason_codes', [])
        
        self.draw_text(self.canvas, f"PERF STATE: {p_state}", 30, y, color=self.text_color if p_state == "NORMAL" else self.warning_color, thickness=2); y += 25
        self.draw_text(self.canvas, f"DEG STATE: {d_state}", 30, y, color=self.text_color if d_state == "NORMAL" else self.warning_color, thickness=2); y += 30
        
        self.draw_text(self.canvas, "REASON CODES:", 30, y, color=self.header_color); y += 20
        rc_str = ", ".join(rc) if rc else "NONE"
        
        # Simple word wrap for reason codes
        if len(rc_str) > 35:
            self.draw_text(self.canvas, rc_str[:35], 30, y); y += 20
            self.draw_text(self.canvas, rc_str[35:], 30, y)
        else:
            self.draw_text(self.canvas, rc_str, 30, y)

        # 4. Camera & Target Embed (Top Right)
        # Assuming camera frame is provided as a numpy array, we resize and place it.
        # Max area: W: 900, H: 500
        cam_x, cam_y = 340, 60
        self.draw_panel(cam_x, cam_y, 900, 520, "CAMERA / TARGET")
        
        frame = state_data.get('camera_frame', None)
        if frame is not None:
            # Resize frame to fit in 880 x 480
            frame_resized = cv2.resize(frame, (880, 480))
            
            # Draw tracking bbox on the resized frame
            cx, cy = metrics.get("center_x"), metrics.get("center_y")
            status = metrics.get("status", "WAITING")
            conf = metrics.get("confidence")
            
            if status != "LOST" and cx is not None and cy is not None:
                # scale bbox
                h_orig, w_orig = frame.shape[:2]
                scale_x, scale_y = 880 / w_orig, 480 / h_orig
                
                bx1 = int(metrics.get("bbox_x1", 0) * scale_x)
                by1 = int(metrics.get("bbox_y1", 0) * scale_y)
                bx2 = int(metrics.get("bbox_x2", 0) * scale_x)
                by2 = int(metrics.get("bbox_y2", 0) * scale_y)
                
                color = (0, 255, 0) if status in ("NEW", "TRACKING") else (0, 165, 255)
                cv2.rectangle(frame_resized, (bx1, by1), (bx2, by2), color, 2)
                
                ccx, ccy = int(cx * scale_x), int(cy * scale_y)
                cv2.circle(frame_resized, (ccx, ccy), 4, color, -1)
                
                cv2.putText(frame_resized, f"DETECTION CONFIDENCE: {conf:.2f}", (bx1, max(20, by1 - 10)), self.font, 0.5, color, 1, cv2.LINE_AA)
            elif status == "WAITING" or status == "LOST":
                cv2.putText(frame_resized, "TARGET: NONE", (20, 30), self.font, 0.7, self.error_color, 2, cv2.LINE_AA)
                
            self.canvas[cam_y+35:cam_y+35+480, cam_x+10:cam_x+10+880] = frame_resized
        else:
            mode_text = state_data.get('mode', '').upper()
            cam_src = state_data.get('camera_source', '').upper()
            
            if "SIMULAT" in cam_src or "SIMULAT" in mode_text:
                cam_cx, cam_cy = cam_x + 440, cam_y + 200
                cv2.rectangle(self.canvas, (cam_cx - 80, cam_cy - 60), (cam_cx + 80, cam_cy + 60), (50, 50, 50), 2)
                self.draw_text(self.canvas, "[ SIMULATED TARGET ]", cam_cx - 75, cam_cy - 70, color=self.header_color, scale=0.5, thickness=1)
                self.draw_text(self.canvas, "DEMO", cam_cx - 25, cam_cy - 10, color=self.text_color, scale=0.6, thickness=2)
                self.draw_text(self.canvas, "TARGET", cam_cx - 35, cam_cy + 20, color=self.text_color, scale=0.6, thickness=2)
                
                info_y = cam_y + 350
                conf = metrics.get('confidence', 0.80)
                conf_val = conf if conf is not None else 0.0
                
                self.draw_text(self.canvas, "SOURCE: SIMULATION", cam_x + 30, info_y, color=self.text_color, scale=0.6, thickness=1)
                self.draw_text(self.canvas, "STATUS: SIMULATED", cam_x + 30, info_y + 25, color=self.text_color, scale=0.6, thickness=1)
                self.draw_text(self.canvas, f"TRACK ID: {metrics.get('track_id', '1')}", cam_x + 30, info_y + 50, color=self.text_color, scale=0.6, thickness=1)
                self.draw_text(self.canvas, f"TRACK STATUS: {metrics.get('status', 'TRACKING')}", cam_x + 30, info_y + 75, color=self.text_color, scale=0.6, thickness=1)
                self.draw_text(self.canvas, f"DETECTION CONFIDENCE: {conf_val:.2f}", cam_x + 30, info_y + 100, color=self.text_color, scale=0.6, thickness=1)
                self.draw_text(self.canvas, f"TARGET CENTER: ({metrics.get('center_x', 640)}, {metrics.get('center_y', 360)})", cam_x + 30, info_y + 125, color=self.text_color, scale=0.6, thickness=1)
                
            elif "REPLAY" in cam_src or "REPLAY" in mode_text:
                self.draw_text(self.canvas, "SOURCE: REPLAY DATA", cam_x+300, cam_y+230, color=self.warning_color, scale=0.7, thickness=2)
                self.draw_text(self.canvas, "STATUS: REPLAY", cam_x+300, cam_y+260, color=self.warning_color, scale=0.7, thickness=2)
            else:
                self.draw_text(self.canvas, "CAMERA: DISCONNECTED / UNAVAILABLE", cam_x+250, cam_y+250, color=self.error_color, scale=0.7, thickness=2)

        # 5. Sensor / Fusion Health (Mid Right 1)
        fusion = state_data.get('fusion', {})
        self.draw_panel(340, 590, 440, 140, "SENSOR / FUSION HEALTH")
        y = 640
        
        q_state = fusion.get('status', 'UNKNOWN')
        color_q = self.text_color if q_state == "SYNCHRONIZED" else self.error_color
        
        self.draw_text(self.canvas, f"Sensor Health: {env.get('health', 'UNKNOWN')}", 350, y); y += 25
        self.draw_text(self.canvas, f"Fusion Quality: {q_state}", 350, y, color=color_q); y += 25
        self.draw_text(self.canvas, f"Timestamp Delta: {safe_fmt(fusion.get('delta_ms'))} ms", 350, y); y += 25
        self.draw_text(self.canvas, f"Camera Source: {state_data.get('camera_source', 'MAC CAMERA')}", 350, y)

        # 6. Adaptive Control (Mid Right 2)
        adapt = state_data.get('adaptation', {})
        self.draw_panel(800, 590, 440, 140, "ADAPTIVE CONTROL")
        y = 640
        self.draw_text(self.canvas, f"State: {adapt.get('state', 'INACTIVE')}", 810, y); y += 25
        self.draw_text(self.canvas, f"Strategy: {adapt.get('strategy', 'NONE')}", 810, y); y += 25
        self.draw_text(self.canvas, f"Validation: {adapt.get('status', 'PENDING')}", 810, y); y += 25
        
        evid = adapt.get('evidence', 'INSUFFICIENT')
        self.draw_text(self.canvas, f"Evidence: {evid}", 810, y, color=self.text_color if evid == "SUFFICIENT" else self.warning_color)

        # 7. System Pipeline Visualization (Bottom Full)
        self.draw_panel(20, 740, 1220, 50, "SYSTEM PIPELINE")
        pipeline = ["SENSE", "DETECT", "TRACK", "MEASURE", "FUSE", "EXPERIMENT", "ANALYZE", "ADAPT", "ASSESS"]
        px = 40
        for i, step in enumerate(pipeline):
            color = self.text_color
            if step == "ADAPT" and adapt.get('state') == "INACTIVE": color = (100, 100, 100)
            self.draw_text(self.canvas, step, px, 775, color=color, scale=0.5, thickness=2)
            if i < len(pipeline) - 1:
                self.draw_text(self.canvas, "->", px + 95, 775, color=self.header_color)
            px += 130
            
        return self.canvas
