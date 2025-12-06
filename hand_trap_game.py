import cv2
import numpy as np
import random
import time
MIN_HAND_AREA = 1500 
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
PRIZE_RADIUS = 30
PRIZE_INTERVAL = 5.0       
TRAP_DELAY = 1.2           
NUM_TRAPS = 3
TRAP_SIZE = (120, 80)      
WARNING_DISTANCE_PX = 120  
DANGER_DISTANCE_PX = 60
SMOOTHING_ALPHA = 0.35
def detect_hand_center(frame):
    h, w = frame.shape[:2]
    fgmask = bg_subtractor.apply(frame)
    fgmask = cv2.GaussianBlur(fgmask, (7, 7), 0)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, None, iterations=1)
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    cut = int(h * 0.4)  
    hand_mask = np.zeros_like(fgmask)
    hand_mask[cut:h, :] = fgmask[cut:h, :]
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, hand_mask
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_HAND_AREA:
        return None, hand_mask
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, hand_mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), hand_mask
def shortest_distance_point_to_rect(px, py, x1, y1, x2, y2):
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0.0
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5
def classify_state(distance_px):
    if distance_px is None:
        return "SAFE", (0, 255, 0)
    if distance_px <= DANGER_DISTANCE_PX:
        return "DANGER", (0, 0, 255)
    elif distance_px <= WARNING_DISTANCE_PX:
        return "WARNING", (0, 255, 255)
    else:
        return "SAFE", (0, 255, 0)
def spawn_random_prize(frame_w, frame_h):
    margin = 60
    y_min = int(frame_h * 0.25)
    y_max = int(frame_h * 0.9)
    cx = random.randint(margin, frame_w - margin)
    cy = random.randint(y_min, y_max)
    return (cx, cy)
def spawn_random_traps(frame_w, frame_h, num_traps, prize_center):
    traps = []
    trap_w, trap_h = TRAP_SIZE
    px, py = prize_center
    for _ in range(num_traps):
        for _ in range(20):  
            x1 = random.randint(0, frame_w - trap_w)
            y1 = random.randint(int(frame_h * 0.35), frame_h - trap_h)
            x2 = x1 + trap_w
            y2 = y1 + trap_h
            dist_to_prize = shortest_distance_point_to_rect(px, py, x1, y1, x2, y2)
            if dist_to_prize > PRIZE_RADIUS + 40:
                trap_type = random.choice(["laser", "spike"])
                traps.append({
                    "rect": (x1, y1, x2, y2),
                    "type": trap_type
                })
                break
    return traps
def draw_laser_trap(frame, rect, time_now):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    thickness = 4
    glow_thickness = 16
    flicker = 0.5 + 0.5 * abs(np.sin(time_now * 8))
    overlay = frame.copy()
    cv2.line(overlay, (x1, cy), (x2, cy), (0, 0, 255), glow_thickness)
    frame[:] = cv2.addWeighted(overlay, 0.3 * flicker, frame, 1 - 0.3 * flicker, 0)
    cv2.line(frame, (x1, cy), (x2, cy), (0, 0, 255), thickness)
    cv2.line(frame, (x1, cy - 1), (x2, cy - 1), (255, 255, 255), 1)
    cv2.line(frame, (x1, cy + 1), (x2, cy + 1), (255, 255, 255), 1)
def draw_spike_trap(frame, rect):
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    num_spikes = max(3, width // 40)
    spike_width = width / num_spikes
    for i in range(num_spikes):
        sx1 = int(x1 + i * spike_width)
        sx2 = int(x1 + (i + 1) * spike_width)
        mid = (sx1 + sx2) // 2
        pts = np.array([
            [sx1, y2],
            [sx2, y2],
            [mid, y1]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 200))
        highlight = np.array([
            [mid, y1 + height // 4],
            [sx1 + (sx2 - sx1)//3, y2 - height // 4],
            [sx2 - (sx2 - sx1)//3, y2 - height // 4]], np.int32)
        cv2.polylines(frame, [highlight], True, (255, 255, 255), 1)
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prize_center = None
    prize_spawn_time = None
    traps = []
    traps_spawned_for_this_prize = False
    last_prize_end_time = time.time()
    score = 0
    smoothed_hand_point = None  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        hand_center, hand_mask = detect_hand_center(frame)
        raw_point = None  
        if hand_center is not None:
            raw_point = (hand_center[0], hand_center[1])
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest)
                if hull is not None and len(hull) > 0:
                    fingertip = tuple(hull[hull[:, :, 1].argmin()][0])
                    raw_point = fingertip  
                    fx, fy = fingertip
                    cv2.circle(frame, (fx, fy), 10, (0, 255, 255), -1)
                    cv2.putText(frame, "Finger", (fx + 8, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.circle(frame, (hand_center[0], hand_center[1]), 6, (255, 0, 0), -1)
        else:
            cv2.putText(frame, "Move your hand in the lower half", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if raw_point is not None:
            rx, ry = raw_point
            if smoothed_hand_point is None:
                smoothed_hand_point = np.array([float(rx), float(ry)], dtype=float)
            else:
                smoothed_hand_point = ((1.0 - SMOOTHING_ALPHA) * smoothed_hand_point + SMOOTHING_ALPHA * np.array([float(rx), float(ry)], dtype=float))
        else:
            smoothed_hand_point = None
        logic_hand_point = None
        if smoothed_hand_point is not None:
            hx, hy = int(smoothed_hand_point[0]), int(smoothed_hand_point[1])
            logic_hand_point = (hx, hy)
            cv2.circle(frame, (hx, hy), 12, (255, 255, 255), 2)
        if prize_center is None:
            if now - last_prize_end_time > PRIZE_INTERVAL:
                prize_center = spawn_random_prize(w, h)
                prize_spawn_time = now
                traps = []
                traps_spawned_for_this_prize = False
        else:
            if not traps_spawned_for_this_prize and (now - prize_spawn_time) > TRAP_DELAY:
                traps = spawn_random_traps(w, h, NUM_TRAPS, prize_center)
                traps_spawned_for_this_prize = True
        nearest_trap_distance = None
        if logic_hand_point is not None and traps:
            hx, hy = logic_hand_point
            dists = [shortest_distance_point_to_rect(hx, hy, *trap["rect"]) for trap in traps]
            nearest_trap_distance = min(dists)
        state, state_color = classify_state(nearest_trap_distance)
        if logic_hand_point is not None and prize_center is not None:
            hx, hy = logic_hand_point
            px, py = prize_center
            dist_to_prize = ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5
            if dist_to_prize <= PRIZE_RADIUS:
                score += 1
                prize_center = None
                traps = []
                traps_spawned_for_this_prize = False
                last_prize_end_time = now
        if prize_center is not None:
            px, py = prize_center
            cv2.circle(frame, (px, py), PRIZE_RADIUS, (0, 215, 255), -1)
            cv2.circle(frame, (px, py), PRIZE_RADIUS + 2, (0, 140, 255), 2)
        for trap in traps:
            rect = trap["rect"]
            ttype = trap["type"]
            if ttype == "laser":
                draw_laser_trap(frame, rect, now)
            else: 
                draw_spike_trap(frame, rect)
        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
        cv2.putText(frame, f"Score: {score}", (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Collect the prize. Avoid lasers & spikes!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if state == "DANGER":
            overlay = frame.copy()
            overlay[:] = (0, 0, 120)  
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv2.putText(frame, "DANGER DANGER", (60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        cv2.imshow("Hand Trap Game", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()