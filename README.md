# Hand Trap Game – Real-Time Hand/Fingertip Danger Detection (Arvyax Assignment)

This project is a proof-of-concept prototype for the Arvyax internship assignment.

It uses a **live webcam feed** to:

- Track the **user’s hand / fingertip** in real time (no MediaPipe, no OpenPose)
- Spawn a **virtual prize** at random positions
- Spawn **virtual traps** (lasers and spikes) around the scene
- Compute the **distance between the fingertip and the traps**
- Classify interaction as **SAFE / WARNING / DANGER**
- Show a clear **“DANGER DANGER”** warning when the hand reaches the danger zone

On top of that, it turns the assignment into a small interactive mini-game:
> Move your hand to collect the prize, avoid the laser and spike traps, and watch the state change as you approach danger.

---

## 🎯 How it Meets the Assignment Requirements

**Objective:**

> Build a prototype that uses a camera feed to track the position of the user’s hand in real time and detect when the hand approaches a virtual object on the screen. When the hand reaches this boundary, the system should trigger a clear on-screen warning: `DANGER DANGER`.

This implementation provides:

### ✅ Real-time hand / fingertip tracking

- Uses **background subtraction (MOG2)** to detect moving regions in the frame.
- Restricts to the **lower 60% of the frame** to avoid detecting the face/head.
- Finds the **largest moving contour** and computes:
  - Its **centroid** (hand region center)
  - A **fingertip-like point** using the **convex hull** (highest hull point).
- Applies an **exponential moving average** to smooth the fingertip, giving a stable yet responsive control point.

No pose APIs (MediaPipe, OpenPose, etc.) are used. Everything is classical computer vision with OpenCV + NumPy.

### ✅ Virtual objects / virtual boundaries

Two kinds of virtual objects are drawn:

- A **prize**: a yellow circle that appears at random positions.
- **Traps**:
  - **Laser traps**: glowing horizontal beams.
  - **Spike traps**: rows of triangular spikes.

Internally, each trap is represented by a **rectangle (`x1, y1, x2, y2`)** that acts as the **virtual boundary** for distance calculations.

### ✅ Dynamic distance-based state logic

For each frame:

- Compute the hand point: smoothed fingertip (or hand center as fallback).
- Compute the **distance to each trap rectangle**.
- Take the **minimum distance** as the distance to danger.
- Classify into states:

- `SAFE` – hand is comfortably far from traps  
- `WARNING` – hand is approaching traps (`distance ≤ WARNING_DISTANCE_PX`)  
- `DANGER` – hand is extremely close or inside a trap (`distance ≤ DANGER_DISTANCE_PX`)

### ✅ Visual state feedback overlay

The live camera feed is augmented with:

- `State: SAFE / WARNING / DANGER` text in the top-left, with color:
  - Green = SAFE
  - Yellow = WARNING
  - Red = DANGER
- Score display for collected prizes.
- A clear **`DANGER DANGER`** overlay:
  - Red-tinted screen
  - Large “DANGER DANGER” text at the center when in DANGER state.

### ✅ Real-time performance (CPU-only)

- Uses lightweight operations: background subtraction, contour/hull, geometry, simple drawing.
- Runs on **CPU only** with a **640×480 webcam feed**.
- No neural nets, no cloud API calls.

### ✅ Libraries

- **Used**: `OpenCV`, `NumPy`, `random`, `time`

---

## 🧩 Game Mechanics

- A **prize** (yellow circle) appears every few seconds.
- After a short delay, **traps** spawn:
  - Each trap is randomly chosen as either:
    - A **laser beam** (glowing red line)
    - A **spike field** (row of spikes)
- You control the **fingertip cursor** with your hand motion:
  - Move your hand in the **lower half of the frame**.
  - The fingertip visual (small yellow dot + white ring) follows your motions.
- If the fingertip:
  - **Touches the prize** → score increases, the scene resets with a new prize later.
  - **Gets close to a trap** → state changes to WARNING.
  - **Touches/enters a trap** → state becomes DANGER and “DANGER DANGER” appears.

