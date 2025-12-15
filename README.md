# Biceps Curl Counter

Computer-vision biceps curl counter using MediaPipe BlazePose and OpenCV. It tracks left and right arms independently, only counting reps that complete a full range of motion and providing on-screen feedback to correct form.

## What’s in the repo
- `demo.py`: runnable script with pose detection, arm state machine, HUD/progress bars, and keyboard controls.
- `requirements.txt`: Python dependencies (MediaPipe, OpenCV, NumPy, Matplotlib, etc.).

## Quick start
1) Install Python 3.9+ and ensure a webcam is available.  
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Run the tracker:
```
python demo.py
```
4) Controls: `r` resets both counters; `Esc` quits.

## How it works
- Pose estimation: MediaPipe BlazePose finds body landmarks each frame.
- Angle and smoothing: elbow angles (shoulder–elbow–wrist) are smoothed with an EMA to avoid jitter.
- State machine: each arm transitions EXTENDED → FLEXING → FLEXED → EXTENDING and increments the counter only after a full extend–flex–extend cycle with sufficient angular velocity.
- Feedback UI: progress bars reflect curl completion per arm; HUD shows left/right rep counts and pose skeleton overlay.

## Troubleshooting
- If the window is black, confirm the webcam is free and `cv2.VideoCapture(0)` opens; adjust the index if you have multiple cameras.
- If reps do not count, ensure a full extension/flexion (about 160° to 50°) and keep the elbow visible to the camera.
- On Apple Silicon or systems without GPU support, performance may improve by reducing camera resolution in `demo.py`.

## References and credits
1. [Guide to Human Pose Estimation with Deep Learning (Nanonets)](https://nanonets.com/blog/human-pose-estimation-2d-guide/)
2. [Mediapipe Pose Classification (Google)](https://google.github.io/mediapipe/solutions/pose_classification.html)
3. [Real-time Human Pose Estimation in the Browser (TF Blog)](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
4. [MediaPipePoseEstimation (Nicknochnack)](https://github.com/nicknochnack/MediaPipePoseEstimation)
