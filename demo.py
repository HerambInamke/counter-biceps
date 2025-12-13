import cv2
import numpy as np
import mediapipe as mp
import time


# -------------------- Math --------------------

def calc_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ab = a - b
    bc = c - b

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    if norm_ab == 0 or norm_bc == 0:
        return 0.0

    cos_angle = np.dot(ab, bc) / (norm_ab * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def angle_to_progress(angle, min_angle=50, max_angle=160):
    return 1.0 - np.clip(
        (angle - min_angle) / (max_angle - min_angle),
        0.0,
        1.0
    )


# -------------------- UI --------------------

def draw_hud(image, left_count, right_count, width):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    cv2.putText(image, "BICEP CURL TRACKER",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (220, 220, 220), 2)

    cv2.putText(image, f"LEFT  {left_count}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 255, 120), 3)

    cv2.putText(image, f"RIGHT  {right_count}",
                (260, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 180, 255), 3)


def draw_progress_bar(image, x, y, w, h, progress, color):
    progress = np.clip(progress, 0.0, 1.0)
    cv2.rectangle(image, (x, y), (x + w, y + h), (80, 80, 80), 2)
    cv2.rectangle(image, (x, y),
                  (x + int(w * progress), y + h),
                  color, -1)


# -------------------- Motion Logic --------------------

class ArmTracker:
    def __init__(self):
        self.state = "EXTENDED"
        self.count = 0
        self.angle_ema = None
        self.prev_angle = None
        self.prev_time = None

        self.ALPHA = 0.3
        self.EXTENDED_ANGLE = 160
        self.FLEXED_ANGLE = 50
        self.MIN_VELOCITY = 30

    def update(self, raw_angle):
        now = time.time()

        if self.angle_ema is None:
            self.angle_ema = raw_angle
        else:
            self.angle_ema = (
                self.ALPHA * raw_angle +
                (1 - self.ALPHA) * self.angle_ema
            )

        velocity = 0
        if self.prev_angle is not None:
            dt = now - self.prev_time
            if dt > 0:
                velocity = (self.prev_angle - self.angle_ema) / dt

        self.prev_angle = self.angle_ema
        self.prev_time = now

        if self.state == "EXTENDED":
            if self.angle_ema < self.EXTENDED_ANGLE and velocity > self.MIN_VELOCITY:
                self.state = "FLEXING"

        elif self.state == "FLEXING":
            if self.angle_ema < self.FLEXED_ANGLE:
                self.state = "FLEXED"

        elif self.state == "FLEXED":
            if velocity < -self.MIN_VELOCITY:
                self.state = "EXTENDING"

        elif self.state == "EXTENDING":
            if self.angle_ema > self.EXTENDED_ANGLE:
                self.count += 1
                self.state = "EXTENDED"

        return self.angle_ema


# -------------------- Main --------------------

def infer():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    left_arm = ArmTracker()
    right_arm = ArmTracker()

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                la = calc_angle(
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                    lm[mp_pose.PoseLandmark.LEFT_WRIST]
                )
                ra = calc_angle(
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                )

                la_s = left_arm.update(la)
                ra_s = right_arm.update(ra)

                lp = angle_to_progress(la_s)
                rp = angle_to_progress(ra_s)

                draw_progress_bar(image, 20, height - 55, 250, 25, lp, (0, 255, 120))
                draw_progress_bar(image, 300, height - 55, 250, 25, rp, (0, 180, 255))

                mp_draw.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            draw_hud(image, left_arm.count, right_arm.count, width)

            cv2.imshow("Advanced Bicep Curl Tracker", image)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                left_arm.count = 0
                right_arm.count = 0
                left_arm.state = "EXTENDED"
                right_arm.state = "EXTENDED"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    infer()
