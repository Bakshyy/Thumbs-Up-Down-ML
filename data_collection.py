import cv2
import os
import mediapipe as mp

# === Settings ===
CAPTURE_COUNT = 250
OUTPUT_DIR = "data"
GESTURES = ["thumbs_up", "thumbs_down"]

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    gesture_dir = os.path.join(OUTPUT_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)

    count = 0
    print(f"Capturing {gesture}... Press 'q' to quit early.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:
        while count < CAPTURE_COUNT:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save the frame
                img_path = os.path.join(gesture_dir, f"{gesture}_{count}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1

            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
