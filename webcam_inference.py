import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("thumbs_model.keras")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

labels = ["Thumbs Up", "Thumbs Down"]

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Crop hand bounding box
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in results.multi_hand_landmarks[0].landmark]) * w
            x_max = max([lm.x for lm in results.multi_hand_landmarks[0].landmark]) * w
            y_min = min([lm.y for lm in results.multi_hand_landmarks[0].landmark]) * h
            y_max = max([lm.y for lm in results.multi_hand_landmarks[0].landmark]) * h

            x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                hand_img = cv2.resize(hand_img, (128, 128)) / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                pred = model.predict(hand_img, verbose=0)[0]
                idx = np.argmax(pred)
                prob = pred[idx] * 100

                # Draw results
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{labels[idx]}: {prob:.2f}%",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Thumbs Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
