import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load your trained model
model = load_model('models/windowsAdvanced.keras')

# Video capture setup
videoCapture = cv.VideoCapture('video/Rangle.mp4')
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2
backSub = cv.createBackgroundSubtractorKNN()

# Trajectory tracking
trajectory = []
max_trajectory_length = 20
ball_detected_count = 0
frames_without_detection = 0

# Preprocessing images for model
def preprocess_for_model(img, size=(224, 224)):
    img = cv.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Get centroid of a rectangle
def get_centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

while True:
    ret, frame = videoCapture.read()
    if not ret: break
    frame = cv.resize(frame, (1280, 720))

    mask = backSub.apply(frame)
    mask = cv.GaussianBlur(mask, (13, 13), 0)
    ret, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT_ALT, 1.5, 1, param1=250, param2=0.84, minRadius=3, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None or (prevCircle is not None and dist(i[0], i[1], prevCircle[0], prevCircle[1]) < dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1])):
                chosen = i

        rx, ry = chosen[0] - chosen[2], chosen[1] - chosen[2]
        rw, rh = chosen[2] * 2, chosen[2] * 2

        height, width = frame.shape[:2]
        x, y, w, h = max(0, rx), max(0, ry), min(rw, width - rx), min(rh, height - ry)

        cut_f = frame[y:y+h, x:x+w]
        cut_m = mask[y:y+h, x:x+w]
        cut_c = cv.bitwise_and(cut_f, cut_f, mask=cut_m)

        if cut_c is not None:
            preprocessed_img = preprocess_for_model(cut_c)
            prediction = model.predict(preprocessed_img)
            predicted_class = 'ball' if prediction[0][0] < 0.5 else 'notBall'

            if predicted_class == 'ball':
                ball_detected_count += 1
                frames_without_detection = 0
                center = get_centroid(x, y, w, h)
                trajectory.append(center)
                if len(trajectory) > max_trajectory_length:
                    trajectory.pop(0)
                cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 255, 0), 3)
            else:
                frames_without_detection += 1
                cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 0, 255), 3)
                print("Not a ball")
        else:
            frames_without_detection += 1
        prevCircle = chosen
    else:
        print("No circles detected")
        frames_without_detection += 1

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    # Game state display
    game_state = "In Play" if ball_detected_count > 3 else "Out of Play"
    if frames_without_detection >= 20:
        game_state = "Out of Play"
        trajectory = []  # Reset trajectory if out of play for too long
        ball_detected_count = 0  # Reset ball count

    cv.putText(frame, f"Game State: {game_state}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("getBall", frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
