from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ----------------------------
# Function to calculate eyebrow distance
# ----------------------------
def eye_brow_distance(leye, reye):
    global points
    distq = dist.euclidean(leye, reye)
    points.append(int(distq))
    return distq

# ----------------------------
# Function to detect emotion and stress status
# ----------------------------
def emotion_finder(face, frame):
    global emotion_classifier

    EMOTIONS = ["angry", "disgust", "scared",
                "happy", "sad", "surprised", "neutral"]

    x, y, w, h = face_utils.rect_to_bb(face)
    roi = frame[y:y+h, x:x+w]

    # Convert to grayscale
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi, verbose=0)[0]
    emotion = EMOTIONS[preds.argmax()]

    # Stress status from emotion
    if emotion in ['angry', 'sad', 'scared']:
        stress_status = "stressed"
    else:
        stress_status = "not stressed"

    return emotion, stress_status

# ----------------------------
# Normalize stress values
# ----------------------------
def normalize_values(points, disp):
    if len(points) == 0 or np.max(points) == np.min(points):
        return 0.0, "Low Stress"

    normalized_value = abs(disp - np.min(points)) / abs(np.max(points) - np.min(points))
    stress_value = np.exp(-normalized_value)

    if stress_value >= 0.75:
        return stress_value, "High Stress"
    else:
        return stress_value, "Low Stress"

# ----------------------------
# Load models and predictors
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    os.path.join(root_dir, "shape_predictor_68_face_landmarks.dat")
)
emotion_classifier = load_model(
    os.path.join(root_dir, "_mini_XCEPTION.102-0.66.hdf5"),
    compile=False
)

# ----------------------------
# Video capture
# ----------------------------
cap = cv2.VideoCapture(0)
points = []

(lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (500, 500))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 0)

    for detection in detections:
        # Emotion detection
        emotion, stress_status = emotion_finder(detection, frame)

        cv2.putText(frame, f"Emotion: {emotion}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"Status: {stress_status}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        shape = predictor(gray, detection)
        shape = face_utils.shape_to_np(shape)

        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]

        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

        distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
        stress_value, stress_label = normalize_values(points, distq)

        cv2.putText(frame,
                    f"Stress Level: {int(stress_value * 100)}%",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)

    cv2.imshow("Stress & Emotion Detection", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()

# Plot stress variation graph
plt.plot(range(len(points)), points, 'ro')
plt.title("Stress Levels Over Time")
plt.xlabel("Frame Count")
plt.ylabel("Eyebrow Distance")
plt.show()
