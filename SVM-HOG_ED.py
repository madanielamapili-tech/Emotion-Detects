import cv2
import numpy as np
from skimage.feature import hog
import joblib

# ===============================
# Load trained SVM model and labels
# ===============================
svm_model = joblib.load('SVM-HOG_model_CK+48.pkl')
emotion_labels = np.load('HOG_emotion_labels_CK+48.npy', allow_pickle=True)

print("[INFO] Model and labels loaded successfully!")
print("[INFO] Emotion classes:", emotion_labels)

# ===============================
# Initialize Haarcascade face detector
# ===============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===============================
# Start webcam
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access the webcam.")
    exit()

# Keep track of detected emotions
detected_emotions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))

        # Extract HOG features (must match training exactly!)
        features = hog(roi_gray, orientations=7, pixels_per_cell=(8, 8),
                       cells_per_block=(4, 4), block_norm='L2-Hys',
                       transform_sqrt=False)
        features = features.reshape(1, -1)

        # Predict emotion
        emotion_index = svm_model.predict(features)[0]
        label = emotion_index  # already string if you trained labels as text
        detected_emotions.append(label)

        # Draw face box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition (SVM + HOG + Haarcascade)", frame)

    # Press 'q' to quit and show detected emotions summary
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===============================
# Display all detected emotions
# ===============================
if detected_emotions:
    print("\n==============================")
    print("Session Detected Emotions:")
    print("==============================")
    unique, counts = np.unique(detected_emotions, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"{emotion}: {count} times")
else:
    print("[INFO] No emotions detected during session.")
