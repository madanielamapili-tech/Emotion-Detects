import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern

# ===============================
# Load Models
# ===============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
svm_model = joblib.load('SVM-LBP_model_FER2013.pkl')
emotion_labels = np.load('LBP_emotion_labels_FER2013.npy', allow_pickle=True)


# ===============================
# Parameters (MUST match training)
# ===============================
radius = 1
n_points = 8 * radius
method = 'uniform'

# ===============================
# LBP Feature Extraction Function
# ===============================
def extract_lbp_features(gray_face):
    lbp = local_binary_pattern(gray_face, n_points, radius, method)
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram
    return hist.reshape(1, -1)   # Reshape for SVM

# ===============================
# Initialize Video and Storage
# ===============================
cap = cv2.VideoCapture(0)
detected_emotions = []  # store all predicted emotions

print("[INFO] Starting real-time Emotion Detection (press 'q' to quit)...")

# ===============================
# Real-time Detection Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))

        features = extract_lbp_features(roi_gray)
        pred = svm_model.predict(features)[0]

        # Handle numeric or string predictions
        label = emotion_labels[int(pred)] if str(pred).isdigit() else pred
        detected_emotions.append(label)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('LBP + SVM Emotion Detection', frame)

    # Exit when pressing 'q'
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
