import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ==============================
# 1. Load the trained SVM model and labels
# ==============================
model_path = "SVM-HOG_model_CK+48.pkl"
labels_path = "HOG_emotion_labels_CK+48.npy"

print("[INFO] Loading trained SVM model and labels...")
svm_clf = joblib.load(model_path)
emotion_labels = np.load(labels_path)

print("[INFO] Model and labels loaded successfully!")

# ==============================
# 2. Initialize Haar Cascade for face detection
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==============================
# 3. Define a function to extract HOG features
# ==============================
def extract_hog_features(gray_face):
    resized_face = cv2.resize(gray_face, (64, 64))
    features, _ = hog(resized_face, orientations=7, pixels_per_cell=(8, 8),
                      cells_per_block=(4, 4), block_norm='L2-Hys', visualize=True)
    return features.reshape(1, -1)

# ==============================
# 4. Start real-time video capture
# ==============================
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time emotion detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]

        # Extract HOG features
        features = extract_hog_features(roi_gray)

        # Predict emotion
        prediction = svm_clf.predict(features)[0]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, prediction, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display frame
    cv2.imshow("Real-Time Emotion Detection (HOG + SVM + Haarcascade)", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()