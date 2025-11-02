# ==========================================================
# Real-Time Emotion Detection using Gabor + SVM + Haarcascade
# ==========================================================
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load face detector and model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
svm_model = joblib.load('SVM-Gabor_model_FER2013.pkl')

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Build same Gabor filters used in training
def build_gabor_filters():
    filters = []
    ksize = [3, 5, 7]
    for theta in np.arange(0, np.pi, np.pi / 4):
        for k in ksize:
            kern = cv2.getGaborKernel((k, k), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

def process_gabor(img, filters):
    feats = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        feats.append(np.mean(fimg))
        feats.append(np.std(fimg))
    return np.array(feats).reshape(1, -1)

filters = build_gabor_filters()
detected_emotions = []

cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time emotion detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        feat = process_gabor(roi_gray, filters)
        pred = svm_model.predict(feat)[0]
        label = pred
        detected_emotions.append(label)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gabor + SVM Emotion Detection", frame)
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
