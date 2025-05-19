import sys
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Check for pose name passed from Flask
pose_name = sys.argv[1] if len(sys.argv) > 1 else "default_pose"

# Load model and labels
model = load_model("model.h5")
labels = np.load("labels.npy")

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to check if the pose is visible in the frame
def is_pose_in_frame(landmarks):
    if landmarks[28].visibility > 0.6 and landmarks[27].visibility > 0.6 and landmarks[15].visibility > 0.6 and landmarks[16].visibility > 0.6:
        return True
    return False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Process the frame to extract pose landmarks
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.pose_landmarks and is_pose_in_frame(result.pose_landmarks.landmark):
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append(lm.x - result.pose_landmarks.landmark[0].x)
            landmarks.append(lm.y - result.pose_landmarks.landmark[0].y)

        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict the pose using the model
        prediction = model.predict(landmarks)
        predicted_pose = labels[np.argmax(prediction)]

        # Display the prediction result
        if prediction[0][np.argmax(prediction)] > 0.75:
            cv2.putText(frame, f"Pose: {predicted_pose}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Pose is either incorrect or not trained", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw pose landmarks
    drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow("Pose Detection", frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
