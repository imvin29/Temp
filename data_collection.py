import mediapipe as mp 
import numpy as np 
import cv2 
import time  
import sys
name = sys.argv[1] if len(sys.argv) > 1 else input("Enter Asana Name: ")


def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True 
    return False
 
cap = cv2.VideoCapture(1)

#name = input("Enter the name of the Asana: ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0


start_time = time.time()
while time.time() - start_time < 10:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    remaining_time = int(10 - (time.time() - start_time))
    
    cv2.putText(frm, f"Starting in: {remaining_time}s", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.imshow("window", frm)
    
    if cv2.waitKey(1) == 27:  
        cap.release()
        cv2.destroyAllWindows()
        exit()

while True:
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        X.append(lst)
        data_size = data_size + 1

    else: 
        cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or data_size > 80:
        cv2.destroyAllWindows()
        cap.release()
        break

np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)