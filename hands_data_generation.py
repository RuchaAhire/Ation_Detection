'''
import cv2
import mediapipe as mp
import pandas as pd

cap = cv2.VideoCapture(4)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)  # Ensure only one hand is detected
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "h"
no_of_frames = 1000

def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only consider the first detected hand
        for lm in hand_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only consider the first detected hand
        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
        for lm in hand_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            frame = draw_landmark_on_image(mpDraw, results, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()

# to open camera
import cv2

for i in range(10):  # Try the first 10 indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Camera index {i}", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Camera index {i} is not available.")
'''

import cv2
import mediapipe as mp
import pandas as pd

# Function to process hand landmarks
def process_hand_landmarks(frame, hands, mpDraw):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

# Function to detect landmarks for a timestep
def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only consider the first detected hand
        for lm in hand_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
    return c_lm

# Initialize camera capture and MediaPipe objects
cap = None
for i in range(10):  # Try the first 10 camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        break
    else:
        print(f"Camera index {i} is not available.")

if cap is None or not cap.isOpened():
    print("No camera available.")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)  # Ensure only one hand is detected
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "hand_landmarks"
no_of_frames = 1000

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame_with_landmarks = None

    frame_with_landmarks = process_hand_landmarks(frame, hands, mpDraw)
    cv2.imshow("Hand Landmarks", frame_with_landmarks)

    if cv2.waitKey(1) == ord('q'):
        break

# Save landmarks to a CSV file
df = pd.DataFrame(lm_list)
df.to_csv(label + ".csv")

cap.release()
cv2.destroyAllWindows()
