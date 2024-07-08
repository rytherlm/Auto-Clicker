import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Helper function to calculate distance between two points.
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

# Initialize the last click time.
last_click_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands.
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions.
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate distance.
            thumb_pinky_distance = calculate_distance(thumb_tip, pinky_tip)

            # Check for left click gesture.
            current_time = time.time()
            if thumb_pinky_distance < 0.02:  # 0.5-second delay
                pyautogui.click(button='left')
                last_click_time = current_time

    # Convert the image back to BGR for display.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image.
    cv2.imshow('Hand Detection and Gesture Recognition', image)

    if cv2.waitKey(1) & 0xFF == 27:  # Using 1ms wait for faster loop
        break

cap.release()
cv2.destroyAllWindows()
