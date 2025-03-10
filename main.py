import cv2
import mediapipe as mp
import math as math
import numpy as np
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
RING_OFFSET_THRESHOLD = 0.03

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, handNumber=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        results = self.hands.process(image)
        landMarkList = []
        if results.multi_hand_landmarks: 
            hand = results.multi_hand_landmarks[handNumber]
            for id, landMark in enumerate(hand.landmark):
                imgH, imgW, imgC = originalImage.shape 
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos])
            if draw:
                mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)
        return landMarkList
    
    def isSizeGesture(self, landmarks):
        if len(landmarks) < 21:
            return False
        middle_folded = landmarks[8][2] < landmarks[12][2]
        ring_folded = landmarks[8][2] < landmarks[16][2]
        pinky_folded = landmarks[8][2] < landmarks[20][2]
        return  middle_folded and ring_folded and pinky_folded and self.computeRingFingerOffset(landmarks) < RING_OFFSET_THRESHOLD
    
    def changeBubbleSize (self, landmarks):
        x1, y1 = landmarks[4][1], landmarks[4][2]
        x2, y2 = landmarks[8][1], landmarks[8][2]
        length = math.hypot(x2-x1, y2-y1)
        return length

    def computeRingFingerOffset(self, landmarks):
        landmarks = np.array(landmarks)

        hand_size = np.linalg.norm(landmarks[0] - landmarks[9])
        if hand_size == 0:
            raise ValueError("Taille de la main invalide (division par zéro)")

        middle_mean = np.mean(landmarks[[9, 10, 11, 12]], axis=0)
        pinky_mean = np.mean(landmarks[[17, 18, 19, 20]], axis=0)
        ring_mean = np.mean(landmarks[[13, 14, 15, 16]], axis=0)
        offset = np.linalg.norm(ring_mean - (middle_mean + pinky_mean) / 2) / hand_size

        return offset
        

handDetector = HandDetector(min_detection_confidence=0.7)
webcamFeed = cv2.VideoCapture(0)
bubble_size = 1.

while True:
    status, image = webcamFeed.read()
    handLandmarks = handDetector.findHandLandMarks(image=image, draw=True)
    if len(handLandmarks) > 0:
        x1, y1 = handLandmarks[4][1], handLandmarks[4][2]
        x2, y2 = handLandmarks[8][1], handLandmarks[8][2]
        if(len(handLandmarks) != 0):

            if handDetector.isSizeGesture(handLandmarks):
                cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(image, "Size Gesture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                length = handDetector.changeBubbleSize(handLandmarks)
                hand_size = math.hypot(handLandmarks[0][1] - handLandmarks[5][1], handLandmarks[0][2] - handLandmarks[5][2])
                bubble_size = length / hand_size

    cv2.putText(image, f"Size: {bubble_size:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("TS IS BUBONIC 💔 PMO" , image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)