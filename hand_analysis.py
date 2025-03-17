import cv2
import mediapipe as mp
import math
import numpy as np

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
RING_OFFSET_THRESHOLD = 0.03

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, 
                                  min_detection_confidence=min_detection_confidence,
                                  min_tracking_confidence=min_tracking_confidence)
        self.MOVEMENT_THRESHOLD = 30 
        self.SMOOTHING_FACTOR = 0.4  
        self.last_center = None

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
        return middle_folded and ring_folded and pinky_folded and self.computeRingFingerOffset(landmarks) < RING_OFFSET_THRESHOLD
    
    def changeBubbleSize(self, landmarks):
        x1, y1 = landmarks[4][1], landmarks[4][2]
        x2, y2 = landmarks[8][1], landmarks[8][2]
        return math.hypot(x2-x1, y2-y1)

    def computeRingFingerOffset(self, landmarks):
        landmarks = np.array(landmarks)
        hand_size = np.linalg.norm(landmarks[0] - landmarks[9])
        middle_mean = np.mean(landmarks[[9, 10, 11, 12]], axis=0)
        pinky_mean = np.mean(landmarks[[17, 18, 19, 20]], axis=0)
        ring_mean = np.mean(landmarks[[13, 14, 15, 16]], axis=0)
        return np.linalg.norm(ring_mean - (middle_mean + pinky_mean)/2) / hand_size
    
    def calculate_hand_center(self, landmarks):
        if not landmarks:
            return None
        x_coords = [lm[1] for lm in landmarks]
        y_coords = [lm[2] for lm in landmarks]
        return (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))
    
    def analyzeMovement(self, image, landmarks):
        direction = []
        movement_vector = (0, 0)  
        current_center = self.calculate_hand_center(landmarks)
        
        if current_center:
            if self.last_center:
                dx = current_center[0] - self.last_center[0]
                dy = current_center[1] - self.last_center[1]
                
                smoothed_dx = dx * self.SMOOTHING_FACTOR + (1 - self.SMOOTHING_FACTOR) * (self.last_center[0] - current_center[0])
                smoothed_dy = dy * self.SMOOTHING_FACTOR + (1 - self.SMOOTHING_FACTOR) * (self.last_center[1] - current_center[1])
                
                movement_vector = (smoothed_dx, smoothed_dy)
                
                if abs(smoothed_dx) > self.MOVEMENT_THRESHOLD or abs(smoothed_dy) > self.MOVEMENT_THRESHOLD:
                    if abs(smoothed_dx) > abs(smoothed_dy):
                        direction.append("RIGHT" if smoothed_dx > 0 else "LEFT")
                    else:
                        direction.append("DOWN" if smoothed_dy > 0 else "UP")

            if self.last_center:
                self.last_center = (
                    self.last_center[0] * (1 - self.SMOOTHING_FACTOR) + current_center[0] * self.SMOOTHING_FACTOR,
                    self.last_center[1] * (1 - self.SMOOTHING_FACTOR) + current_center[1] * self.SMOOTHING_FACTOR
                )
            else:
                self.last_center = current_center
        else:
            self.last_center = None
            
        return direction, movement_vector

def draw_bubble_text(image, text, position, bg_color=(0, 255, 0), text_color=(0, 0, 0), font_scale=0.7, thickness=2, padding=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    
    x, y = position
    x_end = x + text_width + 2 * padding
    y_end = y + text_height + 2 * padding
    
    cv2.rectangle(image, (x, y), (x_end, y_end), bg_color, -1)
    radius = 10
    cv2.circle(image, (x + radius, y + radius), radius, bg_color, -1)
    cv2.circle(image, (x_end - radius, y + radius), radius, bg_color, -1)
    cv2.circle(image, (x + radius, y_end - radius), radius, bg_color, -1)
    cv2.circle(image, (x_end - radius, y_end - radius), radius, bg_color, -1)
    text_position = (x + padding, y + text_height + padding)
    cv2.putText(image, text, text_position, font, font_scale, text_color, thickness)
    
    return x_end  # Retourne la position x de fin pour aligner plusieurs bulles

# handDetector = HandDetector(min_detection_confidence=0.7)
# webcamFeed = cv2.VideoCapture(1)
# bubble_size = 1.0

# while True:
#     status, image = webcamFeed.read()
#     if not status:
#         break

#     handLandmarks = handDetector.findHandLandMarks(image=image, draw=True)
    
#     direction, movement_vector = handDetector.analyzeMovement(image, handLandmarks)
#     movement_text = f"Mouvement: {', '.join(direction) if direction else 'Aucun'} ({movement_vector[0]:.1f}, {movement_vector[1]:.1f})"
#     size_text = f"Size: {bubble_size:.2f}"
#     start_y = image.shape[0] - 60
#     x_pos = 10
    
#     x_pos = draw_bubble_text(image, movement_text, (x_pos, start_y), (0, 255, 0))
    
#     x_pos += 10  
#     draw_bubble_text(image, size_text, (x_pos, start_y), (255, 0, 0), (255, 255, 255))

#     if len(handLandmarks) > 0:
#         x1, y1 = handLandmarks[4][1], handLandmarks[4][2]
#         x2, y2 = handLandmarks[8][1], handLandmarks[8][2]

#         if handDetector.isSizeGesture(handLandmarks):
#             cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
#             cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
#             cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
#             cv2.putText(image, "Size Gesture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#             length = handDetector.changeBubbleSize(handLandmarks)
#             hand_size = math.hypot(handLandmarks[0][1] - handLandmarks[5][1], 
#                                  handLandmarks[0][2] - handLandmarks[5][2])
#             bubble_size = length / hand_size
    
#     cv2.imshow("TS IS BUBONIC 💔 PMO", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# webcamFeed.release()
# cv2.destroyAllWindows()