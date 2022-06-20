import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def detect(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handls in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handls, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handno=0, draw=True):
        lmList = [] 
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) 
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 7, (0,255,255), cv2.FILLED)

        return lmList



def main():
    ptime = 0
    ctime = 0 
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.detect(img)
        plist = detector.findPosition(img)
        if len(plist) != 0:
            print(plist[4])

        ctime = time.time()
        fps = int(1/(ctime-ptime)) 
        ptime = ctime
        # print(int(fps))
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()