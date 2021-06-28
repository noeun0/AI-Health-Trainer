import cv2
import mediapipe as mp
import numpy as np
import time
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
class poseDetector():
    
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw):
        
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # cv2.putText(img, '%.2f' % angle, (x2 - 150, y2 - 150),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        return angle

    def VideoSave(self, writer,img):
        writer.write(img)
        return writer

class ExerciseType(poseDetector):

    def __init__(self, width, height, fps):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps

        self.count = 0
        self.color = (255, 0, 255)

        self.dir = 0
        self.Up = False
        self.Down = False

        self.check = False
        self.Landmarks = {'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
                          'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7,
                          'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
                          'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15,
                          'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
                          'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
                          'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27,
                          'right_ankle': 28, 'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
                          'right_foot_index': 32}
    
    # def BodyPoint(self,body):
    #     p1, p2, p3 = self.Landmarks['left_shoulder'], self.Landmarks['left_elbow'],\
    #             self.Landmarks['left_wrist']
    #     return p1,p2,p3

    def SelectExerciseMode(self, img, Mode, draw=True):
        
        if Mode == 'pushup':
            p1, p2, p3 = self.Landmarks['left_shoulder'], self.Landmarks['left_elbow'],\
                self.Landmarks['left_wrist']
            p4, p5, p6 = self.Landmarks['right_shoulder'], self.Landmarks['right_elbow'],\
                self.Landmarks['right_wrist']
            p7, p8, p9 = self.Landmarks['left_shoulder'], self.Landmarks['left_hip'], \
                self.Landmarks['left_knee']
            p10, p11, p12 = self.Landmarks['left_hip'], self.Landmarks['left_knee'], \
                self.Landmarks['left_ankle']
            p13, p14, p15 = self.Landmarks['right_hip'], self.Landmarks['right_knee'], \
                self.Landmarks['right_ankle']

            Angle_arm_L = self.findAngle(img, p1, p2, p3, draw)
            Angle_arm_R = self.findAngle(img, p4, p5, p6, draw)

            ref_low = 225
            ref_high = 250

            per = np.interp(Angle_arm_L, (ref_low, ref_high), (0, 100))
            per_R = np.interp(Angle_arm_R, (ref_low, ref_high), (0, 100))
            bar = np.interp(Angle_arm_L, (ref_low + 10, ref_high), (self.height-100, 100))

            # Angle_back = self.findAngle(img, p7, p8, p9, draw=False)
            Angle_leg_left = self.findAngle(img, p10, p11, p12, draw=True)
            Angle_leg_right = self.findAngle(img, p13, p14, p15, draw=True)

            if (int(Angle_leg_left) < 180) & (int(Angle_leg_left) > 160) &\
                    (int(Angle_leg_right) < 180) & (int(Angle_leg_right) > 160):

                if (per >= 70) & (per_R >= 70):
                    self.Up = True
                    if (per == 100) & (per_R == 100):
                        self.color = (0, 255, 0)
                        if (self.dir == 0):
                            self.count += .5
                            self.dir = 1
                            self.check = True
                    if (int(per) > 80) & (int(per) < 100) & (self.check == False):
                        cv2.putText(img, 'MORE! MORE! MORE!', (250, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)

                if (per == 0) & (per_R == 0):
                    self.color = (0, 255, 0)
                    
                    if (self.dir == 1):
                        self.count += .5
                        self.dir = 0
                        self.Up = False
                        self.check = False
                    elif (self.Up == True) & (self.dir == 0):
                        self.check = False
                        cv2.putText(img, 'BEND YOUR ARMS MORE !!!', (250, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
            else:
                per = 0
                bar = self.height-100

            self.Draw(img, per, bar)

        elif Mode == 'lunge':
            p1, p2, p3 = self.Landmarks['left_hip'], self.Landmarks['left_knee'], \
                self.Landmarks['left_ankle']
            p4, p5, p6 = self.Landmarks['right_hip'], self.Landmarks['right_knee'], \
                self.Landmarks['right_ankle']
            p7, p8, p9 = self.Landmarks['left_ear'], self.Landmarks['left_shoulder'], \
                self.Landmarks['left_hip']
            p10, p11, p12 = self.Landmarks['left_ear'], self.Landmarks['left_hip'], \
                self.Landmarks['left_knee']
            p13, p14, p15 = self.Landmarks['right_ear'], self.Landmarks['right_hip'], \
                self.Landmarks['right_knee']
            p16, p17, p18 = self.Landmarks['left_hip'], self.Landmarks['left_knee'], \
                self.Landmarks['left_ankle']
            p19, p20, p21 = self.Landmarks['right_hip'], self.Landmarks['right_knee'], \
                self.Landmarks['right_ankle']

            Angle_leg_left = self.findAngle(img, p1, p2, p3, draw=False)
            Angle_leg_right = self.findAngle(img, p4, p5, p6, draw=False)
            Angle_left_side = self.findAngle(img, p7, p8, p9, draw=False)
            Angle_hip_left = 360 - self.findAngle(img, p10, p11, p12, draw)
            Angle_hip_right = 360- self.findAngle(img, p13, p14, p15, draw)
            Angle_Knee_left = self.findAngle(img, p16, p17, p18, draw)
            Angle_Knee_right = self.findAngle(img, p19, p20, p21, draw)

            ref_low = 90
            ref_high = 150

            per = np.interp(Angle_leg_left, (ref_low, ref_high), (0, 100))
            per_R = np.interp(Angle_leg_left, (ref_low, ref_high), (0, 100))
            bar = np.interp(Angle_leg_left, (ref_low + 10, ref_high), (100, self.height-100))

            if ((int(Angle_hip_left) < 220) & (int(Angle_hip_left) > 170)) |\
                ((int(Angle_hip_right) < 220) & (int(Angle_hip_right) > 170)):

                if (per <= 40) & (per_R <= 40):
                    self.Down = True
                    if per == 0:
                        self.color = (0, 255, 0)
                        if self.dir == 0:
                            self.count += .5
                            self.dir = 1
                            self.check = True

                if (per == 100) & (per_R == 100):
                    self.color = (0, 255, 0)
                    if self.dir == 1:
                        self.count += .5
                        self.dir = 0
                        self.Down = False
                        self.check = False
                    elif (self.Down == True) & (self.dir == 0):
                        self.check = False
                        cv2.putText(img, 'MORE LEGS DOWN!!!', (250, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                per = 100 - per
            else:
                per = 0
                bar = self.height-100

            self.Draw(img, per, bar)

        elif Mode == 'squat':  # side
            p1, p2, p3 = self.Landmarks['left_hip'], self.Landmarks['left_knee'], \
                self.Landmarks['left_ankle']
            p4, p5, p6 = self.Landmarks['right_hip'], self.Landmarks['right_knee'], \
                self.Landmarks['right_ankle']
            p7, p8, p9 = self.Landmarks['left_ear'], self.Landmarks['left_shoulder'], \
                self.Landmarks['left_hip']
            p10, p11, p12 = self.Landmarks['left_ear'], self.Landmarks['left_hip'], \
                self.Landmarks['left_knee']
            p13, p14, p15 = self.Landmarks['right_ear'], self.Landmarks['right_hip'], \
                self.Landmarks['right_knee']
            p16, p17, p18 = self.Landmarks['left_hip'], self.Landmarks['left_knee'], \
                self.Landmarks['left_ankle']
            p19, p20, p21 = self.Landmarks['right_hip'], self.Landmarks['right_knee'], \
                self.Landmarks['right_ankle']

            Angle_leg_left = self.findAngle(img, p1, p2, p3, draw=False)
            Angle_leg_right = self.findAngle(img, p4, p5, p6, draw=False)
            Angle_left_side = self.findAngle(img, p7, p8, p9, draw=False)
            Angle_hip_left = 360 - self.findAngle(img, p10, p11, p12, draw)
            Angle_hip_right = 360- self.findAngle(img, p13, p14, p15, draw)
            Angle_Knee_left = self.findAngle(img, p16, p17, p18, draw)
            Angle_Knee_right = self.findAngle(img, p19, p20, p21, draw)
            
            ref_low_hip = 70
            ref_high_hip = 95
            ref_low_knee = 90 
            ref_high_knee = 150 

            per_h_L = np.interp(Angle_hip_left, (ref_low_hip, ref_high_hip), (0, 100))
            per_h_R = np.interp(Angle_hip_right, (ref_low_hip, ref_high_hip), (0, 100))
            per_kn_L = np.interp(Angle_Knee_left, (ref_low_knee, ref_high_knee), (0, 100))
            per_kn_R = np.interp(Angle_Knee_right, (ref_low_knee, ref_high_knee), (0, 100))

            bar = np.interp(Angle_Knee_left, (ref_low_knee + 10, ref_low_knee), (100, self.height-100))

            if (int(Angle_left_side) < 220) & (int(Angle_left_side) > 170):

                if (per_h_L <= 40) & (per_h_R <= 40) & (per_kn_L <= 40) & (per_kn_R <= 40):
                    self.Down = True
                    if (per_h_L == 0) & (per_h_R == 0) & (per_kn_L == 0) & (per_kn_R == 0):
                        self.color = (0, 255, 0)
                        if self.dir == 0:
                            self.count += .5
                            self.dir = 1
                            self.check = True
                    
                if (per_h_L == 100) & (per_h_R == 100) & (per_kn_L == 100) & (per_kn_R == 100):
                    self.color = (0, 255, 0)
                    if self.dir == 1:
                        self.count += .5
                        self.dir = 0
                        self.Down = False
                        self.check = False
                    elif (self.Down == True) & (self.dir == 0):
                        self.check = False
                        cv2.putText(img, 'MORE LEGS DOWN!!!', (250, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                per_h_L = 100 - per_h_L
            else:
                per_h_L = 0
                bar = self.height-100

            self.Draw(img, per_h_L, bar)
        
        elif Mode == 'bridge':
            p1, p2, p3 = self.Landmarks['left_ear'], self.Landmarks['left_hip'], \
                self.Landmarks['left_knee']
            p4, p5, p6 = self.Landmarks['right_ear'], self.Landmarks['right_hip'], \
                self.Landmarks['right_knee']
            p7, p8, p9 = self.Landmarks['left_hip'], self.Landmarks['left_knee'], \
                self.Landmarks['left_ankle']
            p10, p11, p12 = self.Landmarks['right_hip'], self.Landmarks['right_knee'], \
                self.Landmarks['right_ankle']
            p13, p14, p15 = self.Landmarks['left_shoulder'], self.Landmarks['left_elbow'],\
                self.Landmarks['left_wrist']
            p16, p17, p18 = self.Landmarks['right_shoulder'], self.Landmarks['right_elbow'],\
                self.Landmarks['right_wrist']

            Angle_hip_left = 360 - self.findAngle(img, p1, p2, p3, draw)
            Angle_hip_right = 360- self.findAngle(img, p4, p5, p6, draw)
            Angle_Knee_left = self.findAngle(img, p7, p8, p9, draw)
            Angle_Knee_right = self.findAngle(img, p10, p11, p12, draw)
            Angle_arm_left = self.findAngle(img, p13, p14, p15, draw)
            Angle_arm_right = self.findAngle(img, p16, p17, p18, draw)
            print(Angle_hip_left,Angle_hip_right,Angle_Knee_left,Angle_Knee_right)
            
            ref_low_hip = 70
            ref_high_hip = 95
            ref_low_knee = 90 
            ref_high_knee = 150 

            per_h_L = np.interp(Angle_hip_left, (ref_low_hip, ref_high_hip), (0, 100))
            per_h_R = np.interp(Angle_hip_right, (ref_low_hip, ref_high_hip), (0, 100))
            per_kn_L = np.interp(Angle_Knee_left, (ref_low_knee, ref_high_knee), (0, 100))
            per_kn_R = np.interp(Angle_Knee_right, (ref_low_knee, ref_high_knee), (0, 100))

            bar = np.interp(Angle_Knee_left, (ref_low_knee + 10, ref_low_knee), (100, self.height-100))
            if (int(Angle_arm_left) < 220) & (int(Angle_arm_left) > 170):
    
                if (per_h_L <= 40) & (per_h_R <= 40) & (per_kn_L <= 40) & (per_kn_R <= 40):
                    self.Down = True
                    if (per_h_L == 0) & (per_h_R == 0) & (per_kn_L == 0) & (per_kn_R == 0):
                        self.color = (0, 255, 0)
                        if self.dir == 0:
                            self.count += .5
                            self.dir = 1
                            self.check = True
                    
                if (per_h_L == 100) & (per_h_R == 100) & (per_kn_L == 100) & (per_kn_R == 100):
                    self.color = (0, 255, 0)
                    if self.dir == 1:
                        self.count += .5
                        self.dir = 0
                        self.Down = False
                        self.check = False
                    elif (self.Down == True) & (self.dir == 0):
                        self.check = False
                        cv2.putText(img, 'MORE LEGS DOWN!!!', (250, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                per_h_L = 100 - per_h_L
            else:
                per_h_L = 0
                bar = self.height-100

            self.Draw(img, per_h_L, bar)


    def Draw(self, img, per, bar):
        cv2.rectangle(img, (0, 0), (150, 200), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(self.count)), (20, 150), cv2.FONT_HERSHEY_PLAIN, 10,
                    (255, 0, 0), 10)

        cv2.rectangle(img, (self.width -100, 100), (self.width - 50, self.height-100),
                      self.color, 3)
        cv2.rectangle(img, (self.width -100, int(bar)), (self.width - 50, self.height-100),
                      self.color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                    self.color, 3)

        return self.count
def hand_image():
    # For static images:
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    # feed a video:
    videoFile = "test_vid.mp4"
    cap = cv2.VideoCapture(videoFile)
    flag, frame = cap.read()

    # while cap.isOpened():
    while flag:
        image = cv2.flip(frame, 1)
        frame_ID = cap.get(1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_hight, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(
            '/tmp/annotated_image_' + str(frame_ID) + '.png', cv2.flip(annotated_image, 1))
        flag, frame = cap.read()
    hands.close()
def hand_video(flag, frame, detector):
    image = cv2.flip(frame, 1)
    image = detector.findPose(image, draw=False)
    return image
def hand_video1(flag, frame):
    # For static images:
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    image = cv2.flip(frame, 1)
    # frame_ID = cap.get(1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        hands.close()
        return frame
    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return cv2.flip(annotated_image, 1)

def vid_save():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()