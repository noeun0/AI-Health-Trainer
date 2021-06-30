import cv2
from script.hand_video_detector import hand_video
import time
# import script.ExerciseCondition as EC
import numpy as np
import mediapipe as mp
import math
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

            # self.Draw(img, per, bar)
            return self.count, per, bar

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

            # self.Draw(img, per, bar)
            return self.count, per, bar

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
            Angle_hip_right = 360 - self.findAngle(img, p13, p14, p15, draw)
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

            if (int(Angle_left_side) < 120) & (int(Angle_left_side) > 170):

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

            # self.Draw(img, per_h_L, bar)
            return self.count, per_h_L, bar
        
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

            # self.Draw(img, per_h_L, bar)
            return self.count, per_h_L, bar
        
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

# basically take camera input and convert it into a cv object
# later to be processed by gen()
class VideoCamera(object):
   def __init__(self):
      self.video = cv2.VideoCapture(0)
      
   def __del__(self):
      self.video.release()

   def get_frame(self,detector,start, idx, Exercise_type, Exercise):
      width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
      success, img = self.video.read()

      if success:
         img = hand_video(success, img, detector)
         lmList = detector.findPosition(img, draw=False)
         if len(lmList) != 0:
            current = time.time()-start
           # print(current)
            count, per ,bar = detector.SelectExerciseMode(img, Exercise_type[Exercise[idx]])
            if (int(current) == 5) | (count == 10):
               idx += 1
               start = time.time()

         cv2.rectangle(img, (0, height - 200), (150, height),
                  (0, 255, 0), cv2.FILLED)
         cv2.putText(img, '{:.1f}'.format(current), (15, height-75), cv2.FONT_HERSHEY_PLAIN, 5,
                  (255, 0, 0), 5)
         cv2.putText(img, Exercise_type[Exercise[idx]], (15+150, height-50), cv2.FONT_HERSHEY_PLAIN, 5,
                  (0, 0, 255), 5)
         detector.Draw(img, per, bar)

         
      return img

idx=0
# generator that saves the video captured if flag is set
def gen(camera, flag, mode):
   if flag == True:
      
      time_now = time.localtime()
      current_time = time.strftime("%H:%M:%S", time_now)
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      out = cv2.VideoWriter('output_' + str(current_time) + '.avi',fourcc, 20.0, (640,480))

      while True:
         ret, jpeg = cv2.imencode('.jpg', camera.get_frame())
         frame =  jpeg.tobytes()
         
         yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

         cv_frame = camera.get_frame()
         out.write(cv_frame)
   
   else:

      Exercise_type = {1: 'pushup', 2: 'barbellCurl', 3: 'squat', 4: 'lunge', 5: 'bridge'}
      Routine_type = {'legs': [3, 1, 4], 'Arms': [1, 3, 2] , 'bridge_test':[5]}
      Routine = mode

      Exercise = Routine_type[Routine]

      fps = round(camera.video.get(cv2.CAP_PROP_FPS))
      width = int(camera.video.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(camera.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
      # print(fps, width, height)
      
      detector = ExerciseType(width, height, fps)
      start = time.time()
      global idx
      while True:
         
         ret, jpeg = cv2.imencode('.jpg', camera.get_frame(detector, start, idx, Exercise_type, Exercise))
         frame =  jpeg.tobytes()
           
         yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
         
         if cv2.waitKey(1) == 27:
            break