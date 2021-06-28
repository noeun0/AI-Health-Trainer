import cv2
import time
import ExerciseCondition as EC

# 'pose/squat.mp4' pose/Boy1920x1080.mp4 pose/pushup.mp4 pose/pushupvideo.mp4 pose/barbellcurl2.mp4
# SquatsideView.mp4 squat2
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('pose/Fvs.webm')
fps = round(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width, height,fps)

delay = round(1000 / fps)


codec = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('pose/webcam_output.avi', codec, fps, (width, height))

detector = EC.ExerciseType()

time.sleep(2)

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (width*2, height*2))
    # img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img = cv2.flip(img, 0)
    # img = cv2.flip(img, 1)
    
    img = detector.findPose(img, draw=True)
    lmList = detector.findPosition(img, draw=True)

    if len(lmList) != 0:

        detector.SelectExerciseMode(img, 'squat')

    # detector.VideoSave(writer, img) ############
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv2.destroyAllWindows()