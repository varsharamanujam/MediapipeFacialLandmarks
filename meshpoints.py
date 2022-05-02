import cv2 
import mediapipe as mp
import numpy as np
from custom.iris_lm_depth import from_landmarks_to_depth
from videosource import FileSource, WebcamSource
import math

map_face_mesh = mp.solutions.face_mesh


#right iris indices
RIGHT_IRIS =[ 474,475,476,477]
#left iris indices
LEFT_IRIS =[ 469,470,471,472]

L_H_LEFT = [33] # left eye left corner
L_H_RIGHT = [133] #left eye right corner 
R_H_LEFT = [362] #right eye left corner 
R_H_RIGHT = [263] #right eye right corner

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point.ravel()
    x1, y1 = point1.ravel()
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right_distance = euclaideanDistance(iris_center, right_point)
    center_to_left_distance = euclaideanDistance(iris_center, left_point)
    total_distance = euclaideanDistance(right_point, left_point)
    ratio = center_to_right_distance/total_distance
    IrisPosition = ""
    if ratio <= 0.42:
        IrisPosition = "Right"
    elif ratio > 2.7 and ratio <= 3.2:
        IrisPosition = "Center"
    else:
        IrisPosition = "Left"
    return IrisPosition, ratio

# camera object 
camera = cv2.VideoCapture(0)
ret, frame = camera.read() # getting frame from camera 
with map_face_mesh.FaceMesh(
    max_num_faces = 1,refine_landmarks = True,  min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
    # starting Video loop here.
    while True:
        ret, frame = camera.read() # getting frame from camera 
        
        if not ret: 
            break # no more frames break
        #flip the frame
        frame = cv2.flip(frame, 1)
        img_h, img_w= frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            meshpoints = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(meshpoints[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(meshpoints[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
            #print(meshpoints[LEFT_IRIS])
            #print(meshpoints[RIGHT_IRIS])
            #print(meshpoints[L_H_LEFT])
            #print(meshpoints[L_H_RIGHT])
            #drawing circles on the eyes
            cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), 2,cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (0, 255, 0), 2,cv2.LINE_AA)

            cv2.circle(frame, meshpoints[R_H_RIGHT][0], 2, (255, 255, 255), 2,cv2.LINE_AA)
            cv2.circle(frame, meshpoints[R_H_LEFT][0], 2, (255, 255, 255), 2,cv2.LINE_AA)
            
            cv2.circle(frame, meshpoints[L_H_RIGHT][0], 2, (255, 255, 255), 2,cv2.LINE_AA)
            cv2.circle(frame, meshpoints[L_H_LEFT][0], 2, (255, 255, 255), 2,cv2.LINE_AA)
            
            iris_pos_l, ratio_l = iris_position(center_left, meshpoints[R_H_RIGHT], meshpoints[R_H_LEFT][0])
            iris_pos_r, ratio_r = iris_position(center_right, meshpoints[L_H_RIGHT], meshpoints[L_H_LEFT][0])
            
            cv2.putText(frame, "Iris Position Left: {} ratio: {} Iris Position Right: {} ratio: {}".format(iris_pos_l,ratio_l,iris_pos_r,ratio_r), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow('frame', frame)
        key = cv2.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()