import cv2
from cv2 import waitKey 
import mediapipe as mp
import numpy as np
from custom.iris_lm_depth import from_landmarks_to_depth
from videosource import FileSource, WebcamSource
import math

map_face_mesh = mp.solutions.face_mesh # object for face mesh 


#right iris indices
RIGHT_IRIS =[ 474,475,476,477] 
#left iris indices
LEFT_IRIS =[ 469,470,471,472]

L_H_LEFT = [33] # left eye left corner
L_H_RIGHT = [133] #left eye right corner 
R_H_LEFT = [362] #right eye left corner 
R_H_RIGHT = [263] #right eye right corner
L_H_TOP = [27] # left eye top 
L_H_BOTTOM = [23] # left eye bottom
R_H_TOP = [257] # right eye top
R_H_BOTTOM = [253] # right eye bottom

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point.ravel() # ravel flattens the array
    x1, y1 = point1.ravel() # ravel flattens the array
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2) # euclidean distance
    return distance

def right_position(iris_center, right_point, left_point):
    center_to_right_distance = euclaideanDistance(iris_center, right_point) # distance between iris center and right eye corner
    total_distance = euclaideanDistance(right_point, left_point) # distance between right eye corner and left eye corner
    ratio = center_to_right_distance/total_distance # ratio between distance between iris center and right eye corner and distance between right eye corner and left eye corner
    return  ratio # return Iris position and ratio


def left_position(iris_center, right_point, left_point): # same as right_iris_position
    center_to_left_distance = euclaideanDistance(iris_center, left_point) # distance between iris center and left eye corner
    total_distance = euclaideanDistance(right_point, left_point) # distance between right eye corner and left eye corner
    ratio = center_to_left_distance/total_distance # ratio between distance between iris center and left eye corner and distance between right eye corner and left eye corner
    
    return  ratio # return Iris position and ratio

# camera object 
camera = cv2.VideoCapture(0)
ret, frame = camera.read() # getting frame from camera

with map_face_mesh.FaceMesh(
    max_num_faces = 1,refine_landmarks = True,  min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh: # face mesh object
    # starting Video loop here.
    while True: # looping until break
        #get frame from camera every 5 frames
        
        ret, frame = camera.read() # getting frame from camera 
        
        if not ret:  # if frame is not retrived
            break # no more frames break
        #flip the frame
        frame = cv2.flip(frame, 1)
        img_h, img_w= frame.shape[:2] # getting height and width of frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # converting frame to BGR
        results  = face_mesh.process(rgb_frame) # getting results from face mesh
        if results.multi_face_landmarks: # if there are multiple faces
            meshpoints = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int) # getting meshpoints
                    for p in results.multi_face_landmarks[0].landmark # looping through landmarks
                ]
            )
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(meshpoints[LEFT_IRIS]) # getting left iris center and radius
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(meshpoints[RIGHT_IRIS]) # getting right iris center and radius
            center_left = np.array([l_cx, l_cy], dtype=np.int32) # left iris center
            center_right = np.array([r_cx, r_cy], dtype=np.int32) # right iris center
            
            #print(meshpoints[LEFT_IRIS])
            #print(meshpoints[RIGHT_IRIS])
            #print(meshpoints[L_H_LEFT])
            #print(meshpoints[L_H_RIGHT])
            #drawing circles on the eyes
            """cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), 2,cv2.LINE_AA) # left iris
            cv2.circle(frame, center_right, int(r_radius), (0, 255, 0), 2,cv2.LINE_AA) # right iris
 
            cv2.circle(frame, meshpoints[R_H_RIGHT][0], 2, (255, 255, 255), 2,cv2.LINE_AA) # right eye right corner
            cv2.circle(frame, meshpoints[R_H_LEFT][0], 2, (255, 255, 255), 2,cv2.LINE_AA) # right eye left corner
            
            cv2.circle(frame, meshpoints[L_H_RIGHT][0], 2, (255, 255, 255), 2,cv2.LINE_AA) # left eye right corner
            cv2.circle(frame, meshpoints[L_H_LEFT][0], 2, (255, 255, 255), 2,cv2.LINE_AA) # left eye left corner"""
            
            ratio_r = right_position(center_right, meshpoints[R_H_RIGHT], meshpoints[R_H_LEFT][0]) # getting right iris position and ratio
            ratio_l = left_position(center_left, meshpoints[L_H_RIGHT], meshpoints[L_H_LEFT][0]) # getting left iris position and ratio
            
            ratio_rtb = right_position(center_right, meshpoints[R_H_TOP], meshpoints[R_H_BOTTOM][0]) 
            ratio_ltb = left_position(center_left, meshpoints[L_H_TOP], meshpoints[L_H_BOTTOM][0])

            #cv2.putText(frame, "Left Ratio: {}; Left Ratio: {}".format(ratio_l,ratio_r), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # putting text on frame
            cv2.putText(frame, "Left Ratio: {} ; Right Ratio: {}".format(ratio_ltb,ratio_rtb), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # putting text on frame

            if ratio_l <0.34 : # looking at the left
                cv2.putText(frame, "You are Looking Left", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

            elif ratio_l > 0.54 : # looking at the right
                cv2.putText(frame, "You are Looking Right", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            
            #looking at to top
            elif ratio_ltb > 0.485 :
                cv2.putText(frame, "You are Looking Top", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

            #looking at to bottom
            elif ratio_ltb < 0.41 :
                cv2.putText(frame, "You are Looking Bottom", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2) 
            
            
            
        cv2.imshow('frame', frame) # showing frame
        key = cv2.waitKey(2) # waiting for key press
        if key==ord('q') or key ==ord('Q'): # if key is q or Q
            break 
    cv2.destroyAllWindows() # destroying all windows
    camera.release() # releasing camera
