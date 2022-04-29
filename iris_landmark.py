import argparse
import cv2 
import mediapipe as mp
import numpy as np
from custom.iris_lm_depth import from_landmarks_to_depth
from videosource import FileSource, WebcamSource
import time
import utils, math


# variables 
frame_counter =0
CEF_COUNTER =0

# constants
CLOSED_EYES_FRAME =3
FONTS =cv2.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

def landmarksDetection(img,img_draw, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img_draw, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# eye Ratio
def EyeRatio(img,img_draw, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    cv2.line(img_draw, rh_right, rh_left, utils.GREEN, 2)
    cv2.line(img_draw, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]


    # draw lines on left eyes 
    cv2.line(img_draw, lh_right, lh_left, utils.GREEN, 2)
    cv2.line(img_draw, lv_top, lv_bottom, utils.WHITE, 2)

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
   
    return ratio 


mp_face_mesh = mp.solutions.face_mesh

points_idx = [33, 133, 362, 263, 61, 291, 199]
points_idx = list(set(points_idx))
points_idx.sort()

left_eye_landmarks_id = np.array([33, 133])
right_eye_landmarks_id = np.array([362, 263])

dist_coeff = np.zeros((4, 1))

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
SMALL_CIRCLE_SIZE = 1
LARGE_CIRCLE_SIZE = 2


def main(inp):
    if inp is None:
        frame_height, frame_width = (720, 1280)
        source = WebcamSource(width=frame_width, height=frame_height)
    else:
        source = FileSource(inp)
        frame_width, frame_height = (int(i) for i in source.get_image_size())

    image_size = (frame_width, frame_height)

    # pseudo camera internals
    focal_length = frame_width

    landmarks = None
    smooth_left_depth = -1
    smooth_right_depth = -1
    smooth_factor = 0.1

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for idx, (frame, frame_rgb) in enumerate(source):
            frame_draw = frame.copy()
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            mesh_coords = 0
            ratio = 0

            if multi_face_landmarks:
                mesh_coords = landmarksDetection(frame,frame_draw, results, False)
                ratio = EyeRatio(frame,frame_draw, mesh_coords, RIGHT_EYE, LEFT_EYE)
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T

                (
                    left_depth,
                    left_iris_size,
                    left_iris_landmarks,
                    left_eye_contours,
                ) = from_landmarks_to_depth(
                    frame_rgb,
                    landmarks[:, left_eye_landmarks_id],
                    image_size,
                    is_right_eye=False,
                    focal_length=focal_length,
                )

                (
                    right_depth,
                    right_iris_size,
                    right_iris_landmarks,
                    right_eye_contours,
                ) = from_landmarks_to_depth(
                    frame_rgb,
                    landmarks[:, right_eye_landmarks_id],
                    image_size,
                    is_right_eye=True,
                    focal_length=focal_length,
                )

                if smooth_right_depth < 0:
                    smooth_right_depth = right_depth
                else:
                    smooth_right_depth = (
                        smooth_right_depth * (1 - smooth_factor)
                        + right_depth * smooth_factor
                    )

                if smooth_left_depth < 0:
                    smooth_left_depth = left_depth
                else:
                    smooth_left_depth = (
                        smooth_left_depth * (1 - smooth_factor)
                        + left_depth * smooth_factor
                    )

                #print(
                 #   f"depth in cm: {smooth_left_depth / 10:.2f}, {smooth_right_depth / 10:.2f}")
                #print(f"size: {left_iris_size:.2f}, {right_iris_size:.2f}")

            cv2.polylines(frame_draw,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv2.LINE_AA)
            cv2.polylines(frame_draw,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv2.LINE_AA)
            if landmarks is not None:

                # draw subset of facemesh
                for ii in points_idx:
                    pos = (np.array(image_size) * landmarks[:2, ii]).astype(np.int32)
                    #frame = cv2.circle(frame_draw, tuple(pos), LARGE_CIRCLE_SIZE, GREEN, -1)

                # draw eye contours
                eye_landmarks = np.concatenate(
                    [
                        right_eye_contours,
                        left_eye_contours,
                    ]
                )
                for landmark in eye_landmarks:
                    pos = (np.array(image_size) * landmark[:2]).astype(np.int32)
                    #frame = cv2.circle(frame_draw, tuple(pos), SMALL_CIRCLE_SIZE, RED, -1)

                # draw iris landmarks
                iris_landmarks = np.concatenate(
                    [
                        right_iris_landmarks,
                        left_iris_landmarks,
                    ]
                )
                for landmark in iris_landmarks:
                    pos = (np.array(image_size) * landmark[:2]).astype(np.int32)
                    frame_draw = cv2.circle(frame_draw, tuple(pos), SMALL_CIRCLE_SIZE, YELLOW, -1)

                # write depth values into frame
                depth_string = "{:.2f}cm, {:.2f}cm".format(
                    smooth_left_depth / 10, smooth_right_depth / 10
                )
                frame_draw = cv2.putText(
                    frame_draw,
                    depth_string,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    GREEN,
                    2,
                    cv2.LINE_AA,
                )

            source.show(frame_draw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose video file otherwise webcam is used."
    )
    parser.add_argument(
        "-i", metavar="path-to-file", type=str, help="Path to video file"
    )

    args = parser.parse_args()
    main(args.i)
