import cv2
import imutils
import numpy as np
import time
import argparse
from CentroidTracker import CentroidTracker
from TrackableObject import TrackableObject

def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    return bounding_box_cordinates

def draw_centroid(frame, personID, centroid, base_line):
    color = (0,0,255) if centroid[1] > int(base_line) else (0,255,0)
    cv2.putText(frame, f"ID {personID}", (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

def draw_frame(bounding_box_cordinates, frame, num_person, base_line):
    theme_color = (0, 255, 255)
    (H, W) = frame.shape[:2]

    # for check collapse of detection bounding box
    for x,y,w,h in bounding_box_cordinates:
        color = (0,0,255) if int(y+h/2) > int(base_line) else (0,255,0)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

    cv2.line(frame, (0, int(base_line)), (W, int(base_line)), theme_color, 2)
    cv2.putText(frame, f"amount of person: {num_person}", (10, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme_color, 2)

def humanDetector(args):

    video_path = args['video']
    skip_frames = int(args['skipFrames'])

    if str(args["camera"]) == 'true':
        print('[INFO] Opening Web Cam.')
        detectByCamera()
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, skip_frames)

def detectByPathVideo(path, skip_frames):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('[ERROR] Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')

    total_frame = 0
    num_person = 0
    bounding_boxes = []
    trackable_persons = {}

    while video.isOpened():
        check, frame = video.read()
        if not check:
            break
        frame = imutils.resize(frame , width=min(400,frame.shape[1]))
        (H, W) = frame.shape[:2]
        base_line = H*0.6

        if not check:
            print('[ERROR] Can not capture some frame from current video.')
            break

        if total_frame % skip_frames == 0:
            bounding_boxes = detect(frame)

        person_centroids = ct.update(bounding_boxes)
        for (personID, centroid) in person_centroids.items():
            to = trackable_persons.get(personID, None)
            if to is None:
                to = TrackableObject(personID, centroid)
            else:
                y_mean = np.mean([c[1] for c in to.centroids])
                direction = centroid[1] - y_mean
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < base_line:
                        num_person -= 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > base_line:
                        num_person += 1
                        to.counted = True
            trackable_persons[personID] = to
            draw_centroid(frame, personID, centroid, base_line)

        draw_frame(bounding_boxes, frame, num_person, base_line)
        cv2.imshow('output', frame)
            
        key = cv2.waitKey(1)
        if key== ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByCamera():   
    video = cv2.VideoCapture(0)
    time.sleep(2.0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        draw_bounding_box_to_frame(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default="D:\\Projects\\CSProject\\HumanDetectionServer\\videoplayback.mp4", help="path to Video File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-s", "--skipFrames", default=30, help="# of skip frames between detections")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    ct = CentroidTracker(maxDisappeared=40)

    args = argsParser()
    humanDetector(args)